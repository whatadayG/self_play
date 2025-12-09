#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv

# Make the scripts directory importable so we can load the rollout pipeline
try:
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from rollout_pipeline import run_rollout_pipeline, process_rollouts_post_generation
    from rollout_stats import RolloutStats
except Exception as e:
    print(f"Warning: Could not import rollout modules: {e}")
    run_rollout_pipeline = None  # Will be checked before use
    RolloutStats = None


def run_expert_iteration_training(
    *,
    gpu_string: str,
    round_dir: Path,
    train_parquet: Path,
    val_parquet: Path,
    current_model: str,
    max_len_arg: str,
    wandb_project: str,
    experiment_name: str,
    sft_epochs: int = 1,
    entropy_coeff: float = 0.0,  # Disabled to enable Liger fused CE (was 0.001)
    gradient_checkpointing: bool = True,
    wb = None,
) -> Path:
    """Run Expert Iteration training (filtered behavioral cloning with GRPO-style normalization) and return path to log file.

    Args:
        gpu_string: Comma-separated GPU IDs (e.g., "0,1,2,3")
        round_dir: Directory for this round
        train_parquet: Path to training parquet file
        val_parquet: Path to validation parquet file
        current_model: Model checkpoint to start from
        max_len_arg: Maximum sequence length
        wandb_project: W&B project name
        experiment_name: W&B experiment name
        sft_epochs: Number of training epochs (default: 1)
        entropy_coeff: Entropy regularization coefficient (default: 0.0, disabled to enable Liger fused CE)
        gradient_checkpointing: Enable gradient checkpointing (default: True)
        wb: Optional W&B run object from main loop

    Returns:
        Path to sft_train.log
    """
    # Parse GPU configuration
    gpu_list = [g.strip() for g in gpu_string.split(",") if g.strip()]
    num_gpus = len(gpu_list)

    # Use all available GPUs for SFT
    sft_visible = ",".join(gpu_list)
    nproc = num_gpus if num_gpus > 0 else 1

    # Determine batch sizes based on available GPUs
    # Note: Liger fused linear+CE kernel enables larger batch sizes (saves ~13GB memory)
    # Now works in both training AND validation (skip_logits=True forces fused kernel)
    if num_gpus >= 4:
        micro_batch_size_per_gpu = 8  # Increased from 2 (enabled by Liger fused CE)
        train_batch_size = 32  # Increased from 8 (8 per GPU * 4 GPUs)
        val_batch_size_per_gpu = 8  # Can use same as training (Liger works in validation now)
    else:
        micro_batch_size_per_gpu = 4  # Increased from 1 (for 2-GPU systems)
        train_batch_size = 8
        val_batch_size_per_gpu = 4  # Can use same as training (Liger works in validation now)

    print(f"Running Expert Iteration training (filtered BC) with {nproc} GPU(s)")
    print(f"  CUDA_VISIBLE_DEVICES={sft_visible}")
    print(f"  Batch config: micro_batch_size_per_gpu={micro_batch_size_per_gpu}, train_batch_size={train_batch_size}, val_batch_size_per_gpu={val_batch_size_per_gpu}")
    print(f"  Effective global batch size: {micro_batch_size_per_gpu * nproc} (per-GPU) * grad_accum_steps → {train_batch_size}")

    save_path = str(round_dir / "checkpoints")
    sft_cmd = [
        "bash", "scripts/sft_qwen/sft_qwen3.sh",
        str(nproc), save_path,
        f"data.train_files={str(train_parquet)}",
        f"data.val_files={str(val_parquet)}",
        f"data.micro_batch_size_per_gpu={micro_batch_size_per_gpu}",
        f"data.train_batch_size={train_batch_size}",
        f"data.val_batch_size_per_gpu={val_batch_size_per_gpu}",
        f"model.partial_pretrain={current_model}",
        "model.use_liger=true",  # EXPLICIT: Enable Liger fused linear+CE
        f"model.enable_gradient_checkpointing={'true' if gradient_checkpointing else 'false'}",
        f"trainer.total_epochs={sft_epochs}",
        "trainer.save_freq=900",
        "trainer.test_freq=33",
        # "+trainer.val_before_train=true",
        "trainer.checkpoint.save_contents=[\"hf_model\"]",
        f"trainer.entropy_coeff={entropy_coeff}",  # Must be 0 for Liger
        f"data.max_length={max_len_arg}",
        "data.custom_cls.path=verl/verl/utils/dataset/pretokenized_sft_dataset.py",
        "data.custom_cls.name=PreTokenizedSFTDataset",
        f"trainer.project_name={wandb_project}",
        f"trainer.experiment_name={experiment_name}",
        "optim.lr=5e-6",
        "optim.lr_scheduler=wsd",
        "+optim.stable_ratio=0.99",
        "+optim.min_lr_ratio=0.1",
    ]

    # Setup environment
    sft_env = os.environ.copy()
    test_bin = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin"
    if os.path.isdir(test_bin):
        sft_env["PATH"] = f"{test_bin}:{sft_env.get('PATH','')}"
    if sft_visible:
        sft_env["CUDA_VISIBLE_DEVICES"] = sft_visible
    sft_env["RDZV_PORT"] = "29500"

    # Configure W&B (separate run, same project)
    if wb is not None:
        sft_env["WANDB_PROJECT"] = wb.project
        print(f"Configuring SFT to log to W&B project: {wb.project}")

    # Run SFT training
    sft_log = round_dir / "sft_train.log"
    run_tee(sft_cmd, logfile=sft_log, env=sft_env)

    return sft_log


def run_ppo_grpo_training(
    *,
    gpu_string: str,
    round_dir: Path,
    train_parquet: Path,
    val_parquet: Path,
    current_model: str,
    max_len_arg: str,
    wandb_project: str,
    experiment_name: str,
    group_size: int = 8,
    ppo_epochs: int = 10,
    learning_rate: float = 1e-6,
    clip_ratio: float = 0.2,
    entropy_coeff: float = 0.01,
    gradient_checkpointing: bool = True,
    wb = None,
) -> Path:
    """Run PPO/GRPO training using FSDP SFT trainer with PPO loss enabled.

    Args:
        gpu_string: Comma-separated GPU IDs (e.g., "0,1,2,3")
        round_dir: Directory for this round
        train_parquet: Path to training parquet file
        val_parquet: Path to validation parquet file
        current_model: Model checkpoint to start from
        max_len_arg: Maximum sequence length
        wandb_project: W&B project name
        experiment_name: W&B experiment name
        group_size: GRPO group size (default: 8)
        ppo_epochs: Number of training epochs (default: 10)
        learning_rate: Learning rate (default: 1e-6)
        clip_ratio: PPO clip ratio (default: 0.2)
        entropy_coeff: Entropy coefficient (default: 0.01)
        gradient_checkpointing: Enable gradient checkpointing (default: True)
        wb: Optional W&B run object from main loop

    Returns:
        Path to ppo_train.log
    """
    # Parse GPU configuration
    gpu_list = [g.strip() for g in gpu_string.split(",") if g.strip()]
    num_gpus = len(gpu_list)

    # Use all available GPUs for PPO
    ppo_visible = ",".join(gpu_list)
    nproc = num_gpus if num_gpus > 0 else 1

    micro_batch_size_per_gpu = 4
    train_batch_size = 16
    val_batch_size_per_gpu = 4  # Can use same as training (Liger works in validation now)

    print(f"Running PPO/GRPO training with {nproc} GPU(s)")
    print(f"  CUDA_VISIBLE_DEVICES={ppo_visible}")
    print(f"  Batch config: micro_batch_size_per_gpu={micro_batch_size_per_gpu}, train_batch_size={train_batch_size}")
    print(f"  Training epochs: {ppo_epochs}, Learning rate: {learning_rate}")
    print(f"  Clip ratio: {clip_ratio}, Entropy coeff: {entropy_coeff}")

    save_path = str(round_dir / "checkpoints")

    # Use same SFT script but with PPO flags enabled
    ppo_cmd = [
        "bash", "scripts/sft_qwen/sft_qwen3.sh",
        str(nproc), save_path,
        f"data.train_files={str(train_parquet)}",
        f"data.val_files={str(val_parquet)}",
        f"data.micro_batch_size_per_gpu={micro_batch_size_per_gpu}",
        f"data.train_batch_size={train_batch_size}",
        f"model.partial_pretrain={current_model}",
        f"model.enable_gradient_checkpointing={'true' if gradient_checkpointing else 'false'}",
        f"trainer.total_epochs={ppo_epochs}",
        "trainer.save_freq=900",
        "trainer.test_freq=50",
        # "+trainer.val_before_train=true",
        "trainer.checkpoint.save_contents=[\"hf_model\"]",
        f"data.max_length={max_len_arg}",
        "data.custom_cls.path=verl/verl/utils/dataset/pretokenized_sft_dataset.py",
        "data.custom_cls.name=PreTokenizedSFTDataset",
        f"trainer.project_name={wandb_project}",
        f"trainer.experiment_name={experiment_name}",
        f"optim.lr={learning_rate}",
            # note, warmup and cosine annealing have been restored, since this is a multi-epoch operation, it makes far more sense than it used to
        # PPO-specific flags
        "trainer.use_ppo_loss=true",  # Enable PPO mode
        f"trainer.ppo_clip_ratio={clip_ratio}",
        f"trainer.entropy_coeff={entropy_coeff}",
        f"trainer.group_size={group_size}",
    ]

    # Setup environment
    ppo_env = os.environ.copy()
    test_bin = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin"
    if os.path.isdir(test_bin):
        ppo_env["PATH"] = f"{test_bin}:{ppo_env.get('PATH','')}"
    if ppo_visible:
        ppo_env["CUDA_VISIBLE_DEVICES"] = ppo_visible
    ppo_env["RDZV_PORT"] = "29500"

    # Configure W&B (separate run, same project)
    if wb is not None:
        ppo_env["WANDB_PROJECT"] = wb.project
        print(f"Configuring PPO to log to W&B project: {wb.project}")

    # Run PPO training
    ppo_log = round_dir / "ppo_train.log"
    run_tee(ppo_cmd, logfile=ppo_log, env=ppo_env)

    return ppo_log


def parse_sft_metrics(sft_log_path: Path) -> Optional[Dict[str, float]]:
    """Parse SFT training log to extract initial and final validation loss.

    Args:
        sft_log_path: Path to sft_train.log

    Returns:
        Dict with 'initial_val_loss' and 'final_val_loss', or None if parsing fails
    """
    try:
        with open(sft_log_path, "r") as f:
            lines = f.readlines()

        val_losses = []

        # Find all validation loss entries
        for line in lines:
            # Look for the final validation metrics line
            if "Final validation metrics:" in line and "'val/loss':" in line:
                try:
                    # Format: "Final validation metrics: {'val/loss': -0.0046041331804274966}"
                    loss_str = line.split("'val/loss':")[1].strip().rstrip("}")
                    loss = float(loss_str)
                    val_losses.append(loss)
                except (IndexError, ValueError):
                    continue

        if len(val_losses) >= 2:
            return {
                "initial_val_loss": val_losses[0],
                "final_val_loss": val_losses[-1],
            }
        elif len(val_losses) == 1:
            # Only one validation run
            return {
                "initial_val_loss": val_losses[0],
                "final_val_loss": val_losses[0],
            }
        return None

    except Exception as e:
        print(f"Warning: Failed to parse SFT log: {e}")
        return None


def log_rollout_to_wandb(wb, round_num: int, stats: "RolloutStats", sft_metrics: Optional[Dict[str, float]] = None) -> None:
    """Log rollout and SFT statistics to W&B.

    Args:
        wb: W&B run object (or None to skip logging)
        round_num: Current round number
        stats: RolloutStats object with all metrics
        sft_metrics: Optional dict with SFT training metrics (initial_loss, final_loss)
    """
    if wb is None:
        return

    metrics = {
        "round": round_num,
        # CRITICAL: Game performance metrics (actual rewards, not GRPO-normalized)
        "game/reward_mean": stats.game_reward_mean,
        "game/reward_std": stats.game_reward_std,
        "game/reward_p10": stats.game_reward_p10,
        "game/reward_p25": stats.game_reward_p25,
        "game/reward_p50": stats.game_reward_p50,
        "game/reward_p75": stats.game_reward_p75,
        "game/reward_p90": stats.game_reward_p90,
        "game/perfect_score_ratio": stats.perfect_score_ratio,
        "game/zero_reward_ratio": stats.zero_reward_ratio,
        "game/zero_reward_count": stats.zero_reward_count,

        # Conversation metrics
        "game/conversation_length_mean": stats.conversation_length_mean if stats.conversation_length_mean is not None else 0.0,

        # Filtering impact
        "filter/kept_ratio": stats.kept_sequences / stats.total_sequences,
        "filter/trim_threshold": stats.trim_threshold,
        "filter/perfect_score_ratio_before": stats.perfect_score_ratio,
        "filter/perfect_score_ratio_after": stats.perfect_score_ratio_after_trim,
        "filter/perfect_scores_lost": stats.perfect_score_ratio - stats.perfect_score_ratio_after_trim,

        # GRPO diagnostics
        "grpo/weight_pos_ratio": stats.grpo_weight_pos_ratio,
        "grpo/weight_neg_ratio": stats.grpo_weight_neg_ratio,
    }

    # Add SFT metrics if available
    if sft_metrics is not None:
        metrics["sft/initial_val_loss"] = sft_metrics["initial_val_loss"]
        metrics["sft/final_val_loss"] = sft_metrics["final_val_loss"]
        # Proportional delta instead of absolute (will fail if initial_val_loss is 0)
        metrics["sft/val_loss_delta_pct"] = (sft_metrics["initial_val_loss"] - sft_metrics["final_val_loss"]) / sft_metrics["initial_val_loss"]

    wb.log(metrics)


def find_most_recent_run(base_logs_dir: str = "/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo") -> Optional[Path]:
    """Find the most recent run directory based on timestamp."""
    base_path = Path(base_logs_dir)
    if not base_path.exists():
        return None

    # Find all timestamp directories
    run_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.replace("_", "").isdigit():
            run_dirs.append(item)

    if not run_dirs:
        return None

    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _extract_reward_series_from_df(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract the per-sample game-normalized reward series from a dataframe, if present.

    Tries the following in order:
    - column 'game_normalized_reward'
    - column 'normalized_reward'
    - parse 'game_info' JSON for 'game_normalized_reward' or 'normalized_reward'
    Returns None if unavailable.
    """
    try:
        if "game_normalized_reward" in df.columns:
            return df["game_normalized_reward"].astype(float).to_numpy()
        if "normalized_reward" in df.columns:
            return df["normalized_reward"].astype(float).to_numpy()
        if "game_info" in df.columns:
            def _from_game_info(x: Any) -> Optional[float]:
                try:
                    gi = json.loads(x) if isinstance(x, str) else (x if isinstance(x, dict) else None)
                    if isinstance(gi, dict):
                        if "game_normalized_reward" in gi:
                            return float(gi.get("game_normalized_reward"))
                        if "normalized_reward" in gi:
                            return float(gi.get("normalized_reward"))
                    return None
                except Exception:
                    return None
            vals = [v for v in df["game_info"].apply(_from_game_info).tolist() if isinstance(v, (int, float))]
            return np.array(vals, dtype=float) if len(vals) == len(df) else None
    except Exception:
        return None
    return None


def compute_reward_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics for a reward array.

    Returns a dict with keys: count, mean, std, min, max, p10, p20, ..., p90, zero_reward_ratio, zero_reward_count.
    """
    stats: Dict[str, Any] = {}
    if values is None or len(values) == 0:
        return {"count": 0}
    v = np.asarray(values, dtype=float)
    stats["count"] = int(v.size)
    stats["mean"] = float(np.mean(v))
    stats["std"] = float(np.std(v, ddof=0))
    stats["min"] = float(np.min(v))
    stats["max"] = float(np.max(v))
    for q in range(10, 100, 10):
        stats[f"p{q}"] = float(np.percentile(v, q))
    # Compute proportion and count of zero rewards
    stats["zero_reward_ratio"] = float(np.mean(v == 0.0))
    stats["zero_reward_count"] = int(np.sum(v == 0.0))
    return stats


def write_stats_file(stats: Dict[str, Any], out_path: Path) -> None:
    """Write stats to a small human-readable text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        if stats.get("count", 0) == 0:
            f.write("count: 0\n")
            f.write("mean: N/A\n")
            return
        f.write(f"count: {stats['count']}\n")
        f.write(f"mean: {stats['mean']:.6f}\n")
        f.write(f"std: {stats['std']:.6f}\n")
        f.write(f"min: {stats['min']:.6f}\n")
        for q in range(10, 100, 10):
            key = f"p{q}"
            if key in stats:
                f.write(f"{q}%: {stats[key]:.6f}\n")
        f.write(f"max: {stats['max']:.6f}\n")
        # Add failure ratio if available
        if "failure_ratio" in stats and stats["failure_ratio"] is not None:
            f.write(f"failure_ratio: {stats['failure_ratio']:.4f}\n")
        # Add zero reward metrics if available
        if "zero_reward_ratio" in stats and stats["zero_reward_ratio"] is not None:
            f.write(f"zero_reward_ratio: {stats['zero_reward_ratio']:.4f}\n")
        if "zero_reward_count" in stats and stats["zero_reward_count"] is not None:
            f.write(f"zero_reward_count: {stats['zero_reward_count']}\n")
        # Add conversation length statistics if available
        if "conversation_lengths" in stats and stats["conversation_lengths"] is not None:
            conv_stats = stats["conversation_lengths"]
            f.write(f"\n# Conversation Length Statistics\n")
            f.write(f"conv_length_median: {conv_stats.get('median', 0):.2f}\n")
            f.write(f"conv_length_p95: {conv_stats.get('p95', 0):.2f}\n")
            f.write(f"conv_length_max: {conv_stats.get('max', 0):.2f}\n")
            f.write(f"conv_length_mean: {conv_stats.get('mean', 0):.2f}\n")


def compute_and_save_stats(parquet_path: Path, out_path: Path) -> Dict[str, Any]:
    """Load a parquet, compute reward stats, and write to out_path. Returns stats dict."""
    try:
        df = pd.read_parquet(str(parquet_path))
        values = _extract_reward_series_from_df(df)
        stats = compute_reward_stats(values if values is not None else np.array([]))

        # Compute failure ratio and conversation lengths
        from transformers import AutoTokenizer
        failure_count = 0
        conversation_lengths = []

        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
            for idx, row in df.iterrows():
                # Count failures
                text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                if 'I need to think about this.' in text:
                    failure_count += 1

                # Extract conversation length (prefer game_info, fallback to token counting)
                try:
                    game_info = json.loads(row['game_info']) if isinstance(row['game_info'], str) else row['game_info']
                    if 'turn_count' in game_info:
                        conversation_lengths.append(game_info['turn_count'])
                    else:
                        # Fallback: count role markers in tokenized sequence
                        assistant_count = text.count('<|im_start|>assistant')
                        user_count = text.count('<|im_start|>user')
                        conversation_lengths.append(max(assistant_count, user_count))
                except Exception:
                    # Last resort: count from tokens
                    assistant_count = text.count('<|im_start|>assistant')
                    user_count = text.count('<|im_start|>user')
                    conversation_lengths.append(max(assistant_count, user_count))

            stats["failure_ratio"] = float(failure_count / len(df)) if len(df) > 0 else 0.0

            # Compute conversation length statistics
            if conversation_lengths:
                conv_arr = np.array(conversation_lengths)
                stats["conversation_lengths"] = {
                    'median': float(np.median(conv_arr)),
                    'p95': float(np.percentile(conv_arr, 95)),
                    'max': float(np.max(conv_arr)),
                    'mean': float(np.mean(conv_arr)),
                }
            else:
                stats["conversation_lengths"] = None

        except Exception as e:
            print(f"Warning: Could not compute failure rate and conversation lengths: {e}")
            stats["failure_ratio"] = None
            stats["conversation_lengths"] = None

        write_stats_file(stats, out_path)
        return stats
    except Exception as e:
        # best-effort write
        with open(out_path, "w") as f:
            f.write(f"Error computing stats: {e}\n")
        return {"count": 0}


def read_mean_from_stats(stats_path: Path) -> Optional[float]:
    """Parse mean value from an existing stats.txt file."""
    if not stats_path.exists():
        return None
    try:
        with open(stats_path, "r") as f:
            for line in f:
                if line.startswith("mean:"):
                    try:
                        return float(line.split(":", 1)[1].strip())
                    except Exception:
                        return None
    except Exception:
        return None
    return None


def branch_run(src_run: Path, dst_run: Path, rounds_to_link: int = 2) -> None:
    """Create a new run directory that branches from src_run, symlinking early rounds.

    For each of the first `rounds_to_link` rounds, symlink the existing parquet(s) and checkpoints
    into the new run directory and compute stats.txt for the trimmed parquet if available, else the raw parquet.
    """
    dst_run.mkdir(parents=True, exist_ok=True)
    for r in range(rounds_to_link):
        src_round = src_run / f"round_{r:03d}"
        dst_round = dst_run / f"round_{r:03d}"
        dst_round.mkdir(parents=True, exist_ok=True)
        if not src_round.exists():
            continue
        # Symlink key files if present
        for name in ["train.parquet", "train_trimmed.parquet"]:
            s = src_round / name
            d = dst_round / name
            if s.exists() and not d.exists():
                try:
                    os.symlink(os.fspath(s), os.fspath(d))
                except FileExistsError:
                    pass
        # Symlink checkpoints directory
        src_ckpt = src_round / "checkpoints"
        dst_ckpt = dst_round / "checkpoints"
        if src_ckpt.exists() and not dst_ckpt.exists():
            try:
                os.symlink(os.fspath(src_ckpt), os.fspath(dst_ckpt))
            except FileExistsError:
                pass
        # Compute stats for existing parquet(s) (prefer full rollout set)
        parquet = dst_round / "train.parquet"
        if not parquet.exists():
            parquet = dst_round / "train_trimmed.parquet"
        stats_path = dst_round / "stats.txt"
        if parquet.exists() and not stats_path.exists():
            compute_and_save_stats(parquet, stats_path)

def find_run_by_name(run_name: str, base_logs_dir: str = "/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo") -> Optional[Path]:
    """Find a run directory by name or timestamp."""
    base_path = Path(base_logs_dir)
    if not base_path.exists():
        return None
    
    # Try exact match first
    run_path = base_path / run_name
    if run_path.exists() and run_path.is_dir():
        return run_path
    
    # Try partial match for timestamp
    for item in base_path.iterdir():
        if item.is_dir() and run_name in item.name:
            return item
    
    return None


def analyze_run_state(save_root: Path, args=None) -> Tuple[int, str, Optional[str], Dict[str, Any]]:
    """
    Analyze the state of an existing run to determine where to resume.

    Args:
        save_root: Root directory of the run
        args: Optional args object to check KL settings for proper resume detection

    Returns:
        - current_round: The round to resume from
        - phase: 'rollout', 'training', 'finish_round'
        - current_model: Path to the current model to use
        - state_info: Additional state information
    """
    if not save_root.exists():
        return 0, 'rollout', None, {}

    # Find all round directories
    round_dirs = sorted([d for d in save_root.iterdir() if d.is_dir() and d.name.startswith("round_")])

    if not round_dirs:
        return 0, 'rollout', None, {}

    # Check the latest round directory
    latest_round_dir = round_dirs[-1]
    round_num = int(latest_round_dir.name.split("_")[1])

    state_info = {"round_dir": latest_round_dir}

    # Determine current model from previous rounds
    current_model = None
    if round_num > 0:
        prev_round_dir = save_root / f"round_{round_num-1:03d}"
        if prev_round_dir.exists():
            current_model = find_latest_model_from_round(prev_round_dir)

    # Check actual files to determine state (robust approach)
    train_parquet = latest_round_dir / "train.parquet"
    train_trimmed_parquet = latest_round_dir / "train_trimmed.parquet"
    checkpoints_dir = latest_round_dir / "checkpoints"
    sft_log = latest_round_dir / "sft_train.log"
    kl_metadata = latest_round_dir / "kl_metadata.json"

    # Check if SFT training has completed by looking for checkpoints
    has_checkpoints = checkpoints_dir.exists() and any(checkpoints_dir.glob("global_step_*"))
    has_sft_log = sft_log.exists() and sft_log.stat().st_size > 0

    if has_checkpoints and has_sft_log:
        # SFT is complete, finish the round
        if not current_model:
            current_model = find_latest_model_from_round(latest_round_dir)
        return round_num, 'finish_round', current_model, state_info
    elif train_parquet.exists() and train_trimmed_parquet.exists():
        # Check if KL settings have changed (requires reprocessing)
        if args and kl_metadata.exists():
            try:
                with open(kl_metadata, 'r') as f:
                    prev_kl_settings = json.load(f)

                # Extract current KL settings
                use_kl = getattr(args, 'use_kl', False)
                kl_coef = getattr(args, 'kl_coef', 0.001) if use_kl else None
                kl_method = getattr(args, 'kl_method', 'hf_dataparallel') if use_kl else None
                reference_model = getattr(args, 'reference_model', None) if use_kl else None

                current_kl_settings = {
                    "use_kl": use_kl,
                    "kl_coef": kl_coef,
                    "kl_method": kl_method,
                    "reference_model": reference_model,
                }

                # If KL settings changed, need to reprocess rollouts
                if prev_kl_settings != current_kl_settings:
                    print(f"KL settings changed, will reprocess rollouts for round {round_num}")
                    return round_num, 'rollout', current_model, state_info
            except Exception as e:
                print(f"Warning: Could not check KL metadata: {e}")

        # Rollout complete with correct KL settings, need to do SFT
        return round_num, 'training', current_model, state_info
    else:
        # Rollout incomplete or not started
        return round_num, 'rollout', current_model, state_info


def find_latest_model_from_round(round_dir: Path) -> Optional[str]:
    """Find the latest model from a completed round."""
    checkpoints_dir = round_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    
    # Find the latest checkpoint
    latest = None
    for p in checkpoints_dir.glob("global_step_*"):
        if p.is_dir():
            latest = p if (latest is None or p.stat().st_mtime > latest.stat().st_mtime) else latest
    
    if latest is None:
        return None
    
    # Prefer HuggingFace subdirectory
    hf_dir = latest / "huggingface"
    if hf_dir.is_dir():
        return str(hf_dir)
    else:
        return str(latest)


def find_latest_model_before_round(save_root: Path, upto_round_exclusive: int) -> Optional[str]:
    """Return the latest HF model path from the most recent completed round strictly before upto_round_exclusive."""
    for rn in range(upto_round_exclusive - 1, -1, -1):
        rd = save_root / f"round_{rn:03d}"
        if rd.exists():
            m = find_latest_model_from_round(rd)
            if m is not None:
                return m
    return None


def resume_rollout_generation(round_dir: Path, target_sequences: int, current_model: str, args) -> bool:
    """
    Resume rollout generation if needed.

    Returns True if rollouts were generated/completed, False if nothing was done.
    """
    train_parquet = round_dir / "train.parquet"
    train_trimmed_parquet = round_dir / "train_trimmed.parquet"

    # Check if we already have enough data
    existing_sequences = 0
    if train_trimmed_parquet.exists():
        try:
            df = pd.read_parquet(train_trimmed_parquet)
            existing_sequences = len(df)
            print(f"Found {existing_sequences} existing sequences in trimmed data")
        except Exception as e:
            print(f"Error reading existing trimmed data: {e}")
            existing_sequences = 0
    elif train_parquet.exists():
        try:
            df = pd.read_parquet(train_parquet)
            existing_sequences = len(df)
            print(f"Found {existing_sequences} existing sequences in raw data")
        except Exception as e:
            print(f"Error reading existing raw data: {e}")
            existing_sequences = 0

    needed_sequences = target_sequences - existing_sequences
    if needed_sequences <= 0:
        print(f"Already have {existing_sequences} sequences (target: {target_sequences}), skipping rollout generation")
        return True

    print(f"Need {needed_sequences} additional sequences (have: {existing_sequences}, target: {target_sequences})")

    # Generate additional rollouts using rollout_pipeline
    additional_parquet = round_dir / "train_additional.parquet"
    needed_games = max(1, needed_sequences // 2)  # two sequences per game

    # Use rollout pipeline - it handles dual-server logic automatically
    print(f"Calling rollout_pipeline to generate {needed_games} games...")

    # Create a temporary args-like object for the pipeline
    import argparse
    pipeline_args = argparse.Namespace(
        gpus=args.gpus,
        games_per_round=needed_games,  # Number of games to generate
        server_port=args.server_port,
        server_wait_seconds=args.server_wait_seconds,
        server_mem_fraction=args.server_mem_fraction,
        server_log_level=args.server_log_level,
        server_enable_torch_compile=args.server_enable_torch_compile,
        server_disable_cuda_graph=args.server_disable_cuda_graph,
        dual_server=args.dual_server,
    )

    # Generate via pipeline (automatically uses dual-server for 4 GPUs)
    try:
        # Import and call the function directly
        sys.path.insert(0, str(Path(__file__).parent))
        from rollout_pipeline import function_A_start_server_and_generate

        generated_parquet = function_A_start_server_and_generate(
            args=pipeline_args,
            round_dir=round_dir,
            current_model=current_model,
        )

        # Rename to additional_parquet for consistency
        if generated_parquet.exists() and generated_parquet != additional_parquet:
            generated_parquet.rename(additional_parquet)

    except Exception as e:
        print(f"Error during rollout generation: {e}")
        return False

    # Combine with existing data
    dfs_to_combine = []
    if train_parquet.exists():
        dfs_to_combine.append(pd.read_parquet(train_parquet))
    if additional_parquet.exists():
        dfs_to_combine.append(pd.read_parquet(additional_parquet))

    if dfs_to_combine:
        combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        combined_df.to_parquet(train_parquet)
        print(f"Combined data: {len(combined_df)} total sequences")

        # Clean up additional file
        if additional_parquet.exists():
            additional_parquet.unlink()

    return True


def cleanup_old_checkpoints(save_root: Path, current_round: int) -> None:
    """Delete checkpoint subdirectories from the previous 4 rounds to save disk space.

    Called when current_round is a multiple of 5. Deletes checkpoints from rounds
    (current_round - 4) through (current_round - 1).

    Args:
        save_root: Root directory of the run
        current_round: Current round number (should be a multiple of 5)
    """
    import shutil

    for offset in range(1, 5):
        round_to_clean = current_round - offset
        if round_to_clean < 0:
            continue

        round_dir = save_root / f"round_{round_to_clean:03d}"
        checkpoints_dir = round_dir / "checkpoints"

        if checkpoints_dir.exists() and checkpoints_dir.is_dir():
            try:
                shutil.rmtree(checkpoints_dir)
                print(f"  Deleted checkpoints directory: {checkpoints_dir}")
            except Exception as e:
                print(f"  Warning: Failed to delete {checkpoints_dir}: {e}")


def run_tee(cmd: list[str], logfile: Path, env=None):
    print("Running (tee):", " ".join(cmd), "->", str(logfile))
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", buffering=1) as lf:
        lf.write(f"===== CMD: {' '.join(cmd)}\n")
        lf.write(f"===== START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
        finally:
            proc.wait()
            lf.write(f"===== END (code {proc.returncode}): {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)


def main():
    # Load environment variables from .env file (e.g., WANDB_API_KEY)
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--games-per-round", type=int, default=256)
    ap.add_argument("--model-path", default="/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_3600_merged")
    ap.add_argument("--save-root", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--server-port", type=int, default=30001, help="Port for policy model server (default: 30001)")
    ap.add_argument("--server-wait-seconds", type=int, default=900, help="Max seconds to wait for SGLang server readiness")
    ap.add_argument("--wandb-project", default="offline-grpo")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    # New server control flags
    ap.add_argument("--server-mem-fraction", type=float, default=0.85, help="Static GPU memory fraction reserved by server (mem_fraction_static)")
    ap.add_argument("--server-log-level", type=str, default="debug", help="SGLang server log level: debug|info|warning|error")
    # Default: torch.compile enabled; provide flags to disable
    ap.add_argument("--server-enable-torch-compile", dest="server_enable_torch_compile", action="store_true", help="Enable torch.compile in SGLang server")
    ap.add_argument("--server-disable-cuda-graph", dest="server_disable_cuda_graph", action="store_true", help="Disable CUDA graph in SGLang server")
    ap.add_argument("--dual-server", action="store_true", help="Use dual-server mode (2×TP2) when 4 GPUs available (default: disabled)")
    ap.set_defaults(server_enable_torch_compile=False, server_disable_cuda_graph=False)
    
    # Run naming and resume arguments
    ap.add_argument("--run-name", default="", help="Name for this run (used as directory name). If empty, uses timestamp.")
    ap.add_argument("--resume", default="", help="Resume from specific run (timestamp/directory name). Must be explicitly specified.")
    ap.add_argument("--no-resume", action="store_true", help="Force start a new run (default behavior)")

    # Resampling/eval control (default: enabled; pass --disable-resample to turn off)
    ap.add_argument("--enable-resample", action="store_true", help="Enable resample logic based on rolling window of means (default: enabled)")
    ap.add_argument("--disable-resample", dest="enable_resample", action="store_false", help="Disable resample logic")
    ap.add_argument("--resample-window", type=int, default=2, help="How many previous rounds' means to compare against (default: 2)")
    ap.set_defaults(enable_resample=True)

    # Branching support
    ap.add_argument("--branch-from", type=str, default="", help="Existing run name/path to branch from (symlink early rounds)")
    ap.add_argument("--branch-rounds-to-link", type=int, default=2, help="How many early rounds to link into the new run")

    # KL divergence settings
    ap.add_argument("--use-kl", action="store_true", help="Enable KL divergence penalty in reward (disabled by default)")
    ap.add_argument("--reference-model", type=str, default=None, help="Path to reference model for KL divergence computation (required if --use-kl is set)")
    ap.add_argument("--kl-coef", type=float, default=0.001, help="KL divergence penalty coefficient (default: 0.001)")
    ap.add_argument("--kl-method", type=str, default="hf_dataparallel",
                    choices=["hf_dataparallel", "sglang"],
                    help="Method for computing reference model logprobs: 'hf_dataparallel' (default, uses data parallelism with HuggingFace) or 'sglang' (uses tensor parallelism with SGLang)")

    # Game termination settings
    ap.add_argument("--max-turns", type=int, default=10, help="Maximum number of turns per game (default: 10)")
    ap.add_argument("--max-retries-per-turn", type=int, default=3, help="Maximum retries per turn before terminating (default: 3)")

    # Data filtering settings (EXPERT ITERATION MODE ONLY - GRPO does not filter by quality)
    ap.add_argument("--filter-positive-only", action="store_true", default=True, help="[Expert Iteration only] Train only on positive GRPO-normalized examples (above group mean). Ignored in ppo-grpo mode. (default: enabled for expert-iteration)")
    ap.add_argument("--no-filter-positive-only", dest="filter_positive_only", action="store_false", help="[Expert Iteration only] Disable positive-only filtering, train on all examples. Ignored in ppo-grpo mode.")
    ap.add_argument("--filter-percentile", type=float, default=0.0, help="[Expert Iteration only] Keep only sequences above this percentile of GRPO-normalized rewards (0.0-1.0). Set to 0 to disable. Applies after positive-only filter if both are enabled. Ignored in ppo-grpo mode. Default: 0.0 (disabled)")

    # Training mode selection
    ap.add_argument("--training-mode", type=str, default="ppo-grpo",
                    choices=["expert-iteration", "ppo-grpo"],
                    help="Training algorithm: expert-iteration (filtered behavioral cloning) or ppo-grpo (policy gradient). Default: ppo-grpo")

    # SFT/Expert Iteration training settings (for expert-iteration mode)
    ap.add_argument("--sft-epochs", type=int, default=1, help="Number of epochs for Expert Iteration training (default: 1)")
    ap.add_argument("--sft-entropy-coeff", type=float, default=0.001, help="Entropy regularization coefficient for Expert Iteration training (default: 0.001, set to 0 to disable)")
    ap.add_argument("--sft-gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing during Expert Iteration training (default: enabled)")
    ap.add_argument("--sft-no-gradient-checkpointing", dest="sft_gradient_checkpointing", action="store_false", help="Disable gradient checkpointing during Expert Iteration training")

    # PPO/GRPO training settings (for ppo-grpo mode)
    ap.add_argument("--ppo-epochs", type=int, default=10, help="Number of epochs for PPO/GRPO training (default: 10)")
    ap.add_argument("--ppo-learning-rate", type=float, default=1e-6, help="Learning rate for PPO/GRPO training (default: 1e-6)")
    ap.add_argument("--ppo-batch-size-per-gpu", type=int, default=1, help="Batch size per GPU for PPO/GRPO training (default: 1 due to memory constraints)")
    ap.add_argument("--ppo-gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps for PPO/GRPO training (default: 8)")
    ap.add_argument("--ppo-clip-ratio", type=float, default=0.2, help="PPO clip ratio for policy gradient clipping (default: 0.2)")
    ap.add_argument("--ppo-entropy-coeff", type=float, default=0.0, help="Entropy coefficient for PPO/GRPO training (default: 0.0, disabled due to memory)")

    # Rollout generation settings
    ap.add_argument("--rollout-temperature", type=float, default=0.7, help="Temperature for sampling during rollout generation (default: 0.7)")

    # Asymmetric mode (trainee vs opponent) settings
    ap.add_argument("--shy-setup", action="store_true", help="Enable asymmetric training against fixed 'shy' opponent")
    ap.add_argument("--shy-opponent-model", type=str, default="checkpoints/sft_qwen3_8b/global_step_3600_merged/", help="Path to shy opponent model (only used with --shy-setup)")

    args = ap.parse_args()
    gpu_string = args.gpus
    gpu_list = [g for g in gpu_string.split(",") if g]
    tp = len(gpu_list)

    # Handle asymmetric mode (trainee vs shy opponent)
    if args.shy_setup:
        print("=== ASYMMETRIC MODE ENABLED ===")
        print(f"Training against fixed shy opponent: {args.shy_opponent_model}")
        args.opponent_model = args.shy_opponent_model
        # Asymmetric mode requires 4 GPUs for dual-server setup
        if len(gpu_list) != 4:
            raise ValueError(f"Asymmetric mode requires exactly 4 GPUs, but got {len(gpu_list)}")
    else:
        args.opponent_model = None  # Self-play mode

    # If branching is requested, force a new run (disable resume semantics)
    if args.branch_from:
        args.no_resume = True

    # Handle resume logic
    save_root = None
    current_model = args.model_path
    start_round = 0

    if args.resume and not args.no_resume:
        # Resume from specific run (must be explicitly specified)
        save_root = find_run_by_name(args.resume)
        if save_root is None:
            print(f"Error: Could not find run '{args.resume}'")
            print(f"Available runs in /home/nickatomlin/georgiazhou/self_play/logs/offline_grpo/:")
            base_path = Path("/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo")
            if base_path.exists():
                for item in sorted(base_path.iterdir()):
                    if item.is_dir():
                        print(f"  - {item.name}")
            sys.exit(1)
        else:
            print(f"Resuming from specified run: {save_root}")
    
    # Analyze existing run state if resuming
    if save_root is not None:
        current_round, phase, resume_model, state_info = analyze_run_state(save_root, args)
        if resume_model:
            current_model = resume_model
        start_round = current_round
        
        print(f"Resume analysis: round={current_round}, phase={phase}, model={current_model}")
        
        # Handle special resume phases
        if phase == 'rollout':
            round_dir = save_root / f"round_{current_round:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            # Try to resume rollout generation
            success = resume_rollout_generation(round_dir, args.games_per_round, current_model, args)
            if success:
                # For PPO/GRPO mode, disable quality-based filtering
                if args.training_mode == "ppo-grpo":
                    import copy
                    rollout_args = copy.copy(args)
                    rollout_args.filter_positive_only = False
                    rollout_args.filter_percentile = 0.0
                    print(f"PPO/GRPO mode: Disabling quality-based filtering during resume")
                else:
                    rollout_args = args

                # Process the rollouts (trim, stats, examples, etc.)
                rollout_stats = process_rollouts_post_generation(round_dir, save_root, rollout_args)
                # Update phase to training
                phase = 'training'
        
        if phase == 'training':
            round_dir = save_root / f"round_{current_round:03d}"

            # Run SFT training
            print("Resuming SFT training...")
            train_trimmed_parquet = round_dir / "train_trimmed.parquet"

            if not train_trimmed_parquet.exists():
                raise RuntimeError(f"Training data not found: {train_trimmed_parquet}")

            # Load max_length from data metadata (trim_threshold computed during rollout processing)
            data_metadata_path = round_dir / "data_metadata.json"
            if data_metadata_path.exists():
                with open(data_metadata_path, 'r') as f:
                    data_metadata = json.load(f)
                max_len_arg = str(data_metadata["trim_threshold"])
                print(f"Loaded trim_threshold={max_len_arg} from data metadata")
            else:
                # Fallback: compute from train data (legacy support)
                print("Warning: data_metadata.json not found, computing trim_threshold from train data")
                df = pd.read_parquet(train_trimmed_parquet)
                lengths = df["input_ids"].apply(lambda x: len(x))
                pct95 = int(np.percentile(lengths, 95))
                if pct95 > 5000:
                    pct95 = 5000
                max_len_arg = str(pct95)
                print(f"Computed trim_threshold={max_len_arg} from train split (may differ from original)")

            # Get validation parquet path
            val_trimmed_parquet = round_dir / "val_trimmed.parquet"
            if not val_trimmed_parquet.exists():
                # Fallback to using train as val if split wasn't done
                val_trimmed_parquet = train_trimmed_parquet

            # Run training (uses training mode from args)
            if args.training_mode == "expert-iteration":
                train_log = run_expert_iteration_training(
                    gpu_string=args.gpus,
                    round_dir=round_dir,
                    train_parquet=train_trimmed_parquet,
                    val_parquet=val_trimmed_parquet,
                    current_model=current_model,
                    max_len_arg=max_len_arg,
                    wandb_project=args.wandb_project,
                    experiment_name=f"{save_root.name}_round_{current_round}_resume",
                    sft_epochs=args.sft_epochs,
                    entropy_coeff=args.sft_entropy_coeff,
                    gradient_checkpointing=args.sft_gradient_checkpointing,
                    wb=None,  # No W&B context in resume
                )
            else:  # ppo-grpo mode
                train_log = run_ppo_grpo_training(
                    gpu_string=args.gpus,
                    round_dir=round_dir,
                    train_parquet=train_trimmed_parquet,
                    val_parquet=val_trimmed_parquet,
                    current_model=current_model,
                    max_len_arg=max_len_arg,
                    wandb_project=args.wandb_project,
                    experiment_name=f"{save_root.name}_round_{current_round}_resume",
                    group_size=8,  # Default GRPO group size
                    ppo_epochs=args.ppo_epochs,
                    learning_rate=args.ppo_learning_rate,
                    clip_ratio=args.ppo_clip_ratio,
                    entropy_coeff=args.ppo_entropy_coeff,
                    gradient_checkpointing=args.sft_gradient_checkpointing,
                    wb=None,  # No W&B context in resume
                )
            
            # Log completion and update model
            save_path = str(round_dir / "checkpoints")
            log_file = round_dir / "progress.log"
            with open(log_file, "a") as lf:
                lf.write(json.dumps({
                    "event": "sft_done",
                    "timestamp": time.time(),
                    "save_path": save_path
                }) + "\n")
            
            # Update current model
            latest_model = find_latest_model_from_round(round_dir)
            if latest_model:
                current_model = latest_model
            
            # Mark round complete
            with open(log_file, "a") as lf:
                lf.write(json.dumps({
                    "event": "round_complete",
                    "timestamp": time.time(),
                    "next_model": current_model
                }) + "\n")
            
            # Move to next round
            start_round = current_round + 1
        
        elif phase == 'finish_round':
            round_dir = save_root / f"round_{current_round:03d}"
            
            # Just update the model and mark round complete
            latest_model = find_latest_model_from_round(round_dir)
            if latest_model:
                current_model = latest_model
            
            log_file = round_dir / "progress.log"
            with open(log_file, "a") as lf:
                lf.write(json.dumps({
                    "event": "round_complete",
                    "timestamp": time.time(),
                    "next_model": current_model
                }) + "\n")
            
            start_round = current_round + 1
    
    # If no existing run or starting fresh (also handle branching)
    if save_root is None:
        if args.save_root:
            # Explicit save_root overrides everything
            save_root = Path(args.save_root)
        elif args.run_name:
            # Use provided run name
            save_root = Path(f"/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo/{args.run_name}")
        else:
            # Default to timestamp
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_root = Path(f"/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo/{ts}")
        save_root.mkdir(parents=True, exist_ok=True)
        print(f"Starting new run: {save_root}")

        if args.branch_from:
            src = find_run_by_name(args.branch_from) or Path(args.branch_from)
            if src.exists():
                print(f"Branching from {src} into {save_root} (link first {args.branch_rounds_to_link} rounds)")
                branch_run(src_run=src, dst_run=save_root, rounds_to_link=args.branch_rounds_to_link)
                # Set start_round to the next round after linked ones
                start_round = max(start_round, args.branch_rounds_to_link)
                # Prefer using the last linked round's model as starting model
                if start_round > 0:
                    prev_model = find_latest_model_from_round(save_root / f"round_{start_round-1:03d}")
                    if prev_model:
                        current_model = prev_model
            else:
                print(f"Warning: --branch-from path not found: {src}")

    # Run the main loop
    run_offline_grpo_loop(args, save_root, current_model, start_round)


def run_offline_grpo_loop(args, save_root: Path, current_model: str, start_round: int = 0):
    """Main loop for offline GRPO training."""
    gpu_string = args.gpus
    gpu_list = [g for g in gpu_string.split(",") if g]
    tp = len(gpu_list)

    # Initialize wandb run
    wb = None
    try:
        import wandb as _wandb
        run_name = args.wandb_run_name or save_root.name
        wb = _wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, reinit=False)
        if wb is not None:
            wb.config.update({
                "rounds": args.rounds,
                "games_per_round": args.games_per_round,
                "gpus": gpu_string,
                "tp": tp,
                "model_start": current_model,
                "server_mem_fraction": args.server_mem_fraction,
                "server_log_level": args.server_log_level,
                "server_enable_torch_compile": args.server_enable_torch_compile,
                "resumed_from_round": start_round,
            }, allow_val_change=True)
    except Exception:
        wb = None

    for r in range(start_round, args.rounds):
        print(f"=== Round {r} ===")
        round_dir = save_root / f"round_{r:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        log_file = round_dir / "progress.log"

        # For PPO/GRPO mode, disable quality-based filtering (train on all samples)
        # Expert Iteration uses filtering to select high-quality examples
        if args.training_mode == "ppo-grpo":
            # Create a copy of args with filtering disabled for PPO/GRPO mode
            import copy
            rollout_args = copy.copy(args)
            rollout_args.filter_positive_only = False
            rollout_args.filter_percentile = 0.0
            print(f"PPO/GRPO mode: Disabling quality-based filtering (will train on all samples)")
        else:
            # Expert Iteration: use args as-is (respects user's filter settings)
            rollout_args = args
            if args.filter_positive_only:
                print(f"Expert Iteration mode: Positive-only filtering enabled")
            if args.filter_percentile > 0.0:
                print(f"Expert Iteration mode: Percentile filtering at {args.filter_percentile*100:.0f}th percentile")

        # Run rollout pipeline
        rollout_stats = run_rollout_pipeline(
            args=rollout_args,
            save_root=save_root,
            round_dir=round_dir,
            current_model=current_model,
        )

        # Extract values needed for SFT and resampling logic
        train_parquet_for_sft = rollout_stats.trimmed_parquet
        max_len_arg = str(rollout_stats.trim_threshold)

        # Log to W&B
        log_rollout_to_wandb(wb, r, rollout_stats)

        # Prepare stats dict for resampling logic (backward compatibility)
        stats = {"mean": rollout_stats.game_reward_mean, "count": rollout_stats.total_sequences}

        # Optionally trigger resample: compare mean vs previous N means
        if args.enable_resample:
            window = max(1, int(args.resample_window))
            # current mean refers to performance of previous round's model
            current_mean = None
            try:
                current_mean = stats.get("mean", None)
            except Exception:
                pass

            if current_mean is not None and r >= 1:
                # Read previous window means from stats.txt of previous rounds
                prev_means: list[float] = []
                for k in range(1, window + 1):
                    prev_round = r - k
                    if prev_round < 0:
                        break
                    prev_stats_path = save_root / f"round_{prev_round:03d}" / "stats.txt"
                    m = read_mean_from_stats(prev_stats_path)
                    if m is not None:
                        prev_means.append(m)

                if len(prev_means) == window and all(current_mean < pm for pm in prev_means):
                    print(f"[resample] Current mean {current_mean:.6f} is lower than all of last {window} means {prev_means}. Rolling back previous round and aborting current.")
                    # Write a simple text line to master run directory (not round dir)
                    try:
                        master_log = save_root / "resample.log"
                        ts = time.strftime('%Y-%m-%d %H:%M:%S')
                        with open(master_log, "a") as mf:
                            mf.write(f"[{ts}] round={r} current_mean={current_mean:.6f} window={window} prev_means={prev_means} -> rollback r-1 and redo.\n")
                    except Exception:
                        pass
                    # Remove current round directory
                    try:
                        import shutil
                        shutil.rmtree(round_dir, ignore_errors=True)
                    except Exception:
                        pass
                    # Remove previous round directory (redo r-1), and set model to round r-2
                    prev_round_dir = save_root / f"round_{r-1:03d}"
                    try:
                        import shutil
                        shutil.rmtree(prev_round_dir, ignore_errors=True)
                    except Exception:
                        pass
                    # Determine model from before previous round
                    fallback_model = find_latest_model_before_round(save_root, upto_round_exclusive=r-1)
                    new_current_model = fallback_model if fallback_model is not None else args.model_path
                    # Adjust loop to redo previous round
                    start_round = max(0, r - 1)
                    # Update wandb if present
                    if wb is not None:
                        wb.log({"resample/triggered": 1, "resample/target_round": start_round})
                    # Restart loop from (r-1)
                    return run_offline_grpo_loop(args, save_root, new_current_model, start_round)
        
        # Run training (mode selected by --training-mode flag)
        print(f"Running {args.training_mode} training...")
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "training_start",
                "training_mode": args.training_mode,
                "timestamp": time.time(),
                "round": r,
            }) + "\n")

        if args.training_mode == "expert-iteration":
            train_log = run_expert_iteration_training(
                gpu_string=args.gpus,
                round_dir=round_dir,
                train_parquet=train_parquet_for_sft,
                val_parquet=rollout_stats.val_parquet,
                current_model=current_model,
                max_len_arg=max_len_arg,
                wandb_project=args.wandb_project,
                experiment_name=f"{save_root.name}_round_{r}",
                sft_epochs=args.sft_epochs,
                entropy_coeff=args.sft_entropy_coeff,
                gradient_checkpointing=args.sft_gradient_checkpointing,
                wb=wb,
            )
        else:  # ppo-grpo mode
            train_log = run_ppo_grpo_training(
                gpu_string=args.gpus,
                round_dir=round_dir,
                train_parquet=train_parquet_for_sft,
                val_parquet=rollout_stats.val_parquet,
                current_model=current_model,
                max_len_arg=max_len_arg,
                wandb_project=args.wandb_project,
                experiment_name=f"{save_root.name}_round_{r}",
                group_size=8,  # Default GRPO group size
                ppo_epochs=args.ppo_epochs,
                learning_rate=args.ppo_learning_rate,
                clip_ratio=args.ppo_clip_ratio,
                entropy_coeff=args.ppo_entropy_coeff,
                gradient_checkpointing=args.sft_gradient_checkpointing,
                wb=wb,
            )

        # Parse training metrics and log to W&B
        # Note: parse_sft_metrics works for both SFT and PPO logs (looks for validation loss)
        training_metrics = parse_sft_metrics(train_log)
        if training_metrics is not None:
            print(f"Training metrics: initial_val_loss={training_metrics['initial_val_loss']:.6f}, final_val_loss={training_metrics['final_val_loss']:.6f}")
            # Update W&B with training metrics
            log_rollout_to_wandb(wb, r, rollout_stats, training_metrics)

        save_path = str(round_dir / "checkpoints")
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "training_done",
                "training_mode": args.training_mode,
                "timestamp": time.time(),
                "save_path": save_path,
                "training_metrics": training_metrics,
            }) + "\n")

        # Update model path for next round
        latest = None
        for p in Path(save_path).glob("global_step_*"):
            latest = p if (latest is None or p.stat().st_mtime > latest.stat().st_mtime) else latest
        
        if latest is None:
            print("Warning: no SFT checkpoint found; keeping current model for next round.")
        else:
            hf_dir = latest / "huggingface"
            if hf_dir.is_dir():
                current_model = str(hf_dir)
            else:
                print(f"Warning: HF directory not found under {latest}, falling back to checkpoint root")
                current_model = str(latest)
        
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "round_complete",
                "timestamp": time.time(),
                "next_model": current_model
            }) + "\n")

        # Cleanup old checkpoints when round is a multiple of 5
        if r > 0 and r % 5 == 0:
            cleanup_old_checkpoints(save_root, r)
            print(f"Cleaned up checkpoints from rounds {r-4} to {r-1}")

    print("Offline GRPO loop completed.")


if __name__ == "__main__":
    main()
