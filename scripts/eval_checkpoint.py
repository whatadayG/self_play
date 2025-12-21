#!/usr/bin/env python3
"""
Quick evaluation script for checkpoints.

Generates a small number of games (default 128) with each game played only once
(no GRPO grouping), computes statistics, and saves to small_eval_stats.txt.

This is much faster than full GRPO rollout generation which plays each game 8 times.
"""

import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import utilities from rollout_pipeline
scripts_dir = str(Path(__file__).parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from rollout_pipeline import (
    _start_sglang_server,
    _wait_for_sglang,
    _kill_process_tree,
    _extract_reward_series_from_df,
)


def compute_eval_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute summary statistics for evaluation rewards.

    Args:
        values: Array of game-normalized rewards

    Returns:
        Dict with count, mean, std, percentiles, perfect_score_ratio
    """
    stats = {}
    if values is None or len(values) == 0:
        return {"count": 0}

    v = np.asarray(values, dtype=float)
    stats["count"] = int(v.size)
    stats["mean"] = float(np.mean(v))
    stats["std"] = float(np.std(v, ddof=0))
    stats["min"] = float(np.min(v))
    stats["max"] = float(np.max(v))

    # Percentiles
    for q in range(10, 100, 10):
        stats[f"p{q}"] = float(np.percentile(v, q))

    # Perfect score ratio (reward = 1.0)
    stats["perfect_score_ratio"] = float(np.mean(v >= 1.0))

    return stats


def write_eval_stats(stats: Dict[str, Any], out_path: Path) -> None:
    """Write evaluation stats to file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("=== Quick Evaluation Statistics ===\n\n")

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
        f.write(f"perfect_score_ratio: {stats['perfect_score_ratio']:.4f}\n")


def generate_eval_rollouts(
    model_path: str,
    output_path: Path,
    num_games: int = 128,
    gpus: str = "0,1,2,3",
    server_port: int = 8000,
    server_wait_seconds: int = 600,
    server_mem_fraction: float = 0.85,
    server_log_level: str = "info",
    enable_torch_compile: bool = False,
    disable_cuda_graph: bool = False,
) -> Path:
    """Generate evaluation rollouts using SGLang server.

    Args:
        model_path: Path to model checkpoint
        output_path: Where to save the parquet file
        num_games: Number of games to generate (default: 128)
        gpus: Comma-separated GPU IDs
        server_port: SGLang server port
        server_wait_seconds: Max seconds to wait for server startup
        server_mem_fraction: GPU memory fraction for server
        server_log_level: Server log level
        enable_torch_compile: Enable torch.compile in server
        disable_cuda_graph: Disable CUDA graph in server

    Returns:
        Path to generated parquet file
    """
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    num_gpus = len(gpu_list)
    tp = num_gpus if num_gpus > 0 else 1

    print(f"\n{'='*60}")
    print(f"Starting Quick Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Games: {num_games}")
    print(f"GPUs: {gpus} (TP={tp})")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Start SGLang server
    print("Starting SGLang server...")
    server_log = output_path.parent / "eval_sglang_server.log"
    server_proc = _start_sglang_server(
        model_path=model_path,
        gpus=gpus,
        tp=tp,
        mem_util=server_mem_fraction,
        port=server_port,
        enable_torch_compile=enable_torch_compile,
        disable_cuda_graph=disable_cuda_graph,
        log_level=server_log_level,
        log_file=server_log,
    )

    server_url = f"http://127.0.0.1:{server_port}"

    try:
        # Wait for server to be ready
        _wait_for_sglang(server_url, server_proc, timeout_sec=server_wait_seconds, interval_sec=5)

        # Run generation
        print(f"\nGenerating {num_games} evaluation games...")
        test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
        python_bin = test_py if os.path.exists(test_py) else sys.executable

        gen_cmd = [
            python_bin,
            "scripts/generate_rollouts.py",
            "--server-url", server_url,
            "--model-id", model_path,
            "--out", str(output_path),
            "--num-games", str(num_games),
            "--max-new-tokens", "8192",
            "--group-size", "1",  # KEY DIFFERENCE: Only play each game once
            "--temperature", "0.7",
            "--top-p", "0.9",
            "--max-model-len", "32768",
        ]

        start_time = time.time()
        result = subprocess.run(gen_cmd, check=True)
        elapsed = time.time() - start_time

        print(f"\nGeneration completed in {elapsed:.1f}s ({num_games/elapsed:.1f} games/s)")

        return output_path

    finally:
        print("\nStopping SGLang server...")
        _kill_process_tree(server_proc)
        try:
            subprocess.run(["pkill", "-f", "sglang"], check=False)
        except Exception:
            pass


def find_checkpoint_in_round(round_dir: Path) -> Path:
    """Find the most recent checkpoint in a round directory.

    Args:
        round_dir: Path to round directory (e.g., round_001)

    Returns:
        Path to checkpoint (HuggingFace format if available)

    Raises:
        FileNotFoundError: If no checkpoint found
    """
    checkpoints_dir = round_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {round_dir}")

    # Find all global_step_* directories
    checkpoints = list(checkpoints_dir.glob("global_step_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    # Sort by modification time, get most recent
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)

    # Prefer HuggingFace subdirectory if it exists
    hf_dir = latest / "huggingface"
    if hf_dir.exists() and hf_dir.is_dir():
        return hf_dir

    return latest


def find_most_recent_grpo_run(base_dir: str = "/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo") -> Path:
    """Find the most recent GRPO run directory.

    Args:
        base_dir: Base directory containing GRPO runs

    Returns:
        Path to most recent run directory

    Raises:
        FileNotFoundError: If no runs found
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"GRPO logs directory not found: {base_dir}")

    # Find all timestamp directories
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.replace("_", "").isdigit()]

    if not run_dirs:
        raise FileNotFoundError(f"No GRPO runs found in {base_dir}")

    # Sort by modification time, get most recent
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def find_most_recent_round_with_checkpoint(run_dir: Path) -> Path:
    """Find the most recent round with a checkpoint in a GRPO run.

    Args:
        run_dir: Path to GRPO run directory

    Returns:
        Path to round directory with checkpoint

    Raises:
        FileNotFoundError: If no rounds with checkpoints found
    """
    # Find all round directories
    round_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])

    if not round_dirs:
        raise FileNotFoundError(f"No round directories found in {run_dir}")

    # Check rounds in reverse order (most recent first)
    for round_dir in reversed(round_dirs):
        checkpoints_dir = round_dir / "checkpoints"
        if checkpoints_dir.exists() and list(checkpoints_dir.glob("global_step_*")):
            return round_dir

    raise FileNotFoundError(f"No rounds with checkpoints found in {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation of a checkpoint using dialop self-play games"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint to evaluate (optional if --round-path provided)"
    )
    parser.add_argument(
        "--round-path",
        type=str,
        default=None,
        help="Path to round directory (e.g., logs/offline_grpo/20251008_082305/round_001). "
             "Will auto-find checkpoint and save results here."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as round-path, or logs/eval/<model_name>_<timestamp>)"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=128,
        help="Number of games to generate (default: 128)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="Comma-separated GPU IDs (default: 0,1,2,3)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="SGLang server port (default: 8000)"
    )
    parser.add_argument(
        "--server-wait-seconds",
        type=int,
        default=600,
        help="Max seconds to wait for server startup (default: 600)"
    )
    parser.add_argument(
        "--server-mem-fraction",
        type=float,
        default=0.85,
        help="GPU memory fraction for server (default: 0.85)"
    )
    parser.add_argument(
        "--server-log-level",
        type=str,
        default="info",
        help="Server log level (default: info)"
    )
    parser.add_argument(
        "--server-enable-torch-compile",
        action="store_true",
        help="Enable torch.compile in server"
    )
    parser.add_argument(
        "--server-disable-cuda-graph",
        action="store_true",
        help="Disable CUDA graph in server"
    )

    args = parser.parse_args()

    # Path resolution logic
    model_path = None
    output_dir = None

    if args.round_path:
        # User specified a round path - find checkpoint there and use as output dir
        round_dir = Path(args.round_path).expanduser().resolve()
        if not round_dir.exists():
            print(f"Error: Round directory not found: {round_dir}")
            sys.exit(1)

        print(f"Using round directory: {round_dir}")

        try:
            checkpoint = find_checkpoint_in_round(round_dir)
            model_path = str(checkpoint)
            print(f"Found checkpoint: {checkpoint}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Use round directory as output unless explicitly overridden
        output_dir = Path(args.output_dir) if args.output_dir else round_dir

    elif args.model_path:
        # User specified model path explicitly
        model_path = args.model_path

        # Use custom output dir if specified, otherwise default
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = Path(model_path).name
            output_dir = Path(f"logs/eval/{model_name}_{timestamp}")

    else:
        # No paths specified - auto-find most recent checkpoint
        print("No paths specified, auto-finding most recent GRPO checkpoint...")

        try:
            run_dir = find_most_recent_grpo_run()
            print(f"Found most recent GRPO run: {run_dir}")

            round_dir = find_most_recent_round_with_checkpoint(run_dir)
            print(f"Found most recent round with checkpoint: {round_dir}")

            checkpoint = find_checkpoint_in_round(round_dir)
            model_path = str(checkpoint)
            print(f"Found checkpoint: {checkpoint}")

            # Use round directory as output unless explicitly overridden
            output_dir = Path(args.output_dir) if args.output_dir else round_dir

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease specify --model-path or --round-path explicitly.")
            sys.exit(1)

    # Ensure we have a model path at this point
    if not model_path:
        print("Error: Could not determine model path")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluation configuration:")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Games: {args.num_games}")
    print()

    # Save evaluation config
    config_path = output_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "num_games": args.num_games,
            "gpus": args.gpus,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    # Generate rollouts
    parquet_path = output_dir / "eval.parquet"
    generate_eval_rollouts(
        model_path=model_path,
        output_path=parquet_path,
        num_games=args.num_games,
        gpus=args.gpus,
        server_port=args.server_port,
        server_wait_seconds=args.server_wait_seconds,
        server_mem_fraction=args.server_mem_fraction,
        server_log_level=args.server_log_level,
        enable_torch_compile=args.server_enable_torch_compile,
        disable_cuda_graph=args.server_disable_cuda_graph,
    )

    # Compute statistics
    print("\nComputing statistics...")
    df = pd.read_parquet(str(parquet_path))

    # Extract rewards
    rewards = _extract_reward_series_from_df(df)
    if rewards is None or len(rewards) == 0:
        print("Error: No rewards found in generated data")
        sys.exit(1)

    # Compute stats
    stats = compute_eval_stats(rewards)

    # Write stats file
    stats_path = output_dir / "small_eval_stats.txt"
    write_eval_stats(stats, stats_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Games: {stats['count']}")
    print(f"Mean reward: {stats['mean']:.4f}")
    print(f"Std reward: {stats['std']:.4f}")
    print(f"Perfect score ratio: {stats['perfect_score_ratio']:.2%}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {parquet_path.name}")
    print(f"  - {stats_path.name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
