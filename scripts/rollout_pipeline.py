#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import requests

# Import RolloutStats dataclass
scripts_dir = str(Path(__file__).parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
from rollout_stats import RolloutStats


def _kill_process_tree(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


def _run(cmd: list[str], env=None) -> int:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=True)
    return result.returncode


def _start_sglang_server(
    model_path: str,
    gpus: str,
    tp: int,
    mem_util: float,
    port: int,
    enable_torch_compile: bool,
    disable_cuda_graph: bool,
    log_level: str,
) -> subprocess.Popen:
    num_gpus = len([g.strip() for g in gpus.split(",") if g.strip()])
    if tp > num_gpus:
        print(f"[rollout_pipeline] WARNING: Requested TP={tp} but only {num_gpus} GPUs available. Adjusting TP to {num_gpus}", flush=True)
        tp = num_gpus

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("NCCL_P2P_LEVEL", "NVL")

    print(
        f"[rollout_pipeline] Preparing to launch SGLang server: model_path={model_path}, gpus={gpus}, tp={tp}, mem_util={mem_util}, port={port}, torch_compile={enable_torch_compile}, cuda_graph={'disabled' if disable_cuda_graph else 'enabled'}, log_level={log_level}",
        flush=True,
    )

    test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
    python_bin = test_py if os.path.exists(test_py) else sys.executable
    print(f"[rollout_pipeline] Using Python interpreter: {python_bin}", flush=True)

    cmd = [
        python_bin,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--tp",
        "2",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(mem_util),
        "--dtype",
        "bfloat16",
        "--log-level",
        str(log_level),
        "--schedule-conservativeness",
        "1.0",
        "--chunked-prefill-size",
        "512",
        "--schedule-policy",
        "lpm",
    ]
    if enable_torch_compile:
        cmd.append("--enable-torch-compile")
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    print("[rollout_pipeline] Launch command:", " ".join(cmd), flush=True)
    print(f"[rollout_pipeline] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    print(f"[rollout_pipeline] NCCL_P2P_LEVEL={env.get('NCCL_P2P_LEVEL','')}", flush=True)

    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, text=True)
    print(f"[rollout_pipeline] SGLang server process started with PID={proc.pid}", flush=True)
    return proc


def _wait_for_sglang(server_url: str, server_proc: subprocess.Popen, timeout_sec: int = 900, interval_sec: int = 5) -> None:
    base = server_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    deadline = time.time() + timeout_sec
    last_err = None
    start_time = time.time()

    import socket

    def _is_port_listening(h: str, p: int) -> bool:
        try:
            with socket.create_connection((h, p), timeout=0.2):
                return True
        except Exception:
            return False

    host = "127.0.0.1"
    port = 80
    try:
        from urllib.parse import urlparse

        parsed = urlparse(server_url)
        host = parsed.hostname or host
        port = parsed.port or 80
    except Exception:
        pass

    while time.time() < deadline:
        if server_proc.poll() is not None:
            print(f"[rollout_pipeline] Server process exited early with code {server_proc.returncode}", flush=True)
            raise RuntimeError(f"SGLang server process died with exit code {server_proc.returncode}")

        try:
            r = requests.get(f"{base}/models", timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("data"):
                elapsed = int(time.time() - start_time)
                print(f"[rollout_pipeline] SGLang server is ready after {elapsed}s.", flush=True)
                return
            if isinstance(data, list) and len(data) > 0:
                elapsed = int(time.time() - start_time)
                print(f"[rollout_pipeline] SGLang server is ready after {elapsed}s.", flush=True)
                return
        except Exception as e:
            last_err = e
            elapsed = int(time.time() - start_time)
            port_listening = _is_port_listening(host, port)
            print(f"[rollout_pipeline] Still waiting ({elapsed}s). Last HTTP error: {repr(last_err)}. proc_alive={server_proc.poll() is None}, port_listening={port_listening}", flush=True)

        time.sleep(interval_sec)

    print(f"[rollout_pipeline] Timeout after {timeout_sec}s waiting for SGLang server at {base}. Last error: {last_err}", flush=True)
    raise TimeoutError(f"SGLang server at {base} did not become ready within {timeout_sec}s. Last error: {last_err}")


def function_A_start_server_and_generate(
    *,
    args,
    round_dir: Path,
    current_model: str,
) -> Path:
    """Start SGLang server, generate rollouts via scripts/generate_rollouts.py, log events, stop server.

    Returns path to the raw parquet (train.parquet).
    """
    gpu_string = args.gpus
    tp = len([g for g in gpu_string.split(",") if g])

    print("Starting SGLang server...")
    server_proc = _start_sglang_server(
        model_path=current_model,
        gpus=gpu_string,
        tp=tp,
        mem_util=args.server_mem_fraction,
        port=args.server_port,
        enable_torch_compile=args.server_enable_torch_compile,
        disable_cuda_graph=args.server_disable_cuda_graph,
        log_level=args.server_log_level,
    )

    server_url = f"http://127.0.0.1:{args.server_port}"
    log_file = round_dir / "progress.log"
    with open(log_file, "a") as lf:
        lf.write(json.dumps({
            "event": "server_started",
            "timestamp": time.time(),
            "model": current_model,
            "gpus": gpu_string,
            "tp": tp,
        }) + "\n")

    try:
        _wait_for_sglang(server_url, server_proc, timeout_sec=args.server_wait_seconds, interval_sec=5)

        out_parquet = round_dir / "train.parquet"
        test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
        python_bin = test_py if os.path.exists(test_py) else sys.executable
        gen_cmd = [
            python_bin,
            "scripts/generate_rollouts.py",
            "--server-url",
            server_url,
            "--model-id",
            current_model,
            "--out",
            str(out_parquet),
            "--num-games",
            str(args.games_per_round // 2),
            "--max-new-tokens",
            "8192",
            "--group-size",
            "8",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--max-model-len",
            "32768",
        ]
        _run(gen_cmd)
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "generation_done",
                "timestamp": time.time(),
                "out": str(out_parquet),
            }) + "\n")
        return out_parquet
    finally:
        print("Stopping SGLang server...")
        _kill_process_tree(server_proc)
        try:
            subprocess.run(["pkill", "-f", "sglang"], check=False)
        except Exception:
            pass


def _extract_reward_series_from_df(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract game-normalized rewards (actual game performance, not GRPO weights)."""
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


def _decode_tokens_to_text(input_ids: List[int], tokenizer) -> str:
    """Decode token IDs to text."""
    try:
        return tokenizer.decode(input_ids, skip_special_tokens=False)
    except Exception:
        return f"[Failed to decode {len(input_ids)} tokens]"


def _export_example_games(df: pd.DataFrame, output_path: Path, n_per_category: int = 5) -> None:
    """Export best/worst/median games as human-readable plaintext.

    Args:
        df: DataFrame with game data (must have 'game_normalized_reward' or similar)
        output_path: Path to write examples.txt
        n_per_category: Number of examples per category
    """
    try:
        # Load tokenizer for decoding (use same as in generate_rollouts.py)
        from transformers import AutoTokenizer
        # Get model path from first row's game_info if available, else use default
        model_path = "Qwen/Qwen2.5-7B-Instruct"  # Default fallback
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            print(f"Warning: Could not load tokenizer for {model_path}, examples will show raw tokens")
            tokenizer = None

        # Extract rewards
        rewards = _extract_reward_series_from_df(df)
        if rewards is None or len(rewards) == 0:
            print("Warning: No rewards found, skipping example export")
            return

        # Find indices for best/worst/median games
        sorted_indices = np.argsort(rewards)

        # Best games (highest rewards)
        best_indices = sorted_indices[-n_per_category:][::-1]
        # Worst games (lowest rewards)
        worst_indices = sorted_indices[:n_per_category]
        # Median games (around 50th percentile)
        median_idx = len(sorted_indices) // 2
        median_start = max(0, median_idx - n_per_category // 2)
        median_indices = sorted_indices[median_start:median_start + n_per_category]

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ROLLOUT EXAMPLES\n")
            f.write("=" * 80 + "\n\n")

            for category, indices in [("BEST GAMES", best_indices),
                                      ("MEDIAN GAMES", median_indices),
                                      ("WORST GAMES", worst_indices)]:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"{category}\n")
                f.write("=" * 80 + "\n\n")

                for idx in indices:
                    row = df.iloc[idx]
                    reward = rewards[idx]
                    grpo_weight = row.get("sample_weight", None)
                    game_id = row.get("game_id", idx)

                    f.write(f"Game ID: {game_id}\n")
                    f.write(f"Reward (game-normalized): {reward:.4f}\n")
                    f.write(f"GRPO Weight (relative): {grpo_weight:.4f}\n")
                    f.write(f"Sequence Length: {len(row['input_ids'])}\n")
                    f.write("-" * 80 + "\n")

                    # Decode conversation
                    if tokenizer is not None:
                        try:
                            text = _decode_tokens_to_text(row["input_ids"].tolist(), tokenizer)
                            f.write(f"{text}\n")
                        except Exception as e:
                            f.write(f"[Error decoding tokens: {e}]\n")
                    else:
                        f.write(f"[Token IDs: {row['input_ids'][:50].tolist()}... (truncated)]\n")

                    f.write("\n" + "=" * 80 + "\n\n")

        print(f"Exported {n_per_category * 3} example games to {output_path}")

    except Exception as e:
        print(f"Warning: Failed to export examples: {e}")
        # Create empty file so pipeline doesn't break
        output_path.touch()


def function_B_generate_trim_and_log(
    *,
    args,
    save_root: Path,
    round_dir: Path,
    current_model: str,
) -> RolloutStats:
    """Call A to generate, then trim to p95, write files, log metrics, and return RolloutStats."""
    out_parquet = function_A_start_server_and_generate(args=args, round_dir=round_dir, current_model=current_model)

    # Load data
    df = pd.read_parquet(str(out_parquet))
    if "sample_weight" in df.columns and len(df) > 0 and not (df["sample_weight"] > 0).any():
        raise RuntimeError("All rollout sample weights are zero. This likely indicates inference failures or server issues.")

    # Extract game-normalized rewards (ACTUAL game performance, not GRPO weights)
    game_rewards = _extract_reward_series_from_df(df)
    if game_rewards is None or len(game_rewards) == 0:
        raise RuntimeError("Failed to extract game rewards from data")

    # Compute game performance metrics (before trimming)
    game_reward_mean = float(np.mean(game_rewards))
    game_reward_std = float(np.std(game_rewards))
    game_reward_p10 = float(np.percentile(game_rewards, 10))
    game_reward_p25 = float(np.percentile(game_rewards, 25))
    game_reward_p50 = float(np.percentile(game_rewards, 50))
    game_reward_p75 = float(np.percentile(game_rewards, 75))
    game_reward_p90 = float(np.percentile(game_rewards, 90))

    # Perfect score ratio (before trimming)
    perfect_score_ratio = float(np.mean(game_rewards >= 1.0))

    # Compute GRPO weight statistics (these should have mean ~0)
    grpo_weights = df["sample_weight"].astype(float) if "sample_weight" in df.columns else np.zeros(len(df))
    grpo_weight_mean = float(np.mean(grpo_weights))
    grpo_weight_pos_ratio = float(np.mean(grpo_weights > 0))
    grpo_weight_neg_ratio = float(np.mean(grpo_weights < 0))
    grpo_weight_zero_ratio = float(np.mean(grpo_weights == 0))

    # Trim by length (p95)
    lengths = df["input_ids"].apply(lambda x: len(x))
    pct95 = int(np.percentile(lengths, 95))
    if pct95 > 5000:
        pct95 = 5000

    kept = df[lengths <= pct95]
    trimmed_path = round_dir / "train_trimmed.parquet"
    kept.to_parquet(str(trimmed_path))
    print(f"Trimmed to 95th percentile length={pct95}, kept {len(kept)}/{len(df)} samples -> {trimmed_path}")

    # Compute perfect score ratio after trimming
    kept_game_rewards = game_rewards[lengths <= pct95]
    perfect_score_ratio_after_trim = float(np.mean(kept_game_rewards >= 1.0))

    # Export example games
    examples_path = round_dir / "examples.txt"
    _export_example_games(df, examples_path, n_per_category=5)

    # Create RolloutStats object
    stats = RolloutStats(
        raw_parquet=out_parquet,
        trimmed_parquet=trimmed_path,
        examples_txt=examples_path,
        game_reward_mean=game_reward_mean,
        game_reward_std=game_reward_std,
        game_reward_p10=game_reward_p10,
        game_reward_p25=game_reward_p25,
        game_reward_p50=game_reward_p50,
        game_reward_p75=game_reward_p75,
        game_reward_p90=game_reward_p90,
        perfect_score_ratio=perfect_score_ratio,
        perfect_score_ratio_after_trim=perfect_score_ratio_after_trim,
        total_sequences=len(df),
        kept_sequences=len(kept),
        trim_threshold=pct95,
        grpo_weight_mean=grpo_weight_mean,
        grpo_weight_pos_ratio=grpo_weight_pos_ratio,
        grpo_weight_neg_ratio=grpo_weight_neg_ratio,
        grpo_weight_zero_ratio=grpo_weight_zero_ratio,
    )

    # Log to progress.log
    log_file = round_dir / "progress.log"
    with open(log_file, "a") as lf:
        lf.write(json.dumps({
            "event": "trim_done",
            "timestamp": time.time(),
            "pct95": pct95,
            "kept": int(len(kept)),
            "total": int(len(df)),
        }) + "\n")
        lf.write(json.dumps({
            "event": "rollout_metrics",
            "timestamp": time.time(),
            "game_reward_mean": game_reward_mean,
            "game_reward_std": game_reward_std,
            "game_reward_p50": game_reward_p50,
            "perfect_score_ratio": perfect_score_ratio,
            "perfect_score_ratio_after_trim": perfect_score_ratio_after_trim,
            "grpo_weight_mean": grpo_weight_mean,
            "grpo_weight_pos_ratio": grpo_weight_pos_ratio,
        }) + "\n")

    print(f"Rollout stats: game_reward_mean={game_reward_mean:.4f}, perfect_score_ratio={perfect_score_ratio:.3f}")

    return stats


# B.main(): execute the full pipeline when run as a script
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--round-dir", default="")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--games-per-round", type=int, default=256)
    ap.add_argument("--server-port", type=int, default=8000)
    ap.add_argument("--server-wait-seconds", type=int, default=900)
    ap.add_argument("--server-mem-fraction", type=float, default=0.85)
    ap.add_argument("--server-log-level", type=str, default="debug")
    ap.add_argument("--server-disable-torch-compile", dest="server_enable_torch_compile", action="store_false")
    ap.add_argument("--server-disable-cuda-graph", dest="server_disable_cuda_graph", action="store_true")
    ap.set_defaults(server_enable_torch_compile=True, server_disable_cuda_graph=False)
    args = ap.parse_args()

    save_root = Path(args.save_root)
    if args.round_dir:
        round_dir = Path(args.round_dir)
    else:
        round_dir = save_root / f"round_{args.round:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

    result = function_B_generate_trim_and_log(args=args, save_root=save_root, round_dir=round_dir, current_model=args.model_path)
    # Also print JSON result to stdout for easy capture
    print(json.dumps(result.to_dict()))


# Named export used by offline_grpo_loop
run_rollout_pipeline = function_B_generate_trim_and_log


if __name__ == "__main__":
    main()



