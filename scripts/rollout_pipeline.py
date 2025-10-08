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
    log_file: Optional[Path] = None,
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
        "0.5",
        "--chunked-prefill-size",
        "512",
        "--schedule-policy",
        "lpm",
        "--max-running-requests",
        "1024",  # Allow many concurrent requests; SGLang will batch efficiently
    ]
    if enable_torch_compile:
        cmd.append("--enable-torch-compile")
    if disable_cuda_graph:
        cmd.append("--disable-cuda-graph")

    print("[rollout_pipeline] Launch command:", " ".join(cmd), flush=True)
    print(f"[rollout_pipeline] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    print(f"[rollout_pipeline] NCCL_P2P_LEVEL={env.get('NCCL_P2P_LEVEL','')}", flush=True)

    # Redirect output to log file if provided, otherwise to stdout/stderr
    if log_file:
        print(f"[rollout_pipeline] Server output will be logged to: {log_file}", flush=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_file, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        # Store file handle on proc so it doesn't get closed prematurely
        proc._log_file = f
    else:
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
    """Start SGLang server(s), generate rollouts via scripts/generate_rollouts.py, log events, stop server(s).

    For 4 GPUs: Uses 2 servers (TP=2 each) to avoid all-reduce overhead.
    For other GPU counts: Uses single server with appropriate TP.

    Returns path to the raw parquet (train.parquet).
    """
    gpu_string = args.gpus
    gpu_list = [g.strip() for g in gpu_string.split(",") if g.strip()]
    num_gpus = len(gpu_list)

    log_file = round_dir / "progress.log"
    test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
    python_bin = test_py if os.path.exists(test_py) else sys.executable

    # Special case: 4 GPUs → use 2 servers to avoid cross-bridge all-reduce (if enabled)
    if num_gpus == 4 and args.dual_server:
        print("Detected 4 GPUs: using dual-server mode (2×TP2) for better performance...")

        # Start both servers with separate log files
        server1_log = round_dir / "sglang_server1.log"
        server2_log = round_dir / "sglang_server2.log"

        print("Starting server 1 on GPUs 0,1...")
        server1 = _start_sglang_server(
            model_path=current_model,
            gpus=f"{gpu_list[0]},{gpu_list[1]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=12345,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=server1_log,
        )

        print("Starting server 2 on GPUs 2,3...")
        server2 = _start_sglang_server(
            model_path=current_model,
            gpus=f"{gpu_list[2]},{gpu_list[3]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=12346,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=server2_log,
        )

        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "dual_servers_started",
                "timestamp": time.time(),
                "model": current_model,
                "server1_gpus": f"{gpu_list[0]},{gpu_list[1]}",
                "server2_gpus": f"{gpu_list[2]},{gpu_list[3]}",
            }) + "\n")

        try:
            # Wait for both servers in parallel
            print("Waiting for both servers to become ready...")
            import threading

            errors = []

            def wait_server1():
                try:
                    _wait_for_sglang("http://127.0.0.1:12345", server1, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print("Server 1 ready!")
                except Exception as e:
                    errors.append(("server1", e))

            def wait_server2():
                try:
                    _wait_for_sglang("http://127.0.0.1:12346", server2, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print("Server 2 ready!")
                except Exception as e:
                    errors.append(("server2", e))

            t1 = threading.Thread(target=wait_server1)
            t2 = threading.Thread(target=wait_server2)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            if errors:
                error_msg = "; ".join([f"{name}: {err}" for name, err in errors])
                raise RuntimeError(f"Server startup failed: {error_msg}")

            # Generate on both servers in parallel
            num_games_per_server = args.games_per_round // 2  # Each server gets half the unique games
            group_size = 8  # From generate_rollouts.py default

            out1 = round_dir / "train_server1.parquet"
            out2 = round_dir / "train_server2.parquet"

            print(f"Launching parallel generation: {num_games_per_server} unique games per server...")
            gen_cmd1 = [
                python_bin, "scripts/generate_rollouts.py",
                "--server-url", "http://127.0.0.1:12345",
                "--model-id", current_model,
                "--out", str(out1),
                "--num-games", str(num_games_per_server),
                "--max-new-tokens", "8192",
                "--group-size", str(group_size),
                "--temperature", "0.7",
                "--top-p", "0.9",
                "--max-model-len", "32768",
                "--max-turns", str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn", str(getattr(args, 'max_retries_per_turn', 2)),
            ]

            gen_cmd2 = [
                python_bin, "scripts/generate_rollouts.py",
                "--server-url", "http://127.0.0.1:12346",
                "--model-id", current_model,
                "--out", str(out2),
                "--num-games", str(num_games_per_server),
                "--max-new-tokens", "8192",
                "--group-size", str(group_size),
                "--temperature", "0.7",
                "--top-p", "0.9",
                "--max-model-len", "32768",
                "--max-turns", str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn", str(getattr(args, 'max_retries_per_turn', 2)),
            ]

            # Run both in parallel
            proc1 = subprocess.Popen(gen_cmd1)
            proc2 = subprocess.Popen(gen_cmd2)

            # Wait for completion
            ret1 = proc1.wait()
            ret2 = proc2.wait()

            if ret1 != 0 or ret2 != 0:
                raise RuntimeError(f"Generation failed: server1={ret1}, server2={ret2}")

            # Merge parquets and fix game_ids
            print("Merging outputs from both servers...")
            df1 = pd.read_parquet(str(out1))
            df2 = pd.read_parquet(str(out2))

            # Shift server2 game_ids to continue from server1
            offset = num_games_per_server * group_size
            print(f"Offsetting server2 game_ids by {offset}...")
            df2["game_id"] = df2["game_id"] + offset

            # Merge
            df = pd.concat([df1, df2], ignore_index=True)
            out_parquet = round_dir / "train.parquet"
            df.to_parquet(str(out_parquet))
            print(f"Merged {len(df)} sequences into {out_parquet}")

            # Cleanup temp files
            out1.unlink()
            out2.unlink()

            with open(log_file, "a") as lf:
                lf.write(json.dumps({
                    "event": "dual_generation_done",
                    "timestamp": time.time(),
                    "out": str(out_parquet),
                    "server1_sequences": len(df1),
                    "server2_sequences": len(df2),
                }) + "\n")

            return out_parquet

        finally:
            print("Stopping both SGLang servers...")
            _kill_process_tree(server1)
            _kill_process_tree(server2)
            try:
                subprocess.run(["pkill", "-f", "sglang"], check=False)
            except Exception:
                pass

    else:
        # Standard single-server mode
        tp = num_gpus if num_gpus > 0 else 1
        print(f"Using single-server mode (TP={tp})...")

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
                str(args.games_per_round),
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
                "--max-turns",
                str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn",
                str(getattr(args, 'max_retries_per_turn', 2)),
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
        # Extract rewards
        rewards = _extract_reward_series_from_df(df)
        if rewards is None or len(rewards) == 0:
            print("Warning: No rewards found, skipping example export")
            return

        # Group by game_id to avoid duplicates (each game has 2 sequences)
        unique_games = df.groupby('game_id').first().reset_index()

        # Extract rewards for unique games
        game_rewards = _extract_reward_series_from_df(unique_games)
        if game_rewards is None or len(game_rewards) == 0:
            print("Warning: No rewards found for unique games, skipping example export")
            return

        # Find indices for best/worst/median games
        sorted_indices = np.argsort(game_rewards)

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
                    row = unique_games.iloc[idx]
                    reward = game_rewards[idx]
                    grpo_weight = row.get("sample_weight", None)
                    game_id = row.get("game_id", idx)

                    f.write(f"Game ID: {game_id}\n")
                    f.write(f"Reward (game-normalized): {reward:.4f}\n")
                    f.write(f"GRPO Weight (relative): {grpo_weight:.4f}\n")
                    f.write("-" * 80 + "\n")

                    # Parse and format full conversation
                    try:
                        full_conv_str = row.get("full_conversation", "[]")
                        full_conv = json.loads(full_conv_str) if isinstance(full_conv_str, str) else full_conv_str

                        if full_conv:
                            f.write("\n=== Full Conversation (Including Errors) ===\n\n")
                            for entry in full_conv:
                                turn = entry.get("turn", "?")
                                player = entry.get("player", "?")
                                message = entry.get("message", "")
                                retry = entry.get("retry", 0)

                                if player == "error":
                                    f.write(f"Turn {turn} - ERROR (retry {retry}):\n{message}\n\n")
                                else:
                                    retry_str = f" (retry {retry})" if retry > 0 else ""
                                    f.write(f"Turn {turn} - {player}{retry_str}:\n{message}\n\n")
                        else:
                            f.write("[No conversation data available]\n")
                    except Exception as e:
                        f.write(f"[Error parsing conversation: {e}]\n")

                    f.write("\n" + "=" * 80 + "\n\n")

        print(f"Exported {n_per_category * 3} example games to {output_path}")

    except Exception as e:
        print(f"Warning: Failed to export examples: {e}")
        import traceback
        traceback.print_exc()
        # Create empty file so pipeline doesn't break
        output_path.touch()


def process_rollouts_post_generation(round_dir: Path, save_root: Path, args=None) -> RolloutStats:
    """
    Process raw rollouts: compute stats, trim, export examples, log metrics.
    This is called after rollouts are generated (either fresh or resumed).

    Args:
        round_dir: Directory containing train.parquet
        save_root: Root directory for saving (unused, kept for compatibility)
        args: Optional args object with reference_model and kl_coef

    Returns:
        RolloutStats object with all metrics.
    """
    train_parquet = round_dir / "train.parquet"

    if not train_parquet.exists():
        raise RuntimeError(f"Expected train.parquet at {train_parquet} but not found")

    # Load data
    df = pd.read_parquet(str(train_parquet))
    if "sample_weight" in df.columns and len(df) > 0 and not (df["sample_weight"] > 0).any():
        raise RuntimeError("All rollout sample weights are zero. This likely indicates inference failures or server issues.")

    # Apply KL divergence penalty if reference model is specified
    if args and hasattr(args, 'reference_model') and args.reference_model:
        print(f"\n{'='*80}")
        print(f"Applying KL divergence penalty from reference model...")
        print(f"  Reference model: {args.reference_model}")
        print(f"  KL coefficient: {getattr(args, 'kl_coef', 0.1)}")
        print(f"{'='*80}\n")

        # Compute reference logprobs
        ref_logprobs_list = compute_reference_logprobs(
            df=df,
            reference_model_path=args.reference_model,
            gpus=args.gpus,
            server_port=8001,  # Use different port from policy server
            server_wait_seconds=args.server_wait_seconds,
            server_mem_fraction=args.server_mem_fraction,
            server_log_level=args.server_log_level,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            round_dir=round_dir,
        )

        # Calculate KL penalty and adjust rewards
        kl_coef = getattr(args, 'kl_coef', 0.1)
        kl_penalties = []

        for idx, row in df.iterrows():
            # Parse policy logprobs from JSON
            policy_logprobs = json.loads(row["policy_logprobs"]) if isinstance(row["policy_logprobs"], str) else row["policy_logprobs"]
            ref_logprobs = ref_logprobs_list[idx]

            # Compute KL divergence: KL = policy_logprob - ref_logprob
            # Sum over all generated tokens for this sequence
            if len(policy_logprobs) > 0 and len(ref_logprobs) > 0:
                # Align lengths (they should match, but be defensive)
                min_len = min(len(policy_logprobs), len(ref_logprobs))
                kl_per_token = [
                    policy_logprobs[i] - ref_logprobs[i]
                    for i in range(min_len)
                ]
                kl_penalty = sum(kl_per_token)
            else:
                kl_penalty = 0.0

            kl_penalties.append(kl_penalty)

            # Adjust reward: reward_adjusted = reward - kl_coef * kl_penalty
            # sample_weight currently holds the raw game reward
            original_reward = float(row["sample_weight"])
            adjusted_reward = original_reward - kl_coef * kl_penalty
            df.at[idx, "sample_weight"] = adjusted_reward

        # Save KL statistics
        mean_kl = np.mean(kl_penalties)
        std_kl = np.std(kl_penalties)
        print(f"\nKL divergence statistics:")
        print(f"  Mean KL: {mean_kl:.4f}")
        print(f"  Std KL: {std_kl:.4f}")
        print(f"  Mean penalty: {kl_coef * mean_kl:.4f}")

        # Log KL stats
        log_file = round_dir / "progress.log"
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "kl_penalty_applied",
                "timestamp": time.time(),
                "mean_kl": mean_kl,
                "std_kl": std_kl,
                "kl_coef": kl_coef,
                "mean_penalty": kl_coef * mean_kl,
            }) + "\n")

        # Overwrite the parquet with adjusted rewards
        df.to_parquet(str(train_parquet))
        print(f"Saved adjusted rewards to {train_parquet}\n")

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

    # Filter out generation failures BEFORE trimming
    from transformers import AutoTokenizer
    failure_mask = np.zeros(len(df), dtype=bool)
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        for idx, row in df.iterrows():
            text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
            if 'I need to think about this.' in text:
                failure_mask[idx] = True
    except Exception as e:
        print(f"Warning: Could not filter generation failures: {e}")

    failure_count = int(np.sum(failure_mask))
    failure_ratio = float(failure_count / len(df)) if len(df) > 0 else 0.0

    # Check if failure rate exceeds acceptable threshold
    FAILURE_THRESHOLD = 0.05  # 5%
    if failure_ratio > FAILURE_THRESHOLD:
        error_msg = (
            f"FATAL: Generation failure rate ({failure_ratio:.2%}) exceeds threshold ({FAILURE_THRESHOLD:.2%}).\n"
            f"  Failed: {failure_count}/{len(df)} samples\n"
            f"  This indicates serious problems with model inference or server stability.\n"
            f"  Check generation_failures.log for details."
        )
        print(f"\n{'='*80}")
        print(error_msg)
        print(f"{'='*80}\n")

        # Log to progress.log before crashing
        log_file = round_dir / "progress.log"
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "fatal_high_failure_rate",
                "timestamp": time.time(),
                "failure_count": failure_count,
                "total_sequences": len(df),
                "failure_ratio": failure_ratio,
                "threshold": FAILURE_THRESHOLD,
            }) + "\n")

        raise RuntimeError(error_msg)

    # Remove failed samples
    df_no_failures = df[~failure_mask].reset_index(drop=True)
    game_rewards_no_failures = game_rewards[~failure_mask]

    if failure_count > 0:
        print(f"Filtered out {failure_count} failed generations ({failure_ratio:.2%})")
        print(f"  Remaining: {len(df_no_failures)}/{len(df)} samples")

    # Trim by length (p95) on non-failed samples
    lengths = df_no_failures["input_ids"].apply(lambda x: len(x))
    pct95 = int(np.percentile(lengths, 95))
    if pct95 > 5000:
        pct95 = 5000

    kept = df_no_failures[lengths <= pct95]

    # Split into train (90%) and val (10%)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(kept, test_size=0.1, random_state=42)

    train_path = round_dir / "train_trimmed.parquet"
    val_path = round_dir / "val_trimmed.parquet"
    train_df.to_parquet(str(train_path))
    val_df.to_parquet(str(val_path))

    print(f"Trimmed to 95th percentile length={pct95}, kept {len(kept)}/{len(df_no_failures)} non-failed samples")
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Val: {len(val_df)} samples -> {val_path}")

    # Compute perfect score ratio after filtering and trimming
    kept_game_rewards = game_rewards_no_failures[lengths <= pct95]
    perfect_score_ratio_after_trim = float(np.mean(kept_game_rewards >= 1.0))

    # Export example games (from original df including failures, for debugging)
    examples_path = round_dir / "examples.txt"
    _export_example_games(df, examples_path, n_per_category=5)

    # Create RolloutStats object
    stats = RolloutStats(
        raw_parquet=train_parquet,
        trimmed_parquet=train_path,
        val_parquet=val_path,
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
        total_sequences=len(df),  # Original count including failures
        kept_sequences=len(kept),  # After filtering failures AND length trimming
        trim_threshold=pct95,
        grpo_weight_mean=grpo_weight_mean,
        grpo_weight_pos_ratio=grpo_weight_pos_ratio,
        grpo_weight_neg_ratio=grpo_weight_neg_ratio,
        grpo_weight_zero_ratio=grpo_weight_zero_ratio,
        failure_ratio=failure_ratio,  # Ratio of failed generations in raw data
    )

    # Log to progress.log
    log_file = round_dir / "progress.log"
    with open(log_file, "a") as lf:
        lf.write(json.dumps({
            "event": "filter_and_trim_done",
            "timestamp": time.time(),
            "total": int(len(df)),
            "failed": failure_count,
            "after_failure_filter": int(len(df_no_failures)),
            "pct95_threshold": pct95,
            "after_length_trim": int(len(kept)),
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
            "failure_ratio": failure_ratio,
        }) + "\n")

    print(f"\nRollout summary:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Failed generations: {failure_count} ({failure_ratio:.2%})")
    print(f"  After failure filter: {len(df_no_failures)}")
    print(f"  After length trim (p95={pct95}): {len(kept)}")
    print(f"  Game reward (raw data): mean={game_reward_mean:.4f}, perfect_score={perfect_score_ratio:.1%}")

    # Write stats.txt for quick reference (use full rollout set, not trimmed)
    # Import compute_and_save_stats from offline_grpo_loop
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from offline_grpo_loop import compute_and_save_stats
        stats_file = round_dir / "stats.txt"
        compute_and_save_stats(train_parquet, stats_file)
    except Exception as e:
        print(f"Warning: Could not write stats.txt: {e}")

    return stats


def function_B_generate_trim_and_log(
    *,
    args,
    save_root: Path,
    round_dir: Path,
    current_model: str,
) -> RolloutStats:
    """Call A to generate, then call post-processing to trim, compute stats, and log.

    This is the main entry point used by offline_grpo_loop.
    """
    # Generate raw rollouts
    out_parquet = function_A_start_server_and_generate(
        args=args,
        round_dir=round_dir,
        current_model=current_model
    )

    # Process rollouts (trim, stats, examples)
    stats = process_rollouts_post_generation(
        round_dir=round_dir,
        save_root=save_root,
        args=args
    )

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
    ap.add_argument("--dual-server", action="store_true", help="Use dual-server mode (2×TP2) when 4 GPUs available (default: disabled)")
    # Game termination settings
    ap.add_argument("--max-turns", type=int, default=10, help="Maximum number of turns per game (default: 10)")
    ap.add_argument("--max-retries-per-turn", type=int, default=2, help="Maximum retries per turn before terminating (default: 2)")
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


def compute_reference_logprobs(
    *,
    df: pd.DataFrame,
    reference_model_path: str,
    gpus: str,
    server_port: int = 8001,
    server_wait_seconds: int = 900,
    server_mem_fraction: float = 0.85,
    server_log_level: str = "info",
    enable_torch_compile: bool = False,
    disable_cuda_graph: bool = False,
    round_dir: Optional[Path] = None,
) -> List[List[float]]:
    """Compute reference model logprobs for already-generated sequences.

    This function:
    1. Starts a reference model server
    2. Scores all sequences with the reference model (no generation, just scoring)
    3. Returns per-sequence logprobs
    4. Shuts down the server

    Args:
        df: DataFrame containing generated sequences with 'input_ids' and 'policy_logprobs'
        reference_model_path: Path to the reference model checkpoint
        gpus: GPU IDs to use (e.g., "0,1,2,3")
        server_port: Port for reference model server
        server_wait_seconds: Max time to wait for server startup
        server_mem_fraction: Memory fraction for server
        server_log_level: Server log level
        enable_torch_compile: Whether to enable torch.compile
        disable_cuda_graph: Whether to disable CUDA graph
        round_dir: Optional directory for server logs

    Returns:
        List of logprob lists (one list per sequence)
    """
    import requests
    from transformers import AutoTokenizer

    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    tp = len(gpu_list) if len(gpu_list) > 0 else 1

    print(f"\n{'='*80}")
    print(f"Starting reference model server for KL divergence computation...")
    print(f"  Reference model: {reference_model_path}")
    print(f"  GPUs: {gpus} (TP={tp})")
    print(f"{'='*80}\n")

    # Start reference model server
    ref_log_file = round_dir / "sglang_reference.log" if round_dir else None
    server_proc = _start_sglang_server(
        model_path=reference_model_path,
        gpus=gpus,
        tp=tp,
        mem_util=server_mem_fraction,
        port=server_port,
        enable_torch_compile=enable_torch_compile,
        disable_cuda_graph=disable_cuda_graph,
        log_level=server_log_level,
        log_file=ref_log_file,
    )

    server_url = f"http://127.0.0.1:{server_port}"

    try:
        # Wait for server to be ready
        _wait_for_sglang(server_url, server_proc, timeout_sec=server_wait_seconds, interval_sec=5)

        # Load tokenizer to reconstruct messages from input_ids
        print(f"Loading tokenizer from {reference_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(reference_model_path, trust_remote_code=True)

        # Prepare HTTP session
        session = requests.Session()
        completions_url = f"{server_url}/v1/chat/completions"

        all_ref_logprobs = []
        batch_size = 32  # Process in batches for progress tracking
        total_sequences = len(df)

        print(f"\nComputing reference logprobs for {total_sequences} sequences...")
        print(f"Processing in batches of {batch_size}...\n")

        for batch_start in range(0, total_sequences, batch_size):
            batch_end = min(batch_start + batch_size, total_sequences)
            batch_df = df.iloc[batch_start:batch_end]

            batch_ref_logprobs = []

            for idx, row in batch_df.iterrows():
                # Decode input_ids back to messages
                # The input_ids include the full conversation, so we decode and parse it
                input_ids = row["input_ids"]
                full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

                # For SGLang scoring, we need to format as messages
                # Since we have the full conversation, create a single user message with it
                messages = [{"role": "user", "content": full_text}]

                # Request logprobs with max_tokens=0 (scoring mode)
                payload = {
                    "model": reference_model_path,
                    "messages": messages,
                    "max_tokens": 0,  # Don't generate, just score
                    "logprobs": True,
                    "top_logprobs": 1,
                }

                try:
                    resp = session.post(completions_url, json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()

                    # Extract logprobs
                    logprobs_data = data["choices"][0].get("logprobs", None)
                    if logprobs_data and logprobs_data.get("content"):
                        ref_logprobs = [
                            token_data["logprob"]
                            for token_data in logprobs_data["content"]
                        ]
                        batch_ref_logprobs.append(ref_logprobs)
                    else:
                        # No logprobs available, use empty list
                        batch_ref_logprobs.append([])

                except Exception as e:
                    print(f"Warning: Failed to get reference logprobs for sequence {idx}: {e}")
                    batch_ref_logprobs.append([])

            all_ref_logprobs.extend(batch_ref_logprobs)

            # Progress update
            pct = 100.0 * batch_end / total_sequences
            print(f"Progress: {batch_end}/{total_sequences} sequences ({pct:.1f}%)")

        print(f"\n{'='*80}")
        print(f"Reference logprob computation complete!")
        print(f"  Processed: {len(all_ref_logprobs)} sequences")
        print(f"{'='*80}\n")

        return all_ref_logprobs

    finally:
        print("Stopping reference model server...")
        _kill_process_tree(server_proc)
        try:
            subprocess.run(["pkill", "-f", f"sglang.*{server_port}"], check=False)
        except Exception:
            pass


# Named export used by offline_grpo_loop
run_rollout_pipeline = function_B_generate_trim_and_log


if __name__ == "__main__":
    main()



