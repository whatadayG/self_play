#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import requests


def start_sglang_server(model_path: str, gpus: str = "0,1,2,3", tp: int = 4, mem_util: float = 0.6, port: int = 8000) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    
    # Kill any existing server on the port
    try:
        result = subprocess.run(["lsof", "-ti:" + str(port)], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False)
                except Exception:
                    pass
            # Wait a bit for processes to die
            time.sleep(1)
    except Exception:
        pass  # No process on port or lsof not available
    
    # Prefer test_venv python if available (same as used for generate_rollouts.py)
    test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
    python_bin = test_py if os.path.exists(test_py) else sys.executable
    
    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--tp", str(tp),
        "--trust-remote-code",
        "--mem-fraction-static", str(mem_util),
        "--dtype", "bfloat16",
        "--enable-torch-compile"
    ]
    
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def kill_process_tree(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


def run(cmd: list[str], env=None):
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=True)
    return result.returncode


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


def wait_for_sglang(server_url: str, server_proc: subprocess.Popen, timeout_sec: int = 900, interval_sec: int = 5) -> None:
    """Poll the SGLang OpenAI-compatible /v1/models endpoint until ready or timeout."""
    base = server_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    deadline = time.time() + timeout_sec
    last_err = None
    server_output_lines = []
    
    while time.time() < deadline:
        # Check if server process has died
        if server_proc.poll() is not None:
            # Collect any remaining output
            if server_proc.stdout:
                try:
                    remaining = server_proc.stdout.read()
                    if remaining:
                        server_output_lines.append(remaining)
                except Exception:
                    pass
            output = "".join(server_output_lines)
            raise RuntimeError(f"SGLang server process died with exit code {server_proc.returncode}. Output:\n{output}")
        
        # Collect server output for debugging
        if server_proc.stdout:
            try:
                # Non-blocking read of available output
                import select
                while True:
                    ready, _, _ = select.select([server_proc.stdout], [], [], 0)
                    if not ready:
                        break
                    line = server_proc.stdout.readline()
                    if line:
                        server_output_lines.append(line)
                        # Keep only last 100 lines to avoid memory issues
                        if len(server_output_lines) > 100:
                            server_output_lines.pop(0)
                    else:
                        break
            except (ImportError, AttributeError):
                # select not available (e.g. Windows) - skip non-blocking read
                pass
            except Exception:
                pass
        
        # Try to connect to the server
        try:
            r = requests.get(f"{base}/models", timeout=10)
            r.raise_for_status()
            data = r.json()
            # Require at least one model entry
            if isinstance(data, dict) and data.get("data"):
                return
            # Some versions return list
            if isinstance(data, list) and len(data) > 0:
                return
        except Exception as e:
            last_err = e
        
        print("[offline_grpo] waiting for SGLang server...", flush=True)
        time.sleep(interval_sec)
    
    # Timeout reached
    output = "".join(server_output_lines)
    raise TimeoutError(f"SGLang server at {base} did not become ready within {timeout_sec}s. Last error: {last_err}\nServer output:\n{output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--sequences-per-round", type=int, default=512)
    ap.add_argument("--model-path", default="/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_4800_merged")
    ap.add_argument("--save-root", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--server-port", type=int, default=8000)
    ap.add_argument("--server-wait-seconds", type=int, default=900, help="Max seconds to wait for SGLang server readiness")
    ap.add_argument("--wandb-project", default="offline-grpo")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    # Default save_root: logs/offline_grpo/YYYYmmdd_HHMMSS
    if args.save_root:
        save_root = Path(args.save_root)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_root = Path(f"/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo/{ts}")
    save_root.mkdir(parents=True, exist_ok=True)

    current_model = args.model_path

    # Initialize outer-loop wandb run (best-effort)
    wb = None
    try:
        import wandb as _wandb
        run_name = args.wandb_run_name or save_root.name
        wb = _wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, reinit=False)
        # Log initial config snapshot
        if wb is not None:
            wb.config.update({
                "rounds": args.rounds,
                "sequences_per_round": args.sequences_per_round,
                "gpus": args.gpus,
                "tp": args.tp,
                "model_start": current_model,
            }, allow_val_change=True)
    except Exception:
        wb = None

    for r in range(args.rounds):
        print(f"=== Round {r} ===")
        round_dir = save_root / f"round_{r:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1) Start SGLang server
        print("Starting SGLang server...")
        server_proc = start_sglang_server(
            model_path=current_model, gpus=args.gpus, tp=args.tp, mem_util=0.6, port=args.server_port
        )
        # Wait for server to come up by polling /v1/models
        server_url = f"http://127.0.0.1:{args.server_port}"
        try:
            wait_for_sglang(server_url, server_proc, timeout_sec=args.server_wait_seconds, interval_sec=5)
        except Exception as e:
            print(f"Error waiting for SGLang server: {e}")
            kill_process_tree(server_proc)
            # Best effort: kill any lingering server processes
            try:
                subprocess.run(["pkill", "-f", "sglang.launch_server"], check=False)
            except Exception:
                pass
            raise

        # 2) Generate rollouts
        print("Generating rollouts...")
        # crude dynamic length: let generator cap to large max_model_len; after writing Parquet, compute 95th pct length
        out_parquet = round_dir / "train.parquet"
        # Prefer test_venv's python to ensure proper deps
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
            str(args.sequences_per_round // 2),  # two sequences per game
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
        # Log round progress to file
        log_file = round_dir / "progress.log"
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "server_started",
                "timestamp": time.time(),
                "model": current_model,
                "gpus": args.gpus,
                "tp": args.tp,
            }) + "\n")
        try:
            run(gen_cmd)
            with open(log_file, "a") as lf:
                lf.write(json.dumps({
                    "event": "generation_done",
                    "timestamp": time.time(),
                    "out": str(out_parquet)
                }) + "\n")
        finally:
            print("Stopping SGLang server...")
            kill_process_tree(server_proc)
            # Ensure complete teardown of any stray servers to avoid stale state
            try:
                subprocess.run(["pkill", "-f", "sglang"], check=False)
            except Exception:
                pass

        # this is not optional-- if you set max_model_len to a high number to capture all sequences, you will OOM when training
        import pandas as pd
        import numpy as np
        df = pd.read_parquet(str(out_parquet))
        # Loud failure if all weights are zero (silent inference failure)
        if "sample_weight" in df.columns and len(df) > 0 and not (df["sample_weight"] > 0).any():
            raise RuntimeError("All rollout sample weights are zero. This likely indicates inference failures or server issues.")
        lengths = df["input_ids"].apply(lambda x: len(x))
        pct95 = int(np.percentile(lengths, 95))
        # Filter rows longer than pct95
        kept = df[lengths <= pct95]
        trimmed_path = round_dir / "train_trimmed.parquet"
        kept.to_parquet(str(trimmed_path))
        print(f"Trimmed to 95th percentile length={pct95}, kept {len(kept)}/{len(df)} samples -> {trimmed_path}")
        train_parquet_for_sft = trimmed_path
        max_len_arg = str(pct95)
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "trim_done",
                "timestamp": time.time(),
                "pct95": pct95,
                "kept": int(len(kept)),
                "total": int(len(df))
            }) + "\n")
        # Compute rollout performance metrics and log
        try:
            mean_norm_reward = float(kept["sample_weight"].astype(float).mean()) if "sample_weight" in kept.columns else None
            pos_ratio = float((kept["sample_weight"] > 0).mean()) if "sample_weight" in kept.columns else None
            zero_ratio = float((kept["sample_weight"] == 0).mean()) if "sample_weight" in kept.columns else None
            neg_ratio = float((kept["sample_weight"] < 0).mean()) if "sample_weight" in kept.columns else None
            def _get_abs_reward(x):
                try:
                    gi = json.loads(x)
                    # prefer normalized if present
                    if isinstance(gi, dict):
                        return gi.get("reward", gi.get("normalized_reward", None))
                    return None
                except Exception:
                    return None
            abs_vals = kept["game_info"].apply(_get_abs_reward) if "game_info" in kept.columns else []
            abs_vals = [v for v in abs_vals if isinstance(v, (int, float))]
            mean_abs_reward = float(np.mean(abs_vals)) if len(abs_vals) > 0 else None
        except Exception:
            mean_norm_reward = None
            mean_abs_reward = None
            pos_ratio = None
            zero_ratio = None
            neg_ratio = None
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "rollout_metrics",
                "timestamp": time.time(),
                "mean_norm_reward": mean_norm_reward,
                "mean_abs_reward": mean_abs_reward,
                "weight_pos_ratio": pos_ratio,
                "weight_zero_ratio": zero_ratio,
                "weight_neg_ratio": neg_ratio,
            }) + "\n")
        if wb is not None:
            wb.log({
                "round": r,
                "rollout/pct95_length": pct95,
                "rollout/kept": int(len(kept)),
                "rollout/total": int(len(df)),
                "rollout/mean_norm_reward": mean_norm_reward,
                "rollout/mean_abs_reward": mean_abs_reward,
            })

        # 3) Run SFT with weights-aware dataset (pretokenized path expects weights)
        # TODO: one problem with this is that the optimizer state is reset at every SFT.
        # hopefully this is not too much a problem.
        print("Running SFT training...")
        # Use existing SFT launcher, overriding train/val files to our parquet
        # HACK: Force SFT to run on 2 GPUs due to observed 4-GPU NCCL hang.
        # If fewer than 2 GPUs were provided, fall back to available count.
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if len(gpu_ids) >= 2:
            sft_visible = ",".join(gpu_ids[:2])
            nproc = 2
        else:
            sft_visible = ",".join(gpu_ids)
            nproc = max(1, len(gpu_ids))
        print(f"[HACK] Forcing SFT to use {nproc} GPU(s): CUDA_VISIBLE_DEVICES={sft_visible}")
        save_path = str(round_dir / "checkpoints")
        sft_cmd = [
            "bash",
            "scripts/sft_qwen/sft_qwen3.sh",
            str(nproc),
            save_path,
            f"data.train_files={str(train_parquet_for_sft)}",
            f"data.val_files={str(train_parquet_for_sft)}",
            f"model.partial_pretrain={current_model}",
            "trainer.total_epochs=1",
            # Save only at the end: rely on default end-of-training save; no mid-training save
            "trainer.save_freq=-1",
            "trainer.test_freq=200",
            f"data.max_length={max_len_arg}",
            # Switch dataset class to pretokenized
            "data.custom_cls.path=verl/verl/utils/dataset/pretokenized_sft_dataset.py",
            "data.custom_cls.name=PreTokenizedSFTDataset",
            # Make each SFT a distinct run in wandb, but within the same project
            f"trainer.project_name={args.wandb_project}",
            f"trainer.experiment_name={save_root.name}_round_{r}_2gpu_hack",
        ]
        # Ensure torchrun/python from test_venv are available
        sft_env = os.environ.copy()
        test_bin = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin"
        if os.path.isdir(test_bin):
            sft_env["PATH"] = f"{test_bin}:{sft_env.get('PATH','')}"
        # HACK: Restrict SFT to 2 GPUs to avoid NCCL hang
        if sft_visible:
            sft_env["CUDA_VISIBLE_DEVICES"] = sft_visible
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "sft_hack_2gpus",
                "timestamp": time.time(),
                "visible_gpus": sft_visible,
                "nproc": nproc,
            }) + "\n")
        # Tee SFT stdout/err to per-round log file
        sft_log = round_dir / "sft_train.log"
        run_tee(sft_cmd, logfile=sft_log, env=sft_env)
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "sft_done",
                "timestamp": time.time(),
                "save_path": save_path
            }) + "\n")

        # 4) Update model path to the latest SFT checkpoint for next round
        # Prefer the Hugging Face subdirectory if present for robust loading (config/tokenizer present)
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

    print("Offline GRPO loop completed.")


if __name__ == "__main__":
    main()


