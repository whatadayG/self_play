#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path


def start_sglang_server(model_path: str, gpus: str = "0,1,2,3", tp: int = 4, mem_util: float = 0.6) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["MODEL_PATH"] = model_path
    env["TP_SIZE"] = str(tp)
    env["GPU_MEMORY_UTILIZATION"] = str(mem_util)
    # Prefer 0.4 static via mem-fraction-static inside script
    cmd = ["bash", "scripts/start_sglang_server.sh"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--sequences-per-round", type=int, default=512)
    ap.add_argument("--model-path", default="/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_4800_merged")
    ap.add_argument("--save-root", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--server-port", type=int, default=8000)
    args = ap.parse_args()

    # Default save_root: logs/offline_grpo/YYYYmmdd_HHMMSS
    if args.save_root:
        save_root = Path(args.save_root)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_root = Path(f"/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo/{ts}")
    save_root.mkdir(parents=True, exist_ok=True)

    current_model = args.model_path

    for r in range(args.rounds):
        print(f"=== Round {r} ===")
        round_dir = save_root / f"round_{r:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # 1) Start SGLang server
        print("Starting SGLang server...")
        server_proc = start_sglang_server(
            model_path=current_model, gpus=args.gpus, tp=args.tp, mem_util=0.6
        )
        # Wait a few seconds for server to come up
        time.sleep(30)

        # 2) Generate rollouts
        print("Generating rollouts...")
        # crude dynamic length: let generator cap to large max_model_len; after writing Parquet, compute 95th pct length
        out_parquet = round_dir / "train.parquet"
        server_url = f"http://127.0.0.1:{args.server_port}"
        gen_cmd = [
            sys.executable,
            "scripts/generate_rollouts.py",
            "--server-url",
            server_url,
            "--model-id",
            "Qwen/Qwen3-8B-Instruct",
            "--out",
            str(out_parquet),
            "--num-games",
            str(args.sequences_per_round // 2),  # two sequences per game
            "--max-new-tokens",
            "256",
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

        # Optional: compute 95th percentile and keep only those rows (prevent very long outliers from wasting mem)
        try:
            import pandas as pd
            import numpy as np
            df = pd.read_parquet(str(out_parquet))
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
        except Exception as e:
            print(f"Warning: percentile trim failed: {e}; using full dataset")
            train_parquet_for_sft = out_parquet
            max_len_arg = "32768"
            with open(log_file, "a") as lf:
                lf.write(json.dumps({
                    "event": "trim_failed",
                    "timestamp": time.time(),
                    "error": str(e)
                }) + "\n")

        # 3) Run SFT with weights-aware dataset (pretokenized path expects weights)
        print("Running SFT training...")
        # Use existing SFT launcher, overriding train/val files to our parquet
        nproc = len(args.gpus.split(","))
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
            "trainer.save_freq=200",
            "trainer.test_freq=200",
            f"data.max_length={max_len_arg}",
            # Switch dataset class to pretokenized
            "data.custom_cls.path=verl/verl/utils/dataset/pretokenized_sft_dataset.py",
            "data.custom_cls.name=PreTokenizedSFTDataset",
        ]
        run(sft_cmd)
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "sft_done",
                "timestamp": time.time(),
                "save_path": save_path
            }) + "\n")

        # 4) Update model path to the latest SFT checkpoint for next round
        # We use the default FSDP checkpoint directory under save_path
        # The SFT trainer writes checkpoints to default_local_dir; we passed save_path for that.
        # Find the latest global_step directory.
        latest = None
        for p in Path(save_path).glob("global_step_*"):
            latest = p if (latest is None or p.stat().st_mtime > latest.stat().st_mtime) else latest
        if latest is None:
            print("Warning: no SFT checkpoint found; keeping current model for next round.")
        else:
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


