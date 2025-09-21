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


def analyze_run_state(save_root: Path) -> Tuple[int, str, Optional[str], Dict[str, Any]]:
    """
    Analyze the state of an existing run to determine where to resume.
    
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
    
    # Check if SFT training has completed by looking for checkpoints
    has_checkpoints = checkpoints_dir.exists() and any(checkpoints_dir.glob("global_step_*"))
    has_sft_log = sft_log.exists() and sft_log.stat().st_size > 0
    
    if has_checkpoints and has_sft_log:
        # SFT is complete, finish the round
        if not current_model:
            current_model = find_latest_model_from_round(latest_round_dir)
        return round_num, 'finish_round', current_model, state_info
    elif train_parquet.exists() and train_trimmed_parquet.exists():
        # Rollout complete, need to do SFT
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
    
    # Generate additional rollouts
    additional_parquet = round_dir / "train_additional.parquet"
    needed_games = max(1, needed_sequences // 2)  # two sequences per game
    
    # Start server and generate
    gpu_string = args.gpus
    gpu_list = [g for g in gpu_string.split(",") if g]
    tp = len(gpu_list)
    
    server_proc = start_sglang_server(
        model_path=current_model, gpus=gpu_string, tp=tp, 
        mem_util=args.server_mem_fraction, port=args.server_port,
        enable_torch_compile=args.server_enable_torch_compile, 
        log_level=args.server_log_level
    )
    
    server_url = f"http://127.0.0.1:{args.server_port}"
    try:
        wait_for_sglang(server_url, server_proc, timeout_sec=args.server_wait_seconds, interval_sec=5)
        
        # Generate additional rollouts
        test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
        python_bin = test_py if os.path.exists(test_py) else sys.executable
        gen_cmd = [
            python_bin,
            "scripts/generate_rollouts.py",
            "--server-url", server_url,
            "--model-id", current_model,
            "--out", str(additional_parquet),
            "--num-games", str(needed_games),
            "--max-new-tokens", "8192",
            "--group-size", "8",
            "--temperature", "0.7",
            "--top-p", "0.9",
            "--max-model-len", "32768",
        ]
        
        run(gen_cmd)
        
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
        
    finally:
        kill_process_tree(server_proc)
        try:
            subprocess.run(["pkill", "-f", "sglang"], check=False)
        except Exception:
            pass


def start_sglang_server(model_path: str, gpus: str = "0,1,2,3", tp: int = 4, mem_util: float = 0.6, port: int = 8000, enable_torch_compile: bool = True, log_level: str = "info") -> subprocess.Popen:
    # Auto-adjust TP size to match number of available GPUs
    num_gpus = len([g.strip() for g in gpus.split(",") if g.strip()])
    if tp > num_gpus:
        print(f"[offline_grpo] WARNING: Requested TP={tp} but only {num_gpus} GPUs available. Adjusting TP to {num_gpus}", flush=True)
        tp = num_gpus

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    print(f"[offline_grpo] Preparing to launch SGLang server: model_path={model_path}, gpus={gpus}, tp={tp}, mem_util={mem_util}, port={port}, torch_compile={enable_torch_compile}, log_level={log_level}", flush=True)
    
    """
    # Kill any existing server on the port
    try:
        result = subprocess.run(["lsof", "-ti:" + str(port)], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"[offline_grpo] Port {port} is in use by PIDs: {pids}. Sending SIGKILL...", flush=True)
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False)
                except Exception:
                    pass
            # Wait a bit for processes to die
            time.sleep(1)
            print(f"[offline_grpo] Port {port} cleanup done.", flush=True)
        else:
            print(f"[offline_grpo] Port {port} appears free.", flush=True)
    except Exception:
        pass  # No process on port or lsof not available
    """
    
    # Prefer test_venv python if available (same as used for generate_rollouts.py)
    test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
    python_bin = test_py if os.path.exists(test_py) else sys.executable
    print(f"[offline_grpo] Using Python interpreter: {python_bin}", flush=True)
    
    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--tp", str(tp),
        "--trust-remote-code",
        "--mem-fraction-static", str(mem_util),
        "--dtype", "bfloat16",
        "--log-level", str(log_level),
        "--disable-cuda-graph",
    ]
    if enable_torch_compile:
        cmd.append("--enable-torch-compile")
    print("[offline_grpo] Launch command:", " ".join(cmd), flush=True)
    print(f"[offline_grpo] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    
    # Pipe server output directly to our stdout - much simpler!
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, text=True)
    print(f"[offline_grpo] SGLang server process started with PID={proc.pid}", flush=True)
    return proc


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
    start_time = time.time()
    last_http_error_log_time = 0.0
    http_error_log_interval = 10.0
    print(f"[offline_grpo] Waiting for SGLang readiness at {base}/models (timeout={timeout_sec}s, check every {interval_sec}s)", flush=True)
    
    # Parse host/port for port-listening diagnostics
    host = "127.0.0.1"
    port = 80
    try:
        from urllib.parse import urlparse
        parsed = urlparse(server_url)
        host = parsed.hostname or host
        port = parsed.port or 80
    except Exception:
        pass
    
    import socket
    def _is_port_listening(h: str, p: int) -> bool:
        try:
            with socket.create_connection((h, p), timeout=0.2):
                return True
        except Exception:
            return False
    
    while time.time() < deadline:
        # Check if server process has died
        if server_proc.poll() is not None:
            print(f"[offline_grpo] Server process exited early with code {server_proc.returncode}", flush=True)
            raise RuntimeError(f"SGLang server process died with exit code {server_proc.returncode}")
        
        # Try to connect to the server
        try:
            r = requests.get(f"{base}/models", timeout=10)
            r.raise_for_status()
            data = r.json()
            # Require at least one model entry
            if isinstance(data, dict) and data.get("data"):
                elapsed = int(time.time() - start_time)
                print(f"[offline_grpo] SGLang server is ready after {elapsed}s.", flush=True)
                return
            # Some versions return list
            if isinstance(data, list) and len(data) > 0:
                elapsed = int(time.time() - start_time)
                print(f"[offline_grpo] SGLang server is ready after {elapsed}s.", flush=True)
                return
        except Exception as e:
            last_err = e
            now = time.time()
            if now - last_http_error_log_time >= http_error_log_interval:
                last_http_error_log_time = now
                elapsed = int(now - start_time)
                port_listening = _is_port_listening(host, port)
                print(f"[offline_grpo] Still waiting ({elapsed}s). Last HTTP error: {repr(last_err)}. proc_alive={server_proc.poll() is None}, port_listening={port_listening}", flush=True)
        
        time.sleep(interval_sec)
    
    # Timeout reached
    print(f"[offline_grpo] Timeout after {timeout_sec}s waiting for SGLang server at {base}. Last error: {last_err}", flush=True)
    raise TimeoutError(f"SGLang server at {base} did not become ready within {timeout_sec}s. Last error: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--games-per-round", type=int, default=128)
    ap.add_argument("--model-path", default="/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_4800_merged")
    ap.add_argument("--save-root", default="")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--server-port", type=int, default=8000)
    ap.add_argument("--server-wait-seconds", type=int, default=900, help="Max seconds to wait for SGLang server readiness")
    ap.add_argument("--wandb-project", default="offline-grpo")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    # New server control flags
    ap.add_argument("--server-mem-fraction", type=float, default=0.85, help="Static GPU memory fraction reserved by server (mem_fraction_static)")
    ap.add_argument("--server-log-level", type=str, default="debug", help="SGLang server log level: debug|info|warning|error")
    ap.add_argument("--server-enable-torch-compile", action="store_true", help="Enable torch.compile in SGLang server (may slow startup)")
    
    # Resume functionality arguments
    ap.add_argument("--resume", default="", help="Resume from specific run (timestamp/directory name). If empty, resumes from most recent run.")
    ap.add_argument("--no-resume", action="store_true", help="Force start a new run, disable auto-resume")
    
    args = ap.parse_args()
    gpu_string = args.gpus
    gpu_list = [g for g in gpu_string.split(",") if g]
    tp = len(gpu_list)

    # Handle resume logic
    save_root = None
    current_model = args.model_path
    start_round = 0
    
    if not args.no_resume:
        # Try to find existing run to resume
        if args.resume:
            # Resume from specific run
            save_root = find_run_by_name(args.resume)
            if save_root is None:
                print(f"Warning: Could not find run '{args.resume}', starting new run")
            else:
                print(f"Resuming from specified run: {save_root}")
        else:
            # Resume from most recent run
            save_root = find_most_recent_run()
            if save_root is not None:
                print(f"Auto-resuming from most recent run: {save_root}")
    
    # Analyze existing run state if resuming
    if save_root is not None:
        current_round, phase, resume_model, state_info = analyze_run_state(save_root)
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
                # Process the data (trim, etc.)
                train_parquet = round_dir / "train.parquet"
                if train_parquet.exists():
                    df = pd.read_parquet(str(train_parquet))
                    if "sample_weight" in df.columns and len(df) > 0 and not (df["sample_weight"] > 0).any():
                        raise RuntimeError("All rollout sample weights are zero. This likely indicates inference failures or server issues.")
                    
                    lengths = df["input_ids"].apply(lambda x: len(x))
                    pct95 = int(np.percentile(lengths, 95))
                    if pct95 > 5000:
                        pct95 = 5000
                    
                    kept = df[lengths <= pct95]
                    trimmed_path = round_dir / "train_trimmed.parquet"
                    kept.to_parquet(str(trimmed_path))
                    print(f"Trimmed to 95th percentile length={pct95}, kept {len(kept)}/{len(df)} samples -> {trimmed_path}")
                    
                    # Log trim completion
                    log_file = round_dir / "progress.log"
                    with open(log_file, "a") as lf:
                        lf.write(json.dumps({
                            "event": "trim_done",
                            "timestamp": time.time(),
                            "pct95": pct95,
                            "kept": int(len(kept)),
                            "total": int(len(df))
                        }) + "\n")
                    
                    # Update phase to training
                    phase = 'training'
        
        if phase == 'training':
            round_dir = save_root / f"round_{current_round:03d}"
            
            # Run SFT training
            print("Resuming SFT training...")
            train_trimmed_parquet = round_dir / "train_trimmed.parquet"
            
            if not train_trimmed_parquet.exists():
                raise RuntimeError(f"Training data not found: {train_trimmed_parquet}")
            
            # Read training data to get max length
            df = pd.read_parquet(train_trimmed_parquet)
            lengths = df["input_ids"].apply(lambda x: len(x))
            pct95 = int(np.percentile(lengths, 95))
            if pct95 > 5000:
                pct95 = 5000
            max_len_arg = str(pct95)
            
            # Setup SFT training
            if len(gpu_list) >= 2:
                sft_visible = ",".join(gpu_list[:2])
                nproc = 2
            else:
                sft_visible = str(gpu_list[0])
                nproc = 1
            
            print(f"Running SFT training with {nproc} GPU(s): CUDA_VISIBLE_DEVICES={sft_visible}")
            
            save_path = str(round_dir / "checkpoints")
            sft_cmd = [
                "bash", "scripts/sft_qwen/sft_qwen3.sh",
                str(nproc), save_path,
                f"data.train_files={str(train_trimmed_parquet)}",
                f"data.val_files={str(train_trimmed_parquet)}",
                f"model.partial_pretrain={current_model}",
                "trainer.total_epochs=1",
                "trainer.save_freq=500",
                "trainer.test_freq=100",
                f"data.max_length={max_len_arg}",
                "data.custom_cls.path=verl/verl/utils/dataset/pretokenized_sft_dataset.py",
                "data.custom_cls.name=PreTokenizedSFTDataset",
                f"trainer.project_name={args.wandb_project}",
                f"trainer.experiment_name={save_root.name}_round_{current_round}_resume",
            ]
            
            # Setup environment
            sft_env = os.environ.copy()
            test_bin = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin"
            if os.path.isdir(test_bin):
                sft_env["PATH"] = f"{test_bin}:{sft_env.get('PATH','')}"
            if sft_visible:
                sft_env["CUDA_VISIBLE_DEVICES"] = sft_visible
            
            # Run SFT training
            sft_log = round_dir / "sft_train.log"
            run_tee(sft_cmd, logfile=sft_log, env=sft_env)
            
            # Log completion and update model
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
    
    # If no existing run or starting fresh
    if save_root is None:
        if args.save_root:
            save_root = Path(args.save_root)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_root = Path(f"/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo/{ts}")
        save_root.mkdir(parents=True, exist_ok=True)
        print(f"Starting new run: {save_root}")

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

        # 1) Start SGLang server
        print("Starting SGLang server...")
        server_proc = start_sglang_server(
            model_path=current_model, gpus=gpu_string, tp=tp, mem_util=args.server_mem_fraction, port=args.server_port,
            enable_torch_compile=args.server_enable_torch_compile, log_level=args.server_log_level
        )
        
        server_url = f"http://127.0.0.1:{args.server_port}"
        try:
            wait_for_sglang(server_url, server_proc, timeout_sec=args.server_wait_seconds, interval_sec=5)
        except Exception as e:
            print(f"Error waiting for SGLang server: {e}")
            kill_process_tree(server_proc)
            try:
                subprocess.run(["pkill", "-f", "sglang.launch_server"], check=False)
            except Exception:
                pass
            raise

        # 2) Generate rollouts
        print("Generating rollouts...")
        out_parquet = round_dir / "train.parquet"
        test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
        python_bin = test_py if os.path.exists(test_py) else sys.executable
        gen_cmd = [
            python_bin, "scripts/generate_rollouts.py",
            "--server-url", server_url,
            "--model-id", current_model,
            "--out", str(out_parquet),
            "--num-games", str(args.games_per_round // 2),
            "--max-new-tokens", "8192",
            "--group-size", "8",
            "--temperature", "0.7",
            "--top-p", "0.9",
            "--max-model-len", "32768",
        ]
        
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
            try:
                subprocess.run(["pkill", "-f", "sglang"], check=False)
            except Exception:
                pass

        # Process and trim data
        df = pd.read_parquet(str(out_parquet))
        if "sample_weight" in df.columns and len(df) > 0 and not (df["sample_weight"] > 0).any():
            raise RuntimeError("All rollout sample weights are zero. This likely indicates inference failures or server issues.")
        
        lengths = df["input_ids"].apply(lambda x: len(x))
        pct95 = int(np.percentile(lengths, 95))
        if pct95 > 5000:
            pct95 = 5000
        
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

        # Compute and log metrics
        try:
            mean_norm_reward = float(kept["sample_weight"].astype(float).mean()) if "sample_weight" in kept.columns else None
            pos_ratio = float((kept["sample_weight"] > 0).mean()) if "sample_weight" in kept.columns else None
            zero_ratio = float((kept["sample_weight"] == 0).mean()) if "sample_weight" in kept.columns else None
            neg_ratio = float((kept["sample_weight"] < 0).mean()) if "sample_weight" in kept.columns else None
            
            def _get_abs_reward(x):
                try:
                    gi = json.loads(x)
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

        # 3) Run SFT training
        print("Running SFT training...")
        if len(gpu_list) >= 2:
            sft_visible = ",".join(gpu_list[:2])
            nproc = 2
        else:
            sft_visible = str(gpu_list[0])
            nproc = 1
        
        print(f"[HACK] Forcing SFT to use {nproc} GPU(s): CUDA_VISIBLE_DEVICES={sft_visible}")
        save_path = str(round_dir / "checkpoints")
        sft_cmd = [
            "bash", "scripts/sft_qwen/sft_qwen3.sh",
            str(nproc), save_path,
            f"data.train_files={str(train_parquet_for_sft)}",
            f"data.val_files={str(train_parquet_for_sft)}",
            f"model.partial_pretrain={current_model}",
            "trainer.total_epochs=1",
            "trainer.save_freq=500",
            "trainer.test_freq=100",
            f"data.max_length={max_len_arg}",
            "data.custom_cls.path=verl/verl/utils/dataset/pretokenized_sft_dataset.py",
            "data.custom_cls.name=PreTokenizedSFTDataset",
            f"trainer.project_name={args.wandb_project}",
            f"trainer.experiment_name={save_root.name}_round_{r}_2gpu_hack",
        ]
        
        sft_env = os.environ.copy()
        test_bin = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin"
        if os.path.isdir(test_bin):
            sft_env["PATH"] = f"{test_bin}:{sft_env.get('PATH','')}"
        if sft_visible:
            sft_env["CUDA_VISIBLE_DEVICES"] = sft_visible
        
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "sft_hack_2gpus",
                "timestamp": time.time(),
                "visible_gpus": sft_visible,
                "nproc": nproc,
            }) + "\n")
        
        sft_log = round_dir / "sft_train.log"
        run_tee(sft_cmd, logfile=sft_log, env=sft_env)
        
        with open(log_file, "a") as lf:
            lf.write(json.dumps({
                "event": "sft_done",
                "timestamp": time.time(),
                "save_path": save_path
            }) + "\n")

        # 4) Update model path for next round
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
