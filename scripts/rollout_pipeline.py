#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import subprocess
import socket
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
import requests


def _is_port_free(port: int) -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def find_free_ports(n: int = 2, min_port: int = 10000, max_port: int = 65535) -> List[int]:
    """Find n available ports in the given range.

    Returns a list of n ports that are currently free.
    Raises RuntimeError if unable to find enough free ports.
    """
    ports = []
    attempts = 0
    max_attempts = n * 100  # Give up after many failed attempts

    while len(ports) < n and attempts < max_attempts:
        port = random.randint(min_port, max_port)
        if port not in ports and _is_port_free(port):
            ports.append(port)
        attempts += 1

    if len(ports) < n:
        raise RuntimeError(f"Could not find {n} free ports after {max_attempts} attempts")

    print(f"[rollout_pipeline] Found {n} free ports: {ports}")
    return ports


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


def _generate_asymmetric_mode(
    *,
    args,
    round_dir: Path,
    trainee_model: str,
    opponent_model: str,
    gpu_list: List[str],
    python_bin: str,
    opponent_is_openai: bool = False,
) -> Path:
    """Generate rollouts in asymmetric mode (trainee vs opponent).

    If opponent_is_openai is False: uses 2 servers (trainee on GPUs 0-1, opponent on GPUs 2-3).
    If opponent_is_openai is True with 4 GPUs: dual-server mode (2 trainee servers), opponent uses OpenAI API.
    If opponent_is_openai is True with 2 GPUs: single trainee server, opponent uses OpenAI API.
    """
    num_gpus = len(gpu_list)
    use_dual_trainee = opponent_is_openai and num_gpus >= 4

    print("=== ASYMMETRIC MODE: Trainee vs Opponent ===")
    print(f"Trainee model: {trainee_model}")
    print(f"Opponent model: {opponent_model}")
    if opponent_is_openai:
        print("Opponent is an OpenAI API model (no local server needed)")
    if use_dual_trainee:
        print("Using dual-server mode for trainee (2×TP2) for better throughput")

    # Hardcoded path to shy agent prompt
    shy_prompt_path = Path(__file__).parent / "dialop" / "envs" / "data" / "optimization_shy.txt"

    servers_to_cleanup = []

    if use_dual_trainee:
        # Dual trainee servers with OpenAI opponent
        ports = find_free_ports(n=2)
        trainee_port1, trainee_port2 = ports[0], ports[1]

        trainee_log1 = round_dir / "sglang_server_trainee1.log"
        trainee_log2 = round_dir / "sglang_server_trainee2.log"

        print(f"Starting trainee server 1 on GPUs {gpu_list[0]},{gpu_list[1]} (port {trainee_port1})...")
        trainee_server1 = _start_sglang_server(
            model_path=trainee_model,
            gpus=f"{gpu_list[0]},{gpu_list[1]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=trainee_port1,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=trainee_log1,
        )
        servers_to_cleanup.append(trainee_server1)

        print(f"Starting trainee server 2 on GPUs {gpu_list[2]},{gpu_list[3]} (port {trainee_port2})...")
        trainee_server2 = _start_sglang_server(
            model_path=trainee_model,
            gpus=f"{gpu_list[2]},{gpu_list[3]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=trainee_port2,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=trainee_log2,
        )
        servers_to_cleanup.append(trainee_server2)

        trainee_url1 = f"http://127.0.0.1:{trainee_port1}"
        trainee_url2 = f"http://127.0.0.1:{trainee_port2}"

    elif opponent_is_openai:
        # Single trainee server with OpenAI opponent
        trainee_port = find_free_ports(n=1)[0]
        trainee_log = round_dir / "sglang_server_trainee.log"

        print(f"Starting trainee server on GPUs {gpu_list[0]},{gpu_list[1]} (port {trainee_port})...")
        trainee_server = _start_sglang_server(
            model_path=trainee_model,
            gpus=f"{gpu_list[0]},{gpu_list[1]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=trainee_port,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=trainee_log,
        )
        servers_to_cleanup.append(trainee_server)
        trainee_url = f"http://127.0.0.1:{trainee_port}"

    else:
        # Local opponent: trainee on GPUs 0-1, opponent on GPUs 2-3
        asym_ports = find_free_ports(n=2)
        trainee_port, opponent_port = asym_ports[0], asym_ports[1]

        trainee_log = round_dir / "sglang_server_trainee.log"
        opponent_log = round_dir / "sglang_server_opponent.log"

        print(f"Starting trainee server on GPUs {gpu_list[0]},{gpu_list[1]} (port {trainee_port})...")
        trainee_server = _start_sglang_server(
            model_path=trainee_model,
            gpus=f"{gpu_list[0]},{gpu_list[1]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=trainee_port,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=trainee_log,
        )
        servers_to_cleanup.append(trainee_server)

        print(f"Starting opponent server on GPUs {gpu_list[2]},{gpu_list[3]} (port {opponent_port})...")
        opponent_server = _start_sglang_server(
            model_path=opponent_model,
            gpus=f"{gpu_list[2]},{gpu_list[3]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=opponent_port,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=opponent_log,
        )
        servers_to_cleanup.append(opponent_server)

        trainee_url = f"http://127.0.0.1:{trainee_port}"
        opponent_url = f"http://127.0.0.1:{opponent_port}"

    try:
        # Wait for server(s)
        import threading
        errors = []

        if use_dual_trainee:
            print("Waiting for both trainee servers to become ready...")

            def wait_trainee1():
                try:
                    _wait_for_sglang(trainee_url1, trainee_server1, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print(f"Trainee server 1 ready on port {trainee_port1}!")
                except Exception as e:
                    errors.append(("trainee1", e))

            def wait_trainee2():
                try:
                    _wait_for_sglang(trainee_url2, trainee_server2, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print(f"Trainee server 2 ready on port {trainee_port2}!")
                except Exception as e:
                    errors.append(("trainee2", e))

            t1 = threading.Thread(target=wait_trainee1)
            t2 = threading.Thread(target=wait_trainee2)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        elif opponent_is_openai:
            print("Waiting for trainee server to become ready...")
            try:
                _wait_for_sglang(trainee_url, trainee_server, timeout_sec=args.server_wait_seconds, interval_sec=5)
                print(f"Trainee server ready on port {trainee_port}!")
            except Exception as e:
                errors.append(("trainee", e))

        else:
            print("Waiting for both servers to become ready...")

            def wait_trainee():
                try:
                    _wait_for_sglang(trainee_url, trainee_server, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print(f"Trainee server ready on port {trainee_port}!")
                except Exception as e:
                    errors.append(("trainee", e))

            def wait_opponent():
                try:
                    _wait_for_sglang(opponent_url, opponent_server, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print(f"Opponent server ready on port {opponent_port}!")
                except Exception as e:
                    errors.append(("opponent", e))

            t1 = threading.Thread(target=wait_trainee)
            t2 = threading.Thread(target=wait_opponent)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        if errors:
            error_msg = "; ".join([f"{name}: {err}" for name, err in errors])
            raise RuntimeError(f"Server startup failed: {error_msg}")

        # Generate with asymmetric mode
        group_size = 8
        out_parquet = round_dir / "train.parquet"

        if use_dual_trainee:
            # Dual-server mode: split games between two trainee servers
            num_games_per_server = args.games_per_round // 2
            out1 = round_dir / "train_server1.parquet"
            out2 = round_dir / "train_server2.parquet"

            print(f"Launching parallel asymmetric generation: {num_games_per_server} unique games per server...")

            base_cmd = [
                python_bin, "scripts/generate_rollouts.py",
                "--mode", "asymmetric",
                "--model-id", trainee_model,
                "--opponent-model-id", opponent_model,
                "--opponent-instructions", str(shy_prompt_path),
                "--opponent-player-type", "openai",
                "--max-new-tokens", "8192",
                "--max-thinking-tokens", "7000",
                "--group-size", str(group_size),
                "--temperature", str(getattr(args, 'rollout_temperature', 0.7)),
                "--top-p", "0.9",
                "--max-model-len", "32768",
                "--max-turns", str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn", str(getattr(args, 'max_retries_per_turn', 2)),
            ]

            gen_cmd1 = base_cmd + [
                "--server-url", trainee_url1,
                "--out", str(out1),
                "--num-games", str(num_games_per_server),
            ]

            gen_cmd2 = base_cmd + [
                "--server-url", trainee_url2,
                "--out", str(out2),
                "--num-games", str(num_games_per_server),
            ]

            # Run both in parallel
            proc1 = subprocess.Popen(gen_cmd1)
            proc2 = subprocess.Popen(gen_cmd2)

            try:
                ret1 = proc1.wait()
                ret2 = proc2.wait()

                if ret1 != 0 or ret2 != 0:
                    if out1.exists() or out2.exists():
                        print(f"\nGeneration partially failed but found partial results:")
                        if out1.exists():
                            print(f"  Server 1 partial results: {out1}")
                        if out2.exists():
                            print(f"  Server 2 partial results: {out2}")
                        print(f"  Merging available partial results...")
                    else:
                        raise RuntimeError(f"Generation failed: server1={ret1}, server2={ret2}")
            except KeyboardInterrupt:
                print(f"\nInterrupt received during generation. Terminating processes...")
                proc1.terminate()
                proc2.terminate()
                try:
                    proc1.wait(timeout=10)
                    proc2.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc1.kill()
                    proc2.kill()
                print(f"Processes terminated. Checking for partial results...")
                if not (out1.exists() or out2.exists()):
                    raise RuntimeError("No partial results found after interrupt")

            # Merge parquets and fix game_ids
            print("Merging outputs from both servers...")
            dfs_to_merge = []

            if out1.exists():
                try:
                    df1 = pd.read_parquet(str(out1))
                    dfs_to_merge.append(df1)
                    print(f"  Server 1: {len(df1)} sequences")
                except Exception as e:
                    print(f"  Warning: Could not load server 1 results: {e}")

            if out2.exists():
                try:
                    df2 = pd.read_parquet(str(out2))
                    # Shift server2 game_ids to continue from server1
                    offset = num_games_per_server * group_size
                    print(f"  Server 2: {len(df2)} sequences (offsetting game_ids by {offset})")
                    df2["game_id"] = df2["game_id"] + offset
                    dfs_to_merge.append(df2)
                except Exception as e:
                    print(f"  Warning: Could not load server 2 results: {e}")

            if not dfs_to_merge:
                raise RuntimeError("No results available from either server")

            merged = pd.concat(dfs_to_merge, ignore_index=True)
            merged.to_parquet(str(out_parquet))
            print(f"Merged {len(merged)} total sequences → {out_parquet}")

        else:
            # Single-server mode (either single trainee or trainee+opponent)
            print(f"Launching asymmetric generation: {args.games_per_round} unique games...")
            gen_cmd = [
                python_bin, "scripts/generate_rollouts.py",
                "--mode", "asymmetric",
                "--server-url", trainee_url,
                "--model-id", trainee_model,
                "--opponent-model-id", opponent_model,
                "--opponent-instructions", str(shy_prompt_path),
                "--out", str(out_parquet),
                "--num-games", str(args.games_per_round),
                "--max-new-tokens", "8192",
                "--max-thinking-tokens", "7000",
                "--group-size", str(group_size),
                "--temperature", str(getattr(args, 'rollout_temperature', 0.7)),
                "--top-p", "0.9",
                "--max-model-len", "32768",
                "--max-turns", str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn", str(getattr(args, 'max_retries_per_turn', 2)),
            ]

            # Add opponent connection info based on type
            if opponent_is_openai:
                gen_cmd += ["--opponent-player-type", "openai"]
            else:
                gen_cmd += ["--opponent-server-url", opponent_url]

            # Run generation
            proc = subprocess.Popen(gen_cmd)

            try:
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"generate_rollouts.py exited with code {ret}")
            except KeyboardInterrupt:
                print("\n[rollout_pipeline] Caught interrupt, terminating generate_rollouts...")
                proc.terminate()
                proc.wait(timeout=10)
                raise

        if not out_parquet.exists():
            raise FileNotFoundError(f"Expected output {out_parquet} not found after generation")

        return out_parquet

    finally:
        # Cleanup servers
        print("Stopping servers...")
        for server in servers_to_cleanup:
            _kill_process_tree(server)
        try:
            subprocess.run(["pkill", "-f", "sglang"], check=False)
        except Exception as e:
            print(f"Warning: pkill failed: {e}")


def function_A_start_server_and_generate(
    *,
    args,
    round_dir: Path,
    current_model: str,
) -> Path:
    """Start SGLang server(s), generate rollouts via scripts/generate_rollouts.py, log events, stop server(s).

    For self-play with 4 GPUs: Uses 2 servers (TP=2 each) to avoid all-reduce overhead.
    For asymmetric mode: Always uses 2 servers (trainee + opponent).
    For other GPU counts: Uses single server with appropriate TP.

    Returns path to the raw parquet (train.parquet).
    """
    gpu_string = args.gpus
    gpu_list = [g.strip() for g in gpu_string.split(",") if g.strip()]
    num_gpus = len(gpu_list)

    test_py = "/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python"
    python_bin = test_py if os.path.exists(test_py) else sys.executable

    # Check if asymmetric mode (trainee vs opponent)
    is_asymmetric = hasattr(args, 'opponent_model') and args.opponent_model is not None
    opponent_is_openai = getattr(args, 'opponent_is_openai', False)

    # Asymmetric mode: OpenAI opponent needs 2+ GPUs, local opponent needs 4 GPUs
    if is_asymmetric:
        if opponent_is_openai and num_gpus >= 2:
            return _generate_asymmetric_mode(
                args=args,
                round_dir=round_dir,
                trainee_model=current_model,
                opponent_model=args.opponent_model,
                gpu_list=gpu_list,
                python_bin=python_bin,
                opponent_is_openai=True,
            )
        elif not opponent_is_openai and num_gpus == 4:
            return _generate_asymmetric_mode(
                args=args,
                round_dir=round_dir,
                trainee_model=current_model,
                opponent_model=args.opponent_model,
                gpu_list=gpu_list,
                python_bin=python_bin,
                opponent_is_openai=False,
            )

    # Special case: 4 GPUs → use 2 servers to avoid cross-bridge all-reduce (if enabled)
    if num_gpus == 4 and args.dual_server:
        print("Detected 4 GPUs: using dual-server mode (2×TP2) for better performance...")

        # Find two free ports for dual-server mode
        dual_ports = find_free_ports(n=2)
        port1, port2 = dual_ports[0], dual_ports[1]

        # Start both servers with separate log files
        server1_log = round_dir / "sglang_server1.log"
        server2_log = round_dir / "sglang_server2.log"

        print(f"Starting server 1 on GPUs 0,1 (port {port1})...")
        server1 = _start_sglang_server(
            model_path=current_model,
            gpus=f"{gpu_list[0]},{gpu_list[1]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=port1,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=server1_log,
        )

        print(f"Starting server 2 on GPUs 2,3 (port {port2})...")
        server2 = _start_sglang_server(
            model_path=current_model,
            gpus=f"{gpu_list[2]},{gpu_list[3]}",
            tp=2,
            mem_util=args.server_mem_fraction,
            port=port2,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
            log_file=server2_log,
        )

        try:
            # Wait for both servers in parallel
            print("Waiting for both servers to become ready...")
            import threading

            server1_url = f"http://127.0.0.1:{port1}"
            server2_url = f"http://127.0.0.1:{port2}"

            errors = []

            def wait_server1():
                try:
                    _wait_for_sglang(server1_url, server1, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print(f"Server 1 ready on port {port1}!")
                except Exception as e:
                    errors.append(("server1", e))

            def wait_server2():
                try:
                    _wait_for_sglang(server2_url, server2, timeout_sec=args.server_wait_seconds, interval_sec=5)
                    print(f"Server 2 ready on port {port2}!")
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
                "--server-url", server1_url,
                "--model-id", current_model,
                "--out", str(out1),
                "--num-games", str(num_games_per_server),
                "--max-new-tokens", "8192",
                "--max-thinking-tokens", "7000",
                "--group-size", str(group_size),
                "--temperature", str(getattr(args, 'rollout_temperature', 0.7)),
                "--top-p", "0.9",
                "--max-model-len", "32768",
                "--max-turns", str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn", str(getattr(args, 'max_retries_per_turn', 2)),
            ]

            gen_cmd2 = [
                python_bin, "scripts/generate_rollouts.py",
                "--server-url", server2_url,
                "--model-id", current_model,
                "--out", str(out2),
                "--num-games", str(num_games_per_server),
                "--max-new-tokens", "8192",
                "--max-thinking-tokens", "7000",
                "--group-size", str(group_size),
                "--temperature", str(getattr(args, 'rollout_temperature', 0.7)),
                "--top-p", "0.9",
                "--max-model-len", "32768",
                "--max-turns", str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn", str(getattr(args, 'max_retries_per_turn', 2)),
            ]

            # Run both in parallel
            proc1 = subprocess.Popen(gen_cmd1)
            proc2 = subprocess.Popen(gen_cmd2)

            # Wait for completion, handling interrupts
            try:
                ret1 = proc1.wait()
                ret2 = proc2.wait()

                if ret1 != 0 or ret2 != 0:
                    # Check if partial results were created
                    if out1.exists() or out2.exists():
                        print(f"\nGeneration partially failed but found partial results:")
                        if out1.exists():
                            print(f"  Server 1 partial results: {out1}")
                        if out2.exists():
                            print(f"  Server 2 partial results: {out2}")
                        print(f"  Merging available partial results...")
                    else:
                        raise RuntimeError(f"Generation failed: server1={ret1}, server2={ret2}")
            except KeyboardInterrupt:
                print(f"\nInterrupt received during generation. Terminating processes...")
                proc1.terminate()
                proc2.terminate()
                try:
                    proc1.wait(timeout=10)
                    proc2.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc1.kill()
                    proc2.kill()
                print(f"Processes terminated. Checking for partial results...")
                # Continue to merge whatever was generated
                if not (out1.exists() or out2.exists()):
                    raise RuntimeError("No partial results found after interrupt")

            # Merge parquets and fix game_ids (handle partial results)
            print("Merging outputs from both servers...")
            dfs_to_merge = []

            if out1.exists():
                try:
                    df1 = pd.read_parquet(str(out1))
                    dfs_to_merge.append(df1)
                    print(f"  Server 1: {len(df1)} sequences")
                except Exception as e:
                    print(f"  Warning: Could not load server 1 results: {e}")

            if out2.exists():
                try:
                    df2 = pd.read_parquet(str(out2))
                    # Shift server2 game_ids to continue from server1
                    offset = num_games_per_server * group_size
                    print(f"  Server 2: {len(df2)} sequences (offsetting game_ids by {offset})")
                    df2["game_id"] = df2["game_id"] + offset
                    dfs_to_merge.append(df2)
                except Exception as e:
                    print(f"  Warning: Could not load server 2 results: {e}")

            if not dfs_to_merge:
                raise RuntimeError("No results available from either server")

            # Merge
            df = pd.concat(dfs_to_merge, ignore_index=True)
            out_parquet = round_dir / "train.parquet"
            df.to_parquet(str(out_parquet))
            print(f"Merged {len(df)} sequences into {out_parquet}")

            # Cleanup temp files (only if they exist)
            if out1.exists():
                out1.unlink()
            if out2.exists():
                out2.unlink()

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

        # Find a free port for single-server mode
        single_port = find_free_ports(n=1)[0]

        server_proc = _start_sglang_server(
            model_path=current_model,
            gpus=gpu_string,
            tp=tp,
            mem_util=args.server_mem_fraction,
            port=single_port,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            log_level=args.server_log_level,
        )

        server_url = f"http://127.0.0.1:{single_port}"

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
                "--max-thinking-tokens",
                "7000",
                "--group-size",
                "8",
                "--temperature",
                str(getattr(args, 'rollout_temperature', 0.7)),
                "--top-p",
                "0.9",
                "--max-model-len",
                "32768",
                "--max-turns",
                str(getattr(args, 'max_turns', 10)),
                "--max-retries-per-turn",
                str(getattr(args, 'max_retries_per_turn', 2)),
            ]

            # Run generation with interrupt handling
            try:
                _run(gen_cmd)
            except KeyboardInterrupt:
                print(f"\nInterrupt received during generation. Checking for partial results...")
                if not out_parquet.exists():
                    raise RuntimeError("No partial results found after interrupt")
                print(f"Found partial results at {out_parquet}")
            except subprocess.CalledProcessError as e:
                # Check if partial results exist
                if out_parquet.exists():
                    print(f"\nGeneration failed but found partial results at {out_parquet}")
                    print(f"Continuing with partial data...")
                else:
                    raise

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


def _export_single_player_perspective(df: pd.DataFrame, output_path: Path, n_samples: int = 10) -> None:
    """Export games from a single player's perspective showing what they actually see.

    This helps debug what context the model receives during gameplay.

    Args:
        df: DataFrame with game data (can be messages-format or pretokenized-format)
        output_path: Path to write single_player_examples.txt
        n_samples: Number of random games to export
    """
    try:
        # Check data format
        has_messages = "messages" in df.columns
        has_conversation = "conversation" in df.columns

        if not has_messages and not has_conversation:
            print("Warning: No 'messages' or 'conversation' column found, skipping single-player perspective export")
            return

        # Sample random games
        unique_games = df.groupby('game_id').first().reset_index()
        n_samples = min(n_samples, len(unique_games))
        sampled_games = unique_games.sample(n=n_samples, random_state=42)

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SINGLE PLAYER PERSPECTIVE EXAMPLES\n")
            f.write("(What the model actually sees during gameplay)\n")
            f.write("=" * 80 + "\n\n")

            for idx, row in sampled_games.iterrows():
                game_id = row.get("game_id", "?")
                player_name = row.get("player_name", "?")
                reward = row.get("game_normalized_reward", row.get("reward", 0))

                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Game ID: {game_id}, Player: {player_name}\n")
                f.write(f"Reward: {reward:.4f}\n")
                f.write("=" * 80 + "\n\n")

                # Extract conversation data
                try:
                    # Try messages column first (SFT format)
                    if has_messages:
                        messages = row.get("messages", [])
                        if isinstance(messages, str):
                            messages = json.loads(messages)

                        if not messages:
                            f.write("[No messages found]\n\n")
                            continue

                        f.write(f"Total messages in conversation: {len(messages)}\n\n")

                        for msg_idx, msg in enumerate(messages):
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")

                            f.write(f"--- Message {msg_idx} ({role}) ---\n")
                            f.write(content)
                            f.write("\n\n")

                    # Try conversation column (pretokenized format)
                    elif has_conversation:
                        conversation = row.get("conversation", [])
                        if isinstance(conversation, str):
                            conversation = json.loads(conversation)

                        if not conversation:
                            f.write("[No conversation found]\n\n")
                            continue

                        f.write(f"Total turns in conversation: {len(conversation)}\n\n")

                        for turn_idx, turn in enumerate(conversation):
                            player = turn.get("player", "unknown")
                            message = turn.get("message", "")
                            turn_num = turn.get("turn", turn_idx)

                            f.write(f"--- Turn {turn_num} ({player}) ---\n")
                            f.write(message)
                            f.write("\n\n")

                except Exception as e:
                    f.write(f"[Error parsing conversation: {e}]\n\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"Exported {n_samples} single-player perspective examples to {output_path}")

    except Exception as e:
        print(f"Warning: Failed to export single-player examples: {e}")
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
    kl_metadata_path = round_dir / "kl_metadata.json"

    if not train_parquet.exists():
        raise RuntimeError(f"Expected train.parquet at {train_parquet} but not found")

    # Extract current KL settings
    use_kl = getattr(args, 'use_kl', False) if args else False
    kl_coef = getattr(args, 'kl_coef', 0.001) if args else 0.001
    kl_method = getattr(args, 'kl_method', 'hf_dataparallel') if args else 'hf_dataparallel'
    reference_model = getattr(args, 'reference_model', None) if args else None

    current_kl_settings = {
        "use_kl": use_kl,
        "kl_coef": kl_coef if use_kl else None,
        "kl_method": kl_method if use_kl else None,
        "reference_model": reference_model if use_kl else None,
    }

    # Check if KL settings changed from previous processing
    kl_settings_changed = False
    if kl_metadata_path.exists():
        try:
            with open(kl_metadata_path, 'r') as f:
                prev_kl_settings = json.load(f)
            if prev_kl_settings != current_kl_settings:
                print(f"\nKL settings changed:")
                print(f"  Previous: {prev_kl_settings}")
                print(f"  Current: {current_kl_settings}")
                print(f"  Will reprocess from checkpoint...\n")
                kl_settings_changed = True
        except Exception as e:
            print(f"Warning: Could not read KL metadata: {e}")
            kl_settings_changed = True

    # Check if we can skip reprocessing entirely
    train_trimmed = round_dir / "train_trimmed.parquet"
    val_trimmed = round_dir / "val_trimmed.parquet"
    examples_path = round_dir / "examples.txt"
    checkpoint_path = round_dir / "train_before_kl.parquet"

    # Skip reprocessing if:
    # 1. KL metadata exists and matches current settings
    # 2. All output files exist
    if (kl_metadata_path.exists() and not kl_settings_changed and
        train_trimmed.exists() and val_trimmed.exists()):
        print(f"\n{'='*80}")
        print(f"Processing already complete with current KL settings - skipping reprocessing")
        print(f"  KL enabled: {use_kl}")
        if use_kl:
            print(f"  KL coef: {kl_coef}")
            print(f"  KL method: {kl_method}")
        print(f"{'='*80}\n")

        # Load trimmed data to compute stats
        df = pd.read_parquet(str(train_parquet))
        train_df = pd.read_parquet(str(train_trimmed))
        val_df = pd.read_parquet(str(val_trimmed))

        # Extract metrics and return
        game_rewards = _extract_reward_series_from_df(df)
        if game_rewards is None or len(game_rewards) == 0:
            raise RuntimeError("Failed to extract game rewards from data")

        game_reward_mean = float(np.mean(game_rewards))
        game_reward_std = float(np.std(game_rewards))
        game_reward_p10 = float(np.percentile(game_rewards, 10))
        game_reward_p25 = float(np.percentile(game_rewards, 25))
        game_reward_p50 = float(np.percentile(game_rewards, 50))
        game_reward_p75 = float(np.percentile(game_rewards, 75))
        game_reward_p90 = float(np.percentile(game_rewards, 90))
        perfect_score_ratio = float(np.mean(game_rewards >= 1.0))

        grpo_weights = df["sample_weight"].astype(float) if "sample_weight" in df.columns else np.zeros(len(df))
        grpo_weight_mean = float(np.mean(grpo_weights))
        grpo_weight_pos_ratio = float(np.mean(grpo_weights > 0))
        grpo_weight_neg_ratio = float(np.mean(grpo_weights < 0))
        grpo_weight_zero_ratio = float(np.mean(grpo_weights == 0))

        # Get trim threshold from trimmed data
        lengths = train_df["input_ids"].apply(lambda x: len(x))
        trim_threshold = int(lengths.max())

        # Compute perfect score ratio after trim
        kept_game_rewards = _extract_reward_series_from_df(train_df)
        perfect_score_ratio_after_trim = float(np.mean(kept_game_rewards >= 1.0)) if kept_game_rewards is not None else perfect_score_ratio

        # Compute failure ratio (rough estimate from full data)
        failure_ratio = 0.0
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
            failure_count = 0
            for idx, row in df.head(100).iterrows():  # Sample first 100 for speed
                text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                if 'I need to think about this.' in text:
                    failure_count += 1
            failure_ratio = float(failure_count / 100)
        except Exception:
            pass

        zero_reward_ratio = float(np.mean(game_rewards == 0.0))
        zero_reward_count = int(np.sum(game_rewards == 0.0))

        # Compute conversation length mean
        conversation_length_mean = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
            conversation_lengths = []

            for idx, row in df.head(100).iterrows():  # Sample for speed
                try:
                    game_info = json.loads(row['game_info']) if isinstance(row['game_info'], str) else row['game_info']
                    if 'turn_count' in game_info:
                        conversation_lengths.append(game_info['turn_count'])
                    else:
                        text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                        assistant_count = text.count('<|im_start|>assistant')
                        user_count = text.count('<|im_start|>user')
                        conversation_lengths.append(max(assistant_count, user_count))
                except Exception:
                    text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                    assistant_count = text.count('<|im_start|>assistant')
                    user_count = text.count('<|im_start|>user')
                    conversation_lengths.append(max(assistant_count, user_count))

            if conversation_lengths:
                conversation_length_mean = float(np.mean(conversation_lengths))
        except Exception:
            pass

        # Ensure data_metadata.json exists (may not exist for old runs)
        data_metadata_path = round_dir / "data_metadata.json"
        if not data_metadata_path.exists():
            # Create metadata from what we can infer
            data_metadata = {
                "trim_threshold": trim_threshold,
                "filter_positive_only": True,  # Default assumption
                "filter_percentile": 0.0,
            }
            with open(data_metadata_path, 'w') as f:
                json.dump(data_metadata, f, indent=2)
            print(f"Created missing data metadata at {data_metadata_path}")

        stats = RolloutStats(
            raw_parquet=train_parquet,
            trimmed_parquet=train_trimmed,
            val_parquet=val_trimmed,
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
            zero_reward_ratio=zero_reward_ratio,
            total_sequences=len(df),
            kept_sequences=len(train_df),
            trim_threshold=trim_threshold,
            grpo_weight_mean=grpo_weight_mean,
            grpo_weight_pos_ratio=grpo_weight_pos_ratio,
            grpo_weight_neg_ratio=grpo_weight_neg_ratio,
            grpo_weight_zero_ratio=grpo_weight_zero_ratio,
            failure_ratio=failure_ratio,
            conversation_length_mean=conversation_length_mean,
            zero_reward_count=zero_reward_count,
        )
        return stats

    # Need to process - load appropriate data
    if checkpoint_path.exists():
        # Resume or KL settings changed: Load from checkpoint and reprocess
        print(f"Loading from checkpoint for reprocessing: {checkpoint_path}")
        df = pd.read_parquet(str(checkpoint_path))
    else:
        # Fresh generation: Load from train.parquet and create checkpoint
        df = pd.read_parquet(str(train_parquet))
        if "sample_weight" in df.columns and len(df) > 0 and not (df["sample_weight"] > 0).any():
            raise RuntimeError("All rollout sample weights are zero. This likely indicates inference failures or server issues.")

        # Save checkpoint before KL computation (allows iteration on KL logic without re-generating)
        df.to_parquet(str(checkpoint_path))
        print(f"Saved rollout checkpoint (before KL): {checkpoint_path}")
    if use_kl:
        # Validate that reference model is specified
        if not (args and hasattr(args, 'reference_model') and args.reference_model):
            raise ValueError(
                "--use-kl flag is set but --reference-model is not specified. "
                "Please provide --reference-model to enable KL divergence penalty."
            )
        print(f"\n{'='*80}")
        print(f"Applying KL divergence penalty from reference model...")
        print(f"  Reference model: {args.reference_model}")
        print(f"  KL coefficient: {getattr(args, 'kl_coef', 0.001)}")
        print(f"{'='*80}\n")

        # Compute reference logprobs
        # Note: server_port is not currently used by either method (hf_dataparallel or sglang Runtime)
        # but we find a free port anyway for future-proofing
        ref_port = find_free_ports(n=1)[0]
        ref_logprobs_list = compute_reference_logprobs(
            df=df,
            reference_model_path=args.reference_model,
            gpus=args.gpus,
            method=kl_method,
            # Parameters below only used by sglang method
            server_port=ref_port,
            server_wait_seconds=args.server_wait_seconds,
            server_mem_fraction=args.server_mem_fraction,
            server_log_level=args.server_log_level,
            enable_torch_compile=args.server_enable_torch_compile,
            disable_cuda_graph=args.server_disable_cuda_graph,
            round_dir=round_dir,
        )

        # Calculate KL penalty and adjust rewards
        kl_coef = getattr(args, 'kl_coef', 0.001)
        kl_penalties = []

        # Diagnostic counters
        total_policy_tokens = 0
        total_ref_tokens = 0
        length_mismatches = 0
        empty_policy = 0
        empty_ref = 0

        for idx, row in df.iterrows():
            # Parse policy logprobs from JSON
            policy_logprobs = json.loads(row["policy_logprobs"]) if isinstance(row["policy_logprobs"], str) else row["policy_logprobs"]
            ref_logprobs = ref_logprobs_list[idx]

            # Track statistics
            total_policy_tokens += len(policy_logprobs)
            total_ref_tokens += len(ref_logprobs)
            if len(policy_logprobs) == 0:
                empty_policy += 1
            if len(ref_logprobs) == 0:
                empty_ref += 1

            # Debug first few sequences
            if idx < 3:
                print(f"\n[DEBUG] Sequence {idx}:")
                print(f"  Policy logprobs: {len(policy_logprobs)} tokens, first 5: {policy_logprobs[:5]}")
                print(f"  Ref logprobs: {len(ref_logprobs)} tokens, first 5: {ref_logprobs[:5]}")

            # Fail loudly on length mismatch
            if len(policy_logprobs) != len(ref_logprobs):
                raise RuntimeError(
                    f"Length mismatch at sequence {idx}: "
                    f"policy has {len(policy_logprobs)} tokens, "
                    f"reference has {len(ref_logprobs)} tokens. "
                    f"This indicates a bug in logprob collection."
                )

            # Fail loudly on empty logprobs
            if len(policy_logprobs) == 0:
                raise RuntimeError(
                    f"Empty policy logprobs at sequence {idx}. "
                    f"This indicates logprob collection failed during generation."
                )

            # Compute KL divergence: KL = policy_logprob - ref_logprob
            kl_per_token = [
                policy_logprobs[i] - ref_logprobs[i]
                for i in range(len(policy_logprobs))
            ]
            kl_penalty = sum(kl_per_token)

            # Debug output
            if idx < 3:
                print(f"  KL per token (first 5): {kl_per_token[:5]}")
                print(f"  Total KL: {kl_penalty:.4f}")
                print(f"  Avg KL per token: {kl_penalty/len(policy_logprobs):.4f}")

            kl_penalties.append(kl_penalty)

            # Adjust reward: reward_adjusted = reward - kl_coef * kl_penalty
            # sample_weight currently holds the raw game reward
            original_reward = float(row["sample_weight"])
            adjusted_reward = original_reward - kl_coef * kl_penalty
            df.at[idx, "sample_weight"] = adjusted_reward

        # Save KL statistics
        mean_kl = np.mean(kl_penalties)
        std_kl = np.std(kl_penalties)
        avg_policy_tokens = total_policy_tokens / len(df) if len(df) > 0 else 0
        avg_ref_tokens = total_ref_tokens / len(df) if len(df) > 0 else 0

        print(f"\nKL divergence statistics:")
        print(f"  Mean KL: {mean_kl:.4f}")
        print(f"  Std KL: {std_kl:.4f}")
        print(f"  Mean penalty: {kl_coef * mean_kl:.4f}")
        print(f"\nDiagnostic information:")
        print(f"  Total sequences: {len(df)}")
        print(f"  Avg policy tokens per sequence: {avg_policy_tokens:.1f}")
        print(f"  Avg ref tokens per sequence: {avg_ref_tokens:.1f}")
        print(f"  Mean KL per token: {mean_kl / avg_policy_tokens if avg_policy_tokens > 0 else 0:.6f}")
        print(f"  Length mismatches: {length_mismatches}/{len(df)}")
        print(f"  Empty policy logprobs: {empty_policy}/{len(df)}")
        print(f"  Empty ref logprobs: {empty_ref}/{len(df)}")

        # Overwrite the parquet with adjusted rewards
        df.to_parquet(str(train_parquet))
        print(f"Saved adjusted rewards to {train_parquet}\n")

    # Save KL metadata for future resume checks
    with open(kl_metadata_path, 'w') as f:
        json.dump(current_kl_settings, f, indent=2)
    print(f"Saved KL metadata to {kl_metadata_path}")

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
    """
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
    """

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

    # Filter for positive GRPO-normalized examples only (if enabled)
    filter_positive_only = getattr(args, 'filter_positive_only', True) if args else True
    filter_percentile = getattr(args, 'filter_percentile', 0.0) if args else 0.0

    # Save data processing metadata (trim_threshold, etc.) for resume
    data_metadata_path = round_dir / "data_metadata.json"
    data_metadata = {
        "trim_threshold": pct95,
        "filter_positive_only": filter_positive_only,
        "filter_percentile": filter_percentile,
    }
    with open(data_metadata_path, 'w') as f:
        json.dump(data_metadata, f, indent=2)
    print(f"Saved data metadata to {data_metadata_path}")
    length_trimmed_count_before_filtering = len(kept)

    if filter_positive_only:
        # Count positive examples before filtering
        grpo_weights_trimmed = kept["sample_weight"].astype(float)
        positive_mask = grpo_weights_trimmed > 0
        num_positive = int(positive_mask.sum())
        num_negative = int((grpo_weights_trimmed < 0).sum())
        num_zero = int((grpo_weights_trimmed == 0).sum())

        print(f"\nFiltering for positive GRPO-normalized examples...")
        print(f"  Before filtering: {len(kept)} sequences")
        print(f"    Positive (above group mean): {num_positive} ({100*num_positive/len(kept):.1f}%)")
        print(f"    Negative (below group mean): {num_negative} ({100*num_negative/len(kept):.1f}%)")
        print(f"    Zero (at group mean): {num_zero} ({100*num_zero/len(kept):.1f}%)")

        # Keep only positive examples
        kept = kept[positive_mask].reset_index(drop=True)
        print(f"  After positive-only filter: {len(kept)} sequences (kept {100*len(kept)/length_trimmed_count_before_filtering:.1f}%)")
    else:
        print(f"\nPositive-only filtering disabled")

    # Apply percentile-based filtering (if enabled)
    if filter_percentile > 0.0 and filter_percentile <= 1.0 and len(kept) > 0:
        grpo_weights = kept["sample_weight"].astype(float)
        percentile_threshold = np.percentile(grpo_weights, filter_percentile * 100)
        percentile_mask = grpo_weights >= percentile_threshold
        num_above_percentile = int(percentile_mask.sum())

        print(f"\nApplying percentile-based filtering...")
        print(f"  Percentile threshold: {filter_percentile*100:.0f}th percentile (GRPO weight >= {percentile_threshold:.6f})")
        print(f"  Before filtering: {len(kept)} sequences")
        print(f"  Above threshold: {num_above_percentile} ({100*num_above_percentile/len(kept):.1f}%)")

        kept = kept[percentile_mask].reset_index(drop=True)
        print(f"  After percentile filter: {len(kept)} sequences")
    elif filter_percentile > 0.0:
        print(f"\nPercentile filtering requested but invalid threshold ({filter_percentile}), skipping")

    # Warn if we filtered out too many examples
    if len(kept) < 100:
        print(f"\n  WARNING: Only {len(kept)} examples remain after filtering.")
        print(f"  This may be too few for effective training. Consider:")
        print(f"    - Using --no-filter-positive-only to train on all examples (or switch to PPO/GRPO mode)")
        print(f"    - Lowering --filter-percentile (currently {filter_percentile})")
        print(f"    - Increasing --games-per-round to generate more rollouts")
        print(f"  Note: PPO/GRPO mode automatically disables quality-based filtering\n")

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

    # Compute zero reward metrics (both count and ratio)
    zero_reward_ratio = float(np.mean(game_rewards == 0.0))
    zero_reward_count = int(np.sum(game_rewards == 0.0))

    # Compute conversation length statistics
    conversation_length_mean = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        conversation_lengths = []

        for idx, row in df.iterrows():
            # Extract conversation length from game_info if available
            try:
                game_info = json.loads(row['game_info']) if isinstance(row['game_info'], str) else row['game_info']
                if 'turn_count' in game_info:
                    conversation_lengths.append(game_info['turn_count'])
                else:
                    # Fallback: count role markers in tokenized sequence
                    text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                    assistant_count = text.count('<|im_start|>assistant')
                    user_count = text.count('<|im_start|>user')
                    conversation_lengths.append(max(assistant_count, user_count))
            except Exception:
                # Last resort: count from tokens
                text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                assistant_count = text.count('<|im_start|>assistant')
                user_count = text.count('<|im_start|>user')
                conversation_lengths.append(max(assistant_count, user_count))

        if conversation_lengths:
            conversation_length_mean = float(np.mean(conversation_lengths))
    except Exception as e:
        print(f"Warning: Could not compute conversation length mean: {e}")

    # Export example games (from original df including failures, for debugging)
    examples_path = round_dir / "examples.txt"
    _export_example_games(df, examples_path, n_per_category=5)

    # Export single-player perspective examples (shows what model actually sees)
    single_player_examples_path = round_dir / "single_player_examples.txt"
    _export_single_player_perspective(df, single_player_examples_path, n_samples=10)

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
        zero_reward_ratio=zero_reward_ratio,
        total_sequences=len(df),  # Original count including failures
        kept_sequences=len(kept),  # After filtering failures AND length trimming
        trim_threshold=pct95,
        grpo_weight_mean=grpo_weight_mean,
        grpo_weight_pos_ratio=grpo_weight_pos_ratio,
        grpo_weight_neg_ratio=grpo_weight_neg_ratio,
        grpo_weight_zero_ratio=grpo_weight_zero_ratio,
        failure_ratio=failure_ratio,  # Ratio of failed generations in raw data
        conversation_length_mean=conversation_length_mean,
        zero_reward_count=zero_reward_count,
    )

    # Calculate length-trimmed count before positive filtering
    length_trimmed_count = len(df_no_failures[lengths <= pct95])

    print(f"\nRollout summary:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Failed generations: {failure_count} ({failure_ratio:.2%})")
    print(f"  After failure filter: {len(df_no_failures)}")
    print(f"  After length trim (p95={pct95}): {length_trimmed_count}")
    if filter_positive_only:
        print(f"  After positive-only filter: {len(kept)} ({100*len(kept)/length_trimmed_count:.1f}% of trimmed)")
    else:
        print(f"  Positive filtering disabled: {len(kept)}")
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
    ap.add_argument("--server-port", type=int, default=30001, help="Port for policy model server (default: 30001)")
    ap.add_argument("--server-wait-seconds", type=int, default=900)
    ap.add_argument("--server-mem-fraction", type=float, default=0.85)
    ap.add_argument("--server-log-level", type=str, default="debug")
    ap.add_argument("--server-disable-torch-compile", dest="server_enable_torch_compile", action="store_false")
    ap.add_argument("--server-disable-cuda-graph", dest="server_disable_cuda_graph", action="store_true")
    ap.add_argument("--dual-server", action="store_true", help="Use dual-server mode (2×TP2) when 4 GPUs available (default: disabled)")
    # Game termination settings
    ap.add_argument("--max-turns", type=int, default=10, help="Maximum number of turns per game (default: 10)")
    ap.add_argument("--max-retries-per-turn", type=int, default=2, help="Maximum retries per turn before terminating (default: 2)")
    # KL divergence settings
    ap.add_argument("--use-kl", action="store_true", help="Enable KL divergence penalty in reward (disabled by default)")
    ap.add_argument("--reference-model", type=str, default=None, help="Path to reference model for KL divergence computation (required if --use-kl is set)")
    ap.add_argument("--kl-coef", type=float, default=0.001, help="KL divergence penalty coefficient (default: 0.001)")
    ap.add_argument("--kl-method", type=str, default="hf_dataparallel",
                    choices=["hf_dataparallel", "sglang"],
                    help="Method for computing reference model logprobs: 'hf_dataparallel' (default, uses data parallelism with HuggingFace) or 'sglang' (uses tensor parallelism with SGLang)")

    # Data filtering settings (EXPERT ITERATION MODE ONLY - GRPO/PPO does not filter by quality)
    ap.add_argument("--filter-positive-only", action="store_true", default=True, help="[Expert Iteration only] Train only on positive GRPO-normalized examples (above group mean). For PPO/GRPO, use --no-filter-positive-only. (default: enabled)")
    ap.add_argument("--no-filter-positive-only", dest="filter_positive_only", action="store_false", help="[Expert Iteration only] Disable positive-only filtering, train on all examples. Required for PPO/GRPO mode.")
    ap.add_argument("--filter-percentile", type=float, default=0.0, help="[Expert Iteration only] Keep only sequences above this percentile of GRPO-normalized rewards (0.0-1.0). Set to 0 to disable. Applies after positive-only filter if both are enabled. Not used in PPO/GRPO mode. Default: 0.0 (disabled)")

    # Asymmetric mode arguments (trainee vs opponent)
    ap.add_argument("--opponent-model", type=str, default=None, help="Path to opponent model for asymmetric training (default: None = self-play mode)")

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


def compute_reference_logprobs_hf_dataparallel(
    *,
    df: pd.DataFrame,
    reference_model_path: str,
    gpus: str,
    **kwargs  # Accept but ignore other params for API compatibility
) -> List[List[float]]:
    """Compute reference logprobs using HuggingFace model with DataParallel.

    This approach:
    1. Replicates the model on each GPU (data parallelism, NOT tensor parallelism)
    2. Processes sequences in static batches (no continuous batching overhead)
    3. Extracts logprobs for generated tokens only

    More efficient than SGLang for pure scoring workloads (no generation needed).
    Ideal for models that fit on a single GPU.

    Args:
        df: DataFrame with 'input_ids' and 'loss_mask' columns
        reference_model_path: Path to reference model checkpoint
        gpus: GPU IDs (e.g., "0,1,2,3")
        **kwargs: Ignored (for API compatibility with sglang version)

    Returns:
        List of logprob lists (one per sequence, filtered to generated tokens)
    """
    import os
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    # Set GPU visibility
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    num_gpus = len(gpu_list)

    print(f"\n{'='*80}")
    print(f"Loading reference model for KL divergence (HF DataParallel)...")
    print(f"  Reference model: {reference_model_path}")
    print(f"  GPUs: {gpus} ({num_gpus} GPUs)")
    print(f"  Method: Data Parallelism (model replicated on each GPU)")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Wrap with DataParallel if multiple GPUs
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Wrapped model with DataParallel across {num_gpus} GPUs")

    model = model.cuda()
    model.eval()

    print(f"Model loaded successfully\n")

    # Process sequences in batches
    batch_size = 32 * num_gpus  # Scale batch size with number of GPUs
    total_sequences = len(df)
    all_ref_logprobs = []

    print(f"Computing reference logprobs for {total_sequences} sequences...")
    print(f"Batch size: {batch_size} ({batch_size // num_gpus} per GPU)\n")

    with torch.no_grad():
        for batch_start in range(0, total_sequences, batch_size):
            batch_end = min(batch_start + batch_size, total_sequences)
            batch_df = df.iloc[batch_start:batch_end]

            # Prepare batch
            batch_input_ids = []
            batch_loss_masks = []
            max_len = 0

            for idx, row in batch_df.iterrows():
                input_ids = row["input_ids"]
                if isinstance(input_ids, np.ndarray):
                    input_ids = input_ids.tolist()
                loss_mask = row["loss_mask"]
                if isinstance(loss_mask, np.ndarray):
                    loss_mask = loss_mask.tolist()

                batch_input_ids.append(input_ids)
                batch_loss_masks.append(loss_mask)
                max_len = max(max_len, len(input_ids))

            # Pad sequences to same length
            padded_input_ids = []
            padded_loss_masks = []
            for input_ids, loss_mask in zip(batch_input_ids, batch_loss_masks):
                pad_len = max_len - len(input_ids)
                padded_input_ids.append(input_ids + [0] * pad_len)
                padded_loss_masks.append(loss_mask + [0] * pad_len)

            # Convert to tensors
            input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long).cuda()
            loss_masks_tensor = torch.tensor(padded_loss_masks, dtype=torch.long)

            # Forward pass
            outputs = model(input_ids_tensor, use_cache=False)
            logits = outputs.logits  # (batch, seq_len, vocab_size)

            # Compute logprobs
            # logits[:, i, :] predicts token at position i+1
            # So logits[:, :-1, :] predicts tokens at positions 1 onwards
            shift_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab)
            shift_labels = input_ids_tensor[:, 1:].contiguous()  # (batch, seq_len-1)

            # Get log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch, seq_len-1, vocab)

            # Extract logprob for actual token at each position
            batch_size_actual = log_probs.size(0)
            seq_len = log_probs.size(1)

            # Gather logprobs for actual tokens
            batch_indices = torch.arange(batch_size_actual, device=log_probs.device).unsqueeze(1).expand(-1, seq_len)
            seq_indices = torch.arange(seq_len, device=log_probs.device).unsqueeze(0).expand(batch_size_actual, -1)
            token_logprobs = log_probs[batch_indices, seq_indices, shift_labels]  # (batch, seq_len-1)

            # Move to CPU for filtering
            token_logprobs = token_logprobs.cpu().numpy()

            # Filter to generated tokens only using loss_mask
            for i, loss_mask in enumerate(batch_loss_masks):
                # loss_mask[j] corresponds to token at position j
                # token_logprobs[i, j] is logprob for token at position j+1
                # So we need loss_mask[1:] to align with token_logprobs
                loss_mask_shifted = loss_mask[1:]  # Align with predictions

                # Extract logprobs where loss_mask is 1
                ref_logprobs = [
                    float(token_logprobs[i, j])
                    for j in range(len(loss_mask_shifted))
                    if j < len(token_logprobs[i]) and loss_mask_shifted[j] == 1
                ]
                all_ref_logprobs.append(ref_logprobs)

            # Progress update
            pct = 100.0 * batch_end / total_sequences
            print(f"Progress: {batch_end}/{total_sequences} sequences ({pct:.1f}%)")

    print(f"\n{'='*80}")
    print(f"Reference logprob computation complete!")
    print(f"  Processed: {len(all_ref_logprobs)} sequences")
    print(f"{'='*80}\n")

    return all_ref_logprobs


def compute_reference_logprobs_sglang(
    *,
    df: pd.DataFrame,
    reference_model_path: str,
    gpus: str,
    server_port: int = 30004,
    server_wait_seconds: int = 900,
    server_mem_fraction: float = 0.85,
    server_log_level: str = "info",
    enable_torch_compile: bool = False,
    disable_cuda_graph: bool = False,
    round_dir: Optional[Path] = None,
) -> List[List[float]]:
    """Compute reference logprobs using SGLang Runtime API.

    This approach uses SGLang's direct engine API (no HTTP server) with tensor parallelism.
    Note: SGLang still uses continuous batching internally, which has overhead for pure scoring.

    Args:
        df: DataFrame containing generated sequences with 'input_ids' and 'loss_mask'
        reference_model_path: Path to the reference model checkpoint
        gpus: GPU IDs to use (e.g., "0,1,2,3")
        server_port: Unused (kept for API compatibility)
        server_wait_seconds: Unused (kept for API compatibility)
        server_mem_fraction: Memory fraction for engine
        server_log_level: Log level for SGLang
        enable_torch_compile: Whether to enable torch.compile
        disable_cuda_graph: Whether to disable CUDA graph
        round_dir: Unused (kept for API compatibility)

    Returns:
        List of logprob lists (one list per sequence)
    """
    import os
    import torch
    from sglang import Runtime
    from transformers import AutoTokenizer

    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]

    # Use all available GPUs for reference model
    ref_gpus = ",".join(gpu_list)
    tp = len(gpu_list) if len(gpu_list) > 0 else 1

    print(f"\n{'='*80}")
    print(f"Loading reference model for KL divergence (SGLang)...")
    print(f"  Reference model: {reference_model_path}")
    print(f"  GPUs: {ref_gpus} (TP={tp})")
    print(f"  Using direct engine API (no HTTP server)")
    print(f"{'='*80}\n")

    # Set GPU visibility and NCCL environment for topology
    os.environ["CUDA_VISIBLE_DEVICES"] = ref_gpus
    os.environ["NCCL_P2P_LEVEL"] = "NVL"

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(reference_model_path, trust_remote_code=True)

    # Initialize SGLang runtime with efficient settings for scoring
    print(f"Initializing SGLang engine...")
    runtime = Runtime(
        model_path=reference_model_path,
        tp_size=tp,
        mem_fraction_static=0.85,     # Use maximum GPU memory
        # max_total_tokens=0,        # Minimal KV cache (no prefix sharing needed)
        trust_remote_code=True,
        disable_cuda_graph=disable_cuda_graph,
        enable_torch_compile=enable_torch_compile,
        # Prefill optimization - conservative settings for diverse sequences
        chunked_prefill_size=2048,  # Smaller chunks to avoid OOM on long sequences
        schedule_policy="fcfs",      # First-Come-First-Serve (no prefix caching)
        schedule_conservativeness=1.0,  # Conservative scheduling
        disable_radix_cache=True,    # Disable radix cache to save memory
        allow_auto_truncate=True,
        log_level=server_log_level,
    )

    try:
        all_ref_logprobs = []
        batch_size = 64  # Larger batches for direct API (no HTTP overhead)
        total_sequences = len(df)

        print(f"\nComputing reference logprobs for {total_sequences} sequences...")
        print(f"Processing in batches of {batch_size}...\n")

        for batch_start in range(0, total_sequences, batch_size):
            batch_end = min(batch_start + batch_size, total_sequences)
            batch_df = df.iloc[batch_start:batch_end]

            # Prepare batch inputs (decode to text for Runtime API)
            batch_prompts = []
            batch_loss_masks = []
            for idx, row in batch_df.iterrows():
                input_ids = row["input_ids"]
                if isinstance(input_ids, np.ndarray):
                    input_ids = input_ids.tolist()
                # Decode to text for Runtime.generate() API
                prompt_text = tokenizer.decode(input_ids, skip_special_tokens=False)
                batch_prompts.append(prompt_text)

                # Get loss mask to identify generated tokens
                loss_mask = row["loss_mask"]
                if isinstance(loss_mask, np.ndarray):
                    loss_mask = loss_mask.tolist()
                batch_loss_masks.append(loss_mask)

            # Run batch inference to get logprobs
            # We set max_new_tokens=0 to only score the input (no generation)
            response_json_str = runtime.generate(
                prompt=batch_prompts,  # 'prompt' not 'prompts', can be a list
                sampling_params={
                    "max_new_tokens": 0,  # No generation, just score input
                    "temperature": 0,  # Greedy for consistency
                },
                return_logprob=True,  # Return per-token logprobs
                logprob_start_len=0,  # Get logprobs for all tokens
            )

            # Parse the JSON string response
            response_data = json.loads(response_json_str)

            # Handle both single response and batch response
            if isinstance(response_data, list):
                outputs = response_data
            else:
                outputs = [response_data]

            # Extract logprobs from outputs and filter to generated tokens only
            for output, loss_mask in zip(outputs, batch_loss_masks):
                # Get logprobs for the input tokens
                # SGLang returns logprobs in meta_info
                try:
                    ref_logprobs_all = output["meta_info"]["input_token_logprobs"]
                    # Extract just the logprob values (first element of each tuple)
                    ref_logprobs_all = [logprob[0] for logprob in ref_logprobs_all]

                    # Filter to only generated tokens (where loss_mask == 1)
                    # Note: input_token_logprobs[i] is the logprob for token i+1
                    # So we need to align with loss_mask[1:] (skip first token which has no logprob)
                    ref_logprobs_generated = [
                        lp for lp, mask in zip(ref_logprobs_all, loss_mask[1:])
                        if mask == 1
                    ]
                except (KeyError, TypeError, IndexError):
                    # Fallback: use empty list
                    ref_logprobs_generated = []

                all_ref_logprobs.append(ref_logprobs_generated)

            # Progress update
            pct = 100.0 * batch_end / total_sequences
            print(f"Progress: {batch_end}/{total_sequences} sequences ({pct:.1f}%)")

        print(f"\n{'='*80}")
        print(f"Reference logprob computation complete!")
        print(f"  Processed: {len(all_ref_logprobs)} sequences")
        print(f"{'='*80}\n")

        return all_ref_logprobs

    finally:
        print("Shutting down reference model engine...")
        runtime.shutdown()


def compute_reference_logprobs(
    *,
    df: pd.DataFrame,
    reference_model_path: str,
    gpus: str,
    method: str = "hf_dataparallel",
    **kwargs
) -> List[List[float]]:
    """Dispatcher for computing reference model logprobs.

    Calls the appropriate implementation based on the method parameter.

    Args:
        df: DataFrame with 'input_ids' and 'loss_mask' columns
        reference_model_path: Path to reference model checkpoint
        gpus: GPU IDs (e.g., "0,1,2,3")
        method: Method to use: "hf_dataparallel" (default) or "sglang"
        **kwargs: Additional parameters passed to specific implementation

    Returns:
        List of logprob lists (one per sequence)
    """
    if method == "hf_dataparallel":
        return compute_reference_logprobs_hf_dataparallel(
            df=df,
            reference_model_path=reference_model_path,
            gpus=gpus,
            **kwargs
        )
    elif method == "sglang":
        return compute_reference_logprobs_sglang(
            df=df,
            reference_model_path=reference_model_path,
            gpus=gpus,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown KL computation method: {method}. "
            f"Valid options: 'hf_dataparallel' (default), 'sglang'"
        )


# Named export used by offline_grpo_loop
run_rollout_pipeline = function_B_generate_trim_and_log


if __name__ == "__main__":
    main()



