#!/usr/bin/env python3
"""
Generate SFT training data by playing games with OpenAI API models.

Simpler alternative to generate_rollouts.py for SFT data generation:
- No GRPO grouping (each game is unique)
- No logprob collection or tokenization
- No reward normalization
- Direct messages format output

Usage:
    python scripts/generate_sft_data.py \
        --model-id gpt-5-mini \
        --num-games 100 \
        --out data/sft_gpt5mini.parquet
"""
import os
import json
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Manager, Process
import queue

import numpy as np
import pandas as pd

# Make project root importable
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(PROJECT_ROOT) / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        if "OPENAI_API_KEY" in os.environ:
            print(f"[INFO] Loaded OPENAI_API_KEY from {env_path}")
except ImportError:
    pass

# Import the correct game-running logic from generate_rollouts
# Add scripts directory to path
scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from generate_rollouts import run_one_game


def run_one_game_openai(
    model_id: str,
    instructions: str,
    api_key_path: str,
    organization: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    max_turns: int,
    max_retries_per_turn: int,
    game_id: int,
    seed: int,
) -> Dict[str, Any]:
    """Wrapper to run a game with OpenAI models using the correct logic from generate_rollouts.

    Returns:
        Dict with "players" containing player objects, "game_info", etc.
    """
    # Build player config for OpenAI
    player_cfg = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "max_turns": max_turns,
        "max_retries_per_turn": max_retries_per_turn,
        "server_url": "dummy",  # Not used for OpenAI but required by signature
    }

    # Add OpenAI-specific config
    if api_key_path:
        player_cfg["api_key_path"] = api_key_path
    if organization:
        player_cfg["organization"] = organization

    # Call the correct implementation from generate_rollouts
    return run_one_game(
        player_cfg=player_cfg,
        model_id=model_id,
        instructions=instructions,
        game_id=game_id,
        group_size=1,  # No GRPO grouping for SFT
        base_seed=seed,
        player_type="openai"
    )


def process_game_to_messages(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract messages format from game result.

    Returns one row per player with their full conversation history.
    """
    rows = []
    for player_name in result["players"].keys():
        player = result["players"][player_name]

        # Get player's full conversation history
        conversation_history = player.get_conversation_history()

        rows.append({
            "messages": conversation_history,
            "player_name": player_name,
            "game_id": result.get("game_id", 0),
            "game_seed": result.get("game_seed", 0),
            "reward": float(result.get("normalized_reward", 0.0)),
            "game_info": json.dumps(result.get("game_info", {})),
        })
    return rows


def worker_process(
    task_queue,
    result_queue,
    threads_per_proc: int,
    model_id: str,
    instructions: str,
    api_key_path: str,
    organization: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    max_turns: int,
    max_retries_per_turn: int,
    seed: int,
    process_id: int,
):
    """Worker process that runs games in parallel threads."""
    np.random.seed(seed + process_id)

    def thread_worker():
        while True:
            try:
                game_id = task_queue.get_nowait()
            except queue.Empty:
                return

            try:
                result = run_one_game_openai(
                    model_id, instructions, api_key_path, organization,
                    temperature, max_new_tokens, top_p,
                    max_turns, max_retries_per_turn, game_id, seed
                )
                if result is None:
                    print(f"[Process {process_id}] Game {game_id} returned None")
                    result_queue.put([])
                else:
                    rows = process_game_to_messages(result)
                    result_queue.put(rows)
            except Exception as e:
                import traceback
                print(f"[Process {process_id}] Game {game_id} failed: {e}")
                traceback.print_exc()
                result_queue.put([])

    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = [executor.submit(thread_worker) for _ in range(threads_per_proc)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[Process {process_id}] Thread crashed: {e}")


def main():
    ap = argparse.ArgumentParser(description="Generate SFT training data from OpenAI API games")
    ap.add_argument("--model-id", required=True, help="OpenAI model ID (e.g., gpt-5-mini)")
    ap.add_argument("--out", required=True, help="Output parquet file path")
    ap.add_argument("--num-games", type=int, default=100, help="Number of games to play")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-retries-per-turn", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-procs", type=int, default=8, help="Number of worker processes")
    ap.add_argument("--threads-per-proc", type=int, default=16, help="Threads per process")
    ap.add_argument("--progress-interval", type=int, default=10, help="Print progress every N games")
    ap.add_argument("--openai-api-key-path", type=str, help="Path to OpenAI API key JSON file")
    ap.add_argument("--openai-organization", type=str, help="OpenAI organization ID")

    args = ap.parse_args()

    np.random.seed(args.seed)

    # Load instructions
    instructions_path = Path(__file__).parent / "dialop" / "envs" / "data" / "optimization.txt"
    instructions = instructions_path.read_text().strip()

    print(f"Generating SFT data with {args.model_id}")
    print(f"  Games: {args.num_games}")
    print(f"  Workers: {args.num_procs} processes × {args.threads_per_proc} threads = {args.num_procs * args.threads_per_proc} concurrent")
    print(f"  Output: {args.out}")

    start_time = time.time()

    # Create task queue
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for game_id in range(args.num_games):
        task_queue.put(game_id)

    # Spawn worker processes
    processes = []
    for proc_id in range(args.num_procs):
        p = Process(
            target=worker_process,
            args=(
                task_queue, result_queue, args.threads_per_proc,
                args.model_id, instructions,
                args.openai_api_key_path, args.openai_organization,
                args.temperature, args.max_new_tokens, args.top_p,
                args.max_turns, args.max_retries_per_turn,
                args.seed, proc_id,
            )
        )
        p.start()
        processes.append(p)

    # Collect results
    all_rows = []
    completed_games = 0
    last_progress_time = start_time

    while completed_games < args.num_games:
        try:
            rows = result_queue.get(timeout=1.0)
            all_rows.extend(rows)
            completed_games += len(rows) // 2  # Each game produces 2 rows

            # Progress update
            if completed_games % args.progress_interval == 0 or completed_games == args.num_games:
                elapsed = time.time() - last_progress_time
                rate = args.progress_interval / elapsed if elapsed > 0 else 0
                print(f"Progress: {completed_games}/{args.num_games} games ({rate:.1f} games/sec)")
                last_progress_time = time.time()
        except queue.Empty:
            continue

    # Wait for all processes
    for p in processes:
        p.join()

    total_time = time.time() - start_time
    print(f"\nCompleted {args.num_games} games in {total_time:.1f}s ({args.num_games/total_time:.2f} games/sec)")
    print(f"Generated {len(all_rows)} training examples ({len(all_rows)/args.num_games:.1f} per game)")

    # Save to parquet
    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path))
    print(f"Saved to {out_path}")
    print(f"\nSchema: messages (List[Dict]), player_name (str), game_id (int), reward (float), game_info (str)")


if __name__ == "__main__":
    main()
