#!/usr/bin/env python3
import os
import json
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd

# Make project root importable when running from scripts/
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports (now importable as top-level 'dialop')
from dialop.sglang_model_player import SGLangModelPlayer, SGLangConfig
from dialop.envs.optimization import OptimizationEnv


def run_one_game(player_cfg: dict, model_id: str, instructions: str, game_id: int = 0) -> Dict[str, Any]:
    # Build two players sharing the same server config
    console = type("_C", (), {"print": lambda *args, **kwargs: None, "rule": lambda *args, **kwargs: None})()
    cfg = SGLangConfig(
        temperature=player_cfg["temperature"],
        top_p=player_cfg["top_p"],
        max_tokens=player_cfg["max_new_tokens"],
        timeout=player_cfg.get("timeout", 120.0),
    )
    cfg.server_url = player_cfg["server_url"].rstrip("/")
    if not cfg.server_url.endswith("/v1"):
        cfg.server_url += "/v1"

    # Resolve the {unknown_value} placeholder as in the reference rollout
    instr_p1 = instructions.replace("{unknown_value}", "50")
    instr_p2 = instructions.replace("{unknown_value}", "50")

    p1 = SGLangModelPlayer(
        system_prompt=instr_p1,
        role="player-1",
        console=console,
        model_path=model_id,
        config=cfg,
    )
    p2 = SGLangModelPlayer(
        system_prompt=instr_p2,
        role="player-2",
        console=console,
        model_path=model_id,
        config=cfg,
    )

    env = OptimizationEnv()
    obs = env.reset(game_state=None)
    p1.observe(obs["player-1"])  # give initial observation
    p2.observe(obs["player-2"])  # give initial observation

    clean_conversation = []
    full_conversation = []
    done = False
    turn = 0
    max_turns = player_cfg.get("max_turns", 30)
    max_retries = player_cfg.get("max_retries_per_turn", 8)

    while not done and turn < max_turns:
        current = obs["turn_player"]
        player = p1 if current == "player-1" else p2
        retries = 0
        while True:
            try:
                response = player.respond()
                full_conversation.append({"turn": turn, "player": current, "message": response, "retry": retries})
            except Exception as e:
                # Treat generation failure as terminal
                done = True
                break

            obs, error = env.step(response)
            if error:
                retries += 1
                full_conversation.append({"turn": turn, "player": "error", "message": obs[current], "retry": retries})
                player.observe(obs[current])
                if retries >= max_retries:
                    done = True
                    break
                continue

            # valid move
            clean_conversation.append({"turn": turn, "player": current, "message": response})
            # broadcast observation to other player
            other_name = "player-2" if current == "player-1" else "player-1"
            other_player = p2 if current == "player-1" else p1
            if obs[other_name]:
                other_player.observe(obs[other_name])

            done = obs["done"]
            turn += 1
            break

    # Build two sequences (one per player's perspective)
    result = {
        "game_id": game_id,
        "players": {"player-1": p1, "player-2": p2},
        "reward": obs.get("info", {}).get("score", 0.0) if done else 0.0,
        "normalized_reward": obs.get("info", {}).get("score_norm", 0.0) if done else 0.0,
        "game_info": {
            "num_messages": obs.get("info", {}).get("num_msgs", turn),
            "completed": done,
            "turn_count": turn,
            "game_normalized_reward": obs.get("info", {}).get("score_norm", 0.0) if done else 0.0,
        },
        "clean_conversation": clean_conversation,
        "full_conversation": full_conversation,
    }
    return result


def tensorize_player(player, max_model_len: int):
    input_ids = player.get_input_sequence()
    if len(input_ids) > max_model_len:
        input_ids = input_ids[:max_model_len]
    attention_mask = [1] * len(input_ids)
    # assistant mask for loss
    response_loss_mask = player.get_assistant_mask()
    if len(response_loss_mask) > len(input_ids):
        response_loss_mask = response_loss_mask[: len(input_ids)]
    elif len(response_loss_mask) < len(input_ids):
        response_loss_mask = response_loss_mask + [0] * (len(input_ids) - len(response_loss_mask))
    # position ids as simple range (single-head)
    position_ids = list(range(len(input_ids)))
    return input_ids, attention_mask, position_ids, response_loss_mask


def process_game_result(result: Dict[str, Any], max_model_len: int) -> List[Dict[str, Any]]:
    """Process a single game result into training samples for both players."""
    rows = []
    # two sequences (player-1 and player-2)
    for player_name in ["player-1", "player-2"]:
        p = result["players"][player_name]
        input_ids, attention_mask, position_ids, response_loss_mask = tensorize_player(p, max_model_len)
        # Use absolute normalized reward as weight (will be converted to relative in GRPO grouping)
        w = float(result["normalized_reward"])
        rows.append(
            {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "position_ids": np.array(position_ids, dtype=np.int64),
                "loss_mask": np.array(response_loss_mask, dtype=np.int64),
                "sample_weight": w,
                # Store game-normalized reward explicitly
                "game_normalized_reward": float(
                    result.get("game_info", {}).get("game_normalized_reward", result.get("normalized_reward", 0.0))
                ),
                "game_info": json.dumps(result["game_info"]),
                "game_id": result["game_id"],  # Add game_id for grouping
            }
        )
    return rows


def worker_run_games(
    game_ids: List[int],
    player_cfg: Dict[str, Any],
    model_id: str,
    instructions: str,
    max_model_len: int,
    threads_per_proc: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Run a batch of games inside a subprocess, using a small thread pool for I/O.

    Returns a list of processed training rows (two rows per completed game).
    """
    # Ensure different RNG stream per worker
    try:
        np.random.seed(seed + os.getpid())
    except Exception:
        np.random.seed(seed)

    rows_accum: List[Dict[str, Any]] = []

    def one(game_id: int) -> List[Dict[str, Any]]:
        result = run_one_game(player_cfg, model_id, instructions, game_id)
        return process_game_result(result, max_model_len)

    # Small thread pool to overlap request I/O in this process
    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = {executor.submit(one, gid): gid for gid in game_ids}
        for fut in as_completed(futures):
            try:
                rows_accum.extend(fut.result())
            except Exception:
                # Skip failed game; continue with others
                continue

    return rows_accum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:30001", help="URL of the SGLang server (default: http://127.0.0.1:30001)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-games", type=int, default=256, help="Number of unique games to play (each will be played 8 times)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    # Use a large default; external server's KV cache is governed by mem fraction, not this cap
    ap.add_argument("--max-model-len", type=int, default=32768)
    # GRPO grouping size (k). Rewards will be transformed to relative rewards within each group.
    ap.add_argument("--group-size", type=int, default=8, help="Number of times to play each game (creates groups of 2*k sequences)")
    ap.add_argument("--seed", type=int, default=42)
    # Parallelization settings
    ap.add_argument("--num-procs", type=int, default=64, help="Number of worker processes (use up to 32)")
    ap.add_argument("--threads-per-proc", type=int, default=16, help="Threads per process for issuing requests")
    ap.add_argument("--max-workers", type=int, default=1024, help="[deprecated] Ignored; use --num-procs/--threads-per-proc")
    ap.add_argument("--progress-interval", type=int, default=64, help="Print progress every N completed games")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Load instructions used by DialopSelfPlayRollout
    instructions_path = Path(__file__).parent / "dialop" / "envs" / "data" / "optimization.txt"
    instructions = instructions_path.read_text().strip()

    player_cfg = {
        "server_url": args.server_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    # Calculate total games to play: num_games * group_size
    total_games = args.num_games * args.group_size
    print(
        f"Starting generation of {total_games} total games "
        f"({args.num_games} unique games × {args.group_size} plays each) "
        f"with {args.num_procs} processes × {args.threads_per_proc} threads..."
    )
    start_time = time.time()
    
    # Build ordered list of game IDs to preserve grouping logic
    game_ids_all = list(range(total_games))

    # Partition game IDs across processes
    num_procs = max(1, min(args.num_procs, total_games))
    chunk_size = (total_games + num_procs - 1) // num_procs
    game_chunks = [game_ids_all[i : i + chunk_size] for i in range(0, total_games, chunk_size)]

    all_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=num_procs) as pool:
        futures = [
            pool.submit(
                worker_run_games,
                chunk,
                player_cfg,
                args.model_id,
                instructions,
                args.max_model_len,
                args.threads_per_proc,
                args.seed,
            )
            for chunk in game_chunks
        ]

        completed_chunks = 0
        completed_games = 0
        for fut in as_completed(futures):
            try:
                rows = fut.result()
                all_rows.extend(rows)
                # Each game yields two rows
                completed_in_chunk = len(rows) // 2
                completed_games += completed_in_chunk
            except Exception as e:
                print(f"Worker failed: {e}")
            finally:
                completed_chunks += 1
                if completed_games % args.progress_interval == 0 or completed_games >= total_games:
                    elapsed = time.time() - start_time
                    rate = completed_games / elapsed if elapsed > 0 else 0
                    print(
                        f"Completed {completed_games}/{total_games} games "
                        f"({rate:.1f} games/sec) via {completed_chunks}/{len(futures)} workers"
                    )
    print(f"Generation completed in {time.time() - start_time:.1f}s")

    # Group sequences by unique game sets for GRPO normalization
    # Each unique game was played group_size times, creating 2*group_size sequences per unique game
    print(f"Grouping sequences for GRPO normalization...")
    
    # Sort rows by unique_game_id to ensure proper grouping
    all_rows.sort(key=lambda x: x["game_id"] // args.group_size)
    
    # Apply GRPO normalization within each group of 2*group_size sequences
    sequences_per_group = 2 * args.group_size  # 2 players × group_size plays
    total = len(all_rows)
    full_groups = total // sequences_per_group
    
    if full_groups == 0:
        print(f"Warning: not enough samples ({total}) to form a full group of size {sequences_per_group}; leaving weights unchanged.")
    else:
        remainder = total - full_groups * sequences_per_group
        if remainder > 0:
            print(f"Dropping {remainder} leftover samples to form full groups of size {sequences_per_group} for GRPO.")
        # Only keep full groups
        all_rows = all_rows[: full_groups * sequences_per_group]
        
        for g in range(full_groups):
            start = g * sequences_per_group
            end = start + sequences_per_group
            group_rewards = [float(all_rows[j]["sample_weight"]) for j in range(start, end)]
            baseline = float(np.mean(group_rewards))
            for j in range(start, end):
                all_rows[j]["sample_weight"] = float(group_rewards[j - start] - baseline)
        
        print(f"Applied GRPO normalization to {full_groups} groups of {sequences_per_group} sequences each")

    # Save parquet
    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path))
    print(f"Wrote {len(df)} samples to {out_path}")


if __name__ == "__main__":
    main()


