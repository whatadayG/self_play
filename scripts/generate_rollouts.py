#!/usr/bin/env python3
import os
import json
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        # Single-sample group: use absolute normalized reward as weight (equivalent to GRPO with k=1)
        w = float(result["normalized_reward"])
        rows.append(
            {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "position_ids": np.array(position_ids, dtype=np.int64),
                "loss_mask": np.array(response_loss_mask, dtype=np.int64),
                "sample_weight": w,
                "game_info": json.dumps(result["game_info"]),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-games", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    # Use a large default; external server's KV cache is governed by mem fraction, not this cap
    ap.add_argument("--max-model-len", type=int, default=32768)
    # GRPO grouping size (k). Rewards will be transformed to relative rewards within each group.
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    # Parallelization settings
    ap.add_argument("--max-workers", type=int, default=1024, help="Maximum number of concurrent games to run")
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

    print(f"Starting generation of {args.num_games} games with {args.max_workers} workers...")
    start_time = time.time()
    
    # Thread-safe list to collect results
    all_rows: List[Dict[str, Any]] = []
    rows_lock = threading.Lock()
    completed_games = 0
    completed_lock = threading.Lock()

    def safe_append_rows(new_rows: List[Dict[str, Any]]):
        with rows_lock:
            all_rows.extend(new_rows)

    def increment_completed():
        nonlocal completed_games
        with completed_lock:
            completed_games += 1
            return completed_games

    # Use ThreadPoolExecutor for parallel game generation
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all games
        future_to_game_id = {
            executor.submit(run_one_game, player_cfg, args.model_id, instructions, i): i 
            for i in range(args.num_games)
        }
        
        # Process completed games as they finish
        for future in as_completed(future_to_game_id):
            game_id = future_to_game_id[future]
            try:
                result = future.result()
                rows = process_game_result(result, args.max_model_len)
                safe_append_rows(rows)
                
                completed = increment_completed()
                if completed % args.progress_interval == 0 or completed == args.num_games:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"Completed {completed}/{args.num_games} games ({rate:.1f} games/sec)")
                    
            except Exception as e:
                print(f"Game {game_id} failed: {e}")
                # Continue with other games even if one fails

    print(f"Generation completed in {time.time() - start_time:.1f}s")

    # Shuffle rows before grouping to avoid putting both perspectives of a game in the same group
    if len(all_rows) > 1:
        order = np.random.permutation(len(all_rows))
        all_rows = [all_rows[i] for i in order]

    # Convert absolute rewards to GRPO relative rewards in groups of k
    k = max(1, int(args.group_size))
    total = len(all_rows)
    full_groups = total // k
    if full_groups == 0:
        print(f"Warning: not enough samples ({total}) to form a full group of size {k}; leaving weights unchanged.")
    else:
        remainder = total - full_groups * k
        if remainder > 0:
            print(f"Dropping {remainder} leftover samples to form full groups of size {k} for GRPO.")
        # Only keep full groups
        all_rows = all_rows[: full_groups * k]
        for g in range(full_groups):
            start = g * k
            end = start + k
            group_rewards = [float(all_rows[j]["sample_weight"]) for j in range(start, end)]
            baseline = float(np.mean(group_rewards))
            for j in range(start, end):
                all_rows[j]["sample_weight"] = float(group_rewards[j - start] - baseline)

    # Save parquet
    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path))
    print(f"Wrote {len(df)} samples to {out_path}")


if __name__ == "__main__":
    main()


