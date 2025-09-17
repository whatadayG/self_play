#!/usr/bin/env python3
import os
import json
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Local imports
from scripts.dialop.sglang_model_player import SGLangModelPlayer, SGLangConfig
from scripts.dialop.envs.optimization import OptimizationEnv


def run_one_game(player_cfg: dict, model_id: str, instructions: str) -> Dict[str, Any]:
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

    p1 = SGLangModelPlayer(
        system_prompt=instructions,
        role="player-1",
        console=console,
        model_path=model_id,
        config=cfg,
    )
    p2 = SGLangModelPlayer(
        system_prompt=instructions,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-games", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    # Use a large default; external server's KV cache is governed by mem fraction, not this cap
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=42)
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

    rows: List[Dict[str, Any]] = []
    for i in range(args.num_games):
        result = run_one_game(player_cfg, args.model_id, instructions)
        # two sequences (player-1 and player-2)
        for player_name in ["player-1", "player-2"]:
            p = result["players"][player_name]
            input_ids, attention_mask, position_ids, response_loss_mask = tensorize_player(p, args.max_model_len)
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

    # Save parquet
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path))
    print(f"Wrote {len(df)} samples to {out_path}")


if __name__ == "__main__":
    main()


