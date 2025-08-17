import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Ensure RL-matching is on sys.path so we can import rl.verl_env
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_MATCHING_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _RL_MATCHING_ROOT not in sys.path:
    sys.path.insert(0, _RL_MATCHING_ROOT)

import pandas as pd

from rl.verl_env import build_episode_from_line, to_chat_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="RL-matching/dialop/data/optimization.jsonl")
    parser.add_argument("--output", type=str, default="RL-matching/data/matching.parquet")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = build_episode_from_line(line)
            messages = to_chat_messages(episode.prompt)
            rows.append({
                "messages": messages,
                "reward_model": {"ground_truth": json.dumps(episode.ground_truth)},
                "data_source": "matching",
            })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main() 