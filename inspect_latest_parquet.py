#!/usr/bin/env python3
"""
Inspect the most recent training parquet and dump debug text files.

Behavior:
- Finds the latest train.parquet under logs/ by modification time
- Prints average reward and proportion of nonzero rewards
- For the first sample:
  - Detokenizes the full input_ids and writes to temp/debug_input.txt
  - Extracts tokens where loss_mask == 1 (assistant tokens), detokenizes, writes to temp/debug_response.txt
  - Prints the two file paths
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def find_latest_train_parquet(root: Path) -> Optional[Path]:
    candidates = list(root.glob("logs/**/train.parquet"))
    if not candidates:
        # fallback to any parquet named 'train.parquet'
        candidates = list(root.rglob("train.parquet"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def detect_model_id(cli_model: Optional[str]) -> str:
    if cli_model:
        return cli_model
    # Try to read served model from SGLang server
    base = os.environ.get("OPENAI_API_BASE", "http://localhost:8000").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    try:
        import requests
        resp = requests.get(f"{base}/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "data" in data and data["data"]:
            return data["data"][0].get("id")
    except Exception:
        pass
    # Fallback to local checkpoint path commonly used in this repo
    return "/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_4800_merged/"


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def detokenize(tok, token_ids: np.ndarray) -> str:
    # Ensure plain python list of ints
    ids = [int(x) for x in np.asarray(token_ids).tolist()]
    return tok.decode(ids, skip_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="Tokenizer source (path or HF id). If omitted, detect from SGLang server.")
    ap.add_argument("--parquet", type=str, default=None, help="Explicit parquet path to inspect. If omitted, auto-detect latest train.parquet under logs/.")
    args = ap.parse_args()

    repo_root = Path(__file__).parent
    parquet_path = Path(args.parquet) if args.parquet else find_latest_train_parquet(repo_root)
    if parquet_path is None or not parquet_path.exists():
        raise FileNotFoundError("Could not find any train.parquet under logs/ or repo.")

    print(f"Using parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Determine reward column
    reward_col = None
    for col in ["sample_weight", "normalized_reward", "reward"]:
        if col in df.columns:
            reward_col = col
            break

    if reward_col is None:
        # Try to parse from game_info if present
        if "game_info" in df.columns:
            try:
                rewards = []
                for x in df["game_info"]:
                    info = json.loads(x) if isinstance(x, str) else x
                    rewards.append(float(info.get("reward", info.get("score_norm", 0.0))))
                rewards = np.array(rewards, dtype=float)
            except Exception:
                rewards = np.zeros(len(df), dtype=float)
        else:
            rewards = np.zeros(len(df), dtype=float)
    else:
        rewards = df[reward_col].to_numpy(dtype=float)

    avg_reward = float(np.mean(rewards)) if len(rewards) else 0.0
    prop_nonzero = float(np.mean(rewards > 0)) if len(rewards) else 0.0
    print(f"Average reward: {avg_reward:.6f}")
    print(f"Proportion nonzero: {prop_nonzero:.6f}")

    if len(df) == 0:
        print("Parquet is empty; skipping token debug.")
        return

    # Prepare tokenizer
    model_id = detect_model_id(args.model)
    print(f"Tokenizer model: {model_id}")
    tok = load_tokenizer(model_id)

    # First row
    row = df.iloc[0]
    input_ids = row.get("input_ids")
    loss_mask = row.get("loss_mask")
    if input_ids is None or loss_mask is None:
        raise KeyError("Parquet must contain 'input_ids' and 'loss_mask' columns for debugging.")

    input_ids = np.asarray(input_ids)
    loss_mask = np.asarray(loss_mask)
    # Safety: crop mask to length
    if loss_mask.shape[0] > input_ids.shape[0]:
        loss_mask = loss_mask[: input_ids.shape[0]]

    decoded_full = detokenize(tok, input_ids)
    response_ids = input_ids[loss_mask.astype(bool)]
    decoded_resp = detokenize(tok, response_ids)

    # Write debug files
    temp_dir = repo_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    input_file = temp_dir / "debug_input.txt"
    resp_file = temp_dir / "debug_response.txt"
    input_file.write_text(decoded_full)
    resp_file.write_text(decoded_resp)

    print(str(input_file))
    print(str(resp_file))


if __name__ == "__main__":
    main()


