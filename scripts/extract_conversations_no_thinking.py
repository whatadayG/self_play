#!/usr/bin/env python3
"""
Dump conversations from train.parquet in human-readable format.
- Removes all thinking processes
- Labels the shy agent (user) vs non-shy agent (assistant)
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Any
from transformers import AutoTokenizer


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


def identify_assistant_player(input_ids, tokenizer):
    """
    Identify which player is the assistant (non-shy agent being trained).
    Returns player name ('player-1' or 'player-2') or None.
    """
    try:
        # Decode with special tokens
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Find first assistant section
        assistant_start = full_text.find("<|im_start|>assistant\n")
        if assistant_start == -1:
            return None

        assistant_end = full_text.find("<|im_end|>", assistant_start)
        if assistant_end == -1:
            assistant_end = len(full_text)

        assistant_text = full_text[assistant_start:assistant_end]

        return assistant_text
    except Exception:
        return None


def match_assistant_to_conversation(assistant_text, conversation):
    """Match assistant text to a player in the conversation."""
    try:
        for msg in conversation:
            msg_text = msg['message'].strip()
            # Look for the part after [message]/[propose]/[accept]/[reject]
            match = re.search(r'\[(message|propose|accept|reject)\](.*)', msg_text, re.DOTALL)
            if match:
                public_part = match.group(2).strip()[:150]
                if len(public_part) > 20 and public_part in assistant_text:
                    return msg['player']
        return None
    except Exception:
        return None


def extract_public_text(message):
    """
    Extract the public (non-thinking) part of a message.
    This is the part after [message]/[propose]/[accept]/[reject] tag,
    with <think>...</think> sections removed.
    """
    # Find the tag and extract everything after it
    match = re.search(r'\[(message|propose|accept|reject)\](.*)', message, re.DOTALL)
    if not match:
        # No tag found - might be an error message or malformed
        return message

    tag = match.group(1)
    public_text = match.group(2).strip()

    # Remove <think>...</think> sections (including nested ones)
    while '<think>' in public_text.lower():
        public_text = re.sub(r'<think>.*?</think>', '', public_text, flags=re.DOTALL | re.IGNORECASE)

    # Clean up extra whitespace
    public_text = re.sub(r'\n\s*\n', '\n\n', public_text)
    public_text = public_text.strip()

    # Add tag back as a prefix for context
    return f"[{tag}] {public_text}"


def format_single_game(row, reward: float, output_path: Path, tokenizer) -> None:
    """Format a single game dialogue and write to file.

    Args:
        row: DataFrame row for this game
        reward: Game-normalized reward for this game
        output_path: Path to write the formatted dialogue
        tokenizer: Tokenizer for decoding input_ids
    """
    game_id = row.get("game_id", "unknown")
    grpo_weight = row.get("sample_weight", None)

    # Identify which player is shy (user) vs non-shy (assistant)
    input_ids = row.get("input_ids", [])
    full_conv_str = row.get("full_conversation", "[]")
    full_conv = json.loads(full_conv_str) if isinstance(full_conv_str, str) else full_conv_str

    assistant_text = identify_assistant_player(input_ids, tokenizer)
    assistant_player = match_assistant_to_conversation(assistant_text, full_conv) if assistant_text else None

    # Determine player labels
    if assistant_player:
        shy_player = 'player-1' if assistant_player == 'player-2' else 'player-2'
        player_labels = {
            shy_player: "Shy Agent",
            assistant_player: "Non-shy Agent"
        }
    else:
        # Fallback if we can't identify
        player_labels = {
            'player-1': "Player 1",
            'player-2': "Player 2"
        }

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Game ID: {game_id}\n")
        f.write(f"Reward (game-normalized): {reward:.4f}\n")
        if grpo_weight is not None:
            f.write(f"GRPO Weight (relative): {grpo_weight:.4f}\n")
        f.write("-" * 80 + "\n")

        # Parse and format conversation (public parts only)
        try:
            if full_conv:
                f.write("\n=== Conversation (Public Messages Only) ===\n\n")
                for entry in full_conv:
                    turn = entry.get("turn", "?")
                    player = entry.get("player", "?")
                    message = entry.get("message", "")
                    retry = entry.get("retry", 0)

                    # Get player label
                    player_label = player_labels.get(player, player)

                    if player == "error":
                        f.write(f"Turn {turn} - ERROR (retry {retry}):\n{message}\n\n")
                    else:
                        retry_str = f" (retry {retry})" if retry > 0 else ""

                        # Extract public text only
                        public_text = extract_public_text(message)

                        f.write(f"Turn {turn} - {player_label}{retry_str}:\n{public_text}\n\n")
            else:
                f.write("\n[No conversation data available]\n")
        except Exception as e:
            f.write(f"\n[Error parsing conversation: {e}]\n")

        f.write("\n" + "=" * 80 + "\n")


def dump_all_dialogues(parquet_path: Path, output_dir: Path, tokenizer) -> None:
    """Dump all dialogues from parquet file to individual text files.

    Args:
        parquet_path: Path to input parquet file
        output_dir: Directory to write individual dialogue files
        tokenizer: Tokenizer for decoding input_ids
    """
    print(f"Reading parquet file: {parquet_path}")
    df = pd.read_parquet(str(parquet_path))
    print(f"Loaded {len(df)} rows from parquet")

    # Extract rewards
    rewards = _extract_reward_series_from_df(df)
    if rewards is None or len(rewards) == 0:
        print("Warning: No rewards found in parquet file")
        rewards = np.zeros(len(df))

    # Group by game_id to avoid duplicates (each game has 2 sequences)
    print("Grouping by game_id...")
    unique_games = df.groupby('game_id').first().reset_index()
    print(f"Found {len(unique_games)} unique games")

    # Extract rewards for unique games
    game_rewards = _extract_reward_series_from_df(unique_games)
    if game_rewards is None or len(game_rewards) == 0:
        print("Warning: No rewards found for unique games")
        game_rewards = np.zeros(len(unique_games))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dump each game to a separate file
    print(f"\nDumping dialogues to {output_dir}/")
    for idx, row in unique_games.iterrows():
        game_id = row.get("game_id", idx)
        reward = game_rewards[idx]

        # Create filename: game_{game_id}_reward_{reward:.4f}.txt
        filename = f"game_{game_id}_reward_{reward:.4f}.txt"
        output_path = output_dir / filename

        format_single_game(row, reward, output_path, tokenizer)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(unique_games)} games...")

    print(f"\nDone! Wrote {len(unique_games)} dialogue files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Dump conversations from train.parquet without thinking processes"
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to train.parquet file (e.g., logs/offline_grpo/.../round_000/train.parquet)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for dialogue files (default: {parquet_dir}/conversations_no_thinking)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for tokenizer (default: Qwen/Qwen2.5-7B-Instruct)"
    )

    args = parser.parse_args()

    # Expand user paths
    parquet_path = args.parquet_path.expanduser().resolve()

    if not parquet_path.exists():
        print(f"Error: Parquet file not found: {parquet_path}")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        # Use same directory as parquet, with subdirectory
        output_dir = parquet_path.parent / "conversations_no_thinking"

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dump_all_dialogues(parquet_path, output_dir, tokenizer)
    return 0


if __name__ == "__main__":
    exit(main())
