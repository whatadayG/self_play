#!/usr/bin/env python3
"""
Dump all dialogues from a parquet file into individual text documents.

Uses the same formatting logic as examples.txt generation in rollout_pipeline.py.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Any


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


def format_single_game(row, reward: float, output_path: Path) -> None:
    """Format a single game dialogue and write to file.

    Args:
        row: DataFrame row for this game
        reward: Game-normalized reward for this game
        output_path: Path to write the formatted dialogue
    """
    game_id = row.get("game_id", "unknown")
    grpo_weight = row.get("sample_weight", None)

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Game ID: {game_id}\n")
        f.write(f"Reward (game-normalized): {reward:.4f}\n")
        if grpo_weight is not None:
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
                f.write("\n[No conversation data available]\n")
        except Exception as e:
            f.write(f"\n[Error parsing conversation: {e}]\n")

        f.write("\n" + "=" * 80 + "\n")


def dump_all_dialogues(parquet_path: Path, output_dir: Path) -> None:
    """Dump all dialogues from parquet file to individual text files.

    Args:
        parquet_path: Path to input parquet file
        output_dir: Directory to write individual dialogue files
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

        format_single_game(row, reward, output_path)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(unique_games)} games...")

    print(f"\nDone! Wrote {len(unique_games)} dialogue files to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Dump all dialogues from a parquet file into individual text documents"
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to input parquet file (e.g., ~/georgiazhou/self_play/logs/.../train.parquet)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for dialogue files (default: same directory as parquet with '_dialogues' suffix)"
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
        # Use same directory as parquet, with '_dialogues' suffix
        output_dir = parquet_path.parent / f"{parquet_path.stem}_dialogues"

    dump_all_dialogues(parquet_path, output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
