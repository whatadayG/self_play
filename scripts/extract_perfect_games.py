#!/usr/bin/env python3
"""
Extract perfectly played games (reward = 1.0) from all training runs in logs/offline_grpo.

Uses the RAW train.parquet files (not train_trimmed.parquet) to get unfiltered data.
Skips symlinked directories to avoid duplication from branched runs.

Creates a single train.parquet file containing all perfect games across all runs and rounds.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np


def extract_reward_from_df(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract normalized reward array from a dataframe.

    Tries the following fields in order:
    1. game_normalized_reward
    2. normalized_reward
    3. game_info JSON field containing normalized reward

    Returns:
        Array of rewards, or None if not found
    """
    # Try direct columns first
    if "game_normalized_reward" in df.columns:
        return df["game_normalized_reward"].astype(float).to_numpy()
    if "normalized_reward" in df.columns:
        return df["normalized_reward"].astype(float).to_numpy()

    # Try parsing from game_info
    if "game_info" in df.columns:
        rewards = []
        for val in df["game_info"]:
            try:
                if isinstance(val, str):
                    game_info = json.loads(val)
                elif isinstance(val, dict):
                    game_info = val
                else:
                    return None

                if "game_normalized_reward" in game_info:
                    rewards.append(float(game_info["game_normalized_reward"]))
                elif "normalized_reward" in game_info:
                    rewards.append(float(game_info["normalized_reward"]))
                else:
                    return None
            except Exception:
                return None

        if len(rewards) == len(df):
            return np.array(rewards, dtype=float)

    return None


def should_skip_run(run_name: str) -> bool:
    """Check if this entire run should be skipped.

    Args:
        run_name: Name of the run (e.g., "20251016_015645")

    Returns:
        True if this run should be skipped
    """
    # Skip all runs from 20251016 (different semantics from normal runs)
    if run_name.startswith("20251016"):
        return True

    return False


def should_skip_round(run_name: str, round_name: str) -> bool:
    """Check if this round should be skipped based on special filtering rules.

    Args:
        run_name: Name of the run (e.g., "20251013_234837")
        round_name: Name of the round (e.g., "round_007")

    Returns:
        True if this round should be skipped
    """
    # Skip rounds after round_006 in the 20251013_234837 run
    if run_name == "20251013_234837":
        try:
            round_num = int(round_name.replace("round_", ""))
            if round_num > 6:
                return True
        except ValueError:
            pass

    return False


def extract_perfect_games_from_runs(
    base_logs_dir: str = "/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo"
) -> tuple[pd.DataFrame, dict]:
    """Extract all perfect games (reward = 1.0) from all runs in base_logs_dir.

    Returns:
        Tuple of (combined dataframe, stats dictionary)
    """
    base_path = Path(base_logs_dir)
    if not base_path.exists():
        raise ValueError(f"Base logs directory does not exist: {base_logs_dir}")

    # Find all run directories (timestamp format: YYYYMMDD_HHMMSS)
    run_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.replace("_", "").isdigit():
            run_dirs.append(item)

    if not run_dirs:
        raise ValueError(f"No run directories found in {base_logs_dir}")

    # Sort by name (chronological)
    run_dirs.sort(key=lambda x: x.name)

    print(f"Found {len(run_dirs)} runs to process")
    print("=" * 80)

    # Collect all perfect game dataframes
    perfect_dfs = []
    stats = {
        "runs_processed": 0,
        "rounds_processed": 0,
        "rounds_skipped": 0,
        "total_sequences_seen": 0,
        "perfect_sequences_found": 0,
        "by_run": {},
    }

    for run_dir in run_dirs:
        run_name = run_dir.name

        # Check if we should skip this entire run
        if should_skip_run(run_name):
            print(f"\nProcessing run: {run_name}")
            print(f"  SKIPPED (different semantics)")
            continue

        print(f"\nProcessing run: {run_name}")

        # Find all round directories
        round_dirs = sorted([d for d in run_dir.iterdir()
                           if d.is_dir() and d.name.startswith("round_")])

        if not round_dirs:
            print(f"  No rounds found, skipping")
            continue

        run_stats = {
            "rounds_processed": 0,
            "rounds_skipped": 0,
            "total_sequences": 0,
            "perfect_sequences": 0,
        }

        for round_dir in round_dirs:
            round_name = round_dir.name

            # Skip symlinked directories to avoid duplication from branched runs
            if round_dir.is_symlink():
                print(f"  {round_name}: SKIPPED (symlink)")
                stats["rounds_skipped"] += 1
                run_stats["rounds_skipped"] += 1
                continue

            # Check if we should skip this round
            if should_skip_round(run_name, round_name):
                print(f"  {round_name}: SKIPPED (degenerate)")
                stats["rounds_skipped"] += 1
                run_stats["rounds_skipped"] += 1
                continue

            # Use RAW train.parquet (not train_trimmed.parquet which is already filtered)
            train_parquet = round_dir / "train.parquet"

            if not train_parquet.exists():
                continue

            try:
                df = pd.read_parquet(train_parquet)
            except Exception as e:
                print(f"  {round_name}: Error reading parquet: {e}")
                continue

            # Extract rewards
            rewards = extract_reward_from_df(df)
            if rewards is None or len(rewards) == 0:
                print(f"  {round_name}: Could not extract rewards")
                continue

            # Filter for perfect games (reward = 1.0)
            perfect_mask = rewards == 1.0
            num_perfect = int(perfect_mask.sum())

            if num_perfect > 0:
                perfect_df = df[perfect_mask].copy()
                perfect_dfs.append(perfect_df)

            # Update stats
            run_stats["rounds_processed"] += 1
            run_stats["total_sequences"] += len(df)
            run_stats["perfect_sequences"] += num_perfect

            print(f"  {round_name}: {num_perfect}/{len(df)} perfect "
                  f"({num_perfect/len(df)*100:.1f}%)")

        # Update global stats
        if run_stats["rounds_processed"] > 0:
            stats["runs_processed"] += 1
            stats["rounds_processed"] += run_stats["rounds_processed"]
            stats["rounds_skipped"] += run_stats["rounds_skipped"]
            stats["total_sequences_seen"] += run_stats["total_sequences"]
            stats["perfect_sequences_found"] += run_stats["perfect_sequences"]
            stats["by_run"][run_name] = run_stats

            print(f"  Run summary: {run_stats['perfect_sequences']}/{run_stats['total_sequences']} "
                  f"perfect ({run_stats['perfect_sequences']/run_stats['total_sequences']*100:.1f}%)")

    # Combine all perfect games
    if not perfect_dfs:
        raise ValueError("No perfect games found across any runs!")

    combined_df = pd.concat(perfect_dfs, ignore_index=True)

    return combined_df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract perfect games (reward = 1.0) from all training runs"
    )
    parser.add_argument(
        "--output",
        default="perfect_games_combined/train.parquet",
        help="Output parquet file path (default: perfect_games_combined/train.parquet)"
    )
    parser.add_argument(
        "--base-dir",
        default="/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo",
        help="Base directory for runs"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the combined data before saving"
    )

    args = parser.parse_args()

    # Extract perfect games from all runs
    combined_df, stats = extract_perfect_games_from_runs(args.base_dir)

    # Shuffle if requested
    if args.shuffle:
        print("\nShuffling combined data...")
        combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save "all" file
    all_path = output_dir / "all_eq_1.00.parquet"
    combined_df.to_parquet(all_path, index=False)
    print(f"\nSaved complete dataset: {all_path} ({len(combined_df)} sequences)")

    # Split into train (90%) and val (10%)
    split_idx = int(len(combined_df) * 0.9)
    train_df = combined_df.iloc[:split_idx]
    val_df = combined_df.iloc[split_idx:]

    train_path = output_dir / "train_eq_1.00.parquet"
    val_path = output_dir / "val_eq_1.00.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved training split: {train_path} ({len(train_df)} sequences)")
    print(f"Saved validation split: {val_path} ({len(val_df)} sequences)")

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nRuns processed: {stats['runs_processed']}")
    print(f"Rounds processed: {stats['rounds_processed']}")
    print(f"Rounds skipped: {stats['rounds_skipped']}")
    print(f"\nTotal sequences seen: {stats['total_sequences_seen']}")
    print(f"Perfect sequences found: {stats['perfect_sequences_found']} "
          f"({stats['perfect_sequences_found']/stats['total_sequences_seen']*100:.2f}%)")
    print(f"\nOutput directory: {output_dir}")

    # Save stats to JSON
    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")

    return 0


if __name__ == "__main__":
    exit(main())
