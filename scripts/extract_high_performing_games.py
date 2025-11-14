#!/usr/bin/env python3
"""
Extract high-performing games from all training runs in logs/offline_grpo.

Uses the RAW train.parquet files (not train_trimmed.parquet) to get unfiltered data.
Skips symlinked directories to avoid duplication from branched runs.

For each reward threshold (>= 0.95, >= 0.97, >= 0.99, == 1.00), creates:
- all_*.parquet - Complete dataset at this threshold
- train_*.parquet - 90% training split
- val_*.parquet - 10% validation split
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


def extract_games_by_thresholds(
    base_logs_dir: str = "/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo",
    thresholds: List[float] = [0.95, 0.97, 0.99, 1.0]
) -> Tuple[Dict[float, pd.DataFrame], dict]:
    """Extract games meeting various reward thresholds from all runs.

    Args:
        base_logs_dir: Base directory containing run directories
        thresholds: List of reward thresholds to extract (uses >= comparison, except 1.0 uses ==)

    Returns:
        Tuple of (dict mapping threshold to dataframe, stats dictionary)
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

    # Collect dataframes for each threshold
    threshold_dfs = {t: [] for t in thresholds}

    stats = {
        "runs_processed": 0,
        "rounds_processed": 0,
        "rounds_skipped": 0,
        "total_sequences_seen": 0,
        "by_threshold": {t: 0 for t in thresholds},
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
            "by_threshold": {t: 0 for t in thresholds},
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

            # Filter for each threshold
            threshold_counts = {}
            for threshold in thresholds:
                if threshold == 1.0:
                    # Use exact match for 1.0
                    mask = rewards == threshold
                else:
                    # Use >= for other thresholds
                    mask = rewards >= threshold

                count = int(mask.sum())
                threshold_counts[threshold] = count

                if count > 0:
                    filtered_df = df[mask].copy()
                    threshold_dfs[threshold].append(filtered_df)

                run_stats["by_threshold"][threshold] += count

            # Update stats
            run_stats["rounds_processed"] += 1
            run_stats["total_sequences"] += len(df)

            # Print round summary
            threshold_str = ", ".join([
                f">={'=' if t == 1.0 else ''}{t}: {threshold_counts[t]}"
                for t in sorted(thresholds)
            ])
            print(f"  {round_name}: {len(df)} total ({threshold_str})")

        # Update global stats
        if run_stats["rounds_processed"] > 0:
            stats["runs_processed"] += 1
            stats["rounds_processed"] += run_stats["rounds_processed"]
            stats["rounds_skipped"] += run_stats["rounds_skipped"]
            stats["total_sequences_seen"] += run_stats["total_sequences"]
            for t in thresholds:
                stats["by_threshold"][t] += run_stats["by_threshold"][t]
            stats["by_run"][run_name] = run_stats

            # Print run summary
            threshold_str = ", ".join([
                f">={'=' if t == 1.0 else ''}{t}: {run_stats['by_threshold'][t]}"
                for t in sorted(thresholds)
            ])
            print(f"  Run summary: {run_stats['total_sequences']} total ({threshold_str})")

    # Combine dataframes for each threshold
    combined_dfs = {}
    for threshold, dfs in threshold_dfs.items():
        if dfs:
            combined_dfs[threshold] = pd.concat(dfs, ignore_index=True)
        else:
            combined_dfs[threshold] = pd.DataFrame()

    return combined_dfs, stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract high-performing games at multiple reward thresholds"
    )
    parser.add_argument(
        "--output-dir",
        default="high_performing_games",
        help="Output directory for parquet files (default: high_performing_games)"
    )
    parser.add_argument(
        "--base-dir",
        default="/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo",
        help="Base directory for runs"
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.95, 0.97, 0.99, 1.0],
        help="Reward thresholds to extract (default: 0.95 0.97 0.99 1.0)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the combined data before saving"
    )

    args = parser.parse_args()

    # Extract games at different thresholds
    combined_dfs, stats = extract_games_by_thresholds(args.base_dir, args.thresholds)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each threshold to separate files (all, train, val)
    for threshold, df in combined_dfs.items():
        if len(df) == 0:
            print(f"\nNo games found for threshold >={'=' if threshold == 1.0 else ''}{threshold}")
            continue

        # Shuffle if requested
        if args.shuffle:
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Format filename prefix
        if threshold == 1.0:
            prefix = "eq_1.00"
        else:
            prefix = f"gte_{threshold:.2f}"

        # Save "all" file (complete dataset)
        all_path = output_dir / f"all_{prefix}.parquet"
        df.to_parquet(all_path, index=False)
        print(f"\nThreshold >={'=' if threshold == 1.0 else ''}{threshold}:")
        print(f"  All: {len(df)} sequences -> {all_path}")

        # Split into train (90%) and val (10%)
        if len(df) >= 10:  # Only split if we have enough data
            # Simple train/test split (data is already shuffled if requested)
            split_idx = int(len(df) * 0.9)
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]

            train_path = output_dir / f"train_{prefix}.parquet"
            val_path = output_dir / f"val_{prefix}.parquet"

            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)

            print(f"  Train: {len(train_df)} sequences -> {train_path}")
            print(f"  Val: {len(val_df)} sequences -> {val_path}")
        else:
            print(f"  (Not enough data for train/val split, need at least 10 sequences)")

    # Print summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nRuns processed: {stats['runs_processed']}")
    print(f"Rounds processed: {stats['rounds_processed']}")
    print(f"Rounds skipped: {stats['rounds_skipped']}")
    print(f"\nTotal sequences examined: {stats['total_sequences_seen']:,}")
    print(f"\nSequences extracted by threshold:")
    for threshold in sorted(args.thresholds):
        count = stats['by_threshold'][threshold]
        pct = count / stats['total_sequences_seen'] * 100 if stats['total_sequences_seen'] > 0 else 0
        op = "==" if threshold == 1.0 else ">="
        print(f"  {op} {threshold:.2f}: {count:6,} ({pct:5.2f}%)")

    # Save stats to JSON
    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")

    return 0


if __name__ == "__main__":
    exit(main())
