#!/usr/bin/env python3
"""
Filter training parquets across all rounds of a run by reward thresholds.

For each round's training parquet, creates separate filtered parquet files:
- train_filtered_gte_0.95.parquet - Sequences with reward >= 0.95
- train_filtered_gte_0.99.parquet - Sequences with reward >= 0.99
- train_filtered_eq_1.0.parquet - Sequences with reward = 1.0 (perfect)

Supports filtering the most recent run or a specific run by name/timestamp.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def find_most_recent_run(base_logs_dir: str = "/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo") -> Optional[Path]:
    """Find the most recent run directory based on timestamp."""
    base_path = Path(base_logs_dir)
    if not base_path.exists():
        return None

    # Find all timestamp directories
    run_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.replace("_", "").isdigit():
            run_dirs.append(item)

    if not run_dirs:
        return None

    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def find_run_by_name(run_name: str, base_logs_dir: str = "/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo") -> Optional[Path]:
    """Find a run directory by name or timestamp."""
    base_path = Path(base_logs_dir)
    if not base_path.exists():
        return None

    # Try exact match first
    run_path = base_path / run_name
    if run_path.exists() and run_path.is_dir():
        return run_path

    # Try partial match for timestamp
    for item in base_path.iterdir():
        if item.is_dir() and run_name in item.name:
            return item

    return None


def extract_reward_from_parquet(parquet_path: Path) -> Optional[np.ndarray]:
    """Extract normalized reward array from a parquet file.

    Tries the following fields in order:
    1. game_normalized_reward
    2. normalized_reward
    3. game_info JSON field containing normalized reward

    Returns:
        Array of rewards, or None if not found
    """
    try:
        df = pd.read_parquet(parquet_path)

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
                        continue

                    if "game_normalized_reward" in game_info:
                        rewards.append(float(game_info["game_normalized_reward"]))
                    elif "normalized_reward" in game_info:
                        rewards.append(float(game_info["normalized_reward"]))
                except Exception:
                    continue

            if len(rewards) == len(df):
                return np.array(rewards, dtype=float)

        return None
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return None


def filter_and_save_round(round_dir: Path, create_files: bool = True) -> Optional[Dict]:
    """Filter training parquet in a round directory and save filtered versions.

    Args:
        round_dir: Path to round directory
        create_files: If True, create filtered parquet files. If False, only compute stats.

    Returns:
        Dictionary with counts for each threshold, or None if no parquet found
    """
    # Look for training parquet (prefer trimmed, fall back to raw)
    train_parquet = round_dir / "train_trimmed.parquet"
    if not train_parquet.exists():
        train_parquet = round_dir / "train.parquet"

    if not train_parquet.exists():
        return None

    try:
        df = pd.read_parquet(train_parquet)
    except Exception as e:
        print(f"Error reading {train_parquet}: {e}")
        return None

    # Extract rewards
    rewards = None
    reward_column = None

    if "game_normalized_reward" in df.columns:
        rewards = df["game_normalized_reward"].astype(float).to_numpy()
        reward_column = "game_normalized_reward"
    elif "normalized_reward" in df.columns:
        rewards = df["normalized_reward"].astype(float).to_numpy()
        reward_column = "normalized_reward"
    elif "game_info" in df.columns:
        # Parse from game_info JSON
        reward_list = []
        for val in df["game_info"]:
            try:
                if isinstance(val, str):
                    game_info = json.loads(val)
                elif isinstance(val, dict):
                    game_info = val
                else:
                    reward_list.append(None)
                    continue

                if "game_normalized_reward" in game_info:
                    reward_list.append(float(game_info["game_normalized_reward"]))
                elif "normalized_reward" in game_info:
                    reward_list.append(float(game_info["normalized_reward"]))
                else:
                    reward_list.append(None)
            except Exception:
                reward_list.append(None)

        if None not in reward_list and len(reward_list) == len(df):
            rewards = np.array(reward_list, dtype=float)
            reward_column = "game_info (parsed)"

    if rewards is None or len(rewards) == 0:
        print(f"Warning: Could not extract rewards from {train_parquet}")
        return None

    # Create boolean masks for each threshold
    mask_gte_095 = rewards >= 0.95
    mask_gte_099 = rewards >= 0.99
    mask_eq_10 = rewards == 1.0

    # Save filtered parquet files if requested
    if create_files:
        if mask_gte_095.sum() > 0:
            output_path = round_dir / "train_filtered_gte_0.95.parquet"
            df[mask_gte_095].to_parquet(output_path, index=False)
            print(f"  Created {output_path.name} with {mask_gte_095.sum()} sequences")

        if mask_gte_099.sum() > 0:
            output_path = round_dir / "train_filtered_gte_0.99.parquet"
            df[mask_gte_099].to_parquet(output_path, index=False)
            print(f"  Created {output_path.name} with {mask_gte_099.sum()} sequences")

        if mask_eq_10.sum() > 0:
            output_path = round_dir / "train_filtered_eq_1.0.parquet"
            df[mask_eq_10].to_parquet(output_path, index=False)
            print(f"  Created {output_path.name} with {mask_eq_10.sum()} sequences")

    # Compute counts for each threshold
    return {
        "round_dir": str(round_dir),
        "parquet_file": train_parquet.name,
        "total": len(rewards),
        "gte_0.95": int(mask_gte_095.sum()),
        "gte_0.99": int(mask_gte_099.sum()),
        "eq_1.0": int(mask_eq_10.sum()),
        "gte_0.95_pct": float(mask_gte_095.mean() * 100),
        "gte_0.99_pct": float(mask_gte_099.mean() * 100),
        "eq_1.0_pct": float(mask_eq_10.mean() * 100),
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
    }


def process_run(run_dir: Path, create_files: bool = True, concatenate: bool = True) -> Tuple[List[Dict], Optional[Path]]:
    """Process all rounds in a run directory, optionally creating filtered parquet files.

    Args:
        run_dir: Path to run directory
        create_files: If True, create filtered parquet files per round
        concatenate: If True, concatenate all filtered sequences across rounds

    Returns:
        Tuple of (analysis results list, output directory path if concatenated)
    """
    results = []
    output_dir = None

    # Find all round directories
    round_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("round_")])

    # Collect dataframes for concatenation
    dfs_gte_095 = []
    dfs_gte_099 = []
    dfs_eq_10 = []

    for round_dir in round_dirs:
        if create_files:
            print(f"Processing {round_dir.name}...")
        result = filter_and_save_round(round_dir, create_files=create_files)
        if result is not None:
            results.append(result)

            # If concatenating, collect the dataframes
            if concatenate:
                # Read the source parquet
                train_parquet = round_dir / "train_trimmed.parquet"
                if not train_parquet.exists():
                    train_parquet = round_dir / "train.parquet"

                if train_parquet.exists():
                    try:
                        df = pd.read_parquet(train_parquet)

                        # Extract rewards
                        rewards = None
                        if "game_normalized_reward" in df.columns:
                            rewards = df["game_normalized_reward"].astype(float).to_numpy()
                        elif "normalized_reward" in df.columns:
                            rewards = df["normalized_reward"].astype(float).to_numpy()

                        if rewards is not None:
                            # Filter and collect
                            mask_gte_095 = rewards >= 0.95
                            mask_gte_099 = rewards >= 0.99
                            mask_eq_10 = rewards == 1.0

                            if mask_gte_095.sum() > 0:
                                dfs_gte_095.append(df[mask_gte_095])
                            if mask_gte_099.sum() > 0:
                                dfs_gte_099.append(df[mask_gte_099])
                            if mask_eq_10.sum() > 0:
                                dfs_eq_10.append(df[mask_eq_10])
                    except Exception as e:
                        print(f"  Warning: Could not process {train_parquet} for concatenation: {e}")

    # Concatenate and save if requested
    if concatenate and (dfs_gte_095 or dfs_gte_099 or dfs_eq_10):
        output_dir = run_dir / "filtered_combined"
        output_dir.mkdir(exist_ok=True)

        print(f"\nConcatenating filtered sequences across all rounds...")

        # Helper function to split and save train/val
        def save_train_val_split(combined_df: pd.DataFrame, prefix: str):
            """Split dataframe into 90/10 train/val and save both."""
            # Shuffle the data
            shuffled_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

            # Calculate split point
            total_samples = len(shuffled_df)
            train_samples = int(total_samples * 0.9)

            # Split
            train_df = shuffled_df.iloc[:train_samples]
            val_df = shuffled_df.iloc[train_samples:]

            # Save train
            train_path = output_dir / f"train_{prefix}.parquet"
            train_df.to_parquet(train_path, index=False)
            print(f"  Created {train_path.name} with {len(train_df)} sequences")

            # Save val
            val_path = output_dir / f"val_{prefix}.parquet"
            val_df.to_parquet(val_path, index=False)
            print(f"  Created {val_path.name} with {len(val_df)} sequences")

        if dfs_gte_095:
            combined_df = pd.concat(dfs_gte_095, ignore_index=True)
            save_train_val_split(combined_df, "filtered_gte_0.95")

        if dfs_gte_099:
            combined_df = pd.concat(dfs_gte_099, ignore_index=True)
            save_train_val_split(combined_df, "filtered_gte_0.99")

        if dfs_eq_10:
            combined_df = pd.concat(dfs_eq_10, ignore_index=True)
            save_train_val_split(combined_df, "filtered_eq_1.0")

    return results, output_dir


def print_results(results: List[Dict], run_dir: Path):
    """Print analysis results in a formatted table."""
    if not results:
        print("No training parquets found in this run.")
        return

    print(f"\n{'='*100}")
    print(f"Analysis for run: {run_dir}")
    print(f"{'='*100}\n")

    # Print header
    header = f"{'Round':<10} {'Total':<8} {'≥0.95':<10} {'≥0.99':<10} {'=1.0':<10} {'Mean':<8} {'Median':<8}"
    print(header)
    print("-" * 100)

    # Print each round
    for result in results:
        round_name = Path(result["round_dir"]).name
        print(
            f"{round_name:<10} "
            f"{result['total']:<8} "
            f"{result['gte_0.95']:<4} ({result['gte_0.95_pct']:5.1f}%) "
            f"{result['gte_0.99']:<4} ({result['gte_0.99_pct']:5.1f}%) "
            f"{result['eq_1.0']:<4} ({result['eq_1.0_pct']:5.1f}%) "
            f"{result['mean_reward']:<8.4f} "
            f"{result['median_reward']:<8.4f}"
        )

    # Print summary statistics
    print("-" * 100)
    total_sequences = sum(r["total"] for r in results)
    total_gte_095 = sum(r["gte_0.95"] for r in results)
    total_gte_099 = sum(r["gte_0.99"] for r in results)
    total_eq_10 = sum(r["eq_1.0"] for r in results)

    print(
        f"{'TOTAL':<10} "
        f"{total_sequences:<8} "
        f"{total_gte_095:<4} ({total_gte_095/total_sequences*100:5.1f}%) "
        f"{total_gte_099:<4} ({total_gte_099/total_sequences*100:5.1f}%) "
        f"{total_eq_10:<4} ({total_eq_10/total_sequences*100:5.1f}%) "
        f"{'-':<8} "
        f"{'-':<8}"
    )
    print()


def save_results(results: List[Dict], output_file: Path):
    """Save results to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter training parquets by reward thresholds and create separate files"
    )
    parser.add_argument("--run", default="", help="Run name/timestamp to process (default: most recent)")
    parser.add_argument("--output", default="", help="Output JSON file path (default: run_dir/reward_analysis.json)")
    parser.add_argument("--base-dir", default="/home/nickatomlin/georgiazhou/self_play/logs/offline_grpo",
                       help="Base directory for runs")
    parser.add_argument("--no-create-files", dest="create_files", action="store_false",
                       help="Don't create filtered parquet files per round, only show statistics")
    parser.add_argument("--no-concatenate", dest="concatenate", action="store_false",
                       help="Don't create concatenated files across all rounds")
    parser.set_defaults(create_files=True, concatenate=True)

    args = parser.parse_args()

    # Find the run directory
    if args.run:
        run_dir = find_run_by_name(args.run, args.base_dir)
        if run_dir is None:
            print(f"Error: Could not find run '{args.run}'")
            return 1
    else:
        run_dir = find_most_recent_run(args.base_dir)
        if run_dir is None:
            print(f"Error: No runs found in {args.base_dir}")
            return 1

    if args.create_files:
        print(f"Processing run and creating filtered parquet files: {run_dir}\n")
    else:
        print(f"Analyzing run (stats only): {run_dir}\n")

    # Process all rounds
    results, output_dir = process_run(run_dir, create_files=args.create_files, concatenate=args.concatenate)

    if args.create_files:
        print(f"\nCreated filtered parquet files in each round directory:")
        print("  - train_filtered_gte_0.95.parquet (reward >= 0.95)")
        print("  - train_filtered_gte_0.99.parquet (reward >= 0.99)")
        print("  - train_filtered_eq_1.0.parquet (reward = 1.0)")

    if output_dir:
        print(f"\nConcatenated files saved to: {output_dir}")
        print("  Use these for SFT training across all rounds!")

    # Print results
    print_results(results, run_dir)

    # Save results to JSON
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = run_dir / "reward_filter_summary.json"

    save_results(results, output_file)

    return 0


if __name__ == "__main__":
    exit(main())
