"""Utilities for validating parquet files from GRPO training data."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def check_parquet_game_grouping(
    parquet_path: str,
    group_size: int = 8,
    mode: str = "selfplay",
) -> Dict[str, any]:
    """
    Validate that games in parquet are properly grouped with shared initial states.

    For self-play mode: groups of 2*group_size sequences (2 players × group_size plays)
    For asymmetric mode: groups of group_size sequences (1 trainee × group_size plays)

    Args:
        parquet_path: Path to the parquet file to validate
        group_size: Number of replays per unique initial state (default: 8)
        mode: 'selfplay' or 'asymmetric' (default: 'selfplay')

    Returns:
        dict with keys:
            - valid: bool, whether validation passed
            - errors: List[str], error messages
            - stats: Dict[str, any], validation statistics
    """
    df = pd.read_parquet(parquet_path)

    errors = []
    sequences_per_group = 2 * group_size if mode == "selfplay" else group_size

    stats = {
        "total_sequences": len(df),
        "expected_sequences_per_group": sequences_per_group,
        "num_groups": len(df) // sequences_per_group,
        "groups_checked": 0,
        "shared_tokens_mean": 0,
        "shared_tokens_std": 0,
        "shared_tokens_min": float("inf"),
        "shared_tokens_max": 0,
    }

    if "game_id" not in df.columns:
        errors.append("Missing 'game_id' column in parquet file")
        return {"valid": False, "errors": errors, "stats": stats}

    # Sort by game_id
    df = df.sort_values("game_id").reset_index(drop=True)

    # Check each group
    shared_token_counts = []

    for group_idx in range(stats["num_groups"]):
        start_idx = group_idx * sequences_per_group
        end_idx = start_idx + sequences_per_group
        group_df = df.iloc[start_idx:end_idx]

        # Verify all game_ids map to same group
        expected_unique_game_ids = set(
            [game_id // group_size for game_id in group_df["game_id"]]
        )
        if len(expected_unique_game_ids) != 1:
            errors.append(
                f"Group {group_idx}: game_ids don't map to same unique_game_id. "
                f"Found unique_game_ids: {expected_unique_game_ids}"
            )
            continue

        # Check that input_ids have a common prefix (system message with game setup)
        input_ids_list = [row["input_ids"].tolist() for _, row in group_df.iterrows()]

        # Find common prefix length
        common_prefix_length = find_common_prefix_length(input_ids_list)
        shared_token_counts.append(common_prefix_length)

        # Reasonable threshold: at least 100 tokens for system message
        if common_prefix_length < 100:
            errors.append(
                f"Group {group_idx}: Common prefix too short ({common_prefix_length} tokens). "
                f"Expected at least 100 tokens for game setup. "
                f"Games {group_idx * group_size} to {(group_idx + 1) * group_size - 1} "
                f"may not share the same initial state."
            )

        stats["groups_checked"] += 1

    # Compute statistics
    if shared_token_counts:
        stats["shared_tokens_mean"] = float(np.mean(shared_token_counts))
        stats["shared_tokens_std"] = float(np.std(shared_token_counts))
        stats["shared_tokens_min"] = int(min(shared_token_counts))
        stats["shared_tokens_max"] = int(max(shared_token_counts))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "stats": stats,
    }


def find_common_prefix_length(sequences: List[List[int]]) -> int:
    """
    Find the length of the common prefix across all sequences.

    Args:
        sequences: List of token ID lists

    Returns:
        Length of common prefix (number of tokens)
    """
    if not sequences:
        return 0

    min_length = min(len(seq) for seq in sequences)
    common_length = 0

    for i in range(min_length):
        first_token = sequences[0][i]
        if all(seq[i] == first_token for seq in sequences):
            common_length += 1
        else:
            break

    return common_length


def validate_grpo_normalization(
    parquet_path: str,
    group_size: int = 8,
    mode: str = "selfplay",
    tolerance: float = 1e-6,
) -> Dict[str, any]:
    """
    Validate that GRPO normalization was applied correctly.

    Checks that sample_weights within each group sum to approximately zero.

    Args:
        parquet_path: Path to the parquet file to validate
        group_size: Number of replays per unique initial state (default: 8)
        mode: 'selfplay' or 'asymmetric' (default: 'selfplay')
        tolerance: Acceptable error for zero-sum check (default: 1e-6)

    Returns:
        dict with keys:
            - valid: bool, whether validation passed
            - errors: List[str], error messages
            - stats: Dict[str, any], validation statistics
    """
    df = pd.read_parquet(parquet_path)

    errors = []
    sequences_per_group = 2 * group_size if mode == "selfplay" else group_size

    stats = {
        "total_sequences": len(df),
        "num_groups": len(df) // sequences_per_group,
        "groups_checked": 0,
        "max_group_sum_deviation": 0,
        "mean_group_sum_deviation": 0,
    }

    if "sample_weight" not in df.columns:
        errors.append("Missing 'sample_weight' column in parquet file")
        return {"valid": False, "errors": errors, "stats": stats}

    if "game_id" not in df.columns:
        errors.append("Missing 'game_id' column in parquet file")
        return {"valid": False, "errors": errors, "stats": stats}

    # Sort by game_id
    df = df.sort_values("game_id").reset_index(drop=True)

    # Check each group
    group_sum_deviations = []

    for group_idx in range(stats["num_groups"]):
        start_idx = group_idx * sequences_per_group
        end_idx = start_idx + sequences_per_group
        group_df = df.iloc[start_idx:end_idx]

        # Sum of sample_weights should be ~0 (GRPO normalization)
        group_sum = group_df["sample_weight"].sum()
        deviation = abs(group_sum)
        group_sum_deviations.append(deviation)

        if deviation > tolerance:
            errors.append(
                f"Group {group_idx}: Sample weights sum to {group_sum:.6f}, "
                f"expected ~0 (within {tolerance}). GRPO normalization may be incorrect."
            )

        stats["groups_checked"] += 1

    # Compute statistics
    if group_sum_deviations:
        stats["max_group_sum_deviation"] = float(max(group_sum_deviations))
        stats["mean_group_sum_deviation"] = float(np.mean(group_sum_deviations))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "stats": stats,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate GRPO parquet files")
    parser.add_argument("parquet_file", help="Path to parquet file to validate")
    parser.add_argument(
        "--group-size", type=int, default=8, help="GRPO group size (default: 8)"
    )
    parser.add_argument(
        "--mode",
        choices=["selfplay", "asymmetric"],
        default="selfplay",
        help="Game mode (default: selfplay)",
    )
    parser.add_argument(
        "--check-normalization",
        action="store_true",
        help="Also check GRPO normalization (zero-sum sample weights)",
    )

    args = parser.parse_args()

    print(f"Validating GRPO grouping in: {args.parquet_file}")
    print(f"Group size: {args.group_size}, Mode: {args.mode}\n")

    # Check grouping
    result = check_parquet_game_grouping(
        args.parquet_file, group_size=args.group_size, mode=args.mode
    )

    print("=" * 60)
    print("GROUPING VALIDATION RESULTS")
    print("=" * 60)
    print(f"Valid: {result['valid']}")
    print(f"\nStatistics:")
    for key, value in result["stats"].items():
        print(f"  {key}: {value}")

    if result["errors"]:
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result["errors"]:
            print(f"  - {error}")

    # Check normalization if requested
    if args.check_normalization:
        print("\n" + "=" * 60)
        print("NORMALIZATION VALIDATION RESULTS")
        print("=" * 60)

        norm_result = validate_grpo_normalization(
            args.parquet_file, group_size=args.group_size, mode=args.mode
        )

        print(f"Valid: {norm_result['valid']}")
        print(f"\nStatistics:")
        for key, value in norm_result["stats"].items():
            print(f"  {key}: {value}")

        if norm_result["errors"]:
            print(f"\nErrors ({len(norm_result['errors'])}):")
            for error in norm_result["errors"]:
                print(f"  - {error}")

    # Exit with appropriate code
    exit_code = 0 if result["valid"] else 1
    if args.check_normalization:
        exit_code = max(exit_code, 0 if norm_result["valid"] else 1)

    exit(exit_code)
