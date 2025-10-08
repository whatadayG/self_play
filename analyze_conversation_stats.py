#!/usr/bin/env python3
"""
Analyze conversation statistics from offline_grpo round data.

Computes median, 95%ile, and max conversation lengths for:
- Runs with score > 0.5
- Runs with nonzero score

Conversation length is measured in multiple ways:
- turn_count from game_info (if available)
- Counting messages in full_conversation
- Counting <|im_start|>assistant and <|im_start|>user tokens in input_ids
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer


def extract_turn_counts(df: pd.DataFrame, tokenizer: Optional[AutoTokenizer] = None) -> Dict[str, List[int]]:
    """Extract turn counts from multiple sources.

    Returns:
        Dict with keys:
        - 'turn_count_from_game_info': From game_info JSON
        - 'turn_count_from_full_conversation': Counting messages in full_conversation
        - 'turn_count_from_tokens': Counting role markers in tokenized sequence
        - 'messages_with_errors': Count including error messages
    """
    results = {
        'turn_count_from_game_info': [],
        'turn_count_from_full_conversation': [],
        'turn_count_from_tokens': [],
        'num_messages': [],
    }

    for idx, row in df.iterrows():
        # Method 1: From game_info
        try:
            game_info = json.loads(row['game_info']) if isinstance(row['game_info'], str) else row['game_info']
            if 'turn_count' in game_info:
                results['turn_count_from_game_info'].append(game_info['turn_count'])
            else:
                results['turn_count_from_game_info'].append(None)

            if 'num_messages' in game_info:
                results['num_messages'].append(game_info['num_messages'])
            else:
                results['num_messages'].append(None)
        except Exception as e:
            results['turn_count_from_game_info'].append(None)
            results['num_messages'].append(None)

        # Method 2: From full_conversation
        try:
            full_conv = row['full_conversation']
            if isinstance(full_conv, str):
                full_conv = json.loads(full_conv)
            if isinstance(full_conv, list):
                results['turn_count_from_full_conversation'].append(len(full_conv))
            else:
                results['turn_count_from_full_conversation'].append(None)
        except Exception as e:
            results['turn_count_from_full_conversation'].append(None)

        # Method 3: From tokens (if tokenizer provided)
        if tokenizer is not None:
            try:
                text = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
                # Count role markers
                assistant_count = text.count('<|im_start|>assistant')
                user_count = text.count('<|im_start|>user')
                # Turn count is roughly the number of exchanges
                turn_count = max(assistant_count, user_count)
                results['turn_count_from_tokens'].append(turn_count)
            except Exception as e:
                results['turn_count_from_tokens'].append(None)
        else:
            results['turn_count_from_tokens'].append(None)

    return results


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute median, 95%ile, and max for a list of values."""
    if not values or all(v is None for v in values):
        return {'count': 0, 'median': None, 'p95': None, 'max': None, 'mean': None}

    # Filter out None values
    clean_values = [v for v in values if v is not None]

    if not clean_values:
        return {'count': 0, 'median': None, 'p95': None, 'max': None, 'mean': None}

    arr = np.array(clean_values)
    return {
        'count': len(arr),
        'median': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
    }


def analyze_parquet(parquet_path: Path, use_tokenizer: bool = True) -> Dict[str, any]:
    """Analyze conversation statistics from a parquet file.

    Args:
        parquet_path: Path to parquet file
        use_tokenizer: Whether to use tokenizer for token-based counting (slower)

    Returns:
        Dictionary with statistics
    """
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} sequences")

    # Load tokenizer if requested
    tokenizer = None
    if use_tokenizer:
        print("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            tokenizer = None

    # Extract turn counts
    print("Extracting turn counts...")
    turn_data = extract_turn_counts(df, tokenizer)

    # Extract scores
    scores = df['game_normalized_reward'].tolist()

    # Create analysis masks
    score_gt_0_5 = [s > 0.5 for s in scores]
    score_gt_0 = [s > 0 for s in scores]

    results = {
        'total_sequences': len(df),
        'scores_gt_0_5': sum(score_gt_0_5),
        'scores_gt_0': sum(score_gt_0),
    }

    # For each turn count method, compute statistics for both filters
    for method_name, turn_counts in turn_data.items():
        # Filter for score > 0.5
        filtered_gt_0_5 = [tc for tc, mask in zip(turn_counts, score_gt_0_5) if mask and tc is not None]
        stats_gt_0_5 = compute_statistics(filtered_gt_0_5)

        # Filter for score > 0
        filtered_gt_0 = [tc for tc, mask in zip(turn_counts, score_gt_0) if mask and tc is not None]
        stats_gt_0 = compute_statistics(filtered_gt_0)

        results[f'{method_name}_score_gt_0_5'] = stats_gt_0_5
        results[f'{method_name}_score_gt_0'] = stats_gt_0

    return results


def print_results(results: Dict[str, any], output_file: Optional[Path] = None):
    """Print results in a readable format."""
    lines = []

    lines.append("=" * 80)
    lines.append("CONVERSATION LENGTH STATISTICS")
    lines.append("=" * 80)
    lines.append(f"Total sequences: {results['total_sequences']}")
    lines.append(f"Sequences with score > 0.5: {results['scores_gt_0_5']}")
    lines.append(f"Sequences with score > 0: {results['scores_gt_0']}")
    lines.append("")

    # Define method names for display
    method_display_names = {
        'turn_count_from_game_info': 'Turn Count (from game_info)',
        'turn_count_from_full_conversation': 'Message Count (from full_conversation)',
        'turn_count_from_tokens': 'Turn Count (from tokens)',
        'num_messages': 'Num Messages (from game_info)',
    }

    for method_name, display_name in method_display_names.items():
        lines.append("-" * 80)
        lines.append(f"{display_name}")
        lines.append("-" * 80)

        # Score > 0.5
        stats_key = f'{method_name}_score_gt_0_5'
        if stats_key in results:
            stats = results[stats_key]
            lines.append(f"\nFor score > 0.5 (n={stats['count']}):")
            if stats['count'] > 0:
                lines.append(f"  Median: {stats['median']:.2f}")
                lines.append(f"  95th percentile: {stats['p95']:.2f}")
                lines.append(f"  Max: {stats['max']:.2f}")
                lines.append(f"  Mean: {stats['mean']:.2f}")
            else:
                lines.append("  No data available")

        # Score > 0
        stats_key = f'{method_name}_score_gt_0'
        if stats_key in results:
            stats = results[stats_key]
            lines.append(f"\nFor score > 0 (n={stats['count']}):")
            if stats['count'] > 0:
                lines.append(f"  Median: {stats['median']:.2f}")
                lines.append(f"  95th percentile: {stats['p95']:.2f}")
                lines.append(f"  Max: {stats['max']:.2f}")
                lines.append(f"  Mean: {stats['mean']:.2f}")
            else:
                lines.append("  No data available")

        lines.append("")

    lines.append("=" * 80)

    output_text = "\n".join(lines)
    print(output_text)

    if output_file is not None:
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'w') as f:
            f.write(output_text)
            f.write("\n\nRaw JSON:\n")
            f.write(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Analyze conversation statistics from parquet file")
    parser.add_argument(
        "--parquet",
        type=str,
        default="logs/offline_grpo/20251008_082305/round_000/train.parquet",
        help="Path to parquet file (default: most recent round_000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: same directory as parquet, conversation_stats.txt)",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip token-based counting (faster)",
    )

    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Error: Parquet file not found: {parquet_path}")
        return 1

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = parquet_path.parent / "conversation_stats.txt"

    # Run analysis
    results = analyze_parquet(parquet_path, use_tokenizer=not args.no_tokenizer)

    # Print and save results
    print_results(results, output_file)

    return 0


if __name__ == "__main__":
    exit(main())
