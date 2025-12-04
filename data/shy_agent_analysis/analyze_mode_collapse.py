#!/usr/bin/env python3
"""
Analyze mode collapse metrics extracted by measure_mode_collapse.py

Generates comparison tables:
- By training round
- By game outcome (reward percentiles)
- By conversation length (short/medium/long)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def analyze_by_round(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze mode collapse metrics by training round."""
    print("\nAnalyzing by round...")

    results = conversations_df.groupby('round').agg({
        'first_turn_total_info': ['mean', 'std', 'min', 'max'],
        'first_turn_pairs': ['mean', 'std'],
        'first_turn_qualitative': ['mean', 'std'],
        'first_turn_numbers': ['mean', 'std'],
        'proposal_turn': ['mean', 'std'],
        'game_reward': 'mean',
        'conversation_length': 'mean',
        'game_id': 'count',
    })

    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    results = results.rename(columns={'game_id_count': 'n_conversations'})

    return results.round(3)


def analyze_by_outcome(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze mode collapse by game outcome (reward percentiles)."""
    print("\nAnalyzing by outcome...")

    # Create reward percentile bins
    conversations_df['reward_percentile'] = pd.qcut(
        conversations_df['game_reward'],
        q=4,
        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    )

    results = conversations_df.groupby('reward_percentile').agg({
        'first_turn_total_info': ['mean', 'std'],
        'first_turn_pairs': 'mean',
        'proposal_turn': ['mean', 'std'],
        'game_reward': ['mean', 'min', 'max'],
        'conversation_length': 'mean',
        'game_id': 'count',
    })

    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    results = results.rename(columns={'game_id_count': 'n_conversations'})

    return results.round(3)


def analyze_by_length(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze mode collapse by conversation length."""
    print("\nAnalyzing by conversation length...")

    # Create length bins
    def length_category(length):
        if length < 6:
            return 'Short (<6)'
        elif length < 9:
            return 'Medium (6-8)'
        else:
            return 'Long (9+)'

    conversations_df['length_category'] = conversations_df['conversation_length'].apply(length_category)

    results = conversations_df.groupby('length_category').agg({
        'first_turn_total_info': ['mean', 'std'],
        'first_turn_pairs': 'mean',
        'proposal_turn': ['mean', 'std'],
        'game_reward': 'mean',
        'conversation_length': ['mean', 'min', 'max'],
        'game_id': 'count',
    })

    results.columns = ['_'.join(col).strip() for col in results.columns.values]
    results = results.rename(columns={'game_id_count': 'n_conversations'})

    # Reorder rows
    order = ['Short (<6)', 'Medium (6-8)', 'Long (9+)']
    results = results.reindex([cat for cat in order if cat in results.index])

    return results.round(3)


def analyze_by_round_and_outcome(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation: round x outcome."""
    print("\nAnalyzing by round and outcome...")

    conversations_df['reward_category'] = pd.qcut(
        conversations_df['game_reward'],
        q=3,
        labels=['Low', 'Med', 'High']
    )

    results = conversations_df.groupby(['round', 'reward_category']).agg({
        'first_turn_total_info': 'mean',
        'first_turn_pairs': 'mean',
        'game_id': 'count',
    })

    results.columns = ['first_turn_info_mean', 'first_turn_pairs_mean', 'n_conversations']

    return results.round(3)


def analyze_by_round_and_length(conversations_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation: round x length."""
    print("\nAnalyzing by round and length...")

    conversations_df['length_category'] = conversations_df['conversation_length'].apply(
        lambda x: 'Short (<6)' if x < 6 else ('Medium (6-8)' if x < 9 else 'Long (9+)')
    )

    results = conversations_df.groupby(['round', 'length_category']).agg({
        'first_turn_total_info': 'mean',
        'proposal_turn': 'mean',
        'game_id': 'count',
    })

    results.columns = ['first_turn_info_mean', 'proposal_turn_mean', 'n_conversations']

    return results.round(3)


def print_summary_statistics(conversations_df: pd.DataFrame, diversity_df: pd.DataFrame):
    """Print key summary statistics."""
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # First-turn variance over training
    first_turn_by_round = conversations_df.groupby('round')['first_turn_total_info'].agg(['mean', 'std'])

    print("\nFirst-turn information variance across training:")
    print(f"  Round 0: mean={first_turn_by_round.loc[0, 'mean']:.2f}, std={first_turn_by_round.loc[0, 'std']:.2f}")
    last_round = first_turn_by_round.index.max()
    print(f"  Round {last_round}: mean={first_turn_by_round.loc[last_round, 'mean']:.2f}, std={first_turn_by_round.loc[last_round, 'std']:.2f}")

    std_change = ((first_turn_by_round.loc[last_round, 'std'] - first_turn_by_round.loc[0, 'std']) /
                  first_turn_by_round.loc[0, 'std'] * 100)
    print(f"  Variance change: {std_change:+.1f}%")

    if std_change < -10:
        print("  → Significant DECREASE in variance (more mode collapse)")
    elif std_change > 10:
        print("  → INCREASE in variance (more diversity)")
    else:
        print("  → Stable variance")

    # Text diversity trends
    if not diversity_df.empty:
        print("\nText diversity metrics:")
        print(f"  Distinct-2 (Round 0): {diversity_df.loc[0, 'distinct_2']:.4f}")
        print(f"  Distinct-2 (Round {last_round}): {diversity_df.loc[last_round, 'distinct_2']:.4f}")

        d2_change = ((diversity_df.loc[last_round, 'distinct_2'] - diversity_df.loc[0, 'distinct_2']) /
                     diversity_df.loc[0, 'distinct_2'] * 100)
        print(f"  Change: {d2_change:+.1f}%")

        if d2_change < -5:
            print("  → Text becoming more repetitive")
        elif d2_change > 5:
            print("  → Text becoming more diverse")
        else:
            print("  → Stable text diversity")

    # Outcome comparison
    reward_bins = pd.qcut(conversations_df['game_reward'], q=4, labels=False)
    conversations_df['reward_bin'] = reward_bins

    high_perf = conversations_df[reward_bins == 3]['first_turn_total_info'].std()
    low_perf = conversations_df[reward_bins == 0]['first_turn_total_info'].std()

    print(f"\nVariance by performance:")
    print(f"  High-reward games: std={high_perf:.2f}")
    print(f"  Low-reward games: std={low_perf:.2f}")

    if high_perf < low_perf * 0.9:
        print("  → High-performing games show MORE mode collapse")
    elif high_perf > low_perf * 1.1:
        print("  → High-performing games show LESS mode collapse")
    else:
        print("  → Similar variance across performance levels")


def main():
    parser = argparse.ArgumentParser(description="Analyze mode collapse metrics")
    parser.add_argument("run_id", type=str, help="Run identifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/shy_agent_analysis"),
                       help="Directory containing mode collapse data")

    args = parser.parse_args()

    data_dir = args.data_dir
    run_id = args.run_id

    # Load data
    print(f"Loading mode collapse data for run: {run_id}")
    conversations_path = data_dir / f"mode_collapse_conversations_{run_id}.parquet"
    diversity_path = data_dir / f"mode_collapse_diversity_{run_id}.csv"

    if not conversations_path.exists():
        print(f"Error: {conversations_path} not found")
        print("Run measure_mode_collapse.py first!")
        return 1

    conversations_df = pd.read_parquet(conversations_path)
    diversity_df = pd.read_csv(diversity_path) if diversity_path.exists() else pd.DataFrame()

    print(f"Loaded {len(conversations_df)} conversations")

    # Generate analyses
    print("\n" + "=" * 80)
    print("GENERATING ANALYSES")
    print("=" * 80)

    # By round
    by_round = analyze_by_round(conversations_df)
    print("\nBy Round:")
    print(by_round)

    # By outcome
    by_outcome = analyze_by_outcome(conversations_df)
    print("\nBy Outcome (Reward Quartiles):")
    print(by_outcome)

    # By length
    by_length = analyze_by_length(conversations_df)
    print("\nBy Conversation Length:")
    print(by_length)

    # Cross-tabs
    by_round_outcome = analyze_by_round_and_outcome(conversations_df)
    by_round_length = analyze_by_round_and_length(conversations_df)

    # Save all results
    output_prefix = data_dir / f"mode_collapse_analysis_{run_id}"

    by_round.to_csv(f"{output_prefix}_by_round.csv")
    by_outcome.to_csv(f"{output_prefix}_by_outcome.csv")
    by_length.to_csv(f"{output_prefix}_by_length.csv")
    by_round_outcome.to_csv(f"{output_prefix}_by_round_outcome.csv")
    by_round_length.to_csv(f"{output_prefix}_by_round_length.csv")

    print(f"\n\nSaved analysis results to:")
    print(f"  {output_prefix}_by_round.csv")
    print(f"  {output_prefix}_by_outcome.csv")
    print(f"  {output_prefix}_by_length.csv")
    print(f"  {output_prefix}_by_round_outcome.csv")
    print(f"  {output_prefix}_by_round_length.csv")

    # Print summary
    print_summary_statistics(conversations_df, diversity_df)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    exit(main())
