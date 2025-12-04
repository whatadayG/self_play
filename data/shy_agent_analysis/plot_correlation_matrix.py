#!/usr/bin/env python3
"""
Generate a correlation matrix heatmap showing relationships between all key metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def main():
    print("=" * 80)
    print("GENERATING CORRELATION MATRIX")
    print("=" * 80)

    # Load data
    data_dir = Path("data/shy_agent_analysis")
    conversations = pd.read_parquet(data_dir / "conversations_20251110_214435.parquet")
    proposals = pd.read_parquet(data_dir / "proposals_20251110_214435.parquet")

    # Merge datasets
    merged = conversations.merge(proposals, on=['run_id', 'round', 'game_id'], suffixes=('', '_prop'))

    # Select key metrics for correlation analysis
    metrics = {
        'Round': 'round',
        'Conv Length': 'conversation_length',
        'Game Reward': 'game_reward',
        'Total Questions': 'total_questions',
        'Shy Questions': 'shy_questions_count',
        'NonShy Questions': 'nonshy_questions_count',
        'Total Proposals': None,  # Will compute
        'Shy Proposals': 'shy_proposals',
        'NonShy Proposals': 'nonshy_proposals',
        'Total Rejections': None,  # Will compute
        'Shy Rejections': 'shy_rejects',
        'NonShy Rejections': 'nonshy_rejects',
    }

    # Compute totals
    merged['Total Proposals'] = merged['shy_proposals'] + merged['nonshy_proposals']
    merged['Total Rejections'] = merged['shy_rejects'] + merged['nonshy_rejects']

    # Build dataframe with selected metrics
    data_for_corr = pd.DataFrame()
    for display_name, col_name in metrics.items():
        if col_name is None:
            col_name = display_name  # Already computed above
        data_for_corr[display_name] = merged[col_name]

    # Compute correlation matrix
    corr_matrix = data_for_corr.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Mask upper triangle
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax
    )

    ax.set_title('Correlation Matrix: Key Metrics Across All Rounds',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = data_dir / 'correlation_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    # Print some key insights
    print("\n" + "=" * 80)
    print("KEY CORRELATIONS")
    print("=" * 80)

    # Find strongest correlations (excluding diagonal)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'var1': corr_matrix.columns[i],
                'var2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

    corr_pairs_df = pd.DataFrame(corr_pairs)
    corr_pairs_df['abs_corr'] = corr_pairs_df['correlation'].abs()
    corr_pairs_df = corr_pairs_df.sort_values('abs_corr', ascending=False)

    print("\nStrongest positive correlations:")
    positive = corr_pairs_df[corr_pairs_df['correlation'] > 0].head(10)
    for _, row in positive.iterrows():
        print(f"  {row['var1']:20s} ↔ {row['var2']:20s}: {row['correlation']:+.3f}")

    print("\nStrongest negative correlations:")
    negative = corr_pairs_df[corr_pairs_df['correlation'] < 0].head(10)
    for _, row in negative.iterrows():
        print(f"  {row['var1']:20s} ↔ {row['var2']:20s}: {row['correlation']:+.3f}")

    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
