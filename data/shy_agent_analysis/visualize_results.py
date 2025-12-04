#!/usr/bin/env python3
"""
Generate comprehensive visualizations for shy agent analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data(run_id="20251110_214435"):
    """Load all analysis datasets."""
    data_dir = Path("data/shy_agent_analysis")

    questions = pd.read_parquet(data_dir / f"questions_{run_id}.parquet")
    conversations = pd.read_parquet(data_dir / f"conversations_{run_id}.parquet")
    proposals = pd.read_parquet(data_dir / f"proposals_{run_id}.parquet")

    return questions, conversations, proposals


def plot_questions_by_round(conversations, output_dir):
    """Plot question-asking behavior over training rounds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall questions by agent type
    ax = axes[0, 0]
    by_round = conversations.groupby('round').agg({
        'shy_questions_count': 'mean',
        'nonshy_questions_count': 'mean',
        'total_questions': 'mean'
    })

    ax.plot(by_round.index, by_round['shy_questions_count'], 'o-', label='Shy Agent', linewidth=2, markersize=8)
    ax.plot(by_round.index, by_round['nonshy_questions_count'], 's-', label='Non-shy Agent', linewidth=2, markersize=8)
    ax.plot(by_round.index, by_round['total_questions'], '^--', label='Total', linewidth=2, markersize=8, alpha=0.5)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Mean Questions per Conversation', fontsize=12)
    ax.set_title('Question-Asking Behavior Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Non-shy questions by player ID
    ax = axes[0, 1]
    p1_data = conversations[conversations['nonshy_player'] == 'player-1'].groupby('round')['nonshy_questions_count'].mean()
    p2_data = conversations[conversations['nonshy_player'] == 'player-2'].groupby('round')['nonshy_questions_count'].mean()

    ax.plot(p1_data.index, p1_data.values, 'o-', label='Non-shy as P1', linewidth=2, markersize=8)
    ax.plot(p2_data.index, p2_data.values, 's-', label='Non-shy as P2', linewidth=2, markersize=8)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Mean Questions per Conversation', fontsize=12)
    ax.set_title('Non-shy Agent: P1 vs P2', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. Shy questions by player ID
    ax = axes[1, 0]
    shy_p1_data = conversations[conversations['shy_player'] == 'player-1'].groupby('round')['shy_questions_count'].mean()
    shy_p2_data = conversations[conversations['shy_player'] == 'player-2'].groupby('round')['shy_questions_count'].mean()

    ax.plot(shy_p1_data.index, shy_p1_data.values, 'o-', label='Shy as P1', linewidth=2, markersize=8)
    ax.plot(shy_p2_data.index, shy_p2_data.values, 's-', label='Shy as P2', linewidth=2, markersize=8)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Mean Questions per Conversation', fontsize=12)
    ax.set_title('Shy Agent: P1 vs P2', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 4. Percentage change from Round 0
    ax = axes[1, 1]
    baseline_shy = by_round['shy_questions_count'].iloc[0]
    baseline_nonshy = by_round['nonshy_questions_count'].iloc[0]

    pct_change_shy = ((by_round['shy_questions_count'] / baseline_shy) - 1) * 100
    pct_change_nonshy = ((by_round['nonshy_questions_count'] / baseline_nonshy) - 1) * 100

    ax.plot(pct_change_shy.index, pct_change_shy.values, 'o-', label='Shy Agent', linewidth=2, markersize=8)
    ax.plot(pct_change_nonshy.index, pct_change_nonshy.values, 's-', label='Non-shy Agent', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('% Change from Round 0', fontsize=12)
    ax.set_title('Relative Change in Question-Asking', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'questions_by_round.png', dpi=300, bbox_inches='tight')
    print(f"Saved: questions_by_round.png")
    plt.close()


def plot_proposals_rejections_by_round(proposals_df, output_dir):
    """Plot proposal and rejection behavior over training rounds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    by_round = proposals_df.groupby('round').agg({
        'shy_proposals': 'mean',
        'nonshy_proposals': 'mean',
        'shy_rejects': 'mean',
        'nonshy_rejects': 'mean',
    })

    # 1. Proposals
    ax = axes[0, 0]
    ax.plot(by_round.index, by_round['shy_proposals'], 'o-', label='Shy Agent', linewidth=2, markersize=8)
    ax.plot(by_round.index, by_round['nonshy_proposals'], 's-', label='Non-shy Agent', linewidth=2, markersize=8)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Mean Proposals per Conversation', fontsize=12)
    ax.set_title('Proposal Behavior Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Rejections
    ax = axes[0, 1]
    ax.plot(by_round.index, by_round['shy_rejects'], 'o-', label='Shy Agent', linewidth=2, markersize=8)
    ax.plot(by_round.index, by_round['nonshy_rejects'], 's-', label='Non-shy Agent', linewidth=2, markersize=8)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Mean Rejections per Conversation', fontsize=12)
    ax.set_title('Rejection Behavior Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 3. Total proposals and rejections
    ax = axes[1, 0]
    total_proposals = by_round['shy_proposals'] + by_round['nonshy_proposals']
    total_rejections = by_round['shy_rejects'] + by_round['nonshy_rejects']

    ax.plot(by_round.index, total_proposals, 'o-', label='Total Proposals', linewidth=2, markersize=8)
    ax.plot(by_round.index, total_rejections, 's-', label='Total Rejections', linewidth=2, markersize=8)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Mean per Conversation', fontsize=12)
    ax.set_title('Total Proposals and Rejections', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 4. Rejection rate (rejections / proposals)
    ax = axes[1, 1]
    shy_rej_rate = by_round['shy_rejects'] / by_round['shy_proposals']
    nonshy_rej_rate = by_round['nonshy_rejects'] / by_round['nonshy_proposals']

    ax.plot(by_round.index, shy_rej_rate * 100, 'o-', label='Shy Agent', linewidth=2, markersize=8)
    ax.plot(by_round.index, nonshy_rej_rate * 100, 's-', label='Non-shy Agent', linewidth=2, markersize=8)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Rejection Rate (%)', fontsize=12)
    ax.set_title('Rejection Rate (Rejections/Proposals)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'proposals_rejections_by_round.png', dpi=300, bbox_inches='tight')
    print(f"Saved: proposals_rejections_by_round.png")
    plt.close()


def plot_conversation_length_distribution(conversations, output_dir):
    """Plot conversation length distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram for Round 0
    ax = axes[0, 0]
    round0 = conversations[conversations['round'] == 0]['conversation_length']
    ax.hist(round0, bins=range(1, 17), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Conversation Length (turns)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Round 0 Conversation Length Distribution', fontsize=14, fontweight='bold')
    ax.axvline(round0.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean = {round0.mean():.2f}')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Histogram for Round 7
    ax = axes[0, 1]
    round7 = conversations[conversations['round'] == 7]['conversation_length']
    ax.hist(round7, bins=range(1, 17), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Conversation Length (turns)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Round 7 Conversation Length Distribution', fontsize=14, fontweight='bold')
    ax.axvline(round7.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean = {round7.mean():.2f}')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Mean length over rounds
    ax = axes[1, 0]
    by_round = conversations.groupby('round')['conversation_length'].mean()
    ax.plot(by_round.index, by_round.values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Mean Conversation Length', fontsize=12)
    ax.set_title('Mean Conversation Length Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Length distribution heatmap
    ax = axes[1, 1]
    length_counts = []
    for round_num in range(8):
        round_data = conversations[conversations['round'] == round_num]['conversation_length']
        counts = round_data.value_counts().reindex(range(1, 17), fill_value=0)
        length_counts.append(counts.values / len(round_data) * 100)  # Convert to percentage

    im = ax.imshow(np.array(length_counts).T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Conversation Length', fontsize=12)
    ax.set_title('Length Distribution Heatmap (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_yticks(range(0, 16, 2))
    ax.set_yticklabels(range(1, 17, 2))
    plt.colorbar(im, ax=ax, label='Percentage')

    plt.tight_layout()
    plt.savefig(output_dir / 'conversation_length_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: conversation_length_distribution.png")
    plt.close()


def plot_correlations_with_length(conversations, proposals_df, output_dir):
    """Plot correlations between conversation length and various metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Merge data - keep conversation_length from conversations
    merged = conversations.merge(proposals_df, on=['run_id', 'round', 'game_id'], suffixes=('', '_prop'))

    # Sample for plotting (too many points otherwise)
    sample = merged.sample(min(5000, len(merged)))

    # 1. Questions vs Length
    ax = axes[0, 0]
    ax.scatter(sample['conversation_length'], sample['total_questions'], alpha=0.3, s=10)
    ax.set_xlabel('Conversation Length', fontsize=12)
    ax.set_ylabel('Total Questions', fontsize=12)
    corr = merged['conversation_length'].corr(merged['total_questions'])
    ax.set_title(f'Questions vs Length (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. Shy Questions vs Length
    ax = axes[0, 1]
    ax.scatter(sample['conversation_length'], sample['shy_questions_count'], alpha=0.3, s=10, color='orange')
    ax.set_xlabel('Conversation Length', fontsize=12)
    ax.set_ylabel('Shy Questions', fontsize=12)
    corr = merged['conversation_length'].corr(merged['shy_questions_count'])
    ax.set_title(f'Shy Questions vs Length (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Non-shy Questions vs Length
    ax = axes[0, 2]
    ax.scatter(sample['conversation_length'], sample['nonshy_questions_count'], alpha=0.3, s=10, color='green')
    ax.set_xlabel('Conversation Length', fontsize=12)
    ax.set_ylabel('Non-shy Questions', fontsize=12)
    corr = merged['conversation_length'].corr(merged['nonshy_questions_count'])
    ax.set_title(f'Non-shy Questions vs Length (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Proposals vs Length
    ax = axes[1, 0]
    total_proposals = sample['shy_proposals'] + sample['nonshy_proposals']
    ax.scatter(sample['conversation_length'], total_proposals, alpha=0.3, s=10, color='purple')
    ax.set_xlabel('Conversation Length', fontsize=12)
    ax.set_ylabel('Total Proposals', fontsize=12)
    corr = merged['conversation_length'].corr(merged['shy_proposals'] + merged['nonshy_proposals'])
    ax.set_title(f'Proposals vs Length (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Rejections vs Length
    ax = axes[1, 1]
    total_rejections = sample['shy_rejects'] + sample['nonshy_rejects']
    ax.scatter(sample['conversation_length'], total_rejections, alpha=0.3, s=10, color='red')
    ax.set_xlabel('Conversation Length', fontsize=12)
    ax.set_ylabel('Total Rejections', fontsize=12)
    corr = merged['conversation_length'].corr(merged['shy_rejects'] + merged['nonshy_rejects'])
    ax.set_title(f'Rejections vs Length (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6. Shy Rejections vs Length
    ax = axes[1, 2]
    ax.scatter(sample['conversation_length'], sample['shy_rejects'], alpha=0.3, s=10, color='crimson')
    ax.set_xlabel('Conversation Length', fontsize=12)
    ax.set_ylabel('Shy Rejections', fontsize=12)
    corr = merged['conversation_length'].corr(merged['shy_rejects'])
    ax.set_title(f'Shy Rejections vs Length (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlations_with_length.png', dpi=300, bbox_inches='tight')
    print(f"Saved: correlations_with_length.png")
    plt.close()


def plot_summary_metrics_by_length(conversations, proposals_df, output_dir):
    """Plot summary metrics binned by conversation length."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Merge data
    merged = conversations.merge(proposals_df, on=['run_id', 'round', 'game_id'], suffixes=('', '_prop'))

    # Bin by length
    merged['length_bin'] = pd.cut(
        merged['conversation_length'],
        bins=[0, 4, 6, 8, 10, 20],
        labels=['1-4', '5-6', '7-8', '9-10', '11+']
    )

    by_length = merged.groupby('length_bin', observed=True).agg({
        'shy_questions_count': 'mean',
        'nonshy_questions_count': 'mean',
        'shy_proposals': 'mean',
        'nonshy_proposals': 'mean',
        'shy_rejects': 'mean',
        'nonshy_rejects': 'mean',
    })

    # 1. Questions by length
    ax = axes[0, 0]
    x = range(len(by_length))
    width = 0.35
    ax.bar([i - width/2 for i in x], by_length['shy_questions_count'], width, label='Shy', alpha=0.8)
    ax.bar([i + width/2 for i in x], by_length['nonshy_questions_count'], width, label='Non-shy', alpha=0.8)
    ax.set_xlabel('Conversation Length Bin', fontsize=12)
    ax.set_ylabel('Mean Questions', fontsize=12)
    ax.set_title('Questions by Conversation Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(by_length.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Proposals by length
    ax = axes[0, 1]
    ax.bar([i - width/2 for i in x], by_length['shy_proposals'], width, label='Shy', alpha=0.8)
    ax.bar([i + width/2 for i in x], by_length['nonshy_proposals'], width, label='Non-shy', alpha=0.8)
    ax.set_xlabel('Conversation Length Bin', fontsize=12)
    ax.set_ylabel('Mean Proposals', fontsize=12)
    ax.set_title('Proposals by Conversation Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(by_length.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Rejections by length
    ax = axes[1, 0]
    ax.bar([i - width/2 for i in x], by_length['shy_rejects'], width, label='Shy', alpha=0.8)
    ax.bar([i + width/2 for i in x], by_length['nonshy_rejects'], width, label='Non-shy', alpha=0.8)
    ax.set_xlabel('Conversation Length Bin', fontsize=12)
    ax.set_ylabel('Mean Rejections', fontsize=12)
    ax.set_title('Rejections by Conversation Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(by_length.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Combined view
    ax = axes[1, 1]
    total_questions = by_length['shy_questions_count'] + by_length['nonshy_questions_count']
    total_proposals = by_length['shy_proposals'] + by_length['nonshy_proposals']
    total_rejections = by_length['shy_rejects'] + by_length['nonshy_rejects']

    ax.plot(x, total_questions, 'o-', label='Questions', linewidth=2, markersize=8)
    ax.plot(x, total_proposals, 's-', label='Proposals', linewidth=2, markersize=8)
    ax.plot(x, total_rejections, '^-', label='Rejections', linewidth=2, markersize=8)
    ax.set_xlabel('Conversation Length Bin', fontsize=12)
    ax.set_ylabel('Mean Count', fontsize=12)
    ax.set_title('All Metrics by Conversation Length', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(by_length.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_length.png', dpi=300, bbox_inches='tight')
    print(f"Saved: metrics_by_length.png")
    plt.close()


def main():
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    output_dir = Path("data/shy_agent_analysis")

    print("\nLoading data...")
    questions, conversations, proposals = load_data()

    print("\n1. Plotting questions by round...")
    plot_questions_by_round(conversations, output_dir)

    print("\n2. Plotting proposals and rejections by round...")
    plot_proposals_rejections_by_round(proposals, output_dir)

    print("\n3. Plotting conversation length distributions...")
    plot_conversation_length_distribution(conversations, output_dir)

    print("\n4. Plotting correlations with conversation length...")
    plot_correlations_with_length(conversations, proposals, output_dir)

    print("\n5. Plotting summary metrics by length...")
    plot_summary_metrics_by_length(conversations, proposals, output_dir)

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nSaved to: {output_dir}/")


if __name__ == "__main__":
    main()
