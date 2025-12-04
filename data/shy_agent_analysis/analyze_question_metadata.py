#!/usr/bin/env python3
"""
Analyze question metadata extracted by extract_question_metadata.py

Demonstrates various queries and analyses possible with the structured datasets.
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def load_datasets(run_id: str, data_dir: Path = Path("data/shy_agent_analysis")):
    """Load questions and conversations datasets for a run."""
    questions_path = data_dir / f"questions_{run_id}.parquet"
    conversations_path = data_dir / f"conversations_{run_id}.parquet"

    if not questions_path.exists() or not conversations_path.exists():
        raise FileNotFoundError(f"Datasets not found for run {run_id} in {data_dir}")

    questions = pd.read_parquet(questions_path)
    conversations = pd.read_parquet(conversations_path)

    return questions, conversations


def analyze_questions_by_round(questions: pd.DataFrame, conversations: pd.DataFrame):
    """Analyze how question-asking behavior changes across rounds."""
    print("\n" + "=" * 80)
    print("QUESTIONS BY ROUND")
    print("=" * 80)

    # Questions per conversation by round
    by_round = conversations.groupby('round').agg({
        'total_questions': 'mean',
        'shy_questions_count': 'mean',
        'nonshy_questions_count': 'mean',
        'game_id': 'count'  # number of conversations
    }).round(3)
    by_round.columns = ['Mean Total Q', 'Mean Shy Q', 'Mean Non-shy Q', 'N Conversations']

    print("\n", by_round)

    return by_round


def analyze_questions_by_turn(questions: pd.DataFrame):
    """Analyze which turns questions are asked on."""
    print("\n" + "=" * 80)
    print("QUESTIONS BY TURN")
    print("=" * 80)

    # Distribution of questions across turns
    turn_dist = questions.groupby(['turn', 'is_shy']).size().unstack(fill_value=0)
    turn_dist.columns = ['Non-shy', 'Shy']

    print("\nQuestions by turn number:")
    print(turn_dist.head(10))

    # Average turn when questions are asked
    print(f"\nMean turn for shy agent questions: {questions[questions['is_shy']]['turn'].mean():.2f}")
    print(f"Mean turn for non-shy agent questions: {questions[~questions['is_shy']]['turn'].mean():.2f}")

    return turn_dist


def analyze_correlations(conversations: pd.DataFrame):
    """Analyze correlations between conversation metrics."""
    print("\n" + "=" * 80)
    print("CORRELATIONS")
    print("=" * 80)

    metrics = ['conversation_length', 'total_questions', 'shy_questions_count',
               'nonshy_questions_count', 'game_reward', 'grpo_weight']

    corr_matrix = conversations[metrics].corr()

    print("\nCorrelation matrix:")
    print(corr_matrix.round(3))

    # Specific interesting correlations
    print("\nKey correlations:")
    print(f"  Conversation length vs Total questions: {conversations['conversation_length'].corr(conversations['total_questions']):.3f}")
    print(f"  Game reward vs Non-shy questions: {conversations['game_reward'].corr(conversations['nonshy_questions_count']):.3f}")
    print(f"  Game reward vs Shy questions: {conversations['game_reward'].corr(conversations['shy_questions_count']):.3f}")

    return corr_matrix


def analyze_by_player(questions: pd.DataFrame):
    """Analyze question patterns by player ID (player-1 vs player-2)."""
    print("\n" + "=" * 80)
    print("QUESTIONS BY PLAYER ID")
    print("=" * 80)

    player_stats = questions.groupby(['player', 'is_shy']).agg({
        'question_text': 'count',
        'turn': 'mean'
    }).round(2)
    player_stats.columns = ['Count', 'Mean Turn']

    print("\n", player_stats)


def analyze_question_timing(conversations: pd.DataFrame):
    """Analyze when in the conversation questions are asked."""
    print("\n" + "=" * 80)
    print("QUESTION TIMING ANALYSIS")
    print("=" * 80)

    # Parse the turn lists
    conversations = conversations.copy()
    conversations['shy_turns'] = conversations['shy_questions_by_turn'].apply(
        lambda x: json.loads(x) if x else []
    )
    conversations['nonshy_turns'] = conversations['nonshy_questions_by_turn'].apply(
        lambda x: json.loads(x) if x else []
    )

    # Early vs late questions (first half vs second half of conversation)
    def classify_timing(row):
        shy_turns = row['shy_turns']
        nonshy_turns = row['nonshy_turns']
        length = row['conversation_length']

        if length == 0:
            return None

        midpoint = length / 2

        return {
            'shy_early': sum(1 for t in shy_turns if t < midpoint),
            'shy_late': sum(1 for t in shy_turns if t >= midpoint),
            'nonshy_early': sum(1 for t in nonshy_turns if t < midpoint),
            'nonshy_late': sum(1 for t in nonshy_turns if t >= midpoint),
        }

    timing_data = conversations.apply(classify_timing, axis=1)
    timing_df = pd.DataFrame(timing_data.tolist())

    print("\nMean questions by timing (early vs late in conversation):")
    print(timing_df.mean().round(3))


def analyze_progression_over_training(conversations: pd.DataFrame):
    """Analyze how behavior changes over the course of training."""
    print("\n" + "=" * 80)
    print("PROGRESSION OVER TRAINING")
    print("=" * 80)

    # Group into early, mid, late training
    max_round = conversations['round'].max()
    conversations = conversations.copy()

    def training_phase(round_num):
        if round_num < max_round / 3:
            return 'Early'
        elif round_num < 2 * max_round / 3:
            return 'Mid'
        else:
            return 'Late'

    conversations['phase'] = conversations['round'].apply(training_phase)

    phase_stats = conversations.groupby('phase').agg({
        'total_questions': 'mean',
        'nonshy_questions_count': 'mean',
        'shy_questions_count': 'mean',
        'conversation_length': 'mean',
        'game_reward': 'mean',
    }).round(3)

    print("\nMetrics by training phase:")
    print(phase_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze question metadata"
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run identifier (e.g., 20251110_214435)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/shy_agent_analysis"),
        help="Directory containing the datasets"
    )

    args = parser.parse_args()

    print(f"Loading datasets for run: {args.run_id}")
    questions, conversations = load_datasets(args.run_id, args.data_dir)

    print(f"\nLoaded:")
    print(f"  {len(questions)} questions")
    print(f"  {len(conversations)} conversations")

    # Run analyses
    analyze_questions_by_round(questions, conversations)
    analyze_questions_by_turn(questions)
    analyze_by_player(questions)
    analyze_correlations(conversations)
    analyze_question_timing(conversations)
    analyze_progression_over_training(conversations)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    exit(main())
