#!/usr/bin/env python3
"""Analyze question-asking patterns from training logs.

This script analyzes the question metrics logged during training to
validate the hypothesis that Qwen learns to ask more questions as
training progresses.

Usage:
    python analyze_questions.py --log_dir logs/
"""
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_question_logs(log_dir: str) -> list[dict]:
    """Load all question log files from a directory.

    Args:
        log_dir: Path to log directory

    Returns:
        List of log entries sorted by step
    """
    log_path = Path(log_dir)
    all_episodes = []

    for log_file in sorted(log_path.glob("questions_step*.json")):
        with open(log_file) as f:
            data = json.load(f)
            step = data.get("step", 0)
            for episode in data.get("episodes", []):
                episode["step"] = step
                all_episodes.append(episode)

    return sorted(all_episodes, key=lambda x: x.get("step", 0))


def compute_rolling_average(values: list, window: int = 50) -> list:
    """Compute rolling average.

    Args:
        values: List of values
        window: Window size

    Returns:
        List of rolling averages
    """
    if not HAS_NUMPY:
        # Simple fallback
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start:i+1]) / (i - start + 1))
        return result

    import numpy as np
    values = np.array(values)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode='valid').tolist()


def analyze_questions(log_dir: str, output_dir: Optional[str] = None):
    """Analyze question patterns and generate report.

    Args:
        log_dir: Directory containing question logs
        output_dir: Directory for output plots (optional)
    """
    episodes = load_question_logs(log_dir)

    if not episodes:
        print(f"No question logs found in {log_dir}")
        return

    print(f"\n{'='*60}")
    print("Question-Asking Analysis Report")
    print(f"{'='*60}")
    print(f"\nTotal episodes analyzed: {len(episodes)}")

    # Extract metrics
    question_counts = [e.get("question_count", 0) for e in episodes]
    rewards = [e.get("reward", 0) for e in episodes]
    turn_counts = [e.get("turn_count", 0) for e in episodes]

    # Overall statistics
    print(f"\n--- Overall Statistics ---")
    print(f"Average questions per episode: {sum(question_counts)/len(question_counts):.2f}")
    print(f"Average reward: {sum(rewards)/len(rewards):.3f}")
    print(f"Average turns: {sum(turn_counts)/len(turn_counts):.1f}")
    print(f"Max questions in single episode: {max(question_counts)}")
    print(f"Min questions in single episode: {min(question_counts)}")

    # Early vs late comparison
    n_early = min(100, len(episodes) // 4)
    n_late = min(100, len(episodes) // 4)

    if n_early > 0 and n_late > 0:
        early_episodes = episodes[:n_early]
        late_episodes = episodes[-n_late:]

        early_questions = sum(e.get("question_count", 0) for e in early_episodes) / n_early
        late_questions = sum(e.get("question_count", 0) for e in late_episodes) / n_late
        early_reward = sum(e.get("reward", 0) for e in early_episodes) / n_early
        late_reward = sum(e.get("reward", 0) for e in late_episodes) / n_late

        print(f"\n--- Early vs Late Training ---")
        print(f"Early (first {n_early} episodes):")
        print(f"  Avg questions: {early_questions:.2f}")
        print(f"  Avg reward: {early_reward:.3f}")
        print(f"Late (last {n_late} episodes):")
        print(f"  Avg questions: {late_questions:.2f}")
        print(f"  Avg reward: {late_reward:.3f}")

        if early_questions > 0:
            change_pct = (late_questions - early_questions) / early_questions * 100
            print(f"\nQuestion change: {change_pct:+.1f}%")

        if early_reward > 0:
            reward_change_pct = (late_reward - early_reward) / early_reward * 100
            print(f"Reward change: {reward_change_pct:+.1f}%")

    # Correlation analysis
    if HAS_NUMPY and len(episodes) > 10:
        import numpy as np
        corr = np.corrcoef(question_counts, rewards)[0, 1]
        print(f"\n--- Correlation Analysis ---")
        print(f"Question-Reward correlation: {corr:.3f}")

        if corr > 0.3:
            print("  -> Positive correlation: More questions associated with higher rewards")
        elif corr < -0.3:
            print("  -> Negative correlation: More questions associated with lower rewards")
        else:
            print("  -> Weak correlation: Questions and rewards not strongly related")

    # Generate plots if matplotlib available
    if HAS_MATPLOTLIB and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Plot 1: Questions over training
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Rolling average of questions
        ax1 = axes[0, 0]
        if len(question_counts) > 50:
            rolling_questions = compute_rolling_average(question_counts, 50)
            ax1.plot(range(50, len(question_counts) + 1), rolling_questions)
        else:
            ax1.plot(question_counts)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Questions per Episode")
        ax1.set_title("Questions Asked Over Training (Rolling Avg)")
        ax1.grid(True, alpha=0.3)

        # Rolling average of rewards
        ax2 = axes[0, 1]
        if len(rewards) > 50:
            rolling_rewards = compute_rolling_average(rewards, 50)
            ax2.plot(range(50, len(rewards) + 1), rolling_rewards)
        else:
            ax2.plot(rewards)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Reward")
        ax2.set_title("Reward Over Training (Rolling Avg)")
        ax2.grid(True, alpha=0.3)

        # Scatter: Questions vs Reward
        ax3 = axes[1, 0]
        ax3.scatter(question_counts, rewards, alpha=0.3, s=10)
        ax3.set_xlabel("Questions Asked")
        ax3.set_ylabel("Reward")
        ax3.set_title("Questions vs Reward")
        ax3.grid(True, alpha=0.3)

        # Histogram of question counts
        ax4 = axes[1, 1]
        ax4.hist(question_counts, bins=range(max(question_counts) + 2), edgecolor='black')
        ax4.set_xlabel("Questions per Episode")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Distribution of Questions per Episode")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_path / "question_analysis.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")
        plt.close()

    print(f"\n{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze question-asking patterns from training"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing question log files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (optional)"
    )

    args = parser.parse_args()
    analyze_questions(args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
