"""Track questions per dialogue to validate hypothesis.

This module tracks how many questions Qwen asks during each dialogue,
allowing us to test the hypothesis that Qwen will learn to ask more
questions as training progresses.
"""
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import json

# Patterns that indicate a question or information-seeking behavior
QUESTION_PATTERNS = [
    r'\?',  # Contains question mark
    r'(?:^|\s)(what|who|where|when|why|how|which|can|could|would|is|are|do|does|tell me)\b',
    r'score.*(for|of|between)',
    r'(affinity|rating|preference|expertise).*(for|of|between)',
    r'(what|how).*(think|feel|rate|score)',
    r'could you (tell|share|give)',
]


def count_questions(text: str) -> int:
    """Count question-like patterns in text.

    Args:
        text: The text to analyze

    Returns:
        Number of questions detected (0 or 1 per call)
    """
    if not text:
        return 0
    text_lower = text.lower().strip()
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text_lower):
            return 1
    return 0


def classify_question_type(text: str) -> Optional[str]:
    """Classify the type of question asked.

    Args:
        text: The text to analyze

    Returns:
        Question type or None if not a question
    """
    if not text:
        return None
    text_lower = text.lower().strip()

    # Specific cell query (asking about a specific reviewer-paper pair)
    if re.search(r'(score|rating|affinity).*(for|of|between).*\w+.*\w+', text_lower):
        return "specific_cell"

    # General query about a reviewer or paper
    if re.search(r'(what|which).*(reviewer|paper|assignment)', text_lower):
        return "general_query"

    # Confirmation question
    if re.search(r'(is that|does that|would that|can you confirm)', text_lower):
        return "confirmation"

    # Any other question
    if '?' in text:
        return "other"

    return None


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode/game."""
    game_id: int
    question_count: int
    turn_count: int
    reward: float
    qwen_turns: int = 0
    question_types: dict = field(default_factory=dict)

    @property
    def questions_per_turn(self) -> float:
        """Average questions per Qwen turn."""
        return self.question_count / max(1, self.qwen_turns)

    @property
    def questions_per_total_turn(self) -> float:
        """Average questions per total turn."""
        return self.question_count / max(1, self.turn_count)


class QuestionLogger:
    """Log and analyze question-asking patterns across training.

    This logger tracks question metrics over training to help validate
    the hypothesis that Qwen learns to ask more questions as training
    progresses.
    """

    def __init__(self, log_dir: str):
        """Initialize the logger.

        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes: list[dict] = []
        self.current_step = 0

    def log_episode(self, metrics: EpisodeMetrics):
        """Log metrics for a completed episode.

        Args:
            metrics: Episode metrics to log
        """
        self.episodes.append(asdict(metrics))

    def log_from_sample(self, sample_info: dict):
        """Log metrics from a sample's extra_info dict.

        Args:
            sample_info: Dictionary with question_count, turn_count, etc.
        """
        metrics = EpisodeMetrics(
            game_id=sample_info.get("game_id", -1),
            question_count=sample_info.get("question_count", 0),
            turn_count=sample_info.get("turn_count", 0),
            reward=sample_info.get("reward", 0.0),
            qwen_turns=sample_info.get("qwen_turns", sample_info.get("turn_count", 0) // 2),
        )
        self.log_episode(metrics)

    def save(self, step: int):
        """Save logged episodes to disk.

        Args:
            step: Training step number
        """
        self.current_step = step
        path = self.log_dir / f"questions_step{step}.json"
        with open(path, "w") as f:
            json.dump({
                "step": step,
                "episodes": self.episodes,
                "summary": self.get_summary(),
            }, f, indent=2)

    def get_summary(self, window: int = 100) -> dict:
        """Get summary statistics over recent episodes.

        Args:
            window: Number of recent episodes to summarize

        Returns:
            Dictionary with summary statistics
        """
        if not self.episodes:
            return {}

        recent = self.episodes[-window:] if len(self.episodes) >= window else self.episodes

        question_counts = [e["question_count"] for e in recent]
        rewards = [e["reward"] for e in recent]
        turn_counts = [e["turn_count"] for e in recent]

        # Compute correlation between questions and reward
        corr = 0.0
        if len(recent) > 1:
            try:
                import numpy as np
                if np.std(question_counts) > 0 and np.std(rewards) > 0:
                    corr = float(np.corrcoef(question_counts, rewards)[0, 1])
            except ImportError:
                pass

        return {
            "num_episodes": len(recent),
            "avg_questions": sum(question_counts) / len(recent),
            "avg_reward": sum(rewards) / len(recent),
            "avg_turns": sum(turn_counts) / len(recent),
            "max_questions": max(question_counts),
            "min_questions": min(question_counts),
            "question_reward_correlation": corr,
        }

    def get_learning_curve_data(self, window: int = 50) -> list[dict]:
        """Get data for plotting learning curves.

        Args:
            window: Rolling window size for averaging

        Returns:
            List of dicts with step, avg_questions, avg_reward
        """
        if len(self.episodes) < window:
            return []

        curve_data = []
        for i in range(window, len(self.episodes) + 1, window // 2):
            window_episodes = self.episodes[i - window:i]
            curve_data.append({
                "episode": i,
                "avg_questions": sum(e["question_count"] for e in window_episodes) / window,
                "avg_reward": sum(e["reward"] for e in window_episodes) / window,
            })
        return curve_data
