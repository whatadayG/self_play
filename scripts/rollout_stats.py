#!/usr/bin/env python3
"""Data structure for rollout statistics and metrics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RolloutStats:
    """Comprehensive statistics from a rollout generation round.

    This captures all metrics needed for logging, analysis, and debugging.
    """
    # Output files
    raw_parquet: Path
    trimmed_parquet: Path
    examples_txt: Path

    # Game performance metrics (ACTUAL game rewards, not GRPO-normalized)
    game_reward_mean: float
    game_reward_std: float
    game_reward_p10: float
    game_reward_p25: float
    game_reward_p50: float
    game_reward_p75: float
    game_reward_p90: float

    # Perfect score tracking (all as ratios for consistency)
    perfect_score_ratio: float  # Before trimming
    perfect_score_ratio_after_trim: float

    # Sequence filtering metrics
    total_sequences: int  # Before trimming
    kept_sequences: int  # After trimming
    trim_threshold: int  # p95 length cutoff

    # GRPO-normalized weights (for debugging, should have mean ~0)
    grpo_weight_mean: float
    grpo_weight_pos_ratio: float
    grpo_weight_neg_ratio: float
    grpo_weight_zero_ratio: float

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "raw_parquet": str(self.raw_parquet),
            "trimmed_parquet": str(self.trimmed_parquet),
            "examples_txt": str(self.examples_txt),
            "pct95": self.trim_threshold,
            "kept": self.kept_sequences,
            "total": self.total_sequences,
            "mean_norm_reward": self.grpo_weight_mean,
            "stats_mean": self.game_reward_mean,
            "game_reward_mean": self.game_reward_mean,
            "game_reward_std": self.game_reward_std,
            "game_reward_p50": self.game_reward_p50,
            "perfect_score_ratio": self.perfect_score_ratio,
            "perfect_score_ratio_after_trim": self.perfect_score_ratio_after_trim,
        }
