"""SLIME RL training for paper-reviewer matching game."""

from .env_matching import MatchingEnv, build_env
from .gpt_partner import ShyGPTPartner, MockShyPartner
from .question_logger import QuestionLogger, count_questions, EpisodeMetrics
from .data_generator import generate_game, generate_dataset
from .rollout import generate, RolloutSample

__all__ = [
    # Environment
    "MatchingEnv",
    "build_env",
    # GPT Partner
    "ShyGPTPartner",
    "MockShyPartner",
    # Question Tracking
    "QuestionLogger",
    "count_questions",
    "EpisodeMetrics",
    # Data Generation
    "generate_game",
    "generate_dataset",
    # Rollout (for SLIME integration)
    "generate",
    "RolloutSample",
]
