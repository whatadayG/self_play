"""Environment adapter for paper-reviewer matching game.

This module wraps DialOp's OptimizationEnv to match SLIME's expected
environment interface (similar to examples/geo3k_vlm_multi_turn/env_geo3k.py).

SLIME environment interface:
- reset(game_state) -> (obs_dict, info_dict)
- step(response_text) -> (obs_dict, done, info_dict)
"""
import sys
from pathlib import Path
from typing import Optional

# Add self_play/scripts to path for dialop imports
SCRIPTS_PATH = Path(__file__).parent.parent.parent / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import OptimizationGame


class MatchingEnv:
    """SLIME-compatible wrapper for DialOp OptimizationEnv.

    This adapter provides a clean interface for the multi-turn rollout:
    - Tracks whose turn it is (Qwen vs GPT)
    - Provides appropriate observations for each player
    - Returns normalized reward on game completion
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_retries_per_turn: int = 3,
        error_penalty: float = 0.0,
    ):
        """Initialize the environment.

        Args:
            max_turns: Maximum dialogue turns before forced termination
            max_retries_per_turn: Max retries for malformed responses
            error_penalty: Penalty per error (subtracted from reward)
        """
        self.max_turns = max_turns
        self.max_retries_per_turn = max_retries_per_turn
        self.error_penalty = error_penalty

        self.env: Optional[OptimizationEnv] = None
        self.turn_count = 0
        self.current_player = 0  # 0 = Qwen (agent/player-1), 1 = GPT (partner/player-2)
        self.done = False
        self.last_reward = 0.0

    def reset(self, game_state: dict) -> tuple[dict, dict]:
        """Reset environment with a game state.

        Args:
            game_state: Dictionary with table, masks, scales, etc.

        Returns:
            obs: dict with "obs_str" for initial observation to Qwen
            info: dict with metadata (best_score, game_state)
        """
        self.env = OptimizationEnv(
            max_turns=self.max_turns,
            max_retries_per_turn=self.max_retries_per_turn,
            error_penalty=self.error_penalty,
        )

        # Reset with the provided game state
        obs = self.env.reset(game_state=game_state)

        self.turn_count = 0
        self.current_player = 0  # Qwen starts
        self.done = False
        self.last_reward = 0.0

        # Initial observation for Qwen (player-1)
        qwen_obs = obs.get("player-1", "")

        return {
            "obs_str": qwen_obs,
            "role": "user",
        }, {
            "best_score": self.env.best_score,
            "game_state": game_state,
        }

    def step(self, response_text: str) -> tuple[dict, bool, dict]:
        """Process a response and return next observation.

        Args:
            response_text: The response from current player

        Returns:
            obs: dict with "obs_str" for next observation
            done: bool indicating if game ended
            info: dict with reward and metadata
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.turn_count += 1

        # Step the underlying environment
        # OptimizationEnv.step returns (result_dict, is_error)
        result, is_error = self.env.step(response_text)

        # Check if done
        self.done = result.get("done", False)

        # Check for turn limit
        if self.turn_count >= self.max_turns:
            self.done = True

        # Extract reward if game ended
        reward = 0.0
        if self.done:
            # Get normalized reward from environment
            info = result.get("info", {})
            reward = info.get("score_norm", 0.0)

            # Apply error penalty
            if hasattr(self.env, 'total_errors'):
                reward = max(0.0, reward - self.error_penalty * self.env.total_errors)

            self.last_reward = reward

        # Determine next player and observation
        if not is_error:
            # Alternate players on valid moves
            self.current_player = 1 - self.current_player

        # Get observation for next player
        if self.current_player == 0:
            next_player_key = "player-1"
        else:
            next_player_key = "player-2"

        obs_str = result.get(next_player_key, "")

        return {
            "obs_str": obs_str,
            "role": "user",
            "is_error": is_error,
        }, self.done, {
            "reward": reward,
            "turn": self.turn_count,
            "current_player": self.current_player,
            "raw_result": result,
        }

    def get_partner_view(self) -> str:
        """Get the partner's (GPT's) formatted view of the game.

        Returns:
            String representation of partner's visible scores
        """
        if self.env is None or self.env.game is None:
            return ""

        # Get player-2's table (already formatted with headers)
        partner_table = self.env.game.tables[1]
        return _format_table(partner_table)

    def get_agent_view(self) -> str:
        """Get the agent's (Qwen's) formatted view of the game.

        Returns:
            String representation of agent's visible scores
        """
        if self.env is None or self.env.game is None:
            return ""

        # Get player-1's table
        agent_table = self.env.game.tables[0]
        return _format_table(agent_table)

    def is_qwen_turn(self) -> bool:
        """Check if it's Qwen's turn to respond."""
        return self.current_player == 0

    def is_gpt_turn(self) -> bool:
        """Check if it's GPT's turn to respond."""
        return self.current_player == 1

    def get_normalized_reward(self) -> float:
        """Get the final normalized reward (0-1)."""
        return self.last_reward

    def get_game_info(self) -> dict:
        """Get current game information."""
        if self.env is None or self.env.game is None:
            return {}
        return self.env.game.get_game_info()


def _format_table(table: list) -> str:
    """Format a table as a readable string.

    Args:
        table: 2D list with headers

    Returns:
        Formatted string
    """
    if not table:
        return ""

    lines = []
    for row in table:
        cells = []
        for cell in row:
            if cell == "" or cell is None:
                cells.append("-")
            else:
                cells.append(str(cell))
        lines.append(" | ".join(f"{c:>12}" for c in cells))

    return "\n".join(lines)


def build_env(
    max_turns: int = 20,
    max_retries_per_turn: int = 3,
    error_penalty: float = 0.0,
) -> MatchingEnv:
    """Factory function for SLIME config.

    Args:
        max_turns: Maximum dialogue turns
        max_retries_per_turn: Max retries for errors
        error_penalty: Penalty per error

    Returns:
        MatchingEnv instance
    """
    return MatchingEnv(
        max_turns=max_turns,
        max_retries_per_turn=max_retries_per_turn,
        error_penalty=error_penalty,
    )
