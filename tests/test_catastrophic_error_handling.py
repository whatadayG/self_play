#!/usr/bin/env python3
"""Unit tests for catastrophic error handling.

Verifies that unexpected exceptions (generation failures, invalid state, bugs)
properly terminate games with done=True, while normal GameErrors allow retries.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dialop.envs.optimization import OptimizationEnv
from dialop.envs.base_env import GameError
from dialop.base_player import BaseModelPlayer, ModelConfig
from dialop.envs.wrappers import ForceProposal
from rich.console import Console


class TestOptimizationEnvCatastrophicErrors:
    """Test that OptimizationEnv properly handles catastrophic errors."""

    def test_game_error_does_not_terminate_game(self):
        """Normal GameErrors should NOT set done=True (allow retries)."""
        env = OptimizationEnv(one_player=False, max_turns=10, max_retries_per_turn=3)
        env.reset(seed=42)

        # Invalid message (missing tag) triggers GameError
        invalid_message = "This message has no tag"

        obs, error_flag = env.step(invalid_message)

        # GameError should set error flag but NOT terminate game
        assert error_flag is True
        assert obs["done"] is False  # Game continues for retry

    def test_unexpected_exception_terminates_game(self):
        """Unexpected exceptions should set done=True (catastrophic failure)."""
        env = OptimizationEnv(one_player=False, max_turns=10, max_retries_per_turn=3)
        env.reset(seed=42)

        # Create a valid message structure but mock _parse_message to raise unexpected exception
        valid_message = "[message] Test message"

        with patch.object(env, '_parse_message', side_effect=RuntimeError("Unexpected bug")):
            # Capture stdout to suppress error prints
            with patch('sys.stdout', new=StringIO()):
                obs, error_flag = env.step(valid_message)

        # Catastrophic error should set done=True
        assert error_flag is True
        assert obs["done"] is True  # Game terminated

    def test_exception_in_parse_message_terminates_game(self):
        """Exception in _parse_message should terminate the game."""
        env = OptimizationEnv(one_player=False, max_turns=10, max_retries_per_turn=3)
        env.reset(seed=42)

        # Mock _parse_message to raise an unexpected exception
        with patch.object(env, '_parse_message', side_effect=ValueError("Parsing failed unexpectedly")):
            with patch('sys.stdout', new=StringIO()):
                obs, error_flag = env.step("[message] Test")

        assert obs["done"] is True
        assert error_flag is True


class TestBasePlayerGenerationFailures:
    """Test that BaseModelPlayer properly propagates generation failures."""

    def test_generation_exception_propagates(self):
        """Generation exceptions should propagate, not return fallback."""
        console = Console(file=StringIO())

        # Create a mock player that raises on generation
        class FailingPlayer(BaseModelPlayer):
            def _setup_model(self):
                pass

            def _generate_text(self, messages, **kwargs):
                raise RuntimeError("Server connection failed")

        player = FailingPlayer(
            system_prompt="Test",
            role="test-player",
            console=console,
            model_path="test-model"
        )

        player.observe("Test observation")

        # Generation failure should raise, not return fallback
        with pytest.raises(RuntimeError, match="Server connection failed"):
            player.respond()

    def test_no_fallback_response(self):
        """Verify that fallback 'I need to think about this.' is NOT returned."""
        console = Console(file=StringIO())

        class FailingPlayer(BaseModelPlayer):
            def _setup_model(self):
                pass

            def _generate_text(self, messages, **kwargs):
                raise TimeoutError("Request timed out")

        player = FailingPlayer(
            system_prompt="Test",
            role="test-player",
            console=console,
            model_path="test-model"
        )

        player.observe("Test observation")

        # Should raise, not return fallback
        with pytest.raises(TimeoutError):
            response = player.respond()


class TestWrapperExceptionHandling:
    """Test that wrappers log exceptions instead of silently swallowing them."""

    def test_wrapper_logs_exception(self, capsys):
        """Wrapper should log exceptions when inserting word limit fails."""
        # Create a mock environment
        mock_env = Mock()
        mock_env.game = Mock()
        mock_env.game.action_log = []
        mock_env.players = ["player-1", "player-2"]

        # Mock env.step to return valid observation
        obss = {
            "turn_player": "player-1",
            "player-1": "Test observation\nYour turn",
            "done": False
        }
        mock_env.step = Mock(return_value=(obss, False))
        mock_env.reset = Mock(return_value={"turn_player": "player-1", "player-1": "Start\nYour turn"})

        # Create wrapper
        wrapper = ForceProposal(mock_env, ["player-1"])
        wrapper.reset(word_limit=100)

        # Mock _insert_word_limit to raise exception
        with patch.object(wrapper, '_insert_word_limit', side_effect=AttributeError("'NoneType' object has no attribute 'rsplit'")):
            result_obss, resample = wrapper.step("[message] test")

        # Check that exception was logged (not silently swallowed)
        captured = capsys.readouterr()
        assert "WARNING: ForceProposal wrapper failed to insert word limit" in captured.out
        assert "AttributeError" in captured.out

    def test_wrapper_continues_on_exception(self):
        """Wrapper should continue with raw obs when word limit insertion fails."""
        mock_env = Mock()
        mock_env.game = Mock()
        mock_env.game.action_log = []
        mock_env.players = ["player-1", "player-2"]

        # Mock reset to return valid observation
        mock_env.reset = Mock(return_value={
            "turn_player": "player-1",
            "player-1": "Start observation\nYour turn"
        })

        # Mock step to return valid observation
        mock_env.step = Mock(return_value=(
            {
                "turn_player": "player-1",
                "player-1": "Test observation\nYour turn",
                "done": False
            },
            False  # no resample
        ))

        wrapper = ForceProposal(mock_env, ["player-1"])
        wrapper.reset(word_limit=100, game_state=None)

        # Mock _insert_word_limit to raise exception
        with patch.object(wrapper, '_insert_word_limit', side_effect=ValueError("Test error")):
            with patch('sys.stdout', new=StringIO()):  # Suppress logging
                obss, resample = wrapper.step("[message] test")

        # Game should continue (not crash)
        assert "done" in obss
        assert obss["turn_player"] == "player-1"


class TestCatastrophicVsNormalErrors:
    """Test that we distinguish between normal and catastrophic errors correctly."""

    def test_invalid_proposal_format_is_normal_error(self):
        """Invalid proposal format should be GameError (normal, allows retry)."""
        env = OptimizationEnv(one_player=False, max_turns=10, max_retries_per_turn=3)
        env.reset(seed=42)

        # Invalid proposal (missing "Proposal:" header)
        invalid_proposal = "[propose] Here's my proposal\n- BLEU: Sofia"

        obs, error_flag = env.step(invalid_proposal)

        # This is a normal error (GameError) - should allow retry
        assert error_flag is True
        assert obs["done"] is False  # NOT terminated

    def test_missing_message_tag_is_normal_error(self):
        """Missing message tag should be normal error (allows retry)."""
        env = OptimizationEnv(one_player=False, max_turns=10, max_retries_per_turn=3)
        env.reset(seed=42)

        # Missing message tag
        no_tag_message = "I think we should assign Sofia to BLEU"

        obs, error_flag = env.step(no_tag_message)

        # This is a normal error - should allow retry
        assert error_flag is True
        assert obs["done"] is False  # NOT terminated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
