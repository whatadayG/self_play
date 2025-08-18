"""
Comprehensive tests for the matching game interaction setup.
Run this before starting RL training to ensure everything works correctly.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../verl'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))

import pytest
import numpy as np
from verl.interactions.matching_game_interaction import MatchingGameInteraction
from dialop.games.optimization import OptimizationGame


class TestMatchingGameInteraction:
    """Test the matching game interaction wrapper."""
    
    @pytest.fixture
    def sample_game_state(self):
        """Create a sample game state for testing."""
        return {
            "table": [[75, 23, 89, 45, 12, 67, 34, 91],
                     [56, 88, 14, 73, 29, 95, 41, 18],
                     [92, 37, 61, 8, 84, 26, 79, 53],
                     [19, 64, 98, 31, 47, 72, 15, 86],
                     [43, 81, 22, 69, 55, 38, 93, 10],
                     [77, 16, 85, 42, 68, 24, 59, 96],
                     [30, 71, 49, 87, 13, 76, 35, 62],
                     [58, 94, 6, 52, 83, 21, 66, 40]],
            "mask1": [[True, False, True, True, False, True, False, True],
                     [True, True, False, True, False, False, True, False],
                     [False, True, True, False, True, False, True, True],
                     [True, False, True, True, False, True, False, False],
                     [False, True, False, True, True, False, True, False],
                     [True, False, True, False, True, True, False, True],
                     [False, True, False, True, False, True, True, False],
                     [True, False, True, False, True, False, False, True]],
            "mask2": [[False, True, True, False, True, False, True, False],
                     [True, False, True, False, True, True, False, True],
                     [True, False, False, True, False, True, False, True],
                     [False, True, False, False, True, False, True, True],
                     [True, False, True, False, False, True, False, True],
                     [False, True, False, True, False, False, True, False],
                     [True, False, True, False, True, False, False, True],
                     [False, True, False, True, False, True, True, False]],
            "scale1": 2.5,
            "scale2": 3.7,
            "best_assignment_reward": 584.0,
            "action_log": []
        }
    
    @pytest.mark.asyncio
    async def test_game_initialization(self, sample_game_state):
        """Test that the game initializes correctly from game state."""
        interaction = MatchingGameInteraction({"max_turns": 35})
        
        # Start interaction with game state
        instance_id = await interaction.start_interaction(game_state=sample_game_state)
        
        assert instance_id is not None
        assert instance_id in interaction._instance_dict
        
        game_data = interaction._instance_dict[instance_id]
        assert "game" in game_data
        assert isinstance(game_data["game"], OptimizationGame)
        assert game_data["turn_count"] == 0
        assert game_data["proposal_made"] == False
        
        # Check game properties
        game = game_data["game"]
        assert game.best_assignment_reward == sample_game_state["best_assignment_reward"]
        assert game.scales[0] == sample_game_state["scale1"]
        assert game.scales[1] == sample_game_state["scale2"]
        
        await interaction.finalize_interaction(instance_id)
    
    @pytest.mark.asyncio
    async def test_message_parsing(self, sample_game_state):
        """Test different message types are parsed correctly."""
        interaction = MatchingGameInteraction({"max_turns": 35})
        instance_id = await interaction.start_interaction(game_state=sample_game_state)
        
        # Test normal message
        messages = [{"role": "assistant", "content": "Let's think step by step. [message] I see some high values in the table."}]
        should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
        assert should_end == False
        assert reward == 0.0
        
        # Test proposal message
        messages.append({"role": "assistant", "content": "Let's think step by step. [propose] Proposal: Paper 0: Reviewer 2, Paper 1: Reviewer 7..."}])
        should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
        assert should_end == False
        assert "must now accept or reject" in response
        assert interaction._instance_dict[instance_id]["proposal_made"] == True
        
        # Test accept message
        messages.append({"role": "assistant", "content": "Let's think step by step. [accept] I agree with this proposal."})
        should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
        assert should_end == True
        assert "accepted" in response.lower()
        assert reward >= 0.0 and reward <= 1.0  # Normalized reward
        
        await interaction.finalize_interaction(instance_id)
    
    @pytest.mark.asyncio
    async def test_force_proposal_mechanism(self, sample_game_state):
        """Test that force proposal triggers at the right time."""
        interaction = MatchingGameInteraction({"max_turns": 10, "force_proposal_threshold": 3})
        instance_id = await interaction.start_interaction(game_state=sample_game_state)
        
        # Simulate turns until we hit the threshold
        for turn in range(7):  # Turns 1-7
            messages = [{"role": "assistant", "content": f"[message] Turn {turn+1} discussion"}]
            should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
            
            if turn < 7:  # Before threshold
                assert "force_proposal" not in data or data["force_proposal"] == False
            else:  # At or after threshold (turn 8, with 3 turns left)
                assert data.get("force_proposal") == True
                assert "should make a proposal soon" in response
        
        await interaction.finalize_interaction(instance_id)
    
    @pytest.mark.asyncio
    async def test_reward_calculation(self, sample_game_state):
        """Test reward calculation and normalization."""
        interaction = MatchingGameInteraction({})
        instance_id = await interaction.start_interaction(game_state=sample_game_state)
        
        # Manually set a proposal reward for testing
        game = interaction._instance_dict[instance_id]["game"]
        game.proposal_reward = 450.0  # Less than best_assignment_reward
        
        reward = await interaction.calculate_score(instance_id)
        expected_normalized = 450.0 / 584.0
        assert abs(reward - expected_normalized) < 0.001
        
        # Test perfect proposal
        game.proposal_reward = 584.0
        reward = await interaction.calculate_score(instance_id)
        assert reward == 1.0
        
        await interaction.finalize_interaction(instance_id)
    
    @pytest.mark.asyncio
    async def test_game_state_persistence(self, sample_game_state):
        """Test that game state persists correctly across turns."""
        interaction = MatchingGameInteraction({})
        
        # Add some action log to test resumption
        sample_game_state["action_log"] = [
            {"type": "message", "player": 0, "message": {"data": "Hello"}},
            {"type": "message", "player": 1, "message": {"data": "Hi there"}}
        ]
        
        instance_id = await interaction.start_interaction(game_state=sample_game_state)
        game_data = interaction._instance_dict[instance_id]
        
        # Check turn count reflects existing action log
        assert game_data["turn_count"] == 2
        
        # Check game has correct action log
        assert len(game_data["game"].action_log) == 2
        
        await interaction.finalize_interaction(instance_id)


def test_optimization_game_compatibility():
    """Test that OptimizationGame works as expected."""
    # Test game creation from state
    game_state = {
        "table": [[50] * 8 for _ in range(8)],
        "mask1": [[True] * 8 for _ in range(8)],
        "mask2": [[True] * 8 for _ in range(8)],
        "scale1": 1.0,
        "scale2": 1.0,
        "best_assignment_reward": 400.0,
        "action_log": []
    }
    
    game = OptimizationGame.create_from_game_state(game_state, one_player=False)
    assert game is not None
    assert game.num_rows == 8
    assert game.num_cols == 8
    assert game.best_assignment_reward == 400.0
    
    # Test that game can handle messages
    game.message({
        "data": "Test message",
        "from_player": 0,
        "type": "utterance"
    })
    assert len(game.action_log) == 1
    assert game.turn_player == 1  # Should switch turns


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
    
    # Additional manual checks
    print("\n=== Manual Integration Check ===")
    
    # Check that we can import all necessary modules
    try:
        from verl.interactions.matching_game_interaction import MatchingGameInteraction
        from dialop.games.optimization import OptimizationGame
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
    
    # Check interaction config can be loaded
    config_path = Path(__file__).parent.parent / "verl/examples/sglang_multiturn/config/interaction_config/matching_game_interaction_config.yaml"
    if config_path.exists():
        print(f"✓ Interaction config exists at {config_path}")
    else:
        print(f"✗ Interaction config not found at {config_path}")
    
    print("\nAll tests completed!")