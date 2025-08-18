"""
Test integration with verl components to ensure the matching game works with the RL framework.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
import tempfile
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../verl'))

import pytest
from omegaconf import OmegaConf
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config


class TestVerlIntegration:
    """Test that matching game integrates properly with verl."""
    
    def test_interaction_registry(self):
        """Test that the interaction can be loaded from config."""
        config_path = Path(__file__).parent.parent / "verl/examples/sglang_multiturn/config/interaction_config/matching_game_interaction_config.yaml"
        
        # Load interactions from config
        try:
            interactions = initialize_interactions_from_config(str(config_path))
            assert "matching_game" in interactions
            
            # Check the interaction was created correctly
            interaction = interactions["matching_game"]
            assert interaction.config["max_turns"] == 35
            assert interaction.config["force_proposal_threshold"] == 5
            print("✓ Interaction loaded successfully from config")
        except Exception as e:
            pytest.fail(f"Failed to load interaction from config: {e}")
    
    def test_config_loading(self):
        """Test that training config can be loaded."""
        config_path = Path(__file__).parent.parent / "verl/examples/sglang_multiturn/config/matching_game_multiturn_grpo_w_interaction.yaml"
        
        try:
            config = OmegaConf.load(config_path)
            
            # Check key settings
            assert config.data.max_prompt_length == 1024
            assert config.data.max_response_length == 3072
            assert config.actor_rollout_ref.rollout.name == "sglang"
            assert config.actor_rollout_ref.rollout.multi_turn.enable == True
            assert config.actor_rollout_ref.rollout.multi_turn.max_user_turns == 35
            
            print("✓ Training config loaded successfully")
        except Exception as e:
            pytest.fail(f"Failed to load training config: {e}")
    
    @pytest.mark.asyncio
    async def test_multi_turn_flow(self):
        """Test a complete multi-turn flow."""
        from verl.interactions.matching_game_interaction import MatchingGameInteraction
        
        # Create a minimal game state
        game_state = {
            "table": [[50] * 8 for _ in range(8)],
            "mask1": [[True] * 8 for _ in range(8)],
            "mask2": [[True] * 8 for _ in range(8)],
            "scale1": 1.0,
            "scale2": 1.0,
            "best_assignment_reward": 400.0,
            "action_log": []
        }
        
        interaction = MatchingGameInteraction({"max_turns": 10})
        instance_id = await interaction.start_interaction(game_state=game_state)
        
        # Simulate a conversation
        messages = []
        
        # Turn 1: Player 1 message
        messages.append({"role": "assistant", "content": "[message] I see all values are 50."})
        should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
        assert not should_end
        assert reward == 0.0
        
        # Turn 2: Player 2 message
        messages.append({"role": "user", "content": "Yes, let's discuss the assignment."})
        messages.append({"role": "assistant", "content": "[message] We should optimize the total."})
        should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
        assert not should_end
        
        # Turn 3: Make a proposal
        messages.append({"role": "user", "content": "What do you propose?"})
        messages.append({"role": "assistant", "content": "[propose] Proposal: Assign each paper to its corresponding reviewer."})
        should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
        assert not should_end
        assert "accept or reject" in response
        
        # Turn 4: Accept the proposal
        messages.append({"role": "user", "content": "Sounds good."})
        messages.append({"role": "assistant", "content": "[accept] I agree with this assignment."})
        should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
        assert should_end
        assert "accepted" in response.lower()
        assert 0.0 <= reward <= 1.0
        
        print("✓ Multi-turn flow completed successfully")
        await interaction.finalize_interaction(instance_id)
    
    def test_data_format_compatibility(self):
        """Test that our data format works with verl's dataset loader."""
        # Create a temporary parquet file
        sample_data = {
            "prompt": [[
                {"role": "system", "content": "You are player-1 in a matching game."},
                {"role": "user", "content": "Here are your scores..."}
            ]],
            "extra_info": [{
                "game_state": {
                    "table": [[50] * 8 for _ in range(8)],
                    "mask1": [[True] * 8 for _ in range(8)],
                    "mask2": [[True] * 8 for _ in range(8)],
                    "scale1": 1.0,
                    "scale2": 1.0,
                    "best_assignment_reward": 400.0,
                    "action_log": []
                }
            }]
        }
        
        df = pd.DataFrame(sample_data)
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name)
            
            # Try to load it back
            loaded_df = pd.read_parquet(tmp.name)
            assert len(loaded_df) == 1
            assert "prompt" in loaded_df.columns
            assert "extra_info" in loaded_df.columns
            
            # Check the structure
            row = loaded_df.iloc[0]
            assert isinstance(row["prompt"], list)
            assert isinstance(row["extra_info"], dict)
            assert "game_state" in row["extra_info"]
            
            print("✓ Data format is compatible with parquet storage")
            
            # Clean up
            os.unlink(tmp.name)


def test_run_script_syntax():
    """Check that the run script has correct syntax."""
    script_path = Path(__file__).parent.parent / "run_matching_game_rl.sh"
    
    if not script_path.exists():
        pytest.skip(f"Run script not found at {script_path}")
    
    # Basic syntax check
    import subprocess
    result = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
    
    if result.returncode != 0:
        pytest.fail(f"Run script has syntax errors:\n{result.stderr}")
    
    print("✓ Run script syntax is valid")


def check_dependencies():
    """Check that all required dependencies are available."""
    print("\n=== Checking Dependencies ===")
    
    dependencies = [
        ("verl", "verl.interactions.base"),
        ("sglang", "sglang"),
        ("dialop", "dialop.games.optimization"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("omegaconf", "omegaconf"),
    ]
    
    all_available = True
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} is available")
        except ImportError:
            print(f"✗ {name} is NOT available - please install it")
            all_available = False
    
    return all_available


if __name__ == "__main__":
    # Run dependency check first
    if not check_dependencies():
        print("\nPlease install missing dependencies before running tests.")
        sys.exit(1)
    
    # Run tests
    print("\n=== Running Integration Tests ===")
    pytest.main([__file__, "-v", "-s"])
    
    print("\nIntegration tests completed!")