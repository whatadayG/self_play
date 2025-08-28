#!/usr/bin/env python3
"""
Fixed comprehensive test for generate_sequences with proper game handling.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

import torch
import numpy as np
from transformers import AutoTokenizer
from tensordict import TensorDict

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "verl"))
sys.path.insert(0, str(project_root / "dialop"))

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import TASKS_SHORT, WORKERS
from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout


class ProperProcessingClass:
    """Processing class that works with verl."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        """Apply chat template."""
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text += f"System: {content}\n"
            elif role == "user":
                text += f"User: {content}\n" 
            else:
                text += f"Assistant: {content}\n"
                
        if add_generation_prompt:
            text += "Assistant: "
            
        if tokenize:
            return self.tokenizer.encode(text, add_special_tokens=True)
        return text


class ProperConfig:
    max_model_len = 2048
    response_length = 512


def create_valid_proposal(num_papers=8, num_reviewers=8):
    """Create a valid proposal string."""
    proposal = "Proposal:\n"
    for i in range(min(num_papers, num_reviewers)):
        paper = TASKS_SHORT[i]
        reviewer = WORKERS[i]
        proposal += f"{paper}: {reviewer}\n"
    return proposal.strip()


async def test_with_proper_game_flow():
    """Test generate_sequences with proper game flow."""
    print("=== Testing with Proper Game Flow ===")
    
    # Create rollout
    with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
        rollout = DialopSelfPlayRollout()
        rollout.config = ProperConfig()
        rollout.sampling_params = {"temperature": 0.7}
        rollout.processing_class = ProperProcessingClass()
        
        # Create a smarter response generator
        response_counter = {"count": 0}
        
        async def smart_generate_response(messages, obs, player):
            """Generate appropriate responses based on game state."""
            count = response_counter["count"]
            response_counter["count"] += 1
            
            # Player 1 responses
            if count == 0:  # First message from player 1
                return "[message] Hello partner, let's work on this matching task."
            elif count == 2:  # Player 1 makes proposal
                proposal = create_valid_proposal()
                return f"[propose] {proposal}"
            elif count == 4:  # Extra message
                return "[message] I think this is a good assignment."
                
            # Player 2 responses  
            elif count == 1:  # Response from player 2
                return "[message] Hi! I can see my part of the table. Let me share what I know."
            elif count == 3:  # Player 2 accepts
                return "[accept]"
            else:
                return "[message] Continuing discussion..."
                
        rollout._generate_player_response = smart_generate_response
    
    # Create input for one complete game (both players)
    batch_size = 2
    input_data = DataProto(
        batch=TensorDict({
            "dummy": torch.zeros(batch_size, 1)
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array([json.dumps(None), json.dumps(None)]),
            "player_index": np.array([0, 1]),  # Both players from same game
            "game_id": np.array([1, 1]),
            "prompt": np.array(["Start game"] * 2)
        }
    )
    
    # Run generate_sequences
    print("Running generate_sequences...")
    output = await rollout.generate_sequences(input_data)
    
    # Validate output
    print("\n--- Validation Results ---")
    
    # 1. Check structure
    assert isinstance(output, DataProto), "Output should be DataProto"
    assert all(key in output.batch for key in ["input_ids", "attention_mask", "rewards"]), "Missing required batch keys"
    assert all(key in output.non_tensor_batch for key in ["reward_model", "game_info", "player_index"]), "Missing non_tensor_batch keys"
    print("✓ Output structure correct")
    
    # 2. Check sequences
    for i in range(len(output.batch["input_ids"])):
        seq = output.batch["input_ids"][i]
        mask = output.batch["attention_mask"][i] 
        rewards = output.batch["rewards"][i]
        
        # Get valid length
        valid_length = int(mask.sum().item()) if hasattr(mask, 'sum') else len(seq)
        
        print(f"\nPlayer {i}:")
        print(f"  Sequence length: {valid_length}")
        
        # Decode text
        if valid_length > 0:
            text = rollout.processing_class.tokenizer.decode(seq[:valid_length], skip_special_tokens=False)
            print(f"  Text preview: {text[:150]}...")
            
            # Check content
            if i == 0:  # Player 1 perspective
                assert "Assistant: [message] Hello partner" in text, "Missing player 1 first message"
                assert "[propose]" in text, "Missing proposal from player 1"
            else:  # Player 2 perspective  
                assert "User: [message] Hello partner" in text, "Player 2 should see player 1 messages as User"
                assert "[accept]" in text, "Missing accept from player 2"
        
        # 3. Check rewards
        nonzero_rewards = torch.nonzero(rewards).squeeze()
        if nonzero_rewards.numel() > 0:
            reward_pos = nonzero_rewards[-1].item() if nonzero_rewards.dim() > 0 else nonzero_rewards.item()
            reward_val = rewards[reward_pos].item()
            print(f"  Reward: {reward_val:.3f} at position {reward_pos}")
            
            # Reward should be at last valid token
            last_valid = valid_length - 1
            assert reward_pos == last_valid, f"Reward at {reward_pos}, expected at {last_valid}"
            assert reward_val > 0, "Reward should be positive for completed game"
        else:
            print("  WARNING: No rewards found!")
            
    # 4. Check reward consistency
    reward_vals = []
    for i in range(len(output.non_tensor_batch["reward_model"])):
        reward_info = output.non_tensor_batch["reward_model"][i]
        reward_vals.append(reward_info["normalized_reward"])
        
    assert len(set(reward_vals)) == 1, f"All players should have same reward, got {reward_vals}"
    print(f"\n✓ Consistent rewards: {reward_vals[0]:.3f}")
    
    # 5. Check game completion
    for i in range(len(output.non_tensor_batch["game_info"])):
        game_info = output.non_tensor_batch["game_info"][i]
        assert game_info.get("completed", False), f"Game {i} should be completed"
        
    print("\n✓ All validations passed!")
    
    return output


async def test_error_handling():
    """Test that errors are handled gracefully."""
    print("\n\n=== Testing Error Handling ===")
    
    with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
        rollout = DialopSelfPlayRollout()
        rollout.config = ProperConfig()
        rollout.sampling_params = {"temperature": 0.7}
        rollout.processing_class = ProperProcessingClass()
        
        # Mock to always throw error
        async def error_generate_response(messages, obs, player):
            raise Exception("Test error")
            
        rollout._generate_player_response = error_generate_response
    
    # Create input
    input_data = DataProto(
        batch=TensorDict({"dummy": torch.zeros(1, 1)}, batch_size=[1]),
        non_tensor_batch={
            "game_state": np.array([json.dumps(None)]),
            "player_index": np.array([0]),
            "game_id": np.array([1]),
            "prompt": np.array(["Test"])
        }
    )
    
    # Should handle error gracefully
    output = await rollout.generate_sequences(input_data)
    
    # Check empty result
    assert len(output.batch["input_ids"][0]) == 0, "Should return empty sequence on error"
    assert output.non_tensor_batch["game_info"][0].get("error", False), "Should mark error in game_info"
    
    print("✓ Error handling works correctly")


if __name__ == "__main__":
    print("Fixed Comprehensive Test for generate_sequences")
    print("=" * 60)
    
    asyncio.run(test_with_proper_game_flow())
    asyncio.run(test_error_handling())
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    print("\nThe generate_sequences method correctly:")
    print("- Generates complete self-play games")
    print("- Formats sequences from both player perspectives")  
    print("- Places rewards on the last token")
    print("- Maintains consistency across players")
    print("- Handles errors gracefully")