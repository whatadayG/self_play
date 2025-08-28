#!/usr/bin/env python3
"""
Comprehensive test for generate_sequences method of DialopSelfPlayRollout.

This test validates:
1. Correct sequence formatting
2. Proper reward placement
3. Correct attention masks
4. Compatibility with verl's expected data format
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from tensordict import TensorDict

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "verl"))
sys.path.insert(0, str(project_root / "dialop"))

from dialop.envs.optimization import OptimizationEnv
from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout


class TestConfig:
    """Test configuration matching verl's expected structure."""
    max_model_len = 2048
    response_length = 512
    
    
class TestProcessingClass:
    """Processing class with real tokenizer."""
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        """Simple chat template for testing."""
        # Convert messages to text
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
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
            
            
async def create_test_rollout():
    """Create a DialopSelfPlayRollout instance for testing."""
    # We need to mock the parent init since we don't have full SGLang setup
    from unittest.mock import patch, AsyncMock
    
    with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
        rollout = DialopSelfPlayRollout()
        rollout.config = TestConfig()
        rollout.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512
        }
        rollout.processing_class = TestProcessingClass()
        
        # Mock the engine call to generate responses
        async def mock_engine_call(request, params):
            # Generate different responses based on context
            responses = [
                "[message] Hello partner, let's work on this matching task.",
                "[message] I can see some interesting scores in my table.",
                "[propose] Proposal:\nBLEU: Ava Li\nElectra: Daniel Nguyen\nGloVe: Sofia Patel\nGLUE: Andrei Petrov\nLLaMA: Morgan Reed\nRoBERTa: Joseph Santos\nQuAC: Ethan Smith\nSWAG: Noah Wilson",
                "[accept]",
                "[message] Let me share what I see.",
                "[reject]",
                "[message] Good idea!",
            ]
            # Use request state to cycle through responses
            if not hasattr(mock_engine_call, 'counter'):
                mock_engine_call.counter = 0
            response = responses[mock_engine_call.counter % len(responses)]
            mock_engine_call.counter += 1
            return {"text": response}
            
        rollout._handle_engine_call = AsyncMock(side_effect=mock_engine_call)
        
    return rollout


async def test_basic_generate_sequences():
    """Test basic functionality of generate_sequences."""
    print("=== Testing Basic generate_sequences ===")
    
    rollout = await create_test_rollout()
    
    # Create input data with 2 games (4 entries total - 2 players each)
    # Use TensorDict for batch
    batch_size = 4
    input_data = DataProto(
        batch=TensorDict({
            "dummy": torch.zeros(batch_size, 1)  # Placeholder tensor
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array([
                json.dumps(None),  # Game 1, Player 0 - let env create random game
                json.dumps(None),  # Game 1, Player 1
                json.dumps(None),  # Game 2, Player 0
                json.dumps(None),  # Game 2, Player 1
            ]),
            "player_index": np.array([0, 1, 0, 1]),
            "game_id": np.array([1, 1, 2, 2]),
            "prompt": np.array(["Initial observation"] * 4)
        }
    )
    
    # Run generate_sequences
    output = await rollout.generate_sequences(input_data)
    
    # Validate output structure
    assert isinstance(output, DataProto), "Output should be DataProto"
    assert "input_ids" in output.batch, "Missing input_ids"
    assert "attention_mask" in output.batch, "Missing attention_mask"
    assert "rewards" in output.batch, "Missing rewards"
    
    # Check tensor shapes
    batch_size = len(input_data.non_tensor_batch["game_state"])
    assert output.batch["input_ids"].shape[0] == batch_size, f"Wrong batch size: {output.batch['input_ids'].shape[0]} vs {batch_size}"
    assert output.batch["attention_mask"].shape == output.batch["input_ids"].shape, "Mask shape mismatch"
    assert output.batch["rewards"].shape == output.batch["input_ids"].shape, "Rewards shape mismatch"
    
    print(f"✓ Output structure correct")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence lengths: {[len(seq[seq > 0]) for seq in output.batch['input_ids']]}")
    
    return output


async def test_reward_placement(output: DataProto):
    """Test that rewards are placed correctly."""
    print("\n=== Testing Reward Placement ===")
    
    for i in range(len(output.batch["rewards"])):
        rewards = output.batch["rewards"][i]
        nonzero_indices = torch.nonzero(rewards).squeeze()
        
        # Should have exactly one non-zero reward (on last token)
        if len(nonzero_indices.shape) == 0:  # Single element
            nonzero_indices = nonzero_indices.unsqueeze(0)
        
        assert len(nonzero_indices) == 1, f"Example {i}: Expected 1 non-zero reward, got {len(nonzero_indices)}"
        
        # Find last valid token (before padding)
        attention_mask = output.batch["attention_mask"][i]
        last_token_idx = attention_mask.sum() - 1
        
        # Reward should be on the last valid token
        reward_idx = nonzero_indices[0].item()
        assert reward_idx == last_token_idx, f"Example {i}: Reward at {reward_idx}, expected at {last_token_idx}"
        
        # Check reward value
        reward_value = rewards[reward_idx].item()
        assert 0.0 <= reward_value <= 1.0, f"Example {i}: Invalid normalized reward {reward_value}"
        
        print(f"✓ Example {i}: Reward {reward_value:.3f} at position {reward_idx}")


async def test_sequence_content(output: DataProto, rollout):
    """Test that sequences contain valid content."""
    print("\n=== Testing Sequence Content ===")
    
    tokenizer = rollout.processing_class.tokenizer
    
    for i in range(min(2, len(output.batch["input_ids"]))):  # Test first 2 examples
        input_ids = output.batch["input_ids"][i]
        attention_mask = output.batch["attention_mask"][i]
        
        # Get valid tokens (non-padded)
        valid_length = attention_mask.sum().item()
        valid_ids = input_ids[:valid_length]
        
        # Decode to text
        text = tokenizer.decode(valid_ids, skip_special_tokens=False)
        
        print(f"\nExample {i}:")
        print(f"  Sequence length: {valid_length}")
        print(f"  First 100 chars: {text[:100]}...")
        
        # Check for expected content patterns
        assert "User:" in text or "Assistant:" in text, f"Example {i}: Missing role markers"
        assert "[message]" in text or "[propose]" in text or "[accept]" in text, f"Example {i}: Missing dialop actions"
        
        # Verify it's a conversation format
        lines = text.split('\n')
        assert len(lines) > 1, f"Example {i}: Should be multi-line conversation"
        
        print(f"✓ Valid conversation with {len(lines)} lines")


async def test_attention_masks(output: DataProto):
    """Test that attention masks are correct."""
    print("\n=== Testing Attention Masks ===")
    
    for i in range(len(output.batch["attention_mask"])):
        mask = output.batch["attention_mask"][i]
        input_ids = output.batch["input_ids"][i]
        
        # Mask should be 1 for valid tokens, 0 for padding
        valid_length = mask.sum().item()
        
        # All tokens before valid_length should have mask=1
        assert torch.all(mask[:valid_length] == 1), f"Example {i}: Invalid mask before valid_length"
        
        # All tokens after valid_length should have mask=0
        if valid_length < len(mask):
            assert torch.all(mask[valid_length:] == 0), f"Example {i}: Invalid mask after valid_length"
            
        # Input IDs should be pad tokens where mask=0
        if valid_length < len(input_ids):
            pad_token_id = output.batch["input_ids"][i][valid_length:].unique()
            # Should all be the same (pad token)
            assert len(pad_token_id) == 1, f"Example {i}: Multiple token types in padding region"
            
        print(f"✓ Example {i}: Valid mask with length {valid_length}/{len(mask)}")


async def test_non_tensor_batch(output: DataProto):
    """Test non-tensor batch data."""
    print("\n=== Testing Non-Tensor Batch ===")
    
    assert "reward_model" in output.non_tensor_batch, "Missing reward_model in non_tensor_batch"
    assert "game_info" in output.non_tensor_batch, "Missing game_info in non_tensor_batch"
    assert "player_index" in output.non_tensor_batch, "Missing player_index in non_tensor_batch"
    
    # Check each entry
    for i in range(len(output.non_tensor_batch["reward_model"])):
        reward_info = output.non_tensor_batch["reward_model"][i]
        game_info = output.non_tensor_batch["game_info"][i]
        player_idx = output.non_tensor_batch["player_index"][i]
        
        # Validate reward info
        assert "normalized_reward" in reward_info, f"Example {i}: Missing normalized_reward"
        assert "reward" in reward_info, f"Example {i}: Missing raw reward"
        assert 0.0 <= reward_info["normalized_reward"] <= 1.0, f"Example {i}: Invalid normalized reward"
        
        # Validate game info
        assert "completed" in game_info or "error" in game_info, f"Example {i}: Missing completion status"
        
        # Validate player index
        assert player_idx in [0, 1], f"Example {i}: Invalid player index {player_idx}"
        
        print(f"✓ Example {i}: Player {player_idx}, normalized_reward={reward_info['normalized_reward']:.3f}")


async def test_real_game_scenario():
    """Test with a real dialop game to ensure compatibility."""
    print("\n=== Testing Real Game Scenario ===")
    
    # Create a real game
    env = OptimizationEnv()
    initial_obs = env.reset()
    
    # Get game state for serialization
    game_state = {
        "tables": env.game.tables,
        "best_assignment_reward": env.game.best_assignment_reward,
        "seed": 42,
    }
    
    # Create input for both players
    rollout = await create_test_rollout()
    
    batch_size = 2
    input_data = DataProto(
        batch=TensorDict({
            "dummy": torch.zeros(batch_size, 1)
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array([json.dumps(game_state), json.dumps(game_state)]),
            "player_index": np.array([0, 1]),
            "game_id": np.array([1, 1]),
            "prompt": np.array([
                initial_obs["player-1"],  # Player 0's initial observation
                initial_obs["player-2"],  # Player 1's initial observation
            ])
        }
    )
    
    # Run generation
    output = await rollout.generate_sequences(input_data)
    
    # Basic validation
    assert len(output.batch["input_ids"]) == 2, "Should have 2 outputs (one per player)"
    
    # Check that both players got rewards
    for i in range(2):
        rewards = output.batch["rewards"][i]
        max_reward = rewards.max().item()
        assert max_reward > 0, f"Player {i} should have non-zero reward"
        
    print("✓ Real game scenario completed successfully")
    print(f"  Player 0 max reward: {output.batch['rewards'][0].max().item():.3f}")
    print(f"  Player 1 max reward: {output.batch['rewards'][1].max().item():.3f}")


async def main():
    """Run all tests."""
    print("Comprehensive generate_sequences Test")
    print("=" * 50)
    
    # Track results
    results = []
    
    try:
        # Test 1: Basic functionality
        output = await test_basic_generate_sequences()
        results.append(("Basic generate_sequences", True))
        
        # Test 2: Reward placement
        await test_reward_placement(output)
        results.append(("Reward placement", True))
        
        # Test 3: Sequence content
        rollout = await create_test_rollout()
        await test_sequence_content(output, rollout)
        results.append(("Sequence content", True))
        
        # Test 4: Attention masks
        await test_attention_masks(output)
        results.append(("Attention masks", True))
        
        # Test 5: Non-tensor batch
        await test_non_tensor_batch(output)
        results.append(("Non-tensor batch", True))
        
        # Test 6: Real game scenario
        await test_real_game_scenario()
        results.append(("Real game scenario", True))
        
    except Exception as e:
        import traceback
        print(f"\n✗ Test failed with error: {e}")
        traceback.print_exc()
        results.append((f"Failed at current test", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(1 for _, success in results if success)
    print(f"Passed: {passed}/{len(results)}")
    
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {test_name}")
        
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"generate_sequences_test_{timestamp}.log"
    
    with open(log_file, "w") as f:
        f.write("Comprehensive generate_sequences Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Tests passed: {passed}/{len(results)}\n\n")
        
        for test_name, success in results:
            f.write(f"{'PASS' if success else 'FAIL'}: {test_name}\n")
            
        f.write("\nConclusion: ")
        if passed == len(results):
            f.write("All tests passed! generate_sequences produces correctly formatted output.\n")
        else:
            f.write("Some tests failed. Check implementation.\n")
            
    print(f"\nDetailed results saved to: {log_file}")
    
    if passed == len(results):
        print("\n✓ Success! generate_sequences produces correctly formatted output compatible with verl.")
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        

if __name__ == "__main__":
    asyncio.run(main())