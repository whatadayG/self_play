#!/usr/bin/env python3
"""
Simplified test for generate_sequences that bypasses AsyncRolloutRequest issues.
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
from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout


class SimpleProcessingClass:
    """Minimal processing class."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def apply_chat_template(self, messages, **kwargs):
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"
        text += "Assistant: "
        return text


class SimpleConfig:
    max_model_len = 2048
    response_length = 512


async def test_simplified_flow():
    """Test the core logic of generate_sequences without AsyncRolloutRequest complexity."""
    print("Testing simplified generate_sequences flow...")
    
    # Create rollout with simplified mocking
    with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
        rollout = DialopSelfPlayRollout()
        rollout.config = SimpleConfig()
        rollout.sampling_params = {"temperature": 0.7}
        rollout.processing_class = SimpleProcessingClass()
        
        # Override the problematic _generate_player_response method
        async def simple_generate_response(messages, obs, player):
            responses = [
                "[message] Hello partner, let's work together.",
                "[message] I see high scores in my table.",
                "[propose] Proposal:\nBLEU: Ava Li\nElectra: Daniel Nguyen\nGloVe: Sofia Patel",
                "[accept]"
            ]
            # Return different responses based on turn count
            turn = len([m for m in messages if m["role"] == "assistant"])
            return responses[min(turn, len(responses)-1)]
            
        rollout._generate_player_response = simple_generate_response
    
    # Create simple input data
    batch_size = 2
    input_data = DataProto(
        batch=TensorDict({
            "dummy": torch.zeros(batch_size, 1)
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array([json.dumps(None), json.dumps(None)]),
            "player_index": np.array([0, 1]),
            "game_id": np.array([1, 1]),
            "prompt": np.array(["Test prompt"] * 2)
        }
    )
    
    # Run generate_sequences
    output = await rollout.generate_sequences(input_data)
    
    # Check results
    print(f"\nResults:")
    print(f"- Output type: {type(output)}")
    print(f"- Batch keys: {list(output.batch.keys())}")
    print(f"- Non-tensor batch keys: {list(output.non_tensor_batch.keys())}")
    
    # Check sequences
    for i in range(len(output.batch["input_ids"])):
        seq = output.batch["input_ids"][i]
        rewards = output.batch["rewards"][i]
        print(f"\nSequence {i}:")
        print(f"  - Length: {len(seq)}")
        print(f"  - Non-zero rewards: {(rewards != 0).sum().item()}")
        print(f"  - Max reward: {rewards.max().item()}")
        
        # Decode first 100 tokens
        if len(seq) > 0:
            text = rollout.processing_class.tokenizer.decode(seq[:100], skip_special_tokens=False)
            print(f"  - First 100 chars: {text[:100]}...")
        
    # Check reward placement
    for i in range(len(output.batch["rewards"])):
        rewards = output.batch["rewards"][i]
        nonzero = torch.nonzero(rewards)
        if len(nonzero) > 0:
            print(f"\nReward placement for sequence {i}:")
            print(f"  - Position: {nonzero[0].item()}")
            print(f"  - Value: {rewards[nonzero[0]].item()}")
        else:
            print(f"\nNo rewards found for sequence {i}")
            
    return output


async def test_actual_game_generation():
    """Test with actual game generation."""
    print("\n\nTesting with actual dialop game...")
    
    # Create a real dialop game
    env = OptimizationEnv()
    obs = env.reset()
    
    print(f"Game initialized:")
    print(f"- Best possible reward: {env.game.best_assignment_reward}")
    print(f"- Current player: {obs['turn_player']}")
    
    # Simulate a few turns
    messages = [
        "[message] Hello partner",
        "[message] I see some good matches",
        "[propose] Proposal:\nBLEU: Ava Li\nElectra: Daniel Nguyen\nGloVe: Sofia Patel\nGLUE: Andrei Petrov\nLLaMA: Morgan Reed\nRoBERTa: Joseph Santos\nQuAC: Ethan Smith\nSWAG: Noah Wilson",
        "[accept]"
    ]
    
    done = False
    for i, msg in enumerate(messages):
        if done:
            break
            
        print(f"\nTurn {i+1}: {msg[:50]}...")
        obs, error = env.step(msg)
        
        if error:
            print(f"  Error: {error}")
            break
            
        done = obs["done"]
        if done:
            print(f"  Game complete!")
            print(f"  Final reward: {obs['info']['score']}")
            print(f"  Normalized: {obs['info']['score_norm']}")
            

if __name__ == "__main__":
    print("Simplified generate_sequences test")
    print("="*50)
    
    loop = asyncio.get_event_loop()
    
    # Run simplified test
    output = loop.run_until_complete(test_simplified_flow())
    
    # Run actual game test
    loop.run_until_complete(test_actual_game_generation())
    
    print("\nTest completed!")