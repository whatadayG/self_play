#!/usr/bin/env python3
"""
Generate training datapoints using the actual trained model.

This script runs the core rollout logic and outputs the two training datapoints
with tensors decoded back to string format.
"""

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensordict import TensorDict

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "verl"))
sys.path.insert(0, str(project_root / "dialop"))

from dialop.envs.optimization import OptimizationEnv
from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout


class RealModelProcessingClass:
    """Processing class with real model."""
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        """Apply the model's actual chat template."""
        # Check if tokenizer has a chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                **kwargs
            )
        else:
            # Fallback to simple format
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


class RealModelConfig:
    max_model_len = 8192
    response_length = 512


def generate_realistic_example():
    """Generate example using the actual trained model."""
    
    model_path = "/home/nickatomlin/georgiazhou/self_play/old/save_points/global_step_1000_merged"
    
    # Load the actual model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    
    # Load data from parquet file
    df = pd.read_parquet("train.parquet")
    # Select 5 unique games
    unique_games = df['game_id'].unique()[:5]
    
    # Create rollout with real model
    from unittest.mock import patch
    with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
        rollout = DialopSelfPlayRollout()
        rollout.config = RealModelConfig()
        rollout.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 256
        }
        rollout.processing_class = RealModelProcessingClass(model_path)
        
        # Use real model for generation
        def model_generate_response(messages, obs, player):
            """Generate response using the actual model."""
            # Apply chat template
            prompt = rollout.processing_class.apply_chat_template(
                messages + [{"role": "user", "content": obs}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = rollout.processing_class.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=rollout.config.max_model_len
            ).to(model.device)
            
            # Generate with model
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=rollout.sampling_params["max_new_tokens"],
                    temperature=rollout.sampling_params["temperature"],
                    top_p=rollout.sampling_params["top_p"],
                    do_sample=True,
                    pad_token_id=rollout.processing_class.tokenizer.pad_token_id,
                )
            
            # Extract only the generated part
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            response = rollout.processing_class.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            return response
            
        rollout._generate_player_response = model_generate_response
    
    # Create input for 5 games (10 datapoints total, 2 per game)
    batch_size = 10
    
    # Prepare data from parquet file
    game_states = []
    player_indices = []
    game_ids = []
    prompts = []
    
    for game_idx, game_id in enumerate(unique_games):
        game_data = df[df['game_id'] == game_id]
        # Add both players from the same game
        for player_idx in [0, 1]:
            player_data = game_data[game_data['player_index'] == player_idx].iloc[0]
            game_states.append(player_data['game_state'])
            player_indices.append(player_idx)
            game_ids.append(int(game_id))
            prompts.append(player_data['prompt'])
    
    input_data = DataProto(
        batch=TensorDict({
            "dummy": torch.zeros(batch_size, 1)
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array(game_states),
            "player_index": np.array(player_indices),
            "game_id": np.array(game_ids),
            "prompt": np.array(prompts)
        }
    )
    
    # Run generate_sequences
    output = rollout.generate_sequences(input_data)
    
    
    # Save outputs to 5 separate JSON files (one per game)
    all_outputs = []
    
    for game_idx in range(1):
        # Extract data for both players of this game
        game_output_data = {
            "model_path": model_path,
            "generation_params": rollout.sampling_params,
            "game_id": int(unique_games[game_idx]),
            "training_datapoints": []
        }
        
        # Process both players for this game
        for player_offset in range(2):
            i = game_idx * 2 + player_offset  # Index in the batch
            seq = output.batch["input_ids"][i]
            mask = output.batch["attention_mask"][i]
            rewards = output.batch["rewards"][i]
            
            # Get valid sequence length
            valid_length = int(mask.sum().item())
            valid_seq = seq[:valid_length]
            
            # Get reward position
            nonzero_rewards = torch.nonzero(rewards).squeeze()
            if nonzero_rewards.numel() > 0:
                reward_pos = nonzero_rewards[-1].item() if nonzero_rewards.dim() > 0 else nonzero_rewards.item()
                reward_val = rewards[reward_pos].item()
            else:
                reward_pos = -1
                reward_val = 0.0
            
            # Decode the full sequence to string
            decoded_sequence = rollout.processing_class.tokenizer.decode(valid_seq, skip_special_tokens=False)
            
            # Create datapoint
            datapoint = {
                "player_index": player_offset,
                "sequence_length": valid_length,
                "decoded_sequence": decoded_sequence,
                "reward_position": reward_pos,
                "reward_value": reward_val,
                "normalized_reward": output.non_tensor_batch["reward_model"][i]["normalized_reward"],
                "attention_mask": mask[:valid_length].tolist(),
                "game_completed": output.non_tensor_batch["game_info"][i].get("completed", False)
            }
            
            game_output_data["training_datapoints"].append(datapoint)
        
        # Save to separate JSON file
        json_file = f"realistic_dialop_example_{game_idx + 1}.json"
        with open(json_file, "w") as f:
            json.dump(game_output_data, f, indent=2)
        
        all_outputs.append(game_output_data)
        print(f"Saved game {game_idx + 1} to {json_file}")
    
    return all_outputs


if __name__ == "__main__":
    # Run generation
    generate_realistic_example()
