#!/usr/bin/env python3
"""
Generate training datapoints using the actual trained model.

This script runs the core rollout logic and outputs the two training datapoints
with tensors decoded back to string format.
"""

import argparse
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


def generate_realistic_example(num_games=1):
    """Generate example using the actual trained model.
    
    Args:
        num_games: Number of games to generate rollouts for
    """
    
    model_path = "/home/nickatomlin/georgiazhou/self_play/old/save_points/global_step_1000_merged"
    
    # Load the actual model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    
    # Load data from parquet file
    df = pd.read_parquet("data/train.parquet")
    # Select n unique games
    unique_games = df['game_id'].unique()[:num_games]
    print(f"Generating rollouts for {len(unique_games)} games...")
    
    # Create rollout with real model
    from unittest.mock import patch, MagicMock
    
    # Mock the parent class initialization
    def mock_init(self, *args, **kwargs):
        # Set required attributes that parent would set
        self.config = kwargs.get('config', MagicMock())
        pass
    
    with patch('verl.workers.rollout.sglang_rollout.SGLangRollout.__init__', mock_init):
        rollout = DialopSelfPlayRollout(config=RealModelConfig())
        rollout.config = RealModelConfig()
        rollout.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 256
        }
        rollout.processing_class = RealModelProcessingClass(model_path)
        rollout._games_logged = 0
        rollout._failed_games_logged = 0
        rollout.output_dir = "."
        
        # Load game instructions
        from pathlib import Path
        instructions_path = Path(__file__).parent / "dialop" / "dialop" / "envs" / "data" / "optimization.txt"
        if instructions_path.exists():
            rollout.game_instructions = instructions_path.read_text().strip()
        else:
            rollout.game_instructions = "You are playing a cooperative game."
        
    # Use real model for generation - must be async
    async def model_generate_response(messages, obs, player):
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
    
    # Create input for n games (2n datapoints total, 2 per game)
    batch_size = len(unique_games) * 2
    
    # Prepare data from parquet file
    game_states = []
    player_indices = []
    game_ids = []
    prompts = []
    input_ids_list = []
    attention_mask_list = []
    
    for game_idx, game_id in enumerate(unique_games):
        game_data = df[df['game_id'] == game_id]
        # Add both players from the same game
        for player_idx in [0, 1]:
            player_data = game_data[game_data['player_index'] == player_idx].iloc[0]
            game_states.append(player_data['game_state'])
            player_indices.append(player_idx)
            game_ids.append(int(game_id))
            prompts.append(player_data['prompt'])
            
            # Use actual tensors from the data if available
            if 'input_ids' in player_data and player_data['input_ids'] is not None:
                input_ids = torch.tensor(player_data['input_ids'])
                attention_mask = torch.tensor(player_data['attention_mask']) if 'attention_mask' in player_data else torch.ones_like(input_ids)
            else:
                # Tokenize the prompt if no pre-tokenized data
                tokenized = rollout.processing_class.tokenizer(
                    player_data['prompt'],
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                input_ids = tokenized['input_ids'][0]
                attention_mask = tokenized['attention_mask'][0]
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
    
    # Pad sequences to same length
    from torch.nn.utils.rnn import pad_sequence
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=rollout.processing_class.tokenizer.pad_token_id or 0)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    
    input_data = DataProto(
        batch=TensorDict({
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded
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
    
    
    # Save outputs to separate JSON files (one per game)
    all_outputs = []
    
    for game_idx in range(len(unique_games)):
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
            
            # Get valid sequence length
            valid_length = int(mask.sum().item())
            valid_seq = seq[:valid_length]
            
            # Get reward information from non-tensor batch
            # Rewards are provided by the reward model, not included in the batch tensors
            normalized_reward = output.non_tensor_batch["reward_model"][i]["normalized_reward"]
            raw_reward = output.non_tensor_batch["reward_model"][i]["reward"]
            
            # Decode the full sequence to string
            decoded_sequence = rollout.processing_class.tokenizer.decode(valid_seq, skip_special_tokens=False)
            
            # Get response mask and decode tokens where gradient flows
            response_mask = output.batch["response_mask"][i][:valid_length]
            gradient_token_indices = torch.nonzero(response_mask).squeeze()
            if gradient_token_indices.numel() > 0:
                gradient_tokens = valid_seq[gradient_token_indices]
                decoded_gradient_text = rollout.processing_class.tokenizer.decode(gradient_tokens, skip_special_tokens=False)
            else:
                decoded_gradient_text = ""
            
            # Create datapoint
            datapoint = {
                "player_index": player_offset,
                "sequence_length": valid_length,
                "decoded_sequence": decoded_sequence,
                "raw_reward": raw_reward,
                "normalized_reward": normalized_reward,
                "response_mask": output.batch["response_mask"][i][:valid_length].tolist(),
                "num_gradient_tokens": int(response_mask.sum().item()),
                "gradient_text_preview": decoded_gradient_text[:500] + "..." if len(decoded_gradient_text) > 500 else decoded_gradient_text,
                "game_completed": output.non_tensor_batch["game_info"][i].get("completed", False),
                "prompt_length": len(output.batch["prompts"][i]),
                "response_length": len(output.batch["responses"][i])
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
    parser = argparse.ArgumentParser(description="Generate realistic dialop training examples")
    parser.add_argument(
        "--num_games", 
        type=int, 
        default=1, 
        help="Number of games to generate rollouts for (default: 1)"
    )
    args = parser.parse_args()
    
    # Run generation
    generate_realistic_example(num_games=args.num_games)
