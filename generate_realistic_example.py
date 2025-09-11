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
import yaml
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensordict import TensorDict

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "verl"))
sys.path.insert(0, str(project_root / "dialop"))

from dialop.envs.optimization import OptimizationEnv
from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout

import logging
logging.getLogger('verl.workers.rollout.dialop_selfplay_rollout').setLevel(logging.DEBUG)


class RealModelConfig:
    def __init__(self, config_dict):
        self.max_model_len = config_dict['max_model_len']
        self.response_length = config_dict['sampling_params']['max_new_tokens']  # For compatibility
    

def generate_realistic_example(num_games=1, model_path="/home/nickatomlin/georgiazhou/self_play/old/save_points/global_step_1000_merged", use_sglang=True, sglang_url="http://localhost:8000"):
    """Generate example using the actual trained model.
    
    Args:
        num_games: Number of games to generate rollouts for
        model_path: Path to the model checkpoint
        use_sglang: Whether to use SGLang server (True) or local model (False)
        sglang_url: URL of the SGLang server (default: http://localhost:8000)
    """
    
    # Load generation config
    config_path = Path(__file__).parent / "generation_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer (needed for both modes)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model/server setup based on mode
    model = None
    sglang_api_url = None
    
    if use_sglang:
        # SGLang server mode - just prepare the API endpoint
        print(f"Using SGLang server at {sglang_url}")
        # Ensure URL has /v1 for OpenAI-compatible API
        base_url = sglang_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        sglang_api_url = f"{base_url}/chat/completions"
        
        # Verify server is accessible
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            response.raise_for_status()
            print(f"Connected to SGLang server. Available models: {response.json()}")
        except Exception as e:
            print(f"Warning: Could not connect to SGLang server at {sglang_url}: {e}")
            print("Make sure the server is running. Start with:")
            print(f"  python -m sglang.launch_server --model {model_path} --port 8000")
    else:
        # Local model mode with optimizations
        print("Using local model with optimizations")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",  # Enable Flash Attention 2
            device_map="auto"
        )
        model.eval()
        
        # Compile model for faster inference (overhead only on first run)
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    
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
        model_config = RealModelConfig(config)
        rollout = DialopSelfPlayRollout(config=model_config, max_turns=4, max_retries_per_turn=2)
        rollout.config = model_config
        rollout.sampling_params = config['sampling_params']
        rollout.processing_class = tokenizer  # Required by parent class
        rollout.tokenizer = tokenizer  # Keep for direct usage in this script
        rollout._games_logged = 0
        rollout._failed_games_logged = 0
        rollout.output_dir = "."
        
        # Load game instructions
        instructions_path = Path(__file__).parent / "dialop" / "dialop" / "envs" / "data" / "optimization.txt"
        if instructions_path.exists():
            rollout.game_instructions = instructions_path.read_text().strip()
        else:
            raise FileNotFoundError(f"Game instructions file not found at {instructions_path}")
        
    # Use real model for generation - must be async
    async def model_generate_response(messages, obs, player):
        """Generate response using the actual model or SGLang server."""
        if use_sglang:
            # SGLang server mode - use API
            messages_formatted = messages + [{"role": "user", "content": obs}]
            
            payload = {
                "model": model_path,  # Server should be running this model
                "messages": messages_formatted,
                "temperature": rollout.sampling_params["temperature"],
                "top_p": rollout.sampling_params["top_p"],
                "max_tokens": rollout.sampling_params["max_new_tokens"],
                "repetition_penalty": rollout.sampling_params.get("repetition_penalty", 1.0),
                "stream": False,
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('SGLANG_API_KEY', 'EMPTY')}",
            }
            
            try:
                response = requests.post(
                    sglang_api_url,
                    json=payload,
                    headers=headers,
                    timeout=120.0
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"Error calling SGLang server: {e}")
                raise
        else:
            # Local model mode
            # Apply chat template
            prompt = rollout.tokenizer.apply_chat_template(
                messages + [{"role": "user", "content": obs}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = rollout.tokenizer(
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
                    repetition_penalty=rollout.sampling_params.get("repetition_penalty", 1.0),
                    do_sample=True,
                    pad_token_id=rollout.tokenizer.pad_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                )
            
            # Extract only the generated part
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            response = rollout.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
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
                tokenized = rollout.tokenizer(
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
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=rollout.tokenizer.pad_token_id or 0)
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
    # output data structure is now internal to the DialopSelfPlayRollout class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic dialop training examples")
    parser.add_argument(
        "--num_games", 
        type=int, 
        default=1, 
        help="Number of games to generate rollouts for (default: 1)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_4800_merged", 
    )
    parser.add_argument(
        "--no-sglang",
        action="store_true",
        help="Use local model instead of SGLang server (slower but no server needed)"
    )
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the SGLang server (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    
    # Run generation
    generate_realistic_example(
        num_games=args.num_games,
        model_path=args.model_path,
        use_sglang=not args.no_sglang,  # Default is to use SGLang
        sglang_url=args.sglang_url
    )
