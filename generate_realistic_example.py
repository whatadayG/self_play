#!/usr/bin/env python3
"""
Generate training datapoints using the new Player-based dialop rollout.

This script uses the simplified Player abstractions with SGLang server.
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
from tensordict import TensorDict

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "verl"))
sys.path.insert(0, str(project_root / "dialop"))

from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout

import logging
logging.getLogger('verl.workers.rollout.dialop_selfplay_rollout').setLevel(logging.DEBUG)


class RolloutConfig:
    """Configuration that matches what the rollout worker expects."""
    def __init__(self, config_dict, sglang_url, model_name):
        self.max_model_len = config_dict['max_model_len']
        self.response_length = config_dict['sampling_params']['max_new_tokens']
        self.temperature = config_dict['sampling_params']['temperature']
        self.top_p = config_dict['sampling_params']['top_p']
        self.max_new_tokens = config_dict['sampling_params']['max_new_tokens']
        
        # SGLang server settings
        self.base_url = sglang_url
        self.served_model_name = model_name
    

def generate_realistic_example(num_games=1, model_name="Qwen/Qwen2.5-7B-Instruct", sglang_url="http://localhost:8000"):
    """Generate example using the Player-based rollout with SGLang server.
    
    Args:
        num_games: Number of games to generate rollouts for
        model_name: Name/path of the model on the SGLang server
        sglang_url: URL of the SGLang server (default: http://localhost:8000)
    """
    
    # Load generation config
    config_path = Path(__file__).parent / "generation_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify SGLang server is accessible
    print(f"Using SGLang server at {sglang_url}")
    base_url = sglang_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    
    try:
        import requests
        response = requests.get(f"{base_url}/models", timeout=5)
        response.raise_for_status()
        print(f"Connected to SGLang server. Available models: {response.json()}")
    except Exception as e:
        print(f"Error: Could not connect to SGLang server at {sglang_url}: {e}")
        print("Make sure the server is running. Start with:")
        print(f"  ./start_sglang_server.sh")
        sys.exit(1)
    
    # Load data from parquet file
    df = pd.read_parquet("data/train.parquet")
    # Select n unique games
    unique_games = df['game_id'].unique()[:num_games]
    print(f"Generating rollouts for {len(unique_games)} games...")
    
    # Create rollout configuration
    rollout_config = RolloutConfig(config, sglang_url, model_name)
    
    # Create sampling parameters that match train.sh
    sampling_params = {
        'temperature': config['sampling_params']['temperature'],
        'top_p': config['sampling_params']['top_p'], 
        'max_new_tokens': config['sampling_params']['max_new_tokens'],
        'repetition_penalty': config['sampling_params'].get('repetition_penalty', 1.0),
    }
    
    # Initialize the rollout worker with minimal setup
    from unittest.mock import patch, MagicMock
    
    def mock_init(self, *args, **kwargs):
        # Minimal initialization for parent class
        self.config = kwargs.get('config', MagicMock())
        pass
    
    with patch('verl.workers.rollout.sglang_rollout.SGLangRollout.__init__', mock_init):
        rollout = DialopSelfPlayRollout(
            config=rollout_config,
            max_turns=5,
            max_retries_per_turn=0
        )
        rollout.config = rollout_config
        rollout.sampling_params = sampling_params
        
        # Set output directory for debug logs
        rollout.output_dir = "."
    
    # Prepare data from parquet file
    # The new rollout expects minimal input and generates full games
    game_states = []
    player_indices = []
    game_ids = []
    prompts = []
    
    # Create batch with both player perspectives for each game
    batch_size = len(unique_games) * 2
    
    for game_idx, game_id in enumerate(unique_games):
        game_data = df[df['game_id'] == game_id]
        # Add both players from the same game
        for player_idx in [0, 1]:
            player_data = game_data[game_data['player_index'] == player_idx].iloc[0]
            game_states.append(player_data['game_state'])
            player_indices.append(player_idx)
            game_ids.append(int(game_id))
            prompts.append(player_data['prompt'])
    
    # Create minimal input batch
    # The new rollout doesn't need pre-tokenized inputs
    input_data = DataProto(
        batch=TensorDict({
            # Minimal tensor batch - rollout will handle tokenization
            "input_ids": torch.zeros(batch_size, 1),  # Placeholder
            "attention_mask": torch.ones(batch_size, 1),  # Placeholder
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array(game_states),
            "player_index": np.array(player_indices),
            "game_id": np.array(game_ids),
            "prompt": np.array(prompts)
        }
    )
    
    print("\nRunning dialop self-play rollouts...")
    print(f"SGLang server URL: {sglang_url}")
    print(f"Model: {model_name}")
    print(f"Max turns: 30, Max retries per turn: 8")
    print(f"Temperature: {sampling_params['temperature']}, Top-p: {sampling_params['top_p']}")
    print(f"Max new tokens: {sampling_params['max_new_tokens']}")
    print("")
    
    # Run generate_sequences - this will:
    # 1. Create SGLangModelPlayer instances for both players
    # 2. Run complete games using the Player abstractions
    # 3. Log debug information automatically
    output = rollout.generate_sequences(input_data)
    
    print(f"\nRollout complete! Check current directory for debug logs:")
    print(f"  dialop_selfplay_debug_*.txt")
    
    return output


# Usage:
# 1. Start SGLang server first (with your existing script)
# 2. Run this script:
#    python generate_realistic_example.py --num-games 5
#
# The script expects:
# - SGLang server running at http://localhost:8000 (or use --sglang-url)
# - Model loaded on the server (specify with --model-name if different)
# - data/train.parquet file with game data
#
# Output:
# - Debug logs will be written to dialop_selfplay_debug_*.txt files
# - These contain full game transcripts and player perspectives

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic dialop training examples using Player-based rollout")
    parser.add_argument(
        "--num-games", 
        type=int, 
        default=1, 
        help="Number of games to generate rollouts for (default: 1)"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name on the SGLang server (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the SGLang server (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    
    # Run generation
    output = generate_realistic_example(
        num_games=args.num_games,
        model_name=args.model_name,
        sglang_url=args.sglang_url
    )
