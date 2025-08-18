#!/usr/bin/env python3
"""
Convert self-play data from evaluate_opt.py to VERL training format.

This script takes JSONL output from evaluate_opt.py and converts it to parquet format
suitable for VERL training, creating two training instances per game (one from each 
player's perspective).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any
import sys
import os

# Add dialop to path
sys.path.append(os.path.dirname(__file__))


def load_optimization_prompt(player_idx: int, scale: float) -> str:
    """Load and format the optimization prompt for a specific player."""
    prompt_path = "/home/nickatomlin/georgiazhou/self_play/dialop/dialop/prompts/optimization.txt"
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    # Calculate unknown value based on player's scale
    unknown_value = int(50 * scale)
    return prompt_template.replace("{unknown_value}", str(unknown_value))


def format_messages_from_player_perspective(action_log: List[Dict], player_idx: int) -> List[Dict]:
    """
    Format the action log into a conversation from a specific player's perspective.
    
    Args:
        action_log: The game's action log with all messages
        player_idx: 0 for player-1, 1 for player-2
    
    Returns:
        List of message dicts with role and content
    """
    messages = []
    player_name = f"player-{player_idx + 1}"
    other_player_idx = 1 - player_idx
    other_player_name = f"player-{other_player_idx + 1}"
    
    for action in action_log:
        if action.get("type") == "message":
            msg = action.get("message", {})
            from_player = msg.get("from_player")
            content = msg.get("content", "")
            
            if from_player == player_name:
                # This player's message is assistant
                messages.append({
                    "role": "assistant",
                    "content": content
                })
            elif from_player == other_player_name:
                # Other player's message is user
                messages.append({
                    "role": "user", 
                    "content": content
                })
    
    return messages


def convert_game_to_training_instances(game_data: Dict, game_idx: int) -> List[Dict]:
    """
    Convert a single game's data into two training instances (one per player perspective).
    
    Args:
        game_data: The game data from evaluate_opt.py
        game_idx: Index of this game (used for grouping in GRPO)
    
    Returns:
        List of two training instances
    """
    instances = []
    
    # Extract key information
    game_info = game_data.get("game_info", {})
    action_log = game_data.get("action_log", [])
    reward = game_data.get("reward_normalized", 0.0)
    
    # Game state for interaction initialization
    game_state = {
        "table": game_info.get("table", []),
        "mask1": game_info.get("mask1", []),
        "mask2": game_info.get("mask2", []),
        "scale1": game_info.get("scale1", 1.0),
        "scale2": game_info.get("scale2", 1.0),
        "best_assignment_reward": game_info.get("best_assignment_reward", 0),
        "action_log": []  # Empty for fresh start
    }
    
    # Create instance from player 1's perspective
    scale1 = game_state["scale1"]
    system_prompt1 = load_optimization_prompt(0, scale1)
    messages1 = format_messages_from_player_perspective(action_log, 0)
    
    # System message
    prompt1 = [{
        "role": "system",
        "content": system_prompt1
    }]
    
    # Add initial user message to start the conversation
    prompt1.append({
        "role": "user",
        "content": "Here are your reviewer-paper similarity scores. The table shows similarity scores between 8 reviewers and 8 papers. Higher scores indicate better matches. Start the discussion with your partner to find the best assignment."
    })
    
    instances.append({
        "prompt": prompt1,
        "messages": messages1,  # The actual conversation from player 1's perspective
        "reward_model": {
            "ground_truth": reward  # Normalized reward for GRPO
        },
        "extra_info": {
            "index": game_idx,  # Critical for GRPO grouping!
            "perspective": "player-1",
            "game_state": game_state,
            "proposal_reward": game_info.get("proposal_reward", 0),
            "best_assignment_reward": game_info.get("best_assignment_reward", 0)
        },
        "data_source": "optimization_selfplay"
    })
    
    # Create instance from player 2's perspective
    scale2 = game_state["scale2"]
    system_prompt2 = load_optimization_prompt(1, scale2)
    messages2 = format_messages_from_player_perspective(action_log, 1)
    
    # System message
    prompt2 = [{
        "role": "system",
        "content": system_prompt2
    }]
    
    # Add initial user message
    prompt2.append({
        "role": "user",
        "content": "Here are your reviewer-paper similarity scores. The table shows similarity scores between 8 reviewers and 8 papers. Higher scores indicate better matches. Start the discussion with your partner to find the best assignment."
    })
    
    instances.append({
        "prompt": prompt2,
        "messages": messages2,  # The actual conversation from player 2's perspective
        "reward_model": {
            "ground_truth": reward  # Same reward for both perspectives
        },
        "extra_info": {
            "index": game_idx,  # Same index for GRPO grouping!
            "perspective": "player-2",
            "game_state": game_state,
            "proposal_reward": game_info.get("proposal_reward", 0),
            "best_assignment_reward": game_info.get("best_assignment_reward", 0)
        },
        "data_source": "optimization_selfplay"
    })
    
    return instances


def main():
    parser = argparse.ArgumentParser(description="Convert self-play data to VERL format")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input JSONL file from evaluate_opt.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output parquet file for VERL training")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Maximum number of games to process")
    
    args = parser.parse_args()
    
    # Load the self-play data
    print(f"Loading self-play data from {args.input}...")
    games = []
    with open(args.input, 'r') as f:
        for line in f:
            games.append(json.loads(line.strip()))
            if args.max_games and len(games) >= args.max_games:
                break
    
    print(f"Loaded {len(games)} games")
    
    # Convert each game to training instances
    all_instances = []
    for game_idx, game_data in enumerate(games):
        instances = convert_game_to_training_instances(game_data, game_idx)
        all_instances.extend(instances)
        
        if (game_idx + 1) % 10 == 0:
            print(f"Processed {game_idx + 1} games...")
    
    print(f"Created {len(all_instances)} training instances from {len(games)} games")
    
    # Convert to dataframe and save as parquet
    df = pd.DataFrame(all_instances)
    
    # Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Saved {len(df)} training instances to {output_path}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total instances: {len(df)}")
    print(f"Unique games (groups): {df['extra_info'].apply(lambda x: x['index']).nunique()}")
    print(f"Average reward: {df['reward_model'].apply(lambda x: x['ground_truth']).mean():.3f}")
    print(f"Reward distribution:")
    rewards = df['reward_model'].apply(lambda x: x['ground_truth'])
    print(f"  Min: {rewards.min():.3f}")
    print(f"  Max: {rewards.max():.3f}")
    print(f"  Std: {rewards.std():.3f}")


if __name__ == "__main__":
    main()