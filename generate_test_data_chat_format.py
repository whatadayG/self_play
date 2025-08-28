#!/usr/bin/env python3
"""
Generate test data in chat format for dialop self-play.
This creates minimal entries that RLHFDataset can process.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add dialop to path
sys.path.insert(0, str(Path(__file__).parent / "dialop"))

from dialop.envs.optimization import OptimizationEnv


def create_chat_format_data(num_games=2):
    """Create data in the format RLHFDataset expects."""
    entries = []
    
    for game_idx in range(num_games):
        # Create a new game instance
        env = OptimizationEnv()
        obs = env.reset()
        
        # Create entries for both players
        for player_idx in range(2):
            player_name = env.players[player_idx]
            initial_obs = obs[player_name]
            
            # Create minimal chat format
            # The messages field is what RLHFDataset looks for
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are playing a cooperative paper-reviewer matching game. Work with your partner to find the best assignment."
                    },
                    {
                        "role": "user", 
                        "content": initial_obs  # The game state/observation
                    }
                ],
                # Store game info in non-chat fields
                "game_state": json.dumps({
                    "tables": env.game.tables,
                    "best_assignment_reward": env.game.best_assignment_reward,
                    "seed": game_idx,
                }),
                "player_index": player_idx,
                "game_id": game_idx,
                # Add empty reward field
                "reward_model": {
                    "normalized_reward": 0.0,  # Will be filled by rollout
                }
            }
            entries.append(entry)
            
    return entries


def main():
    output_dir = os.path.expanduser("~/data/dialop_selfplay_init")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create small datasets
    train_data = create_chat_format_data(2)  # 4 entries total
    test_data = create_chat_format_data(1)   # 2 entries total
    
    # Save as parquet
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = os.path.join(output_dir, "train_chat_format.parquet")
    test_path = os.path.join(output_dir, "test_chat_format.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Created chat format data:")
    print(f"  Train: {train_path} ({len(train_df)} entries)")
    print(f"  Test: {test_path} ({len(test_df)} entries)")
    
    # Show sample
    print("\nSample entry:")
    sample = train_data[0]
    print(f"  Messages: {len(sample['messages'])} messages")
    print(f"  System: {sample['messages'][0]['content'][:50]}...")
    print(f"  User prompt length: {len(sample['messages'][1]['content'])} chars")
    print(f"  Game ID: {sample['game_id']}")
    print(f"  Player index: {sample['player_index']}")


if __name__ == "__main__":
    main()