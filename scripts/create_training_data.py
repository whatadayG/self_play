#!/usr/bin/env python3
"""
Create training data that's separate from the 134 test examples.
We'll generate fresh gamestates for training.
"""

import pandas as pd
import json
import numpy as np
from dialop.envs.optimization import OptimizationEnv

def create_training_data(output_path: str, num_samples: int = 5000):
    """
    Create training data with fresh gamestates.
    
    Args:
        output_path: Path to save the parquet file
        num_samples: Number of unique gamestates to generate
    """
    
    print(f"Generating {num_samples} training gamestates (separate from test data)...")
    
    data = []
    env = OptimizationEnv()
    
    for i in range(num_samples):
        if i % 500 == 0:
            print(f"Generated {i}/{num_samples} gamestates...")
            
        # Generate a new random gamestate
        obss = env.reset(game_state=None)
        
        # Extract the complete gamestate
        table_values = env.game.table.values if hasattr(env.game.table, 'values') else env.game.table
        
        gamestate = {
            "table": table_values.tolist() if hasattr(table_values, 'tolist') else table_values,
            "mask1": env.game.masks[0].tolist() if hasattr(env.game.masks[0], 'tolist') else env.game.masks[0],
            "mask2": env.game.masks[1].tolist() if hasattr(env.game.masks[1], 'tolist') else env.game.masks[1],
            "best_assignment_reward": float(env.game.best_assignment_reward),
            "scale1": float(env.game.scales[0]),
            "scale2": float(env.game.scales[1]),
            "scales": [float(env.game.scales[0]), float(env.game.scales[1])],
            "action_log": [],
            "proposal_reward": 0,
            "result": {"norm": 0.0, "score": 0.0}
        }
        
        # Create the training example
        messages = [
            {"content": "You are a helpful assistant for reviewer-paper matching.", "role": "system"},
            {"content": "Start matching reviewers to papers.", "role": "user"}
        ]
        
        entry = {
            "messages": messages,
            "reward_model": {"ground_truth": json.dumps(gamestate)},
            "data_source": "matching"
        }
        
        data.append(entry)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    
    print(f"\nTraining dataset created!")
    print(f"- Saved {len(data)} unique gamestates to {output_path}")
    print(f"- These are completely separate from the 134 test examples")
    print(f"- Each gamestate is randomly generated for training diversity")

if __name__ == "__main__":
    # Create 5k training samples (can increase later)
    create_training_data(
        "/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/training_5k.parquet",
        num_samples=5000
    )
    
    # Create small validation set (separate from test)
    create_training_data(
        "/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/validation_100.parquet",
        num_samples=100
    )