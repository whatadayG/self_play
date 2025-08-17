#!/usr/bin/env python3
"""
Create a proper dynamic training dataset with many different gamestates.
Each entry has a complete gamestate that can be used by the matching interaction.
"""

import pandas as pd
import json
import numpy as np
from dialop.envs.optimization import OptimizationEnv

def serialize_numpy(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(v) for v in obj]
    else:
        return obj

def create_dynamic_training_data(output_path: str, num_samples: int = 10000):
    """
    Create training data with many different gamestates.
    Each sample has a unique, randomly generated gamestate.
    
    Args:
        output_path: Path to save the parquet file
        num_samples: Number of unique gamestates to generate
    """
    
    print(f"Generating {num_samples} unique gamestates...")
    
    data = []
    env = OptimizationEnv()
    
    for i in range(num_samples):
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples} gamestates...")
            
        # Generate a new random gamestate
        obss = env.reset(game_state=None)
        
        # Extract the complete gamestate and convert to serializable format
        gamestate = {
            "table": serialize_numpy(env.game.table),
            "mask1": serialize_numpy(env.game.masks[0]),
            "mask2": serialize_numpy(env.game.masks[1]),
            "best_assignment_reward": float(env.game.best_assignment_reward),
            "scale1": float(env.game.scales[0]),
            "scale2": float(env.game.scales[1]),
            "scales": [float(env.game.scales[0]), float(env.game.scales[1])],
            "action_log": [],
            # Add additional fields that might be expected
            "proposal_reward": 0,
            "result": {"norm": 0.0, "score": 0.0}
        }
        
        # Create the training example
        # The messages will be populated by VERL's multi-turn interaction
        messages = [
            {"content": "You are a helpful assistant for reviewer-paper matching.", "role": "system"},
            {"content": obss["player-1"], "role": "user"}  # Initial observation
        ]
        
        entry = {
            "messages": messages,
            "reward_model": {"ground_truth": json.dumps(gamestate)},  # Proper gamestate
            "data_source": "matching"
        }
        
        data.append(entry)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    
    print(f"Saved {len(data)} unique gamestates to {output_path}")
    print("Each gamestate is different - providing massive diversity for training!")

if __name__ == "__main__":
    # Create training data with 10k unique gamestates
    create_dynamic_training_data(
        "/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/dynamic_10k.parquet",
        num_samples=10000
    )
    
    # Create smaller test set
    create_dynamic_training_data(
        "/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/dynamic_test.parquet",
        num_samples=100
    )