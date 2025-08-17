#!/usr/bin/env python3
"""
Creates a dynamic training dataset for VERL that generates new gamestates on every epoch
instead of using fixed gamestates.
"""

import pandas as pd
import json
import numpy as np
from dialop.envs.optimization import OptimizationEnv

def create_dynamic_training_data(output_path: str, num_samples: int = 1000):
    """
    Create training data where each sample represents a request for a NEW gamestate
    to be generated dynamically during training.
    
    Args:
        output_path: Path to save the parquet file
        num_samples: Number of training samples (each will generate a new gamestate)
    """
    
    print(f"Creating {num_samples} dynamic training samples...")
    
    # Create data structure that tells VERL to generate new gamestates
    data = []
    
    for i in range(num_samples):
        # Each sample has minimal conversation to trigger gamestate generation
        messages = [
            {"content": "You are a helpful assistant for reviewer-paper matching.", "role": "system"},
            {"content": "Generate a new matching game with random similarity scores.", "role": "user"}
        ]
        
        # Empty ground_truth signals that a new gamestate should be generated
        reward_model_data = {
            "ground_truth": "GENERATE_NEW_GAMESTATE"  # Special flag for dynamic generation
        }
        
        entry = {
            "messages": messages,
            "reward_model": reward_model_data,
            "data_source": "matching"
        }
        
        data.append(entry)
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(data)
    df.to_parquet(output_path, index=False)
    
    print(f"Saved {len(data)} dynamic training samples to {output_path}")
    print("Each sample will generate a new random gamestate during training!")

if __name__ == "__main__":
    create_dynamic_training_data(
        "/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/dynamic_matching.parquet",
        num_samples=1000  # Generate 1000 unique gamestates during training
    )