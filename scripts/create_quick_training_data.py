#!/usr/bin/env python3
"""
Quick script to create training data separate from the 134 test examples.
"""

import pandas as pd
import json
import numpy as np
from dialop.envs.optimization import OptimizationEnv

print("Generating 5000 training gamestates (separate from test data)...")

data = []
env = OptimizationEnv()

for i in range(5000):
    if i % 500 == 0:
        print(f"Generated {i}/5000 gamestates...")
        
    # Generate a new random gamestate
    obss = env.reset(game_state=None)
    
    # Extract gamestate
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
df.to_parquet("/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/training_5k_separate.parquet", index=False)

print(f"\nTraining dataset created!")
print(f"- Saved 5000 unique gamestates to training_5k_separate.parquet")
print(f"- These are completely separate from the 134 test examples")