#!/usr/bin/env python3
"""Create a mini training set quickly."""

import pandas as pd
import json
import numpy as np
from dialop.envs.optimization import OptimizationEnv

data = []
env = OptimizationEnv()

print("Generating 100 training gamestates...")
for i in range(100):
    obss = env.reset(game_state=None)
    
    table_values = env.game.table.values if hasattr(env.game.table, 'values') else env.game.table
    
    gamestate = {
        "table": table_values.tolist() if hasattr(table_values, 'tolist') else table_values,
        "mask1": env.game.masks[0].tolist(),
        "mask2": env.game.masks[1].tolist(),
        "best_assignment_reward": float(env.game.best_assignment_reward),
        "scale1": float(env.game.scales[0]),
        "scale2": float(env.game.scales[1]),
        "scales": [float(env.game.scales[0]), float(env.game.scales[1])],
        "action_log": [],
        "proposal_reward": 0,
        "result": {"norm": 0.0, "score": 0.0}
    }
    
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

df = pd.DataFrame(data)
df.to_parquet("/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/training_mini.parquet", index=False)
print("Saved 100 training gamestates!")