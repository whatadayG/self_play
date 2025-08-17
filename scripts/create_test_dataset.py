#!/usr/bin/env python3
"""Create a smaller test dataset first."""

import pandas as pd
import json
from dialop.envs.optimization import OptimizationEnv

# Generate 1000 samples for quick testing
num_samples = 1000
data = []
env = OptimizationEnv()

print(f"Generating {num_samples} gamestates...")
for i in range(num_samples):
    if i % 100 == 0:
        print(f"Progress: {i}/{num_samples}")
        
    obss = env.reset(game_state=None)
    
    gamestate = {
        "table": env.game.table.values.tolist(),
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
df.to_parquet("/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/dynamic_1k.parquet", index=False)
print(f"Saved {len(data)} gamestates!")