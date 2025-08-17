#!/usr/bin/env python3
"""Create a test dataset for VERL validation."""

import pandas as pd
import os

# Create test data - just a small subset of dynamic samples
test_data = []
for i in range(10):  # Just 10 test samples
    messages = [
        {"content": "You are a helpful assistant for reviewer-paper matching.", "role": "system"},
        {"content": "Generate a new matching game with random similarity scores.", "role": "user"}
    ]
    
    reward_model_data = {
        "ground_truth": "GENERATE_NEW_GAMESTATE"  # Same as training
    }
    
    entry = {
        "messages": messages,
        "reward_model": reward_model_data,
        "data_source": "matching"
    }
    
    test_data.append(entry)

# Save to the expected location
os.makedirs("/home/nickatomlin/data/rlhf/gsm8k", exist_ok=True)
df = pd.DataFrame(test_data)
df.to_parquet("/home/nickatomlin/data/rlhf/gsm8k/test.parquet", index=False)
print("Test data created at /home/nickatomlin/data/rlhf/gsm8k/test.parquet")