#!/usr/bin/env python3
"""Create minimal dataset files that trigger dynamic generation."""

import pandas as pd
import json

# Create minimal training data - just 1 example as a template
# The actual data will be generated dynamically
train_data = []
for i in range(1):  # Just 1 template example
    messages = [
        {"content": "You are a helpful assistant for reviewer-paper matching.", "role": "system"},
        {"content": "Start a new matching game.", "role": "user"}
    ]
    
    # Empty ground truth signals dynamic generation
    entry = {
        "messages": messages,
        "reward_model": {"ground_truth": "DYNAMIC"},
        "data_source": "matching"
    }
    
    train_data.append(entry)

# Save training and test data
df_train = pd.DataFrame(train_data)
df_train.to_parquet("/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/minimal_train.parquet", index=False)

# Test data
df_test = pd.DataFrame(train_data)
df_test.to_parquet("/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/minimal_test.parquet", index=False)

print("Minimal dataset created - actual data will be generated dynamically during training!")