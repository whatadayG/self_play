#!/usr/bin/env python3
import pandas as pd
import json
import sys

def inspect_parquet(filepath):
    """Inspect a parquet file and print one complete datapoint"""
    print(f"Loading parquet file: {filepath}")
    
    # Load the parquet file
    df = pd.read_parquet(filepath)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    if len(df) == 0:
        print("\nNo data in the file!")
        return
    
    print(f"\nFirst datapoint (index 0):")
    print("=" * 80)
    
    # Get the first row
    first_row = df.iloc[0]
    
    # Print each field
    for column in df.columns:
        value = first_row[column]
        print(f"\n[{column}]:")
        
        # Try to pretty print if it's JSON-like
        if isinstance(value, str):
            try:
                # Check if it's JSON
                parsed = json.loads(value)
                print(json.dumps(parsed, indent=2))
            except:
                # Not JSON, print as-is
                print(value)
        elif isinstance(value, dict):
            print(json.dumps(value, indent=2))
        elif isinstance(value, list):
            # Check if it's a list of dicts (like messages)
            try:
                print(json.dumps(value, indent=2))
            except:
                print(value)
        else:
            print(value)
        
        print("-" * 40)

if __name__ == "__main__":
    filepath = "./scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_train.parquet"
    inspect_parquet(filepath)