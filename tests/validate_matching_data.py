"""
Validate parquet data files before RL training.
Checks data format, game states, and prompt structure.
"""

import pandas as pd
import pyarrow.parquet as pq
import json
import numpy as np
from pathlib import Path
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from dialop.games.optimization import OptimizationGame


def validate_game_state(game_state, idx):
    """Validate a single game state has all required fields."""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ["table", "mask1", "mask2", "scale1", "scale2", "best_assignment_reward"]
    for field in required_fields:
        if field not in game_state:
            errors.append(f"Game {idx}: Missing required field '{field}'")
    
    # Validate table dimensions
    if "table" in game_state:
        table = game_state["table"]
        if not isinstance(table, list) or len(table) != 8:
            errors.append(f"Game {idx}: Table should be 8x8, got {len(table)} rows")
        elif len(table[0]) != 8:
            errors.append(f"Game {idx}: Table should be 8x8, got {len(table[0])} columns")
        
        # Check value ranges
        try:
            table_array = np.array(table)
            if np.any(table_array < 0) or np.any(table_array > 100):
                warnings.append(f"Game {idx}: Table values outside expected range [0, 100]")
        except:
            errors.append(f"Game {idx}: Could not convert table to numpy array")
    
    # Validate masks
    for mask_name in ["mask1", "mask2"]:
        if mask_name in game_state:
            mask = game_state[mask_name]
            if not isinstance(mask, list) or len(mask) != 8 or len(mask[0]) != 8:
                errors.append(f"Game {idx}: {mask_name} should be 8x8 boolean array")
    
    # Validate scales
    for scale_name in ["scale1", "scale2"]:
        if scale_name in game_state:
            scale = game_state[scale_name]
            if not isinstance(scale, (int, float)) or scale <= 0:
                errors.append(f"Game {idx}: {scale_name} should be positive number, got {scale}")
    
    # Validate best_assignment_reward
    if "best_assignment_reward" in game_state:
        reward = game_state["best_assignment_reward"]
        if not isinstance(reward, (int, float)) or reward <= 0:
            errors.append(f"Game {idx}: best_assignment_reward should be positive, got {reward}")
    
    # Try to create OptimizationGame from state
    try:
        game = OptimizationGame.create_from_game_state(game_state, one_player=False)
        # Basic sanity check
        if game.best_assignment_reward != game_state["best_assignment_reward"]:
            warnings.append(f"Game {idx}: Loaded game has different best_assignment_reward")
    except Exception as e:
        errors.append(f"Game {idx}: Failed to create OptimizationGame: {str(e)}")
    
    return errors, warnings


def validate_prompt_format(prompt, idx):
    """Validate the prompt is in correct ChatML format."""
    errors = []
    warnings = []
    
    if not isinstance(prompt, list):
        errors.append(f"Row {idx}: Prompt should be a list of messages")
        return errors, warnings
    
    if len(prompt) < 2:
        warnings.append(f"Row {idx}: Prompt has only {len(prompt)} messages, expected at least 2")
    
    # Check message format
    for msg_idx, msg in enumerate(prompt):
        if not isinstance(msg, dict):
            errors.append(f"Row {idx}: Message {msg_idx} is not a dictionary")
            continue
        
        if "role" not in msg:
            errors.append(f"Row {idx}: Message {msg_idx} missing 'role' field")
        elif msg["role"] not in ["system", "user", "assistant"]:
            warnings.append(f"Row {idx}: Message {msg_idx} has unusual role '{msg['role']}'")
        
        if "content" not in msg:
            errors.append(f"Row {idx}: Message {msg_idx} missing 'content' field")
    
    # Check expected structure
    if len(prompt) >= 1 and prompt[0].get("role") != "system":
        warnings.append(f"Row {idx}: First message is not a system message")
    
    return errors, warnings


def validate_parquet_file(file_path):
    """Validate a parquet file for RL training."""
    print(f"\nValidating {file_path}...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"ERROR: Could not read parquet file: {e}")
        return False
    
    print(f"Loaded {len(df)} rows")
    
    all_errors = []
    all_warnings = []
    
    # Check required columns
    required_columns = ["prompt", "extra_info"]
    for col in required_columns:
        if col not in df.columns:
            all_errors.append(f"Missing required column: {col}")
    
    if all_errors:
        print("ERRORS:", "\n".join(all_errors))
        return False
    
    # Validate each row
    for idx, row in df.iterrows():
        # Validate prompt
        errors, warnings = validate_prompt_format(row["prompt"], idx)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Validate extra_info
        extra_info = row["extra_info"]
        if not isinstance(extra_info, dict):
            all_errors.append(f"Row {idx}: extra_info should be a dictionary")
            continue
        
        # Check for game_state
        if "game_state" in extra_info:
            game_errors, game_warnings = validate_game_state(extra_info["game_state"], idx)
            all_errors.extend(game_errors)
            all_warnings.extend(game_warnings)
        else:
            all_errors.append(f"Row {idx}: extra_info missing 'game_state'")
        
        # Stop after 10 rows to avoid too much output
        if idx >= 10 and (all_errors or all_warnings):
            remaining = len(df) - idx - 1
            print(f"(Checked first 10 rows, skipping remaining {remaining})")
            break
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Errors: {len(all_errors)}")
    print(f"Warnings: {len(all_warnings)}")
    
    if all_errors:
        print("\nERRORS (first 10):")
        for error in all_errors[:10]:
            print(f"  - {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more")
    
    if all_warnings:
        print("\nWARNINGS (first 10):")
        for warning in all_warnings[:10]:
            print(f"  - {warning}")
        if len(all_warnings) > 10:
            print(f"  ... and {len(all_warnings) - 10} more")
    
    # Sample data check
    print("\nSample data from first row:")
    first_row = df.iloc[0]
    print(f"Prompt messages: {len(first_row['prompt'])}")
    if first_row['prompt']:
        print(f"First message role: {first_row['prompt'][0].get('role', 'N/A')}")
        print(f"First message preview: {first_row['prompt'][0].get('content', '')[:100]}...")
    
    if "game_state" in first_row["extra_info"]:
        gs = first_row["extra_info"]["game_state"]
        print(f"Game state has fields: {list(gs.keys())}")
        print(f"Best assignment reward: {gs.get('best_assignment_reward', 'N/A')}")
    
    return len(all_errors) == 0


def check_data_statistics(file_path):
    """Compute statistics about the data."""
    df = pd.read_parquet(file_path)
    
    print(f"\nData Statistics for {file_path}:")
    print(f"Total entries: {len(df)}")
    
    # Prompt length statistics
    prompt_lengths = df["prompt"].apply(len)
    print(f"Prompt message counts: min={prompt_lengths.min()}, max={prompt_lengths.max()}, mean={prompt_lengths.mean():.1f}")
    
    # Game statistics
    rewards = []
    scales = []
    
    for _, row in df.iterrows():
        if "game_state" in row["extra_info"]:
            gs = row["extra_info"]["game_state"]
            if "best_assignment_reward" in gs:
                rewards.append(gs["best_assignment_reward"])
            if "scale1" in gs and "scale2" in gs:
                scales.extend([gs["scale1"], gs["scale2"]])
    
    if rewards:
        rewards = np.array(rewards)
        print(f"Best assignment rewards: min={rewards.min():.1f}, max={rewards.max():.1f}, mean={rewards.mean():.1f}")
    
    if scales:
        scales = np.array(scales)
        print(f"Scale values: min={scales.min():.1f}, max={scales.max():.1f}, mean={scales.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Validate matching game parquet data")
    parser.add_argument("--train", type=str, help="Path to training parquet file")
    parser.add_argument("--test", type=str, help="Path to test parquet file")
    parser.add_argument("--stats", action="store_true", help="Show data statistics")
    
    args = parser.parse_args()
    
    if not args.train and not args.test:
        print("Please provide at least one file to validate (--train or --test)")
        return
    
    all_valid = True
    
    if args.train:
        valid = validate_parquet_file(args.train)
        all_valid = all_valid and valid
        if args.stats and valid:
            check_data_statistics(args.train)
    
    if args.test:
        valid = validate_parquet_file(args.test)
        all_valid = all_valid and valid
        if args.stats and valid:
            check_data_statistics(args.test)
    
    if all_valid:
        print("\n✓ All validations passed! Data is ready for RL training.")
    else:
        print("\n✗ Validation failed. Please fix the errors before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()