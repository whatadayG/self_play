#!/usr/bin/env python3
"""
Generate matching game data in verl format by running the OptimizationGame
and saving game states to parquet files.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# Add dialop to path
sys.path.append(os.path.dirname(__file__))

from dialop.games.optimization import OptimizationGame


def generate_game_states(num_games=100):
    """Generate game states using OptimizationGame."""
    games = []
    
    for i in range(num_games):
        # Create a new game
        game = OptimizationGame({}, one_player=False)
        obs0, obs1 = game.reset()
        
        # Extract game state
        game_state = {
            "table": game.table.values.tolist(),
            "mask1": game.masks[0].tolist(),
            "mask2": game.masks[1].tolist(), 
            "scale1": game.scales[0],
            "scale2": game.scales[1],
            "best_assignment_reward": game.best_assignment_reward,
            "action_log": []
        }
        
        games.append(game_state)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} games...")
    
    return games


def convert_to_verl_format(game_states):
    """Convert game states to verl parquet format."""
    data = []
    
    # Load the optimization prompt
    prompt_path = "/home/nickatomlin/georgiazhou/self_play/dialop/dialop/prompts/optimization.txt"
    with open(prompt_path, 'r') as f:
        system_prompt_template = f.read()

    for idx, game_state in enumerate(game_states):
        # Create prompts for player 1 perspective
        scale1 = game_state["scale1"]
        unknown_value = int(50 * scale1)  # Scale the unknown value
        
        # Replace the placeholder in the prompt
        system_content = system_prompt_template.replace("{unknown_value}", str(unknown_value))
        
        system_msg = {
            "role": "system",
            "content": system_content
        }
        
        # Initial observation about the table
        user_msg = {
            "role": "user",
            "content": f"Here are your reviewer-paper similarity scores. The table shows similarity scores between 8 reviewers and 8 papers. Higher scores indicate better matches. Start the discussion with your partner to find the best assignment."
        }
        
        # Create the entry in verl format
        entry = {
            "prompt": [system_msg, user_msg],  # Changed from 'messages' to 'prompt'
            "reward_model": {
                "ground_truth": game_state["best_assignment_reward"]  # Put ground_truth here like gsm8k
            },
            "extra_info": {
                "game_state": game_state,  # Full game state for initialization
                "index": idx,
                "unknown_value": unknown_value
            },
            "data_source": "optimization_game"
        }
        
        data.append(entry)
      
    

    
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate matching game data for verl")
    parser.add_argument("--num-train", type=int, default=100, help="Number of training games")
    parser.add_argument("--num-test", type=int, default=20, help="Number of test games")
    parser.add_argument("--output-dir", type=str, default="data/matching_game", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--print-examples", type=int, default=0, help="Number of detailed examples to print")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_train} training games...")
    train_games = generate_game_states(args.num_train)
    train_data = convert_to_verl_format(train_games)
    
    print(f"\nGenerating {args.num_test} test games...")
    test_games = generate_game_states(args.num_test)
    test_data = convert_to_verl_format(test_games)
    
    # Save to parquet
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\nData saved to:")
    print(f"  Train: {train_path} ({len(train_df)} games)")
    print(f"  Test: {test_path} ({len(test_df)} games)")
    
    # Print sample for verification
    print("\nSample entry structure:")
    sample = train_data[0]
    print(f"  prompt: {len(sample['prompt'])} messages")
    print(f"  reward_model: {sample['reward_model']}")
    print(f"  extra_info keys: {list(sample['extra_info'].keys())}")
    print(f"  game_state keys: {list(sample['extra_info']['game_state'].keys())}")
    
    # Print detailed examples if requested
    if args.print_examples > 0:
        print("\n" + "="*80)
        print(f"DETAILED EXAMPLES (first {args.print_examples} entries)")
        print("="*80)
        
        for i in range(min(args.print_examples, len(train_data))):
            entry = train_data[i]
            print(f"\n--- Entry {i} ---")
            print(f"Data source: {entry['data_source']}")
            print(f"Reward model: {entry['reward_model']}")
            
            print(f"\nPrompt messages:")
            for j, msg in enumerate(entry['prompt']):
                print(f"  Message {j} - Role: {msg['role']}")
                content_preview = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                print(f"  Content preview: {content_preview}")
            
            print(f"\nGame state info:")
            gs = entry['extra_info']['game_state']
            print(f"  Table shape: {len(gs['table'])}x{len(gs['table'][0])}")
            print(f"  Scale1: {gs['scale1']:.2f}, Scale2: {gs['scale2']:.2f}")
            print(f"  Best assignment reward: {gs['best_assignment_reward']}")
            print(f"  Unknown value (scaled): {entry['extra_info']['unknown_value']}")
            
            # Show a sample of the table
            print(f"\n  Table preview (first 3x3):")
            for row in range(min(3, len(gs['table']))):
                row_vals = gs['table'][row][:3]
                row_str = "    " + " ".join(f"{val:3d}" for val in row_vals) + " ..."
                print(row_str)
            print("    ...")
            
            print(f"\n  Mask1 preview (first 3x3) - Player 1 visibility:")
            for row in range(min(3, len(gs['mask1']))):
                row_vals = gs['mask1'][row][:3]
                row_str = "    " + " ".join("T" if val else "F" for val in row_vals) + " ..."
                print(row_str)
            print("    ...")
            
            print("\n" + "="*80)


if __name__ == "__main__":
    main()