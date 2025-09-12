#!/usr/bin/env python3
"""
Generate minimal game initializations for dialop self-play.

Instead of pre-generating full conversations, this script creates minimal
game initialization data that the DialopSelfPlayRollout will use to
generate complete games on-demand during training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add dialop to path
project_root = Path(__file__).parent.parent.parent.parent
dialop_path = project_root / "dialop"
sys.path.insert(0, str(dialop_path))

from dialop.envs.optimization import OptimizationEnv


def generate_game_initializations(num_games: int) -> list:
    """Generate game initialization data.
    
    Each entry contains:
    - game_state: The initial state of a dialop game
    - player_index: Which player's perspective (0 or 1)
    
    Args:
        num_games: Number of unique games to generate
        
    Returns:
        List of game initialization dicts
    """
    game_inits = []
    
    for game_idx in tqdm(range(num_games), desc="Generating game initializations"):
        # Create a new game instance
        env = OptimizationEnv()
        obs = env.reset()
        
        # Extract the initial game state using the game's get_game_info() method
        # This provides the exact format expected by create_from_game_state
        game_state = env.game.get_game_info()
        
        # Create entries for both players
        for player_idx in range(2):
            # Create a minimal prompt that just identifies the player
            # The actual game observation will be provided by the rollout worker
            prompt = f"You are {env.players[player_idx]} in a cooperative matching game."
            
            game_init = {
                "game_id": game_idx,
                "game_state": game_state,
                "player_index": player_idx,
                "player_name": env.players[player_idx],
                "prompt": prompt,  # Simple prompt instead of full observation
            }
            game_inits.append(game_init)
            
    return game_inits


def create_minimal_prompts(game_inits: list[dict]) -> list[dict]:
    """Convert game initializations to minimal prompts for verl.
    
    Args:
        game_inits: List of game initialization dicts
        
    Returns:
        List of prompt dicts compatible with verl's format
    """
    data = []
    
    for init in game_inits:
        # Create a simple data entry that will be used by the rollout worker
        # The rollout worker will handle the actual game play
        entry = {
            "prompt": init["prompt"],  # Simple player identification
            "game_state": json.dumps(init["game_state"]),  # Serialize to JSON string for parquet compatibility
            "player_index": init["player_index"],
            "game_id": init["game_id"],
        }
        data.append(entry)
        
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate minimal game initializations for dialop self-play"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{os.environ['HOME']}/data/dialop_selfplay_init",
        help="Output directory for game initialization files",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=1000,
        help="Number of training games to generate",
    )
    parser.add_argument(
        "--num_test", 
        type=int,
        default=100,
        help="Number of test games to generate",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating game initializations...")
    print(f"Output directory: {output_dir}")
    
    # Generate training data
    print(f"\nGenerating {args.num_train} training games...")
    train_inits = generate_game_initializations(args.num_train)
    train_data = create_minimal_prompts(train_inits)
    
    # Generate test data
    print(f"\nGenerating {args.num_test} test games...")
    test_inits = generate_game_initializations(args.num_test)
    test_data = create_minimal_prompts(test_inits)
    
    # Save as parquet files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\nSaved game initializations:")
    print(f"  Training: {train_path} ({len(train_data)} entries)")
    print(f"  Test: {test_path} ({len(test_data)} entries)")
    
    # Also save as JSON for inspection
    train_json_path = output_dir / "train_sample.json"
    with open(train_json_path, "w") as f:
        json.dump(train_data[:5], f, indent=2)
    print(f"\nSaved sample of 5 training entries to: {train_json_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total training entries: {len(train_data)} (from {args.num_train} games)")
    print(f"  Total test entries: {len(test_data)} (from {args.num_test} games)")
    print(f"  Each game generates 2 entries (one per player)")
    
    # Print example
    print(f"\nExample game initialization:")
    example = train_data[0]
    print(f"  Game ID: {example['game_id']}")
    print(f"  Player index: {example['player_index']}")
    print(f"  Prompt: {example['prompt']}")
    # Game state is now a JSON string
    game_state_data = json.loads(example['game_state'])
    print(f"  Game state keys: {list(game_state_data.keys())}")
    

if __name__ == "__main__":
    main()