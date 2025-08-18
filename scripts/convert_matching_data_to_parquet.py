"""
Convert matching game data from evaluate_opt.py format to verl parquet format.

This script converts the game state data from the dialop format into the 
format expected by verl for RL training.
"""

import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import argparse


def convert_game_state_to_verl_format(game_state, prompt_template=None):
    """Convert a single game state to verl format.
    
    Args:
        game_state: Dict containing table, mask1, mask2, scale1, scale2, etc.
        prompt_template: Optional template for the system prompt
        
    Returns:
        Dict in verl format with prompt and extra_info
    """
    # Default prompt template if not provided
    if prompt_template is None:
        prompt_template = """You are player-{player_id} in a reviewer-paper matching game. 
Your goal is to work with your partner to find the best assignment of reviewers to papers.
You have partial information about reviewer-paper similarity scores.
Remember that unknown values to both players are worth 50 points.
You should discuss with your partner and eventually make a proposal using [propose] tag."""

    # Create the messages in ChatML format
    # For player 1
    player1_prompt = [
        {
            "role": "system",
            "content": prompt_template.format(player_id=1)
        },
        {
            "role": "user",
            "content": f"Here are your reviewer-paper similarity scores (scaled by {game_state['scale1']}): {game_state.get('table', 'Table data not available')}"
        }
    ]
    
    # Create verl format entry
    verl_entry = {
        "prompt": player1_prompt,  # Initial conversation
        "extra_info": {
            "game_state": game_state,  # Pass entire game state to start_interaction
            "player_1_data": f"Scale: {game_state['scale1']}",
            "player_2_data": f"Scale: {game_state['scale2']}",
            "best_assignment_reward": game_state.get("best_assignment_reward", 0.0),
            "ground_truth": game_state.get("best_assignment", None)
        }
    }
    
    return verl_entry


def main():
    parser = argparse.ArgumentParser(description="Convert matching game data to verl parquet format")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file (e.g., optimization.jsonl)")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of games to convert")
    
    args = parser.parse_args()
    
    # Read input JSONL file
    games = []
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            game = json.loads(line)
            games.append(game)
    
    print(f"Loaded {len(games)} games from {args.input}")
    
    # Convert to verl format
    verl_data = []
    for game in games:
        verl_entry = convert_game_state_to_verl_format(game)
        verl_data.append(verl_entry)
    
    # Create DataFrame and save as parquet
    df = pd.DataFrame(verl_data)
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(verl_data)} entries to {output_path}")
    
    # Print sample entry for verification
    print("\nSample entry:")
    print(json.dumps(verl_data[0], indent=2)[:500] + "...")


if __name__ == "__main__":
    main()