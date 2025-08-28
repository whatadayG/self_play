#!/usr/bin/env python3
"""Simple test to demonstrate dialop interaction."""

import asyncio
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'dialop'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl'))

from dialop.envs.optimization import OptimizationEnv
import json


async def test_dialop_env():
    """Test basic dialop environment functionality."""
    print("=== Testing Dialop OptimizationEnv ===\n")
    
    # Create environment
    env = OptimizationEnv()
    obs = env.reset()
    
    print("Initial game state:")
    print(f"- Current player: {obs['turn_player']}")
    print(f"- Done: {obs['done']}")
    
    # Show player observations
    print(f"\nPlayer-1 sees:\n{obs['player-1'][:300]}...")
    print(f"\nPlayer-2 sees:\n{obs['player-2'][:300]}...")
    
    # Simulate a few turns
    print("\n=== Simulating conversation ===")
    
    # Player 1 sends a message
    msg1 = "[message] Hello partner, I see we have a paper-reviewer matching task."
    obs1, error = env.step(msg1)
    print(f"\nPlayer 1: {msg1}")
    print(f"Next turn: {obs1['turn_player']}")
    
    # Player 2 responds
    msg2 = "[message] Yes, let me share what I see in my table."
    obs2, error = env.step(msg2)
    print(f"\nPlayer 2: {msg2}")
    print(f"Next turn: {obs2['turn_player']}")
    
    # Get game info
    game_info = env.game.get_game_info()
    print(f"\n=== Game Info ===")
    print(f"Best possible reward: {game_info['best_assignment_reward']}")
    print(f"Table dimensions: {len(game_info['table'])}x{len(game_info['table'][0])}")
    
    return game_info


async def test_data_format():
    """Show what the data format looks like for verl."""
    print("\n\n=== Data Format for VERL ===\n")
    
    env = OptimizationEnv()
    obs = env.reset()
    game_info = env.game.get_game_info()
    
    # Create sample data entry
    data_entry = {
        "data_source": "dialop_optimization_train_0_player1",
        "messages": [
            {
                "role": "system",
                "content": env.instructions[0]  # Game instructions
            },
            {
                "role": "user", 
                "content": obs["player-1"]  # Initial observation for player 1
            }
        ],
        "reward_model": {
            "ground_truth": None,  # Not applicable for self-play
            "best_reward": game_info["best_assignment_reward"],
            "game_state": game_info,
            "player_index": 0,
        },
        "extra_info": {
            "split": "train",
            "game_id": 0,
            "player_id": "player-1",
        }
    }
    
    print("Sample data entry structure:")
    print(json.dumps(data_entry, indent=2)[:1000] + "...")
    
    return data_entry


if __name__ == "__main__":
    game_info = asyncio.run(test_dialop_env())
    data_entry = asyncio.run(test_data_format())