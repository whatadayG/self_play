#!/usr/bin/env python3
"""
End-to-end workflow test and pre-flight checks for matching game RL training.
Run this before starting RL to ensure everything is set up correctly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import tempfile
import pandas as pd

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def check(condition, success_msg, failure_msg):
    """Pretty print check results."""
    if condition:
        print(f"{GREEN}✓{RESET} {success_msg}")
        return True
    else:
        print(f"{RED}✗{RESET} {failure_msg}")
        return False


def test_file_structure():
    """Check that all required files exist."""
    print("\n=== Checking File Structure ===")
    
    base_path = Path(__file__).parent.parent
    required_files = [
        ("verl/verl/interactions/matching_game_interaction.py", "Matching game interaction class"),
        ("verl/examples/sglang_multiturn/config/interaction_config/matching_game_interaction_config.yaml", "Interaction config"),
        ("verl/examples/sglang_multiturn/config/matching_game_multiturn_grpo_w_interaction.yaml", "Training config"),
        ("run_matching_game_rl.sh", "Run script"),
        ("scripts/convert_matching_data_to_parquet.py", "Data conversion script"),
        ("scripts/dialop/games/optimization.py", "OptimizationGame class"),
    ]
    
    all_exist = True
    for file_path, description in required_files:
        full_path = base_path / file_path
        exists = check(
            full_path.exists(),
            f"{description} exists at {file_path}",
            f"{description} NOT FOUND at {file_path}"
        )
        all_exist = all_exist and exists
    
    return all_exist


def test_python_imports():
    """Test that all Python imports work."""
    print("\n=== Testing Python Imports ===")
    
    # Add paths
    base_path = Path(__file__).parent.parent
    sys.path.insert(0, str(base_path / "verl"))
    sys.path.insert(0, str(base_path / "scripts"))
    
    imports_ok = True
    
    # Test verl imports
    try:
        from verl.interactions.matching_game_interaction import MatchingGameInteraction
        check(True, "MatchingGameInteraction imports successfully", "")
    except Exception as e:
        check(False, "", f"Failed to import MatchingGameInteraction: {e}")
        imports_ok = False
    
    # Test dialop imports
    try:
        from dialop.games.optimization import OptimizationGame
        check(True, "OptimizationGame imports successfully", "")
    except Exception as e:
        check(False, "", f"Failed to import OptimizationGame: {e}")
        imports_ok = False
    
    return imports_ok


def test_sample_game_creation():
    """Test creating a game instance."""
    print("\n=== Testing Game Creation ===")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from dialop.games.optimization import OptimizationGame
        
        # Create from empty state
        game = OptimizationGame({}, one_player=False)
        game.reset()
        
        check(
            game.num_rows == 8 and game.num_cols == 8,
            f"Created 8x8 game successfully",
            f"Game dimensions incorrect: {game.num_rows}x{game.num_cols}"
        )
        
        # Create from saved state
        game_state = {
            "table": [[50] * 8 for _ in range(8)],
            "mask1": [[True] * 8 for _ in range(8)],
            "mask2": [[False] * 8 for _ in range(8)],
            "scale1": 2.0,
            "scale2": 3.0,
            "best_assignment_reward": 400.0,
            "action_log": []
        }
        
        game2 = OptimizationGame.create_from_game_state(game_state)
        check(
            game2.best_assignment_reward == 400.0,
            "Game loaded from state successfully",
            "Failed to load game from state"
        )
        
        return True
    except Exception as e:
        check(False, "", f"Failed to create game: {e}")
        return False


def test_data_conversion():
    """Test the data conversion process."""
    print("\n=== Testing Data Conversion ===")
    
    base_path = Path(__file__).parent.parent
    sys.path.insert(0, str(base_path / "scripts"))
    
    try:
        # Create sample game data
        sample_game = {
            "table": [[i*j for j in range(8)] for i in range(8)],
            "mask1": [[True] * 8 for _ in range(8)],
            "mask2": [[True] * 8 for _ in range(8)],
            "scale1": 1.5,
            "scale2": 2.5,
            "best_assignment_reward": 500.0,
            "action_log": []
        }
        
        # Import conversion function
        from convert_matching_data_to_parquet import convert_game_state_to_verl_format
        
        # Convert
        verl_entry = convert_game_state_to_verl_format(sample_game)
        
        # Validate structure
        checks = [
            check("prompt" in verl_entry, "Prompt field exists", "Missing prompt field"),
            check("extra_info" in verl_entry, "Extra info field exists", "Missing extra_info field"),
            check(
                isinstance(verl_entry["prompt"], list) and len(verl_entry["prompt"]) >= 2,
                "Prompt has correct structure",
                "Prompt structure incorrect"
            ),
            check(
                "game_state" in verl_entry["extra_info"],
                "Game state included in extra_info",
                "Game state missing from extra_info"
            ),
        ]
        
        return all(checks)
    except Exception as e:
        check(False, "", f"Data conversion failed: {e}")
        return False


def test_interaction_workflow():
    """Test the interaction workflow."""
    print("\n=== Testing Interaction Workflow ===")
    
    try:
        import asyncio
        sys.path.insert(0, str(Path(__file__).parent.parent / "verl"))
        from verl.interactions.matching_game_interaction import MatchingGameInteraction
        
        async def run_test():
            # Create interaction
            interaction = MatchingGameInteraction({"max_turns": 10})
            
            # Create game state
            game_state = {
                "table": [[50] * 8 for _ in range(8)],
                "mask1": [[True] * 8 for _ in range(8)],
                "mask2": [[True] * 8 for _ in range(8)],
                "scale1": 1.0,
                "scale2": 1.0,
                "best_assignment_reward": 400.0,
                "action_log": []
            }
            
            # Start interaction
            instance_id = await interaction.start_interaction(game_state=game_state)
            
            # Test message
            messages = [{"role": "assistant", "content": "[message] Test message"}]
            should_end, response, reward, data = await interaction.generate_response(instance_id, messages)
            
            await interaction.finalize_interaction(instance_id)
            
            return not should_end and reward == 0.0
        
        result = asyncio.run(run_test())
        check(result, "Interaction workflow works correctly", "Interaction workflow failed")
        return result
        
    except Exception as e:
        check(False, "", f"Interaction test failed: {e}")
        return False


def test_config_consistency():
    """Check that configs are consistent."""
    print("\n=== Checking Config Consistency ===")
    
    base_path = Path(__file__).parent.parent
    
    try:
        import yaml
        
        # Load interaction config
        with open(base_path / "verl/examples/sglang_multiturn/config/interaction_config/matching_game_interaction_config.yaml") as f:
            interaction_config = yaml.safe_load(f)
        
        # Load training config
        with open(base_path / "verl/examples/sglang_multiturn/config/matching_game_multiturn_grpo_w_interaction.yaml") as f:
            training_config = yaml.safe_load(f)
        
        # Check max turns consistency
        interaction_max_turns = interaction_config["interaction"][0]["config"]["max_turns"]
        training_max_turns = training_config["actor_rollout_ref"]["rollout"]["multi_turn"]["max_user_turns"]
        
        check(
            interaction_max_turns == training_max_turns,
            f"Max turns consistent: {interaction_max_turns}",
            f"Max turns mismatch: interaction={interaction_max_turns}, training={training_max_turns}"
        )
        
        return interaction_max_turns == training_max_turns
        
    except Exception as e:
        check(False, "", f"Config check failed: {e}")
        return False


def create_test_data():
    """Create a small test dataset."""
    print("\n=== Creating Test Data ===")
    
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "data" / "matching_game"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create 5 test games
        test_games = []
        for i in range(5):
            game = {
                "table": [[j*10 + i for j in range(8)] for _ in range(8)],
                "mask1": [[True if (i+j) % 2 == 0 else False for j in range(8)] for i in range(8)],
                "mask2": [[True if (i+j) % 2 == 1 else False for j in range(8)] for i in range(8)],
                "scale1": 1.0 + i * 0.5,
                "scale2": 2.0 + i * 0.3,
                "best_assignment_reward": 400.0 + i * 50,
                "action_log": []
            }
            test_games.append(game)
        
        # Convert to verl format
        sys.path.insert(0, str(base_path / "scripts"))
        from convert_matching_data_to_parquet import convert_game_state_to_verl_format
        
        verl_data = [convert_game_state_to_verl_format(game) for game in test_games]
        
        # Save as parquet
        df = pd.DataFrame(verl_data)
        train_path = data_dir / "test_train.parquet"
        test_path = data_dir / "test_val.parquet"
        
        df[:3].to_parquet(train_path)
        df[3:].to_parquet(test_path)
        
        check(True, f"Created test data at {data_dir}", "")
        print(f"  - Train: {train_path} (3 games)")
        print(f"  - Val: {test_path} (2 games)")
        
        return True
        
    except Exception as e:
        check(False, "", f"Failed to create test data: {e}")
        return False


def main():
    """Run all pre-flight checks."""
    print("=" * 60)
    print("MATCHING GAME RL - PRE-FLIGHT CHECKS")
    print("=" * 60)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_file_structure()
    all_passed &= test_python_imports()
    all_passed &= test_sample_game_creation()
    all_passed &= test_data_conversion()
    all_passed &= test_interaction_workflow()
    all_passed &= test_config_consistency()
    
    # Offer to create test data
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{GREEN}All checks passed!{RESET}")
        
        response = input("\nWould you like to create test data files? (y/n): ")
        if response.lower() == 'y':
            create_test_data()
        
        print("\n=== Next Steps ===")
        print("1. Convert your game data:")
        print(f"   python scripts/convert_matching_data_to_parquet.py \\")
        print(f"     --input /path/to/optimization.jsonl \\")
        print(f"     --output data/matching_game/train.parquet")
        print("\n2. Validate your data:")
        print(f"   python tests/validate_matching_data.py \\")
        print(f"     --train data/matching_game/train.parquet \\")
        print(f"     --test data/matching_game/test.parquet")
        print("\n3. Start RL training:")
        print(f"   ./run_matching_game_rl.sh")
        
    else:
        print(f"{RED}Some checks failed!{RESET} Please fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()