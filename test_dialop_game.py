#!/usr/bin/env python3
"""
Test dialop game to understand reward assignment.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "dialop"))

from dialop.envs.optimization import OptimizationEnv


def test_game_with_actual_proposal():
    """Test a game with a properly formatted proposal."""
    
    env = OptimizationEnv()
    obs = env.reset()
    
    print("Game initialized:")
    print(f"Best possible reward: {env.game.best_assignment_reward}")
    print(f"Number of papers: {env.game.num_rows}")
    print(f"Number of reviewers: {env.game.num_cols}")
    
    # Get the actual paper and reviewer names from the module
    from dialop.games.optimization import TASKS_SHORT, WORKERS
    print("\nPapers:", TASKS_SHORT[:env.game.num_rows])
    print("Reviewers:", WORKERS[:env.game.num_cols])
    
    # Show initial observation
    print(f"\nInitial observation for {obs['turn_player']}:")
    print(obs[obs['turn_player']][:200] + "...")
    
    # Exchange some messages
    messages = [
        "[message] Hello partner, let's work on this matching task.",
        "[message] Hi! Yes, let me look at what I can see in my table.",
    ]
    
    for msg in messages:
        print(f"\n{obs['turn_player']} says: {msg}")
        obs, error = env.step(msg)
        if error:
            print(f"Error: {error}")
            return
    
    # Create a proper proposal with actual paper/reviewer names
    proposal = "[propose] Proposal:\n"
    for i, (paper, reviewer) in enumerate(zip(TASKS_SHORT[:env.game.num_rows], 
                                              WORKERS[:env.game.num_cols])):
        proposal += f"{paper}: {reviewer}\n"
    
    print(f"\n{obs['turn_player']} proposes:")
    print(proposal)
    
    obs, error = env.step(proposal)
    if error:
        print(f"Error with proposal: {error}")
        # Try to understand what went wrong
        print("\nDebug info:")
        print(f"Papers: {env.game.papers}")
        print(f"Reviewers: {env.game.reviewers}")
        return
    
    print(f"\nProposal made. Current player: {obs['turn_player']}")
    
    # Accept the proposal
    print(f"\n{obs['turn_player']} says: [accept]")
    obs, error = env.step("[accept]")
    
    if error:
        print(f"Error: {error}")
        return
        
    if obs["done"]:
        print("\nGame completed!")
        print(f"Final reward: {obs['info']['score']}")
        print(f"Normalized reward: {obs['info']['score_norm']}")
        print(f"Best possible: {env.game.best_assignment_reward}")
        print(f"Number of messages: {obs['info']['num_msgs']}")
    else:
        print("Game not done?")
        

def test_game_variations():
    """Test different game scenarios."""
    
    from dialop.games.optimization import TASKS_SHORT, WORKERS
    
    print("\n" + "="*50)
    print("Testing game variations")
    print("="*50)
    
    # Test 1: Quick accept without messages
    print("\nTest 1: Quick proposal and accept")
    env = OptimizationEnv()
    obs = env.reset()
    
    # Make immediate proposal
    proposal = "[propose] Proposal:\n"
    for paper, reviewer in zip(TASKS_SHORT[:env.game.num_rows], 
                              WORKERS[:env.game.num_cols]):
        proposal += f"{paper}: {reviewer}\n"
    
    obs, error = env.step(proposal)
    if not error:
        obs, error = env.step("[accept]")
        if obs["done"]:
            print(f"Success! Reward: {obs['info']['score_norm']:.3f}")
        else:
            print("Failed to complete")
    else:
        print(f"Proposal error: {error}")
    
    # Test 2: Try to find what makes a valid proposal
    print("\nTest 2: Understanding proposal format")
    env = OptimizationEnv()
    obs = env.reset()
    
    print(f"Papers available: {TASKS_SHORT[:env.game.num_rows]}")
    print(f"Reviewers available: {WORKERS[:env.game.num_cols]}")
    
    # Check the game's internal proposal parsing
    test_proposals = [
        "[propose] BLEU: Ava Li",  # Single assignment
        "[propose] Proposal:\nBLEU: Ava Li",  # With "Proposal:"
        "[propose] \nBLEU: Ava Li\nElectra: Daniel Nguyen",  # Multiple
    ]
    
    for prop in test_proposals:
        print(f"\nTrying: {prop}")
        env2 = OptimizationEnv()
        env2.reset()
        obs2, error2 = env2.step(prop)
        if error2:
            print(f"  Error: {error2}")
        else:
            print(f"  Success! Proposal state: {env2.game.proposal is not None}")


if __name__ == "__main__":
    print("Testing dialop game mechanics")
    print("="*50)
    
    test_game_with_actual_proposal()
    test_game_variations()