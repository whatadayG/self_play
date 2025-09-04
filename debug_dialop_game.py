#!/usr/bin/env python3
"""
Debug dialop game environment to understand reward structure and failure modes.
"""

import json
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "dialop"))

from dialop.envs.optimization import OptimizationEnv
from dialop.envs.base_env import GameError

def analyze_game_state(env, obs):
    """Analyze and log the current game state."""
    logger.info("\n" + "="*60)
    logger.info(f"Turn player: {obs['turn_player']}")
    logger.info(f"Done: {obs['done']}")
    
    if "info" in obs:
        logger.info(f"Score: {obs['info'].get('score', 'N/A')}")
        logger.info(f"Normalized score: {obs['info'].get('score_norm', 'N/A')}")
    
    # Show what each player sees
    for player in ["player-1", "player-2"]:
        logger.info(f"\n{player} sees:")
        player_obs = obs[player]
        # Show first 300 chars
        if len(player_obs) > 300:
            logger.info(f"{player_obs[:300]}...")
        else:
            logger.info(player_obs)

def run_scripted_game():
    """Run a game with pre-scripted moves to understand the flow."""
    env = OptimizationEnv()
    
    # Test 1: Check initial state
    logger.info("TEST 1: Initial game state")
    obs = env.reset()
    analyze_game_state(env, obs)
    
    # Log best possible reward
    logger.info(f"\nBest possible reward: {env.game.best_assignment_reward}")
    
    # Test 2: Simple message exchange
    logger.info("\n\nTEST 2: Message exchange")
    moves = [
        "[message] Hello, let's work together to find good assignments.",
        "[message] Yes, I agree. Let me share what I know about the papers.",
        "[message] Based on my information, I think we should focus on matching expertise.",
    ]
    
    for i, move in enumerate(moves):
        logger.info(f"\n--- Move {i+1}: {obs['turn_player']} ---")
        logger.info(f"Action: {move}")
        
        obs, error = env.step(move)
        if error:
            logger.error(f"ERROR: {obs[obs['turn_player']]}")
        else:
            logger.info("Move successful")
            
        if obs["done"]:
            break
    
    # Test 3: Make a proposal
    if not obs["done"]:
        logger.info("\n\nTEST 3: Making a proposal")
        
        # Try different proposal formats
        proposals = [
            "[propose] reviewer 1 -> paper 1, reviewer 2 -> paper 2, reviewer 3 -> paper 3",
            "[propose] Assignments: R1->P1, R2->P2, R3->P3",
            "[propose] BERT: Alice Chen\nRoBERTa: Bob Smith\nGPT: Carol Jones"
        ]
        
        for proposal in proposals:
            logger.info(f"\nTrying proposal format: {proposal}")
            test_env = OptimizationEnv()
            test_env.reset()
            
            # Quick message exchange
            test_env.step("[message] Let's assign reviewers")
            test_env.step("[message] OK")
            
            # Try proposal
            obs, error = test_env.step(proposal)
            if error:
                logger.warning(f"Proposal failed: {obs[obs['turn_player']]}")
            else:
                logger.info("Proposal accepted by game")
                
                # Check if other player can accept
                obs, error = test_env.step("[accept]")
                if not error and obs["done"]:
                    logger.info(f"SUCCESS! Final score: {obs['info']['score']}")
                    logger.info(f"Normalized: {obs['info']['score_norm']}")
                    break

def test_max_turns():
    """Test what happens when we reach max turns."""
    logger.info("\n\nTEST 4: Max turns scenario")
    
    env = OptimizationEnv()
    obs = env.reset()
    
    turn_count = 0
    max_turns = 50  # Try many turns
    
    while not obs["done"] and turn_count < max_turns:
        # Just send messages
        obs, error = env.step("[message] Still discussing...")
        turn_count += 1
        
        if turn_count % 10 == 0:
            logger.info(f"Turn {turn_count}: Game still running")
    
    logger.info(f"Game ended at turn {turn_count}")
    logger.info(f"Done: {obs['done']}")
    
    if obs["done"] and "info" in obs:
        logger.info(f"Final reward: {obs['info']['score_norm']}")

def test_error_recovery():
    """Test error handling and recovery."""
    logger.info("\n\nTEST 5: Error handling")
    
    env = OptimizationEnv()
    obs = env.reset()
    
    # Try invalid moves
    invalid_moves = [
        "[invalid] This is not a valid action",
        "[propose]",  # Empty proposal
        "[accept]",   # Accept without proposal
        "Just text without action tag"
    ]
    
    for move in invalid_moves:
        logger.info(f"\nTrying invalid move: {move}")
        obs, error = env.step(move)
        
        if error:
            logger.info(f"Error (as expected): {obs[obs['turn_player']][:100]}...")
            
            # Try to recover
            obs, error = env.step("[message] Let me try a valid move instead.")
            if not error:
                logger.info("Recovery successful!")

if __name__ == "__main__":
    # Run all tests
    run_scripted_game()
    test_max_turns()
    test_error_recovery()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("DEBUGGING SUMMARY")
    logger.info("="*80)
    logger.info("1. Check if conversations are hitting 4096 token limit")
    logger.info("2. Check if games are ending due to max turns (30 in training)")
    logger.info("3. Check if proposals are formatted correctly")
    logger.info("4. Check if error recovery is exhausting retries")
    logger.info("5. Consider increasing max_response_length if needed")