import json
import random
from typing import Any, Optional

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import OptimizationGame


def compute_score_matching(solution_str: str, ground_truth: dict, extra_info: Optional[dict] = None) -> float:
    """Dynamic reward function that generates new gamestates for each evaluation.
    
    When ground_truth contains "GENERATE_NEW_GAMESTATE", we create a fresh random gamestate
    and evaluate the proposal against it.
    """
    try:
        # Check if we need to generate a new gamestate
        if isinstance(ground_truth, str) and ground_truth == "GENERATE_NEW_GAMESTATE":
            # Generate a completely new random gamestate
            env = OptimizationEnv()
            # Reset with no game_state triggers random generation
            obss = env.reset(game_state=None)
            
            # Extract the generated gamestate
            generated_gamestate = {
                "table": env.game.table,
                "mask1": env.game.masks[0],
                "mask2": env.game.masks[1],
                "best_assignment_reward": env.game.best_assignment_reward,
                "scale1": env.game.scales[0],
                "scale2": env.game.scales[1],
                # Add other necessary fields
                "action_log": []
            }
            
            # Now evaluate the proposal against this new gamestate
            return evaluate_proposal_against_gamestate(solution_str, generated_gamestate)
            
        elif isinstance(ground_truth, dict):
            # Use existing gamestate (backward compatibility)
            if isinstance(ground_truth, str):
                ground_truth = json.loads(ground_truth)
            return evaluate_proposal_against_gamestate(solution_str, ground_truth)
        else:
            print(f"Unexpected ground_truth format: {type(ground_truth)}")
            return 0.0
            
    except Exception as e:
        print(f"Error in compute_score_matching: {e}")
        return 0.0


def evaluate_proposal_against_gamestate(solution_str: str, gamestate: dict) -> float:
    """Evaluate a proposal string against a specific gamestate."""
    try:
        # Create environment and game from the gamestate
        env = OptimizationEnv()
        env.game = OptimizationGame.create_from_game_state(gamestate, one_player=False)
        
        # Parse the proposal from the solution string
        proposal_ids = env._parse_proposal(solution_str)
        
        if not proposal_ids:
            # No valid proposal found
            return 0.0
        
        # Apply the proposal to get the reward
        env.game.propose(None, env.game.turn_player, proposal_ids=proposal_ids)
        _ = env.game.proposal_response({"accept": True}, env.game.turn_player)
        
        # Calculate normalized reward
        best = float(env.game.best_assignment_reward) if env.game.best_assignment_reward else 1.0
        proposal_reward = float(env.game.proposal_reward) if env.game.proposal_reward else 0.0
        
        normalized_reward = proposal_reward / max(best, 1e-6)
        return max(0.0, min(1.0, normalized_reward))
        
    except Exception as e:
        print(f"Error evaluating proposal: {e}")
        return 0.0