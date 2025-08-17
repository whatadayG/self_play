import json
from typing import Any, Optional
from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import OptimizationGame


# Store generated gamestates for consistency within episodes
_gamestate_cache = {}


def compute_score_matching(solution_str: str, ground_truth: Any, extra_info: Optional[dict] = None) -> float:
    """Dynamic reward function that generates new gamestates as needed.
    
    For each unique conversation/episode, generates a fresh gamestate and evaluates
    the proposal against it.
    """
    try:
        # Get episode ID from extra_info if available
        episode_id = str(extra_info.get("episode_id", "default")) if extra_info else "default"
        
        # Check if this is a signal for dynamic generation
        if ground_truth == "DYNAMIC" or ground_truth == "GENERATE_NEW_GAMESTATE":
            # Check cache first
            if episode_id in _gamestate_cache:
                gamestate = _gamestate_cache[episode_id]
            else:
                # Generate new gamestate
                env = OptimizationEnv()
                obss = env.reset(game_state=None)  # None triggers random generation
                
                gamestate = {
                    "table": env.game.table,
                    "mask1": env.game.masks[0], 
                    "mask2": env.game.masks[1],
                    "best_assignment_reward": env.game.best_assignment_reward,
                    "scale1": env.game.scales[0],
                    "scale2": env.game.scales[1],
                    "action_log": []
                }
                
                # Cache for this episode
                _gamestate_cache[episode_id] = gamestate
                
                # Clean old cache entries if too large
                if len(_gamestate_cache) > 1000:
                    # Remove oldest entries
                    keys = list(_gamestate_cache.keys())
                    for k in keys[:500]:
                        del _gamestate_cache[k]
        else:
            # Use provided gamestate (backward compatibility)
            if isinstance(ground_truth, str):
                gamestate = json.loads(ground_truth)
            else:
                gamestate = ground_truth
                
        # Evaluate proposal against gamestate
        return evaluate_proposal(solution_str, gamestate)
        
    except Exception as e:
        print(f"Error in reward calculation: {e}")
        return 0.0


def evaluate_proposal(solution_str: str, gamestate: dict) -> float:
    """Evaluate a proposal against a gamestate."""
    try:
        env = OptimizationEnv()
        env.game = OptimizationGame.create_from_game_state(gamestate, one_player=False)
        
        # Parse proposal
        proposal_ids = env._parse_proposal(solution_str)
        if not proposal_ids:
            return 0.0
            
        # Apply proposal
        env.game.propose(None, env.game.turn_player, proposal_ids=proposal_ids)
        _ = env.game.proposal_response({"accept": True}, env.game.turn_player)
        
        # Calculate normalized reward
        best = float(env.game.best_assignment_reward) if env.game.best_assignment_reward else 1.0
        reward = float(env.game.proposal_reward) if env.game.proposal_reward else 0.0
        
        return max(0.0, min(1.0, reward / max(best, 1e-6)))
        
    except Exception as e:
        print(f"Error evaluating proposal: {e}")
        return 0.0