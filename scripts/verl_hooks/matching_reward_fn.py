import json
from typing import Any, Optional

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import OptimizationGame


def compute_score_matching(solution_str: str, ground_truth: dict, extra_info: Optional[dict] = None) -> float:
	"""Stateless reward function used by VERL's custom_reward_function path.
	Parses a [propose] string, evaluates proposal_reward / best_assignment_reward in [0,1]."""
	try:
		env = OptimizationEnv()
		env.game = OptimizationGame.create_from_game_state(ground_truth, one_player=False)
		proposal_ids = env._parse_proposal(solution_str)
		env.game.propose(None, env.game.turn_player, proposal_ids=proposal_ids)
		_ = env.game.proposal_response({"accept": True}, env.game.turn_player)
		best = float(env.game.best_assignment_reward) if env.game.best_assignment_reward else 1.0
		rew = float(env.game.proposal_reward) / max(best, 1e-6)
		return max(0.0, min(1.0, rew))
	except Exception:
		return 0.0 