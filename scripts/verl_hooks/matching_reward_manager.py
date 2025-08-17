import json
from collections import defaultdict
from typing import Any, Optional, Callable

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Import local matching environment and game utilities
from dialop.envs.optimization import OptimizationEnv


def compute_score_matching(solution_str: str, ground_truth: dict, extra_info: dict | None = None) -> float | dict:
	"""
	Compute normalized reward for matching from a model response and ground truth game state.
	- Parses the response as a [propose] message content using OptimizationEnv's parsing logic
	- Computes proposal reward / best_assignment_reward
	Returns a float reward in [0, 1]. If parsing fails or invalid, returns 0.0.
	"""
	try:
		# Initialize env and game from ground truth state
		env = OptimizationEnv()
		from dialop.games.optimization import OptimizationGame
		env.game = OptimizationGame.create_from_game_state(ground_truth, one_player=False)
		# Parse proposal from the solution string
		proposal_ids = env._parse_proposal(solution_str)
		# Register proposal in game and compute reward
		env.game.propose(None, env.game.turn_player, proposal_ids=proposal_ids)
		# Accept to finalize and set proposal_reward
		_ = env.game.proposal_response({"accept": True}, env.game.turn_player)
		best = float(env.game.best_assignment_reward) if env.game.best_assignment_reward else 1.0
		rew = float(env.game.proposal_reward) / max(best, 1e-6)
		return max(0.0, min(1.0, rew))
	except Exception:
		return 0.0


@register("matching")
class MatchingRewardManager(AbstractRewardManager):
	"""Reward manager for matching; consumes raw chat and computes reward at final token."""

	def __init__(self, tokenizer, num_examine: int = 0, compute_score: Optional[Callable] = None, reward_fn_key: str = "reward_model", **kwargs):
		self.tokenizer = tokenizer
		self.num_examine = num_examine
		self._external_compute_score = compute_score  # ignored; we compute with env
		self._reward_fn_key = reward_fn_key

	def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
		reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
		reward_extra_info: dict[str, list] = defaultdict(list)

		already_print_data_sources: dict[str, int] = {}

		for i in range(len(data)):
			item = data[i]
			prompt_ids = item.batch["prompts"]
			prompt_len = prompt_ids.shape[-1]
			valid_prompt_len = item.batch["attention_mask"][:prompt_len].sum()
			valid_prompt_ids = prompt_ids[-valid_prompt_len:]

			response_ids = item.batch["responses"]
			valid_response_len = item.batch["attention_mask"][prompt_len:].sum()
			valid_response_ids = response_ids[:valid_response_len]

			prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
			response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

			# Ground truth carries the serialized game state
			gt = item.non_tensor_batch.get("reward_model", {}).get("ground_truth")
			if isinstance(gt, str):
				try:
					gt = json.loads(gt)
				except Exception:
					gt = None
			data_source = item.non_tensor_batch.get("data_source", "matching")

			reward = 0.0
			if gt is not None:
				reward = float(compute_score_matching(response_str, gt))

			reward_tensor[i, max(0, int(valid_response_len.item()) - 1)] = reward

			if data_source not in already_print_data_sources:
				already_print_data_sources[data_source] = 0
			if already_print_data_sources[data_source] < self.num_examine:
				already_print_data_sources[data_source] += 1
				print("[prompt]", prompt_str)
				print("[response]", response_str)
				print("[score]", reward)

		if return_dict:
			return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
		else:
			return reward_tensor 