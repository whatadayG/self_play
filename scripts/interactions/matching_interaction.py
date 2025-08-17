import json
import os
from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import OptimizationGame


class MatchingInteraction(BaseInteraction):
	"""Multi-turn interaction for matching.
	- Keeps an OptimizationEnv per instance, seeded from ground truth game_state.
	- Policy controls one player (policy_player). The partner is implicit in env observations.
	- Termination: full proposal accepted or max_turns reached.
	"""

	def __init__(self, config: dict):
		super().__init__(config)
		self._instance_dict: dict[str, dict[str, Any]] = {}
		self.max_turns: int = int(config.get("max_turns", 12))
		self.require_proposal: bool = bool(config.get("require_proposal", True))
		self.accept_on_full_proposal: bool = bool(config.get("accept_on_full_proposal", True))
		self.policy_player: str = str(config.get("policy_player", "player-1"))

	async def start_interaction(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
		if instance_id is None:
			instance_id = str(uuid4())
		# Build env from ground truth
		if isinstance(ground_truth, str):
			game_state = json.loads(ground_truth)
		else:
			game_state = ground_truth
		env = OptimizationEnv()
		env.game = OptimizationGame.create_from_game_state(game_state, one_player=False)
		# Prepare first observation
		obss = {
			"player-1": "",
			"player-2": "",
			"turn_player": f"player-{env.game.turn_player+1}",
			"done": False,
		}
		obss = env.reset(game_state=game_state)
		self._instance_dict[instance_id] = {
			"env": env,
			"turn": 0,
			"last_obs": obss,
			"reward": 0.0,
		}
		return instance_id

	async def generate_response(self, instance_id: str, messages: list[dict[str, Any]], **kwargs) -> tuple[bool, str, float, dict]:
		state = self._instance_dict[instance_id]
		env: OptimizationEnv = state["env"]
		turn = state["turn"]
		obss = state["last_obs"]

		# Extract assistant content (policy response) from messages
		content = ""
		for i in range(len(messages) - 1, -1, -1):
			item = messages[i]
			if item.get("role") == "assistant":
				content = item.get("content", "")
				break

		# Step env with the policy response. Force thinking structure via user_think=True
		new_obss, resample = env.step(content, user_think=True)
		# Termination conditions
		done = new_obss.get("done", False)
		reward = 0.0
		if done and env.game.proposal is not None and self.accept_on_full_proposal:
			best = float(env.game.best_assignment_reward) if env.game.best_assignment_reward else 1.0
			reward = float(env.game.proposal_reward) / max(best, 1e-6)

		turn += 1
		should_stop = bool(done) or (turn >= self.max_turns)
		state["turn"] = turn
		state["last_obs"] = new_obss
		state["reward"] = reward

		# Provide textual feedback to continue or stop
		if should_stop:
			response = "Conversation finished."
		else:
			# In multi-turn, we return an instruction to continue
			response = "Continue the conversation with a valid next turn."
		return should_stop, response, reward, {}

	async def calculate_score(self, instance_id: str, **kwargs) -> float:
		return float(self._instance_dict[instance_id]["reward"])

	async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
		if instance_id in self._instance_dict:
			del self._instance_dict[instance_id] 