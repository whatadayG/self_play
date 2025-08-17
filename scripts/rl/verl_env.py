import json
from dataclasses import dataclass
from typing import List, Dict, Any

from dialop.envs.optimization import OptimizationEnv


@dataclass
class MatchingEpisode:
	prompt: str
	ground_truth: Dict[str, Any]


def build_initial_prompt_from_game_state(game_state: Dict[str, Any]) -> str:
	"""Render the initial observation text for the current player as a single prompt string.
	We reuse OptimizationEnv to format the table and history into text.
	"""
	env = OptimizationEnv()
	obss = env.reset(game_state=game_state)
	current_player = obss["turn_player"]
	return obss[current_player]


def to_chat_messages(prompt: str) -> List[Dict[str, str]]:
	"""Convert a single-system style prompt to VERL chat format (system+user)."""
	return [
		{"role": "system", "content": "You are a helpful assistant for reviewer-paper matching."},
		{"role": "user", "content": prompt},
	]


def build_episode_from_line(line: str) -> MatchingEpisode:
	"""Parse a line from optimization.jsonl and produce a MatchingEpisode."""
	state = json.loads(line)
	prompt = build_initial_prompt_from_game_state(state)
	return MatchingEpisode(prompt=prompt, ground_truth=state) 