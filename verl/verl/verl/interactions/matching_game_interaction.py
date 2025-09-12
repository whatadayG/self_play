# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import re
from typing import Any, Optional
from uuid import uuid4

# Add the dialop path to import OptimizationGame
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../../scripts'))

from dialop.games.optimization import OptimizationGame
from verl.interactions.base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MatchingGameInteraction(BaseInteraction):
    """An interaction wrapper for the OptimizationGame (reviewer-paper matching).
    
    This interaction:
    - Uses the existing OptimizationGame initialization
    - Handles multi-turn dialogue between two players
    - Computes normalized rewards when proposals are accepted
    - Manages game state across turns
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.max_turns = config.get("max_turns", 35)
        self.force_proposal_threshold = config.get("force_proposal_threshold", 5)  # Force proposal in last N turns

    async def start_interaction(
        self, 
        instance_id: Optional[str] = None, 
        game_state: Optional[dict] = None,
        ground_truth: Optional[float] = None,
        **kwargs
    ) -> str:
        """Initialize a game instance using OptimizationGame.
        
        Args:
            instance_id: Unique identifier for this game instance
            game_state: Dictionary containing the game state (from parquet extra_info)
                Should include: table, mask1, mask2, scale1, scale2, best_assignment_reward, action_log
            ground_truth: The best assignment reward (from reward_model in parquet)
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Create OptimizationGame instance from game state
        if game_state is not None:
            game = OptimizationGame.create_from_game_state(game_state, one_player=False)
        else:
            # Create a new random game
            game = OptimizationGame({}, one_player=False)
            game.reset()
        
        self._instance_dict[instance_id] = {
            "game": game,
            "turn_count": len(game.action_log),  # Resume from existing turns
            "proposal_made": False,
            "game_state": game_state,
            "ground_truth": ground_truth or game.best_assignment_reward
        }
        
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        """Process player responses and manage game flow.
        
        Returns:
            - should_terminate_sequence (bool): True if game should end
            - response_content (str): Feedback message to the player
            - current_turn_score (float): Normalized reward (0.0 during game, final reward on accept)
            - additional_data (dict): Extra game state information
        """
        game_data = self._instance_dict[instance_id]
        game = game_data["game"]
        
        # Get the last assistant message (the actual player response)
        last_msg = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_msg = messages[i].get("content", "")
                break
        
        # Increment turn count
        game_data["turn_count"] += 1
        
        # Check if we need to force a proposal
        turns_remaining = self.max_turns - game_data["turn_count"]
        force_proposal = turns_remaining <= self.force_proposal_threshold and not game_data["proposal_made"]
        
        # Parse the message for game actions
        if "[accept]" in last_msg.lower():
            # Calculate final reward
            reward = await self.calculate_score(instance_id)
            return True, "The proposal has been accepted! Game complete.", reward, {"final_reward": reward}
        
        elif "[reject]" in last_msg.lower():
            # Continue negotiation
            if turns_remaining <= 2:
                return False, "Proposal rejected. You must make a new proposal soon as the game is ending.", 0.0, {"force_proposal": True}
            return False, "Proposal rejected. Please continue the negotiation.", 0.0, {}
        
        elif "[propose]" in last_msg.lower():
            # Mark that a proposal has been made
            game_data["proposal_made"] = True
            return False, "Proposal made. The other player must now accept or reject.", 0.0, {}
        
        else:
            # Normal message
            if force_proposal:
                return False, "Time is running out. You should make a proposal soon.", 0.0, {"force_proposal": True}
            return False, "", 0.0, {}  # Empty response for normal conversation flow

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate normalized reward based on the current proposal.
        
        Returns normalized score between 0 and 1 based on proposal quality.
        """
        game_data = self._instance_dict[instance_id]
        game = game_data["game"]
        
        # Get proposal reward from the game
        proposal_reward = game.proposal_reward
        best_reward = game.best_assignment_reward
        
        # Normalize the reward
        if best_reward > 0:
            normalized_reward = proposal_reward / best_reward
        else:
            normalized_reward = 0.0
        
        return normalized_reward

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up the game instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]