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
import asyncio
from typing import Any, Optional
from uuid import uuid4

# Add dialop to path for imports
dialop_path = os.path.join(os.path.dirname(__file__), '../../../../dialop')
if os.path.exists(dialop_path):
    sys.path.insert(0, dialop_path)

try:
    from dialop.envs.optimization import OptimizationEnv
    from dialop.envs.base_env import GameError
except ImportError:
    # Try alternative path
    alt_dialop_path = os.path.join(os.path.dirname(__file__), '../../../../../dialop')
    if os.path.exists(alt_dialop_path):
        sys.path.insert(0, alt_dialop_path)
        from dialop.envs.optimization import OptimizationEnv
        from dialop.envs.base_env import GameError
    else:
        raise ImportError("Could not find dialop module. Make sure PYTHONPATH includes dialop directory.")

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DialopSelfplayInteraction(BaseInteraction):
    """A self-play interaction for dialop optimization environment.
    
    This interaction manages a conversation between two agents (both the same LLM)
    playing the dialop optimization game. The interaction:
    - Manages the dialop environment state
    - Converts between dialop message format and chat format
    - Determines when conversations should terminate
    - Computes rewards for both agents
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self._env_class = OptimizationEnv  # Can be made configurable later
        
    async def start_interaction(
        self, instance_id: Optional[str] = None, game_state: Optional[dict] = None, **kwargs
    ) -> str:
        """Initialize a new dialop game instance.
        
        Args:
            instance_id: Unique ID for this interaction instance
            game_state: Optional pre-existing game state to load
            
        Returns:
            The instance ID
        """
        if instance_id is None:
            instance_id = str(uuid4())
            
        # Create new environment
        env = self._env_class()
        obs = env.reset(game_state=game_state)
        
        # Store instance data
        self._instance_dict[instance_id] = {
            "env": env,
            "current_player": obs["turn_player"],
            "done": obs["done"],
            "reward": 0.0,
            "conversation_history": [],
            "game_state": game_state,
        }
        
        return instance_id
    
    def _parse_assistant_message(self, content: str) -> tuple[str, str]:
        """Parse assistant response to extract message type and content.
        
        Args:
            content: Raw assistant message
            
        Returns:
            (message_type, message_content) tuple
        """
        content = content.strip()
        
        # Check for message types
        if content.startswith("[message]"):
            return "message", content[len("[message]"):].strip()
        elif content.startswith("[propose]"):
            return "propose", content[len("[propose]"):].strip()
        elif content.startswith("[accept]"):
            return "accept", ""
        elif content.startswith("[reject]"):
            return "reject", ""
        else:
            # Default to message type if no type specified
            return "message", content
            
    def _format_dialop_message(self, msg_type: str, content: str) -> str:
        """Format message in dialop's expected format."""
        if msg_type in ["accept", "reject"]:
            return f"[{msg_type}]"
        else:
            return f"[{msg_type}] {content}"
    
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """Process the current agent's response and get next player's observation.
        
        This method:
        1. Extracts the last assistant message 
        2. Converts it to dialop format
        3. Steps the environment
        4. Returns the next player's observation
        
        Args:
            instance_id: The instance ID
            messages: Current conversation history
            
        Returns:
            (should_terminate, response_content, turn_score, additional_data)
        """
        instance = self._instance_dict[instance_id]
        env = instance["env"]
        
        # Extract last assistant message
        assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break
                
        if assistant_msg is None:
            # This is the first turn - return the initial observation
            current_player = instance["current_player"]
            initial_obs = env._init_from_action_log()[current_player]
            return False, initial_obs, 0.0, {"player": current_player}
        
        # Parse the assistant's message
        msg_type, content = self._parse_assistant_message(assistant_msg)
        formatted_msg = self._format_dialop_message(msg_type, content)
        
        # Step the environment
        try:
            obs, error = env.step(formatted_msg)
            
            if error:
                # Error occurred - return the error message for the current player
                # The environment already formatted the error message for the current player
                current_player = obs["turn_player"]
                error_obs = obs.get(current_player, "")
                
                # Don't update conversation history for failed moves
                return False, error_obs, 0.0, {"error": True, "player": current_player}
                
            # Get the player who made the move (before turn switches)
            move_player = instance["current_player"]
            
            # Update instance state for successful moves
            instance["current_player"] = obs["turn_player"]
            instance["done"] = obs["done"]
            instance["conversation_history"].append({
                "player": env.players.index(move_player),  # Use the player who made the move
                "message": formatted_msg
            })
            
            if obs["done"]:
                # Game is over - compute final reward
                reward = obs["info"]["score"]
                normalized_reward = obs["info"]["score_norm"]
                instance["reward"] = normalized_reward
                
                # Return empty response to signal completion
                return True, "", normalized_reward, {
                    "done": True,
                    "raw_reward": reward,
                    "normalized_reward": normalized_reward,
                    "num_messages": obs["info"]["num_msgs"]
                }
            else:
                # Get next player's observation
                next_player = obs["turn_player"]
                next_obs = obs[next_player]
                
                # Return the observation for the next turn
                return False, next_obs, 0.0, {"player": next_player}
                
        except GameError as e:
            # Handle game errors by returning error for current player
            # This shouldn't happen if env.step properly catches GameErrors
            current_player = instance["current_player"]
            error_msg = f"Game Error: {str(e)}"
            return False, error_msg, 0.0, {"error": True, "game_error": str(e), "player": current_player}
            
    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Return the final normalized reward for the game."""
        instance = self._instance_dict.get(instance_id, {})
        return instance.get("reward", 0.0)
        
    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up the instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]