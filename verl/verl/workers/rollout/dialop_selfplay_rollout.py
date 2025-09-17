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

"""
Custom rollout worker for dialop self-play using Player abstractions.

This rollout worker generates complete self-play games during each GRPO iteration,
ensuring both agents use the current policy via SGLangModelPlayer.
"""


import logging
import os
import sys
from typing import Any, Dict, List, Optional, Literal, Tuple
import json
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# Add dialop to path  
project_root = Path(__file__).parent.parent.parent.parent.parent
dialop_path = project_root / "scripts"
sys.path.insert(0, str(dialop_path))

from dialop.envs.optimization import OptimizationEnv
from dialop.sglang_model_player import SGLangModelPlayer
from dialop.base_player import SGLangConfig

from verl.workers.rollout.sglang_rollout import SGLangRollout
from verl.workers.rollout.schemas import AsyncRolloutRequest, AsyncRolloutRequestStateEnum, FinishReasonTypeEnum, TokenizationSanityCheckModeEnum
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "DEBUG"))

# Type aliases
PlayerName = Literal["player-1", "player-2"]
PlayerIndex = Literal[0, 1]


class MockConsole:
    """Mock console for SGLangModelPlayer that suppresses output during rollout."""
    def print(self, *args, **kwargs):
        pass
    
    def rule(self, *args, **kwargs):
        pass


class DialopSelfPlayRollout(SGLangRollout):
    """Rollout worker for dialop self-play using Player abstractions.
    
    This worker:
    1. Takes minimal game initializations as input
    2. Creates SGLangModelPlayer instances for both players
    3. Runs complete self-play games using Player abstractions
    4. Returns both players' perspectives as training data
    """
    
    def __init__(self, *args, max_turns=30, max_retries_per_turn=8, **kwargs):
        """Initialize the self-play rollout worker.
        
        Args:
            max_turns: Maximum number of turns per game (default: 30)
            max_retries_per_turn: Maximum retries for errors per turn (default: 8)
        """
        super().__init__(*args, **kwargs)
        logger.info(f"[DIALOP] Parent SGLangRollout initialized successfully")
        self.env_class = OptimizationEnv
        self.max_turns = max_turns
        self.max_retries_per_turn = max_retries_per_turn
        
        # Store the model path from initialization
        self.model_path = args[0] if args else "default"
        
        # Load game instructions
        instructions_path = project_root / "scripts" / "dialop" / "envs" / "data" / "optimization.txt"
        self.game_instructions = instructions_path.read_text().strip()
        
        # Create mock console for players
        self.console = MockConsole()
        
        logger.info(f"Initialized DialopSelfPlayRollout with max_turns={self.max_turns}, "
                   f"max_retries_per_turn={self.max_retries_per_turn}")
        
        # Track if we've logged debug data
        self._has_logged_debug = False
        
        # Output directory for debug logs
        self.output_dir = os.environ.get('VERL_OUTPUT_DIR', os.getcwd())
        
    def _create_players(self, game_state: Optional[Dict]) -> Dict[PlayerName, SGLangModelPlayer]:
        """Create SGLangModelPlayer instances for both players.
        
        Args:
            game_state: Initial game state (used to determine scale values)
            
        Returns:
            Dictionary mapping player names to SGLangModelPlayer instances
        """
        # Determine scale values for unknown_value placeholder
        scales = [1, 1]  # Default scales
        if game_state and "scales" in game_state:
            scales = game_state["scales"]
        
        players = {}
        for i, player_name in enumerate(["player-1", "player-2"]):
            # Replace unknown_value placeholder with scale-specific value
            unknown_value = int(50 * scales[i])
            instructions = self.game_instructions.replace("{unknown_value}", str(unknown_value))
            
            # Create SGLang config
            config = SGLangConfig(
                temperature=self.sampling_params['temperature'],
                max_tokens=self.sampling_params['max_new_tokens'],
                top_p=self.sampling_params['top_p'],
            )
            
            # Add server URL if available (external mode)
            if hasattr(self, 'sglang_url'):
                config.server_url = self.sglang_url
            
            # Use model name if available (external mode), otherwise use model path
            model_identifier = getattr(self, 'model_name', self.model_path)
            
            # Create player with internal engine or external server
            players[player_name] = SGLangModelPlayer(
                system_prompt=instructions,
                role=player_name,
                console=self.console,
                model_path=model_identifier,
                config=config,
                # Pass engine and processing class for internal mode
                engine=self._engine if hasattr(self, '_engine') else None,
                processing_class=self.processing_class if hasattr(self, 'processing_class') else None
            )
        
        return players
        
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate complete self-play games for each input.
        
        Args:
            prompts: DataProto containing game initializations
                  Expected fields in non_tensor_batch:
                  - game_state: Initial game state dict
                  - player_index: Which player's perspective (0 or 1)
                  
        Returns:
            DataProto with complete conversations and rewards
        """
        logger.info(f"[DIALOP] generate_sequences called with {len(prompts) if hasattr(prompts, '__len__') else 'unknown'} items")
        logger.info(f"[DIALOP] kwargs: {kwargs}")
        
        # Log the structure of the input
        if hasattr(prompts, 'batch') and hasattr(prompts.batch, 'keys'):
            logger.info(f"[DIALOP] Tensor batch keys: {list(prompts.batch.keys())}")
            for key, value in prompts.batch.items():
                if hasattr(value, 'shape'):
                    logger.info(f"[DIALOP]   {key}: shape {value.shape}")
        if hasattr(prompts, 'non_tensor_batch'):
            logger.info(f"[DIALOP] Non-tensor batch keys: {list(prompts.non_tensor_batch.keys())}")
            for key, value in prompts.non_tensor_batch.items():
                if hasattr(value, 'shape'):
                    logger.info(f"[DIALOP]   {key}: shape {value.shape}")
                elif hasattr(value, '__len__'):
                    logger.info(f"[DIALOP]   {key}: length {len(value)}")
        
        results = []
        
        # Group items by game_id to avoid running the same game multiple times
        game_groups = {}
        
        # Determine the length of prompts
        try:
            prompts_length = len(prompts)
            logger.info(f"[DIALOP] Processing {prompts_length} prompt items")
            
            # Check distributed environment
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                logger.info(f"[DIALOP] Distributed info: rank={rank}, world_size={world_size}, items_per_rank={prompts_length}")
            else:
                logger.info(f"[DIALOP] Not in distributed mode")
        except:
            logger.warning("[DIALOP] Cannot determine prompts length, treating as single item")
            prompts_length = 1
            
        for i in range(prompts_length):
            logger.debug(f"[DIALOP] Processing item {i}")
            try:
                item = prompts[i] if hasattr(prompts, '__getitem__') else prompts
                
                # Extract game_id
                game_id_data = item.non_tensor_batch.get("game_id", i)
                if hasattr(game_id_data, '__getitem__'):
                    game_id = int(game_id_data[0]) if len(game_id_data.shape) > 0 else int(game_id_data)
                else:
                    game_id = int(game_id_data)
                    
                if game_id not in game_groups:
                    game_groups[game_id] = []
                game_groups[game_id].append((i, item))
                logger.debug(f"[DIALOP] Added item {i} to game_id {game_id}")
            except Exception as e:
                logger.error(f"[DIALOP] Error processing item {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Run each unique game once
        game_results_cache = {}
        
        logger.info(f"[DIALOP] Running {len(game_groups)} unique games")
        
        for game_idx, (game_id, items) in enumerate(game_groups.items()):
            logger.info(f"[DIALOP] Running game {game_idx+1}/{len(game_groups)} (game_id: {game_id})")
            
            # Get game state from first item
            first_item = items[0][1]
            game_state_str = first_item.non_tensor_batch.get("game_state", None)
            
            # Handle numpy array extraction
            if hasattr(game_state_str, '__getitem__'):
                game_state_str = game_state_str[0] if len(game_state_str.shape) > 0 else game_state_str
                
            # Deserialize from JSON if it's a string
            if isinstance(game_state_str, (str, np.str_)):
                game_state = json.loads(game_state_str)
                if game_state == "null" or game_state is None:
                    game_state = None
            else:
                game_state = game_state_str
            
            # Run game once
            try:
                logger.info(f"[DIALOP] Starting self-play game {game_id}")
                game_result = self._run_selfplay_game(game_state)
                game_results_cache[game_id] = game_result
                logger.info(f"[DIALOP] Completed game {game_id} successfully")
            except Exception as e:
                logger.error(f"[DIALOP] Error in self-play game {game_id}: {e}")
                import traceback
                traceback.print_exc()
                game_results_cache[game_id] = None
        
        # Now create results for each player perspective
        logger.info(f"[DIALOP] Creating results for {prompts_length} player perspectives")
        
        for i in range(prompts_length):
            logger.debug(f"[DIALOP] Formatting result {i+1}/{prompts_length}")
            item = prompts[i] if hasattr(prompts, '__getitem__') else prompts
            
            # Get game_id and player_index
            game_id_data = item.non_tensor_batch.get("game_id", i)
            if hasattr(game_id_data, '__getitem__'):
                game_id = int(game_id_data[0]) if len(game_id_data.shape) > 0 else int(game_id_data)
            else:
                game_id = int(game_id_data)
                
            player_index_data = item.non_tensor_batch.get("player_index", 0)
            if hasattr(player_index_data, '__getitem__'):
                player_index = int(player_index_data[0]) if len(player_index_data.shape) > 0 else int(player_index_data)
            else:
                player_index = int(player_index_data)
            
            # Get cached game result
            game_result = game_results_cache.get(game_id)
            
            if game_result is not None:
                # Format result for the specified player's perspective
                player_name = f"player-{player_index + 1}"
                player_data = self._format_player_perspective(
                    game_result, player_name, player_index
                )
                results.append(player_data)
            else:
                # Create empty result on error
                results.append(self._create_empty_result())
                
        # Log debug information on first batch
        if not self._has_logged_debug and game_results_cache:
            logger.info(f"Logging debug information for first batch of {len(game_results_cache)} games")
            
            for game_id, game_result in game_results_cache.items():
                if game_result is not None:
                    # Log successful games
                    self._log_debug(game_result, game_id, success=True)
                else:
                    # Log failed games (minimal info)
                    self._log_failed_game(game_id)
            
            self._has_logged_debug = True
        
        # Create output DataProto
        logger.info(f"[DIALOP] Creating output DataProto from {len(results)} results")
        output = self._create_output_proto(results)
        logger.info(f"[DIALOP] generate_sequences completed successfully")
        return output
        
    def _run_selfplay_game(self, game_state: Optional[Dict]) -> Dict[str, Any]:
        logger.info(f"[DIALOP] _run_selfplay_game called with game_state: {game_state is not None}")
        """Run a complete self-play game using Player abstractions.
        
        Message flow pattern:
        1. Initial: Both players get system message + initial observation as user message
        2. Player 1 responds -> added as assistant message to Player 1's history
        3. Player 2 observes "Partner: [message]..." -> added as user message to Player 2's history  
        4. Player 2 responds -> added as assistant message to Player 2's history
        5. Player 1 observes "Partner: [message]..." -> added as user message to Player 1's history
        
        This ensures each player has the pattern:
        - First mover: system -> user (initial obs) -> assistant (response) -> user (partner msg) -> ...
        - Second mover: system -> user (initial obs) -> user (partner msg) -> assistant (response) -> ...
        
        Args:
            game_state: Initial game state or None for random
            
        Returns:
            Dictionary containing:
            - players: Dict[PlayerName, SGLangModelPlayer] with final player states
            - reward: Final game reward
            - normalized_reward: Normalized reward (0-1)
            - game_info: Additional game metadata
            - token_limit_hit: Whether game ended due to token limit
        """
        # Initialize environment
        logger.info(f"[DIALOP] Creating environment: {self.env_class}")
        env = self.env_class()
        logger.info(f"[DIALOP] Resetting environment...")
        obs = env.reset(game_state=game_state)
        logger.info(f"[DIALOP] Environment reset complete")
        
        # Create players
        logger.info(f"[DIALOP] Creating players...")
        players = self._create_players(game_state)
        logger.info(f"[DIALOP] Players created: {list(players.keys())}")
        
        # Give initial observations to players
        # At the start, both players need to see their view of the game
        for player_name in players:
            players[player_name].observe(obs[player_name])
        
        # Track conversations for debugging
        clean_conversation = []  # Only valid moves
        full_conversation = []   # All attempts including errors
        
        # Run game to completion
        done = False
        turn_count = 0
        token_limit_hit = False
        failed_due_to_max_retries = False
        
        logger.info(f"[DIALOP] Starting game loop (max_turns={self.max_turns})")
        
        while not done and turn_count < self.max_turns:
            # Get current player
            current_player_name = obs["turn_player"]
            current_player = players[current_player_name]
            logger.debug(f"[DIALOP] Turn {turn_count+1}: {current_player_name}'s turn")
            
            # Try to get a valid response with retries
            retries = 0
            valid_move = False
            
            while not valid_move:
                # Generate response using Player abstraction
                try:
                    logger.debug(f"[DIALOP] Generating response (attempt {retries+1})")
                    response = current_player.respond()
                    logger.debug(f"[DIALOP] Generated response: {response[:100]}...")
                    
                    # Add to full conversation
                    full_conversation.append({
                        "turn": turn_count,
                        "player": current_player_name,
                        "message": response,
                        "retry": retries
                    })
                    
                    # Check token limit using player's conversation length
                    # The player tracks input/output tokens during respond()
                    total_tokens = len(current_player.get_input_sequence())
                    logger.debug(f"[DIALOP] Total tokens so far: {total_tokens}")
                    if total_tokens > self.config.max_model_len - 500:  # Leave buffer
                        logger.info(f"Approaching token limit: {total_tokens} tokens")
                        token_limit_hit = True
                        done = True
                        break
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    token_limit_hit = True
                    done = True
                    break
                
                # Step environment
                obs, error = env.step(response)
                
                if error:
                    # Error occurred - current player needs to see error and retry
                    retries += 1
                    logger.debug(f"Game error at turn {turn_count}, retry {retries}/{self.max_retries_per_turn}")
                    logger.debug(f"Error observation for {current_player_name}: {obs[current_player_name]}")
                    
                    # Add error to full conversation
                    full_conversation.append({
                        "turn": turn_count,
                        "player": "error",
                        "message": obs[current_player_name],
                        "retry": retries
                    })
                    
                    # Give error observation to current player
                    current_player.observe(obs[current_player_name])
                    
                    if retries >= self.max_retries_per_turn:
                        # Max retries reached
                        logger.error(f"Max retries reached for {current_player_name} at turn {turn_count}")
                        done = True
                        failed_due_to_max_retries = True
                        break
                    
                    continue
                
                # Valid move - update all players with new observations
                valid_move = True
                done = obs["done"]
                
                # Add to clean conversation
                clean_conversation.append({
                    "turn": turn_count,
                    "player": current_player_name,
                    "message": response
                })
                
                # Give observations to OTHER players only
                # The current player already has their response in their message history
                for player_name in players:
                    if player_name != current_player_name and obs[player_name]:
                        # Other player gets the partner's message
                        logger.debug(f"Player {player_name} observing: {obs[player_name][:100]}...")
                        players[player_name].observe(obs[player_name])
                
                turn_count += 1
                
        # Extract final results
        if done and "info" in obs and not (failed_due_to_max_retries or token_limit_hit):
            # Game completed normally
            reward = obs["info"]["score"]
            normalized_reward = obs["info"]["score_norm"]
            num_messages = obs["info"]["num_msgs"]
        else:
            # Game didn't complete normally
            reward = 0.0
            normalized_reward = 0.0
            num_messages = turn_count
            
            # Log why the game ended
            if token_limit_hit:
                logger.info(f"Game ended due to token limit")
            elif turn_count >= self.max_turns:
                logger.info(f"Game ended due to max turns ({self.max_turns})")
            elif retries >= self.max_retries_per_turn:
                logger.info(f"Game ended due to max retries ({self.max_retries_per_turn})")
            
        return {
            "players": players,
            "reward": reward,
            "normalized_reward": normalized_reward,
            "game_info": {
                "num_messages": num_messages,
                "completed": done and not token_limit_hit, # NOTE: haven't been that careful to make sure that training handles completion totally correctly
                # at the very worst, what it would do is assign 0 reward to sequences which reach the token limit, which is not unreasonable
                # the entire issue is avoided if the token limit is large enough. also really long sequences imply a lot of errors anyway.
                "best_possible_reward": env.game.best_assignment_reward if hasattr(env.game, 'best_assignment_reward') else 0,
                "turn_count": turn_count,
                "token_limit_hit": token_limit_hit
            },
            "clean_conversation": clean_conversation,
            "full_conversation": full_conversation
        }
        
    def _format_player_perspective(
        self, 
        game_result: Dict[str, Any],
        player_name: PlayerName,
        player_index: PlayerIndex
    ) -> Dict[str, Any]:
        """Format game result from a specific player's perspective for training.
        
        Uses SGLangModelPlayer's built-in methods to get input sequences and masks.
        
        Args:
            game_result: Complete game result with player objects
            player_name: Name of the player ("player-1" or "player-2")
            player_index: Index of the player (0 or 1)
            
        Returns:
            Formatted data for training
        """
        player = game_result["players"][player_name]
        
        # Get the full input sequence from the player (already tokenized)
        input_ids = player.get_input_sequence()
        
        # Truncate if needed
        if len(input_ids) > self.config.max_model_len:
            logger.info(f"Truncating sequence from {len(input_ids)} to {self.config.max_model_len} tokens")
            input_ids = input_ids[:self.config.max_model_len]
        
        # Get assistant mask from player
        assistant_mask = player.get_assistant_mask()
        
        # Truncate mask to match input_ids
        if len(assistant_mask) > len(input_ids):
            assistant_mask = assistant_mask[:len(input_ids)]
        elif len(assistant_mask) < len(input_ids):
            # Pad with zeros if mask is shorter
            assistant_mask = assistant_mask + [0] * (len(input_ids) - len(assistant_mask))
        
        # Create other masks
        attention_mask = [1] * len(input_ids)
        
        # For dialop self-play: empty prompt, full sequence as response
        prompt_ids = []
        response_ids = input_ids
        prompt_mask = []
        response_mask = attention_mask
        response_loss_mask = assistant_mask  # Use the assistant mask for loss
        
        # Reward assignment (on last token)
        rewards = [0.0] * len(response_ids)
        if response_ids:  # Avoid index error on empty sequence
            rewards[-1] = game_result["normalized_reward"]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "prompt_ids": torch.tensor(prompt_ids),
            "response_ids": torch.tensor(response_ids), 
            "attention_mask": torch.tensor(attention_mask),
            "prompt_mask": torch.tensor(prompt_mask),
            "response_mask": torch.tensor(response_mask),
            "response_loss_mask": torch.tensor(response_loss_mask),
            "rewards": torch.tensor(rewards),
            "non_tensor_batch": {
                "reward_model": {
                    "normalized_reward": game_result["normalized_reward"],
                    "reward": game_result["reward"],
                },
                "game_info": game_result["game_info"],
                "player_index": player_index
            }
        }
        
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result for error cases."""
        return {
            "input_ids": torch.tensor([]),
            "prompt_ids": torch.tensor([]),
            "response_ids": torch.tensor([]),
            "attention_mask": torch.tensor([]),
            "prompt_mask": torch.tensor([]),
            "response_mask": torch.tensor([]),
            "response_loss_mask": torch.tensor([]),
            "rewards": torch.tensor([]),
            "non_tensor_batch": {
                "reward_model": {
                    "normalized_reward": 0.0,
                    "reward": 0.0,
                },
                "game_info": {
                    "error": True
                },
                "player_index": 0
            }
        }
        
    def _create_output_proto(self, results: List[Dict[str, Any]]) -> DataProto:
        """Create output DataProto from results."""
        logger.info(f"[DIALOP] _create_output_proto called with {len(results)} results")
        
        if not results:
            logger.warning("[DIALOP] No results to process, returning empty DataProto")
            return DataProto.empty()
            
        # Stack tensors - need to pad to same length
        from tensordict import TensorDict
        
        # Get pad token ID
        pad_token_id = 0  # Default pad token
        
        # Use configured lengths
        max_prompt_len = 1  # Minimal for empty prompts
        max_response_len = self.config.response_length if hasattr(self.config, 'response_length') else 30700
        max_len = max_prompt_len + max_response_len
        
        # Pad sequences
        padded_data = {
            "prompts": [],
            "responses": [],
            "response_mask": [],
            "input_ids": [],
            "attention_mask": [],
            "position_ids": [],
        }
        
        for r in results:
            # Handle empty prompts
            if len(r["prompt_ids"]) == 0:
                prompt_ids = torch.full((max_prompt_len,), pad_token_id)
                prompt_mask = torch.zeros(max_prompt_len)
            else:
                # This shouldn't happen in our case
                prompt_ids = r["prompt_ids"]
                prompt_mask = r["prompt_mask"]
            padded_data["prompts"].append(prompt_ids)
            
            # Pad responses
            response_len = len(r["response_ids"])
            if response_len > max_response_len:
                # Truncate
                response_ids = r["response_ids"][:max_response_len]
                response_mask = r["response_mask"][:max_response_len]
                response_loss_mask = r["response_loss_mask"][:max_response_len]
                # Move reward to new last position
                rewards = r["rewards"][:max_response_len]
                rewards[-1] = r["non_tensor_batch"]["reward_model"]["normalized_reward"]
            else:
                # Right pad
                pad_len = max_response_len - response_len
                response_ids = torch.cat([r["response_ids"], torch.full((pad_len,), pad_token_id)])
                response_mask = torch.cat([r["response_mask"], torch.zeros(pad_len)])
                response_loss_mask = torch.cat([r["response_loss_mask"], torch.zeros(pad_len)])
                rewards = torch.cat([r["rewards"], torch.zeros(pad_len)])
            
            padded_data["responses"].append(response_ids)
            padded_data["response_mask"].append(response_loss_mask)
            
            # Combine prompt and response
            input_ids = torch.cat([prompt_ids, response_ids])
            attention_mask = torch.cat([prompt_mask, response_mask])
            padded_data["input_ids"].append(input_ids)
            padded_data["attention_mask"].append(attention_mask)
            
            # Position ids
            position_ids = torch.arange(max_len)
            padded_data["position_ids"].append(position_ids)
        
        # Stack all tensors
        batch_dict = {k: torch.stack(v) for k, v in padded_data.items()}
        batch = TensorDict(batch_dict, batch_size=[len(results)])
        
        # Collect non-tensor data
        non_tensor_batch = {
            "reward_model": np.array([r["non_tensor_batch"]["reward_model"] for r in results], dtype=object),
            "game_info": np.array([r["non_tensor_batch"]["game_info"] for r in results], dtype=object),
        }
        
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch
        )
    
    def _format_clean_conversation(self, conversation: List[Dict]) -> str:
        """Format clean conversation as a string from third-person perspective."""
        lines = ["=== Clean Conversation (Valid Moves Only) ==="]
        for entry in conversation:
            lines.append(f"Turn {entry['turn']} - {entry['player']}: {entry['message']}")
        return "\n".join(lines)
    
    def _format_full_conversation(self, conversation: List[Dict]) -> str:
        """Format full conversation including errors as a string."""
        lines = ["=== Full Conversation (Including Errors) ==="]
        for entry in conversation:
            if entry['player'] == 'error':
                lines.append(f"Turn {entry['turn']} - ERROR (retry {entry['retry']}): {entry['message']}")
            else:
                retry_str = f" (retry {entry['retry']})" if entry['retry'] > 0 else ""
                lines.append(f"Turn {entry['turn']} - {entry['player']}{retry_str}: {entry['message']}")
        return "\n".join(lines)
    
    def _log_debug(self, game_result: Dict[str, Any], game_id: int, success: bool = True) -> None:
        """Log comprehensive debug information for one game."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dialop_selfplay_debug_{timestamp}_game_{game_id}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("DIALOP SELF-PLAY DEBUG LOG\n")
                f.write(f"Generated at: {timestamp}\n")
                f.write(f"Game ID: {game_id}\n")
                f.write(f"Success: {success}\n")
                f.write(f"Reward: {game_result['reward']}\n")
                f.write(f"Normalized Reward: {game_result['normalized_reward']}\n")
                f.write(f"Turn Count: {game_result['game_info']['turn_count']}\n")
                f.write(f"Completed: {game_result['game_info']['completed']}\n")
                f.write("\n" + "="*80 + "\n\n")
                
                # Clean and full conversations
                f.write(self._format_clean_conversation(game_result['clean_conversation']))
                f.write("\n\n" + "="*80 + "\n\n")
                f.write(self._format_full_conversation(game_result['full_conversation']))
                f.write("\n\n" + "="*80 + "\n\n")
                
                # Player perspectives
                for player_name, player in game_result['players'].items():
                    f.write(f"\n=== {player_name.upper()} PERSPECTIVE ===\n\n")
                    
                    # Pretty conversation as seen by player
                    f.write(f"--- Conversation History ---\n")
                    f.write(player.get_pretty_conversation())
                    f.write("\n\n")
                    
                    # Input string
                    f.write(f"--- Input String (get_input_string()) ---\n")
                    try:
                        f.write(player.get_input_string())
                    except Exception as e:
                        f.write(f"Error getting input string: {e}")
                    f.write("\n\n")
                    
                    # Masked sequences
                    f.write(f"--- Masked Sequences (get_masked_sequences_pretty()) ---\n")
                    try:
                        f.write(player.get_masked_sequences_pretty())
                    except Exception as e:
                        f.write(f"Error getting masked sequences: {e}")
                    f.write("\n\n" + "="*80 + "\n")
            
            logger.info(f"Saved debug log to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save debug log: {e}")
            import traceback
            traceback.print_exc()
    
    def _log_failed_game(self, game_id: int) -> None:
        """Log minimal information for a failed game."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dialop_selfplay_debug_{timestamp}_game_{game_id}_FAILED.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("DIALOP SELF-PLAY DEBUG LOG - FAILED GAME\n")
                f.write(f"Generated at: {timestamp}\n")
                f.write(f"Game ID: {game_id}\n")
                f.write(f"Status: FAILED\n")
                f.write("\nThis game failed to generate. Check logs for error details.\n")
            
            logger.info(f"Saved failed game log to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save failed game log: {e}")
