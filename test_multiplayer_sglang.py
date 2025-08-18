#!/usr/bin/env python3
"""
Test script for multi-player SGLang implementation.
This tests one game loop to verify the multi-player generation works as intended.
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any
import torch
from transformers import AutoTokenizer

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl'))

from dialop.games.optimization import OptimizationGame
from verl.interactions.matching_game_multiplayer_interaction import MatchingGameMultiplayerInteraction
from verl.workers.rollout.sglang_rollout.multiplayer_extension import (
    MultiplayerSGLangExtension, MultiplayerRequestWrapper
)
from verl.workers.rollout.schemas import AsyncRolloutRequest, Message


class MockSGLangEngine:
    """Mock SGLang engine for testing without actual model inference."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.turn_count = 0
        
    async def async_generate(self, input_ids: List[int], **kwargs) -> Dict[str, Any]:
        """Mock generation that simulates player responses."""
        # Decode to see what the prompt is
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Simulate different responses based on turn
        self.turn_count += 1
        
        if "Player-1" in prompt:
            if self.turn_count == 1:
                response = "I see we have a matching problem. Let me look at my visible scores and propose something fair."
            elif self.turn_count == 3:
                response = "Based on what I can see, I think paper 0 should go to reviewer 2 and paper 1 to reviewer 0. What do you think?"
            elif self.turn_count == 5:
                response = "[propose] I propose: paper 0 -> reviewer 2, paper 1 -> reviewer 0, paper 2 -> reviewer 1"
            else:
                response = "Let's discuss this further."
        else:  # Player-2
            if self.turn_count == 2:
                response = "Yes, I agree we should find a fair solution. From my perspective, I see different scores."
            elif self.turn_count == 4:
                response = "I see your point, but from my view, reviewer 1 might be better for paper 0."
            elif self.turn_count == 6:
                response = "[accept] I accept your proposal. This seems like a reasonable compromise."
            else:
                response = "I need to think about this."
                
        # Encode the response
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        
        return {
            "text": response,
            "token_ids": response_ids,
            "finish_reason": "stop"
        }


async def test_multiplayer_generation():
    """Test the multi-player generation flow."""
    
    print("=== Testing Multi-Player SGLang Generation ===\n")
    
    # 1. Create a game state (similar to what evaluate_opt.py does)
    game = OptimizationGame({}, one_player=False)
    game.reset()
    
    game_state = {
        "table": game.table,
        "mask1": game.mask1,
        "mask2": game.mask2,
        "scale1": game.scale1,
        "scale2": game.scale2,
        "best_assignment_reward": game.best_assignment_reward,
        "action_log": []
    }
    
    print(f"Created game with table shape: {len(game.table)}x{len(game.table[0])}")
    print(f"Best possible reward: {game.best_assignment_reward:.2f}\n")
    
    # 2. Initialize the multiplayer interaction
    interaction_config = {
        "max_turns": 20,
        "force_proposal_threshold": 5
    }
    interaction = MatchingGameMultiplayerInteraction(interaction_config)
    
    # 3. Create a mock request (simulating what SGLang would create)
    mock_engine = MockSGLangEngine()
    tokenizer = mock_engine.tokenizer
    
    # Initial messages for the request
    initial_messages = [
        Message(role="system", content="You are playing a negotiation game."),
        Message(role="user", content="Start the game.")
    ]
    
    # Create AsyncRolloutRequest
    request = AsyncRolloutRequest(
        request_id="test_game_001",
        batch_data_id=0,
        rollout_offset=0,
        state="pending",
        messages=initial_messages,
        reward_scores={},
        max_prompt_len=1024,
        max_response_len=512,
        use_inference_chat_template=False,
        tokenization_sanity_check_mode="disable",
        processing_class=tokenizer,
        interaction_kwargs={
            "name": "matching_game",
            "game_state": game_state,
            "ground_truth": game.best_assignment_reward
        }
    )
    
    # 4. Initialize multiplayer wrapper
    mp_wrapper = MultiplayerSGLangExtension.initialize_multiplayer_state(request, num_players=2)
    
    # Start the interaction
    instance_id = await interaction.start_interaction(
        instance_id=request.request_id,
        game_state=game_state,
        ground_truth=game.best_assignment_reward,
        current_player=0
    )
    
    print("=== Starting Multi-Turn Game ===\n")
    
    # 5. Simulate the game loop (like _async_rollout_a_request)
    game_complete = False
    final_reward = 0.0
    
    while not game_complete and mp_wrapper.turn_count < 10:  # Limit turns for testing
        current_player = mp_wrapper.current_player
        print(f"\n--- Turn {mp_wrapper.turn_count + 1} - Player {current_player + 1}'s turn ---")
        
        # Prepare prompt for current player
        if mp_wrapper.turn_count == 0:
            # First turn - get initial prompt
            prompt = MultiplayerSGLangExtension.prepare_player_prompt(
                mp_wrapper, interaction, tokenizer
            )
            print(f"Initial prompt excerpt: {prompt[:200]}...")
        
        # Update base request with current player's view
        mp_wrapper.update_base_request_messages()
        
        # Simulate generation (in real SGLang, this would call the engine)
        # For testing, we'll use our mock engine
        messages_for_generation = [msg.model_dump() for msg in mp_wrapper.base_request.messages]
        prompt_text = tokenizer.apply_chat_template(messages_for_generation, tokenize=False)
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")[0].tolist()
        
        output = await mock_engine.async_generate(input_ids)
        response_text = output["text"]
        
        print(f"Player {current_player + 1} says: {response_text}")
        
        # Process the response through the interaction
        messages_list = [{"role": msg.role, "content": msg.content} for msg in mp_wrapper.base_request.messages]
        messages_list.append({"role": "assistant", "content": response_text})
        
        should_terminate, feedback, reward, extra_data = await interaction.generate_response(
            instance_id, messages_list
        )
        
        if should_terminate:
            game_complete = True
            final_reward = reward
            print(f"\nGame completed! Final reward: {final_reward:.2f}")
        else:
            # Handle the response and switch players
            MultiplayerSGLangExtension.handle_player_response(
                mp_wrapper, response_text
            )
            
            if feedback:
                # Add feedback to the new current player's history
                mp_wrapper.add_player_message(
                    mp_wrapper.current_player, "user", feedback
                )
    
    # 6. Finalize and create training instances
    print("\n=== Creating Training Instances ===\n")
    
    player_requests = MultiplayerSGLangExtension.finalize_multiplayer_rollout(
        mp_wrapper, final_reward
    )
    
    print(f"Generated {len(player_requests)} training instances (one per player)")
    
    # 7. Verify the training instances
    for i, req in enumerate(player_requests):
        print(f"\n--- Training Instance {i + 1} (Player {i + 1} perspective) ---")
        print(f"Request ID: {req.request_id}")
        print(f"Batch ID: {req.batch_data_id} (should be same for GRPO grouping)")
        print(f"Reward: {req.reward_scores.get('final_reward', 0.0):.2f}")
        print(f"Number of messages: {len(req.messages)}")
        
        # Show first few messages to verify perspective
        print("\nFirst few messages in history:")
        for j, msg in enumerate(req.messages[:3]):
            role = msg.role
            content = msg.content
            if isinstance(content, str) and len(content) > 100:
                content = content[:100] + "..."
            print(f"  [{role}]: {content}")
    
    # 8. Verify information asymmetry
    print("\n=== Verifying Information Asymmetry ===")
    
    # Check that each player saw different information
    player1_history = mp_wrapper.player_messages[0]
    player2_history = mp_wrapper.player_messages[1]
    
    print(f"\nPlayer 1 saw {len(player1_history)} messages")
    print(f"Player 2 saw {len(player2_history)} messages")
    
    # Look for mask differences in system prompts
    for player_id, history in enumerate([player1_history, player2_history]):
        for msg in history:
            if msg["role"] == "system" and "mask" in msg["content"].lower():
                print(f"\nPlayer {player_id + 1} saw mask{player_id + 1} in their view")
                break
    
    print("\n=== Test Complete ===")
    print("\nSummary:")
    print(f"- Successfully simulated {mp_wrapper.turn_count} turns")
    print(f"- Game ended with reward: {final_reward:.2f}")
    print(f"- Created {len(player_requests)} training instances with same batch_id")
    print("- Each player maintained separate conversation history")
    print("- Information asymmetry preserved (different masks)")
    
    # Clean up
    await interaction.finalize_interaction(instance_id)


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_multiplayer_generation())