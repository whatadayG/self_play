#!/usr/bin/env python3
"""
Simple test for multi-player logic without async complexity.
Tests the core multiplayer state management and perspective switching.
"""

import os
import sys
import json
from typing import Dict, List, Any

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'verl'))

from dialop.games.optimization import OptimizationGame
from verl.workers.rollout.sglang_rollout.multiplayer_extension import (
    MultiplayerRequestWrapper, MultiplayerSGLangExtension
)
from verl.workers.rollout.schemas import AsyncRolloutRequest, Message


def create_mock_request(game_state: Dict[str, Any]) -> AsyncRolloutRequest:
    """Create a mock AsyncRolloutRequest for testing."""
    # Mock processing class
    class MockProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "mock_prompt"
        def encode(self, text, **kwargs):
            return [1, 2, 3, 4, 5]
        def decode(self, ids, **kwargs):
            return "mock_decoded"
    
    initial_messages = [
        Message(role="system", content="You are playing a negotiation game."),
        Message(role="user", content="Start the game.")
    ]
    
    # Create a minimal request
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
        processing_class=MockProcessor(),
        interaction_kwargs={
            "name": "matching_game",
            "game_state": game_state,
            "ground_truth": 100.0
        }
    )
    
    return request


def test_multiplayer_state_management():
    """Test the core multiplayer state management."""
    
    print("=== Testing Multi-Player State Management ===\n")
    
    # 1. Create a game
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
    
    print(f"Created game with:")
    print(f"- Table shape: {len(game.table)}x{len(game.table[0])}")
    print(f"- Player 1 scale: {game.scale1}")
    print(f"- Player 2 scale: {game.scale2}")
    print(f"- Best reward: {game.best_assignment_reward:.2f}\n")
    
    # 2. Create multiplayer wrapper
    request = create_mock_request(game_state)
    mp_wrapper = MultiplayerRequestWrapper(request, num_players=2)
    
    print("Initial state:")
    print(f"- Current player: {mp_wrapper.current_player}")
    print(f"- Turn count: {mp_wrapper.turn_count}")
    print(f"- Player messages: {mp_wrapper.player_messages}\n")
    
    # 3. Simulate a conversation
    print("=== Simulating Conversation ===\n")
    
    # Turn 1: Player 0 speaks
    print("Turn 1 - Player 0's turn:")
    mp_wrapper.add_player_message(0, "system", "You are Player-1. You see mask1.")
    mp_wrapper.add_player_message(0, "user", "Start negotiating.")
    
    player0_response = "I propose we start by looking at the highest scores I can see."
    print(f"Player 0 says: {player0_response}")
    
    MultiplayerSGLangExtension.handle_player_response(
        mp_wrapper, 
        player0_response,
        f"Player-1 says: {player0_response}"
    )
    
    print(f"After turn 1:")
    print(f"- Current player: {mp_wrapper.current_player}")
    print(f"- Turn count: {mp_wrapper.turn_count}")
    print(f"- Player 0 messages: {len(mp_wrapper.player_messages[0])}")
    print(f"- Player 1 messages: {len(mp_wrapper.player_messages[1])}\n")
    
    # Turn 2: Player 1 speaks
    print("Turn 2 - Player 1's turn:")
    mp_wrapper.add_player_message(1, "system", "You are Player-2. You see mask2.")
    
    player1_response = "I agree. From my perspective, I see different values."
    print(f"Player 1 says: {player1_response}")
    
    MultiplayerSGLangExtension.handle_player_response(
        mp_wrapper,
        player1_response,
        f"Player-2 says: {player1_response}"
    )
    
    print(f"After turn 2:")
    print(f"- Current player: {mp_wrapper.current_player}")
    print(f"- Turn count: {mp_wrapper.turn_count}")
    print(f"- Player 0 messages: {len(mp_wrapper.player_messages[0])}")
    print(f"- Player 1 messages: {len(mp_wrapper.player_messages[1])}\n")
    
    # 4. Continue for a few more turns
    for turn in range(3, 6):
        current = mp_wrapper.current_player
        response = f"Turn {turn} response from Player {current}"
        
        MultiplayerSGLangExtension.handle_player_response(
            mp_wrapper,
            response,
            f"Player-{current + 1} says: {response}"
        )
    
    # 5. Show final conversation histories
    print("=== Final Conversation Histories ===\n")
    
    for player_id in range(2):
        print(f"Player {player_id} perspective:")
        for msg in mp_wrapper.player_messages[player_id]:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 80:
                content = content[:80] + "..."
            print(f"  [{role:9}] {content}")
        print()
    
    # 6. Test creating training instances
    print("=== Creating Training Instances ===\n")
    
    # Set a final reward
    final_reward = 0.85
    player_requests = MultiplayerSGLangExtension.finalize_multiplayer_rollout(
        mp_wrapper, final_reward
    )
    
    print(f"Created {len(player_requests)} training instances")
    
    for i, req in enumerate(player_requests):
        print(f"\nTraining instance {i + 1}:")
        print(f"- Request ID: {req.request_id}")
        print(f"- Batch ID: {req.batch_data_id} (same for GRPO)")
        print(f"- Player perspective: {req.interaction_kwargs.get('player_perspective', 'unknown')}")
        print(f"- Reward: {req.reward_scores.get('final_reward', 0.0):.2f}")
        print(f"- Messages: {len(req.messages)}")
    
    # 7. Verify batch IDs are the same
    batch_ids = [req.batch_data_id for req in player_requests]
    if len(set(batch_ids)) == 1:
        print("\n✓ All training instances have the same batch_id (good for GRPO)")
    else:
        print("\n✗ Training instances have different batch_ids (bad for GRPO)")
    
    # 8. Verify rewards are the same
    rewards = [req.reward_scores.get('final_reward', 0.0) for req in player_requests]
    if len(set(rewards)) == 1:
        print("✓ All training instances have the same reward")
    else:
        print("✗ Training instances have different rewards")
    
    print("\n=== Test Complete ===")


def test_information_asymmetry():
    """Test that information asymmetry is maintained."""
    
    print("\n=== Testing Information Asymmetry ===\n")
    
    # Create a small game for clarity
    game = OptimizationGame({}, one_player=False)
    game.reset()
    
    print("Game masks:")
    print(f"Mask1 (Player 1 sees): {game.mask1}")
    print(f"Mask2 (Player 2 sees): {game.mask2}")
    
    # Verify masks are different
    mask1_flat = [val for row in game.mask1 for val in row]
    mask2_flat = [val for row in game.mask2 for val in row]
    
    overlap = sum(1 for m1, m2 in zip(mask1_flat, mask2_flat) if m1 and m2)
    total = len(mask1_flat)
    
    print(f"\nMask statistics:")
    print(f"- Total cells: {total}")
    print(f"- Player 1 sees: {sum(mask1_flat)} cells")
    print(f"- Player 2 sees: {sum(mask2_flat)} cells")
    print(f"- Overlap: {overlap} cells")
    print(f"- Information asymmetry: {100 * (1 - overlap/total):.1f}%")


if __name__ == "__main__":
    test_multiplayer_state_management()
    test_information_asymmetry()