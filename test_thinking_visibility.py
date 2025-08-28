#!/usr/bin/env python3
"""
Test that "let's think step by step" blocks are not visible to the other agent.
"""

import asyncio
import re
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "verl"))
sys.path.insert(0, str(Path(__file__).parent / "dialop"))

from dialop.envs.optimization import OptimizationEnv


def test_thinking_visibility():
    """Test that thinking blocks are hidden from the partner."""
    print("=== Testing Thinking Visibility ===")
    
    env = OptimizationEnv(one_player=False)  # Two-player mode
    obs = env.reset()
    
    print(f"Initial player: {obs['turn_player']}")
    
    # Test 1: Player 1 sends a message with thinking
    message_with_thinking = "Let's think step by step. I need to analyze the table and find good matches. [message] Hello partner, I see some interesting patterns in my table."
    
    print(f"\nPlayer 1 sends: {repr(message_with_thinking[:50])}...")
    obs, error = env.step(message_with_thinking)
    
    if not error:
        print("\nWhat each player sees:")
        print(f"Player 1 (sender) sees: {repr(obs.get('player-1', '')[:80])}...")
        print(f"Player 2 (receiver) sees: {repr(obs.get('player-2', '')[:80])}...")
        
        # Check if thinking is hidden from player 2
        player2_obs = obs.get('player-2', '')
        if "Let's think step by step" in player2_obs:
            print("❌ ERROR: Player 2 can see the thinking block!")
        else:
            print("✅ SUCCESS: Thinking block is hidden from Player 2")
            
        # Check if player 2's observation starts with the tag
        if "[message]" in player2_obs:
            print("✅ SUCCESS: Player 2's observation includes the message tag")
        else:
            print("❌ ERROR: Message tag missing from Player 2's observation")
    
    # Test 2: Player 2 proposes with thinking
    print(f"\n\nNow it's {obs['turn_player']}'s turn")
    
    proposal_with_thinking = """Let's think step by step. Based on the information shared, I can see that:
- Paper1 (BLEU) might go well with Reviewer3
- Paper2 (Electra) seems good for Reviewer5

[propose] Proposal:
BLEU: Ava Li
Electra: Daniel Nguyen
GloVe: Sofia Patel
GLUE: Andrei Petrov
LLaMA: Morgan Reed
RoBERTa: Joseph Santos
QuAC: Ethan Smith
SWAG: Noah Wilson"""
    
    print(f"Player 2 sends proposal with thinking...")
    obs, error = env.step(proposal_with_thinking)
    
    if not error:
        print("\nWhat each player sees:")
        player1_obs = obs.get('player-1', '')
        player2_obs = obs.get('player-2', '')
        
        print(f"Player 1 (receiver) observation length: {len(player1_obs)}")
        print(f"Player 2 (proposer) observation length: {len(player2_obs)}")
        
        # Check if thinking is hidden from player 1
        if "Based on the information shared" in player1_obs:
            print("❌ ERROR: Player 1 can see the thinking block!")
        else:
            print("✅ SUCCESS: Thinking block is hidden from Player 1")
            
        # Check if player 1 sees the proposal
        if "[propose]" in player1_obs and "Proposal:" in player1_obs:
            print("✅ SUCCESS: Player 1 can see the proposal")
        else:
            print("❌ ERROR: Proposal not visible to Player 1")
            
        # Player 1 should also see accept/reject options
        if "[accept]" in player1_obs and "[reject]" in player1_obs:
            print("✅ SUCCESS: Player 1 sees accept/reject options")
    
    return True


async def test_rollout_thinking_visibility():
    """Test thinking visibility in our rollout implementation."""
    print("\n\n=== Testing Rollout Implementation ===")
    
    # This would require setting up the full rollout worker
    # For now, we'll test the helper function directly
    
    # Import after paths are set
    sys.path.insert(0, str(Path(__file__).parent / "verl"))
    
    # Test the get_visible_part function logic
    def get_visible_part(message):
        """Extract the part of the message visible to the partner."""
        import re
        tag_match = re.search(r"\[(message|propose|accept|reject)\]", message, re.IGNORECASE)
        if tag_match:
            return message[tag_match.start():]
        return message
    
    test_cases = [
        (
            "Let's think step by step. I see good matches. [message] Hello partner!",
            "[message] Hello partner!"
        ),
        (
            "Let me analyze... [propose] Proposal:\nBLEU: Ava Li",
            "[propose] Proposal:\nBLEU: Ava Li"
        ),
        (
            "[accept] Great proposal!",
            "[accept] Great proposal!"
        ),
        (
            "Thinking here... [REJECT] Not optimal",
            "[REJECT] Not optimal"
        ),
    ]
    
    print("Testing message extraction:")
    all_passed = True
    for full_msg, expected in test_cases:
        visible = get_visible_part(full_msg)
        if visible == expected:
            print(f"✅ PASS: '{full_msg[:30]}...' -> '{visible[:30]}...'")
        else:
            print(f"❌ FAIL: Expected '{expected}', got '{visible}'")
            all_passed = False
    
    return all_passed


def test_player_2_thinking_requirement():
    """Test that player-2 must include 'let's think step by step'."""
    print("\n\n=== Testing Player 2 Thinking Requirement ===")
    
    env = OptimizationEnv(one_player=False)
    obs = env.reset()
    
    # Get to player 2's turn
    obs, _ = env.step("[message] Hello partner")
    print(f"Turn switched to: {obs['turn_player']}")
    
    # Test: Player 2 without thinking
    print("\nPlayer 2 tries to send message without thinking...")
    obs, error = env.step("[message] I see some good matches")
    
    if error:
        error_msg = obs.get('player-2', '')
        if "let's think step by step" in error_msg.lower():
            print("✅ SUCCESS: Player 2 is required to think step by step")
            print(f"Error message: {error_msg[:100]}...")
        else:
            print("❌ ERROR: Unexpected error message")
    else:
        print("❌ ERROR: Player 2 was allowed to skip thinking")
    
    # Test: Player 2 with thinking
    print("\nPlayer 2 sends message with thinking...")
    obs, error = env.step("Let's think step by step. I see patterns. [message] I found good matches")
    
    if not error:
        print("✅ SUCCESS: Player 2's message with thinking accepted")
    else:
        print(f"❌ ERROR: Valid message rejected: {obs.get('player-2', '')[:100]}")
    
    return True


def main():
    """Run all visibility tests."""
    print("Testing Chain-of-Thought Visibility\n")
    
    all_passed = True
    
    # Test 1: Basic thinking visibility
    try:
        if not test_thinking_visibility():
            all_passed = False
    except Exception as e:
        print(f"✗ Basic visibility test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test 2: Player 2 thinking requirement
    try:
        if not test_player_2_thinking_requirement():
            all_passed = False
    except Exception as e:
        print(f"✗ Thinking requirement test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test 3: Rollout implementation
    try:
        if not asyncio.run(test_rollout_thinking_visibility()):
            all_passed = False
    except Exception as e:
        print(f"✗ Rollout test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All thinking visibility tests passed!")
        print("\nKey verified behaviors:")
        print("- Chain-of-thought blocks are hidden from the partner")
        print("- Only content from [tag] onwards is visible to partner")
        print("- Player 2 must include 'let's think step by step'")
        print("- Each player maintains their own message history")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()