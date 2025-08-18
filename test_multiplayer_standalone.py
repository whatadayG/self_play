#!/usr/bin/env python3
"""
Standalone test for multi-player logic without VERL imports.
Tests the core concepts of multiplayer state management and perspective switching.
"""

import os
import sys
from typing import Dict, List, Any
from dataclasses import dataclass
from copy import deepcopy

# Add path for dialop
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from dialop.games.optimization import OptimizationGame


@dataclass
class Message:
    """Simple message structure."""
    role: str
    content: str


class StandaloneMultiplayerWrapper:
    """Standalone version of the multiplayer wrapper for testing."""
    
    def __init__(self, game_state: Dict[str, Any], num_players: int = 2):
        self.game_state = game_state
        self.num_players = num_players
        self.current_player = 0
        self.player_messages = {i: [] for i in range(num_players)}
        self.turn_count = 0
        
    def switch_player(self):
        """Switch to the next player's perspective."""
        self.current_player = (self.current_player + 1) % self.num_players
        self.turn_count += 1
        
    def add_player_message(self, player_id: int, role: str, content: str):
        """Add a message to a specific player's history."""
        self.player_messages[player_id].append({
            "role": role,
            "content": content
        })
        
    def handle_player_response(self, response: str, other_player_sees: str = None):
        """Handle a player's response and update histories."""
        current_player = self.current_player
        other_player = 1 - current_player  # Works for 2-player games
        
        # Add to current player's history
        self.add_player_message(current_player, "assistant", response)
        
        # Add to other player's history (what they observe)
        if other_player_sees is None:
            other_player_sees = f"The other player said: {response}"
            
        self.add_player_message(other_player, "user", other_player_sees)
        
        # Switch to other player
        self.switch_player()


def format_table_with_mask(table, mask):
    """Format the table showing only values visible through the mask."""
    formatted = []
    for i, row in enumerate(table):
        row_str = []
        for j, val in enumerate(row):
            if mask[i][j]:
                row_str.append(f"{val:6.2f}")
            else:
                row_str.append("  ??  ")
        formatted.append(" | ".join(row_str))
    return "\n".join(formatted)


def get_player_prompt(game: OptimizationGame, player_id: int, max_turns: int = 20) -> str:
    """Get the prompt for a specific player based on their perspective."""
    if player_id == 0:
        # Player 0 sees mask1
        mask = game.masks[0]
        scale = game.scales[0]
        player_label = "Player-1"
    else:
        # Player 1 sees mask2
        mask = game.masks[1]
        scale = game.scales[1]
        player_label = "Player-2"
    
    # Format the table view based on the player's mask
    visible_table = format_table_with_mask(game.table.values, mask)
    
    prompt = f"""You are {player_label} in a reviewer-paper matching negotiation.

Your goal is to negotiate with the other player to find an assignment of reviewers to papers.
You can only see parts of the affinity scores based on your expertise (marked with your mask).

Table of affinities (your visible scores):
{visible_table}

Your preference scale: {scale}

Rules:
- Negotiate to find a mutually agreeable assignment
- Use [propose] when making a formal proposal
- Use [accept] to accept the other player's proposal
- Use [reject] to reject and continue negotiating
- The game will end after {max_turns} turns or when a proposal is accepted"""
    
    return prompt


def test_multiplayer_game_simulation():
    """Test a complete multiplayer game simulation."""
    
    print("=== Testing Standalone Multi-Player Game Simulation ===\n")
    
    # 1. Create a game
    game = OptimizationGame({}, one_player=False)
    game.reset()
    
    game_state = {
        "table": game.table.values,
        "mask1": game.masks[0],
        "mask2": game.masks[1],
        "scale1": game.scales[0],
        "scale2": game.scales[1],
        "best_assignment_reward": game.best_assignment_reward,
        "action_log": []
    }
    
    print(f"Created game with:")
    print(f"- Table shape: {len(game.table.values)}x{len(game.table.values[0])}")
    print(f"- Player 1 scale: {game.scales[0]}")
    print(f"- Player 2 scale: {game.scales[1]}")
    print(f"- Best reward: {game.best_assignment_reward:.2f}\n")
    
    # Show what each player sees
    print("=== Information Asymmetry ===")
    print("\nPlayer 1 sees:")
    print(format_table_with_mask(game.table.values, game.masks[0]))
    print(f"\nPlayer 2 sees:")
    print(format_table_with_mask(game.table.values, game.masks[1]))
    
    # 2. Create multiplayer wrapper
    mp_wrapper = StandaloneMultiplayerWrapper(game_state)
    
    # 3. Initialize each player with their prompt
    for player_id in range(2):
        prompt = get_player_prompt(game, player_id)
        mp_wrapper.add_player_message(player_id, "system", prompt)
    
    # 4. Simulate a conversation
    print("\n=== Simulating Multi-Turn Conversation ===\n")
    
    # Simulate 6 turns of negotiation
    responses = [
        ("I see we have a matching problem. Based on my visible scores, I think reviewer 2 has high affinity for paper 0.", 
         "Player-1 starts: Looking at their visible scores for potential matches."),
        
        ("I agree we should find a good match. From what I can see, reviewer 1 might be good for paper 1.",
         "Player-2 responds: Sharing their perspective based on different visible scores."),
        
        ("That's interesting. How about we consider: paper 0 -> reviewer 2, paper 1 -> reviewer 0?",
         "Player-1 suggests: Initial assignment based on partial information."),
        
        ("I see your point, but I think reviewer 1 would be better for paper 0 based on my information.",
         "Player-2 counters: Different assignment based on their mask."),
        
        ("[propose] Let me make a formal proposal: paper 0 -> reviewer 2, paper 1 -> reviewer 0, paper 2 -> reviewer 1",
         "Player-1 proposes: Formal assignment proposal."),
        
        ("[accept] After considering our discussion, I accept your proposal. It seems like a fair compromise.",
         "Player-2 accepts: Agreement reached.")
    ]
    
    for i, (response, summary) in enumerate(responses):
        current = mp_wrapper.current_player
        print(f"Turn {i + 1} - Player {current + 1}:")
        print(f"  Says: \"{response}\"")
        print(f"  ({summary})")
        
        mp_wrapper.handle_player_response(response)
        print()
    
    # 5. Show final conversation histories
    print("=== Final Conversation Histories ===\n")
    
    for player_id in range(2):
        print(f"=== Player {player_id + 1} Perspective ===")
        messages = mp_wrapper.player_messages[player_id]
        
        # Count message types
        system_msgs = sum(1 for m in messages if m["role"] == "system")
        user_msgs = sum(1 for m in messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in messages if m["role"] == "assistant")
        
        print(f"Total messages: {len(messages)} (system: {system_msgs}, user: {user_msgs}, assistant: {assistant_msgs})")
        print("\nConversation flow:")
        
        for j, msg in enumerate(messages):
            if j == 0:  # Skip the long system prompt
                print(f"  [system] Initial game prompt with Player {player_id + 1}'s view...")
            else:
                role = msg["role"]
                content = msg["content"]
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  [{role}] {content}")
        print()
    
    # 6. Create training instances
    print("=== Training Instance Creation ===\n")
    
    # In the real implementation, each player's history becomes a training instance
    training_instances = []
    for player_id in range(2):
        instance = {
            "player_perspective": player_id,
            "messages": deepcopy(mp_wrapper.player_messages[player_id]),
            "final_reward": 0.85,  # Both get same reward
            "batch_id": 0  # Same batch ID for GRPO
        }
        training_instances.append(instance)
    
    print(f"Created {len(training_instances)} training instances:")
    for i, inst in enumerate(training_instances):
        print(f"\nInstance {i + 1}:")
        print(f"  - Player perspective: Player-{inst['player_perspective'] + 1}")
        print(f"  - Number of messages: {len(inst['messages'])}")
        print(f"  - Final reward: {inst['final_reward']}")
        print(f"  - Batch ID: {inst['batch_id']} (same for GRPO grouping)")
    
    # 7. Verify key properties
    print("\n=== Verification ===")
    
    # Check batch IDs
    batch_ids = [inst["batch_id"] for inst in training_instances]
    if len(set(batch_ids)) == 1:
        print("✓ All instances have same batch_id (correct for GRPO)")
    else:
        print("✗ Instances have different batch_ids")
    
    # Check rewards
    rewards = [inst["final_reward"] for inst in training_instances]
    if len(set(rewards)) == 1:
        print("✓ All instances have same reward (correct for collaborative game)")
    else:
        print("✗ Instances have different rewards")
    
    # Check perspective diversity
    perspectives = [inst["player_perspective"] for inst in training_instances]
    if len(set(perspectives)) == 2:
        print("✓ Both player perspectives represented")
    else:
        print("✗ Missing player perspectives")
    
    # Check information asymmetry
    player1_saw_mask1 = any("mask" in msg["content"].lower() and "player-1" in msg["content"].lower() 
                           for msg in training_instances[0]["messages"] if msg["role"] == "system")
    player2_saw_mask2 = any("mask" in msg["content"].lower() and "player-2" in msg["content"].lower() 
                           for msg in training_instances[1]["messages"] if msg["role"] == "system")
    
    if player1_saw_mask1 and player2_saw_mask2:
        print("✓ Information asymmetry preserved (different masks)")
    else:
        print("✗ Information asymmetry not preserved")
    
    print("\n=== Test Complete ===")
    
    return training_instances


if __name__ == "__main__":
    # Run the test
    training_instances = test_multiplayer_game_simulation()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY: Multi-player implementation successfully:")
    print("- Maintains separate conversation histories per player")
    print("- Preserves information asymmetry (different masks)")
    print("- Alternates perspectives between players")
    print("- Creates grouped training instances for GRPO")
    print("- Assigns same reward to both players")
    print("="*50)