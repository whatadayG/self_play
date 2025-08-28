#!/usr/bin/env python3
"""
Generate human-readable example datapoints from DialopSelfPlayRollout.

This script runs the rollout and saves a formatted example showing:
1. The complete conversation
2. Both player perspectives with rewards
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from tensordict import TensorDict

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "verl"))
sys.path.insert(0, str(project_root / "dialop"))

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import TASKS_SHORT, WORKERS
from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout


class ExampleProcessingClass:
    """Processing class for example generation."""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def apply_chat_template(self, messages, **kwargs):
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text += f"System: {content}\n"
            elif role == "user":
                text += f"User: {content}\n"
            else:
                text += f"Assistant: {content}\n"
        text += "Assistant: "
        return text


class ExampleConfig:
    max_model_len = 2048
    response_length = 512


def create_valid_proposal(papers, reviewers):
    """Create a valid proposal string."""
    proposal = "Proposal:\n"
    for paper, reviewer in zip(papers, reviewers):
        proposal += f"{paper}: {reviewer}\n"
    return proposal.strip()


async def generate_example_game():
    """Generate an example self-play game with realistic responses."""
    
    # Create rollout
    with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
        rollout = DialopSelfPlayRollout()
        rollout.config = ExampleConfig()
        rollout.sampling_params = {"temperature": 0.7}
        rollout.processing_class = ExampleProcessingClass()
        
        # Track the conversation for display
        conversation_log = []
        
        # Create realistic response generator
        async def realistic_responses(messages, obs, player):
            """Generate realistic game responses."""
            # Count assistant messages to determine turn
            assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
            turn = len(assistant_msgs)
            
            response = ""
            if player == "player-1":
                if turn == 0:
                    response = "[message] Hello partner! I can see a reviewer-paper similarity table. Let me share what I observe in my view."
                elif turn == 1:
                    response = "[message] I see that BLEU has high scores with Ava Li (85) and Daniel Nguyen (78). Also, RoBERTa matches well with Joseph Santos (92). What do you see on your end?"
                elif turn == 2:
                    # Make a proposal
                    response = f"[propose] {create_valid_proposal(TASKS_SHORT[:8], WORKERS[:8])}"
                else:
                    response = "[message] This assignment gives us a good overall score. What do you think?"
            else:  # player-2
                if turn == 0:
                    response = "[message] Hi! Yes, I have my own view of the table. I can see different values than you."
                elif turn == 1:
                    response = "[message] From my perspective, I see GloVe has a strong match with Sofia Patel (88), and SWAG works well with Noah Wilson (90). Let me check the other papers."
                elif turn == 2:
                    response = "[message] Let me review your proposal. The assignments look reasonable based on what I can see."
                elif turn == 3:
                    response = "[accept]"
                else:
                    response = "[message] I agree with this assignment."
                    
            # Log the conversation
            conversation_log.append({
                "player": player,
                "turn": turn,
                "observation": obs[:100] + "..." if len(obs) > 100 else obs,
                "response": response
            })
            
            return response
            
        rollout._generate_player_response = realistic_responses
    
    # Create input for both players of the same game
    batch_size = 2
    input_data = DataProto(
        batch=TensorDict({
            "dummy": torch.zeros(batch_size, 1)
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array([json.dumps(None), json.dumps(None)]),
            "player_index": np.array([0, 1]),  # Player 0 and Player 1
            "game_id": np.array([1, 1]),  # Same game!
            "prompt": np.array(["Start game"] * 2)
        }
    )
    
    # Run generate_sequences
    print("Generating example game...")
    output = await rollout.generate_sequences(input_data)
    
    # Create formatted output
    example_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2 (tokenizer only - responses are scripted)",
        "game_conversation": conversation_log,
        "player_perspectives": []
    }
    
    # Format each player's perspective
    for i in range(2):
        seq = output.batch["input_ids"][i]
        mask = output.batch["attention_mask"][i]
        rewards = output.batch["rewards"][i]
        
        # Get valid sequence
        valid_length = int(mask.sum().item())
        valid_seq = seq[:valid_length]
        
        # Decode text
        text = rollout.processing_class.tokenizer.decode(valid_seq, skip_special_tokens=False)
        
        # Find reward position and value
        nonzero_rewards = torch.nonzero(rewards).squeeze()
        if nonzero_rewards.numel() > 0:
            reward_pos = nonzero_rewards[-1].item() if nonzero_rewards.dim() > 0 else nonzero_rewards.item()
            reward_val = rewards[reward_pos].item()
        else:
            reward_pos = -1
            reward_val = 0.0
            
        # Get game info
        reward_info = output.non_tensor_batch["reward_model"][i]
        game_info = output.non_tensor_batch["game_info"][i]
        
        player_data = {
            "player_index": i,
            "player_name": f"player-{i+1}",
            "sequence_length": valid_length,
            "token_ids_sample": valid_seq[:20].tolist(),  # First 20 tokens
            "decoded_text": text,
            "reward": {
                "value": reward_val,
                "normalized_value": reward_info["normalized_reward"],
                "raw_value": reward_info["reward"],
                "position": reward_pos,
                "placed_on_last_token": reward_pos == valid_length - 1
            },
            "game_completed": game_info.get("completed", False),
            "num_turns": len([line for line in text.split('\n') if line.startswith("Assistant:")])
        }
        
        example_data["player_perspectives"].append(player_data)
    
    # Save to file
    output_file = "example_dialop_selfplay_datapoint.json"
    with open(output_file, "w") as f:
        json.dump(example_data, f, indent=2)
    
    print(f"\nExample saved to: {output_file}")
    
    # Also create a readable text version
    text_output = []
    text_output.append("=" * 80)
    text_output.append("DIALOP SELF-PLAY EXAMPLE DATAPOINT")
    text_output.append("=" * 80)
    text_output.append(f"Generated: {example_data['timestamp']}")
    text_output.append("")
    
    # Show the actual game conversation
    text_output.append("COMPLETE GAME CONVERSATION:")
    text_output.append("-" * 40)
    for turn in conversation_log:
        text_output.append(f"\nTurn {turn['turn']+1} - {turn['player']}:")
        text_output.append(f"Sees: {turn['observation']}")
        text_output.append(f"Says: {turn['response']}")
    
    text_output.append("\n" + "=" * 80)
    text_output.append("TRAINING DATA PERSPECTIVES:")
    text_output.append("=" * 80)
    
    # Show both player perspectives
    for i, player in enumerate(example_data["player_perspectives"]):
        text_output.append(f"\n{'='*40}")
        text_output.append(f"PLAYER {i} PERSPECTIVE ('{player['player_name']}')")
        text_output.append(f"{'='*40}")
        text_output.append(f"Sequence length: {player['sequence_length']} tokens")
        text_output.append(f"Number of turns: {player['num_turns']}")
        text_output.append(f"Game completed: {player['game_completed']}")
        text_output.append(f"\nReward:")
        text_output.append(f"  - Normalized: {player['reward']['normalized_value']:.4f}")
        text_output.append(f"  - Raw: {player['reward']['raw_value']}")
        text_output.append(f"  - Position: token {player['reward']['position']}")
        text_output.append(f"  - On last token: {player['reward']['placed_on_last_token']}")
        text_output.append(f"\nDecoded conversation:")
        text_output.append("-" * 40)
        # Format the conversation nicely
        lines = player["decoded_text"].split('\n')
        for line in lines:
            if line.strip():
                text_output.append(line)
        text_output.append("-" * 40)
    
    text_output.append("\n" + "=" * 80)
    text_output.append("KEY OBSERVATIONS:")
    text_output.append("=" * 80)
    text_output.append("1. Both players see the same game but from different perspectives")
    text_output.append("2. Player 0 sees their messages as 'Assistant:' and other as 'User:'")
    text_output.append("3. Player 1 sees their messages as 'Assistant:' and other as 'User:'")
    text_output.append("4. Both players receive the same reward (cooperative game)")
    text_output.append("5. Rewards are placed on the last token of each sequence")
    
    # Save text version
    text_file = "example_dialop_selfplay_readable.txt"
    with open(text_file, "w") as f:
        f.write("\n".join(text_output))
    
    print(f"Readable version saved to: {text_file}")
    
    return example_data


if __name__ == "__main__":
    print("Generating example DIALOP self-play datapoints...")
    print("-" * 60)
    
    # Run the generation
    example = asyncio.run(generate_example_game())
    
    print("\nGeneration complete!")
    print("\nFiles created:")
    print("1. example_dialop_selfplay_datapoint.json - Full data in JSON format")
    print("2. example_dialop_selfplay_readable.txt - Human-readable formatted version")
    
    # Quick summary
    print("\nQuick Summary:")
    print(f"- Game completed: {example['player_perspectives'][0]['game_completed']}")
    print(f"- Final reward: {example['player_perspectives'][0]['reward']['normalized_value']:.4f}")
    print(f"- Player 0 sequence: {example['player_perspectives'][0]['sequence_length']} tokens")
    print(f"- Player 1 sequence: {example['player_perspectives'][1]['sequence_length']} tokens")