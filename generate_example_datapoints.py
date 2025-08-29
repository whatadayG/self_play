#!/usr/bin/env python3
"""
Generate human-readable example datapoints from DialopSelfPlayRollout.

This script runs the rollout and saves a formatted example showing:
1. The complete conversation
2. Both player perspectives with rewards
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import torch
import numpy as np
import pandas as pd
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


def convert_game_state_format(parquet_game_state):
    """Convert game state from parquet format to dialop format."""
    game_state_json = json.loads(parquet_game_state)
    table1 = game_state_json['tables'][0]
    table2 = game_state_json['tables'][1]
    
    # Extract values table and masks
    values = []
    mask1 = []
    mask2 = []
    
    for i in range(1, len(table1)):
        value_row = []
        mask1_row = []
        mask2_row = []
        
        for j in range(1, len(table1[0])):
            # Get value from either table
            val1 = table1[i][j]
            val2 = table2[i][j]
            
            if val1 != '':
                value_row.append(int(val1))
                mask1_row.append(True)
            else:
                mask1_row.append(False)
                
            if val2 != '':
                if val1 == '':
                    value_row.append(int(val2))
                mask2_row.append(True)
            else:
                mask2_row.append(False)
                if val1 == '':
                    value_row.append(0)
                    
        values.append(value_row)
        mask1.append(mask1_row)
        mask2.append(mask2_row)
    
    # Create dialop format game state
    dialop_game_state = {
        "table": values,
        "mask1": mask1,
        "mask2": mask2,
        "scale1": 1.0,  # Default scale
        "scale2": 1.0,  # Default scale
        "best_assignment_reward": game_state_json.get("best_assignment_reward", 0),
        "action_log": []  # Empty action log for new games
    }
    
    return dialop_game_state


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


def generate_example_games():
    """Generate multiple example self-play games with realistic responses."""
    
    # Create rollout
    with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
        rollout = DialopSelfPlayRollout()
        rollout.config = ExampleConfig()
        rollout.sampling_params = {"temperature": 0.7}
        rollout.processing_class = ExampleProcessingClass()
        
        # Track the conversation for display
        conversation_log = []
        
        # Create realistic response generator
        def realistic_responses(messages, obs, player):
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
            # Determine which game this is based on the messages
            # Since we're processing games in order, we can infer from the total logs
            game_idx = len([log for log in conversation_log if log['turn'] == 0 and log['player'] == player]) - 1
            if game_idx < 0:
                game_idx = 0
            
            conversation_log.append({
                "player": player,
                "turn": turn,
                "game_idx": game_idx,
                "observation": obs[:100] + "..." if len(obs) > 100 else obs,
                "response": response
            })
            
            return response
            
        rollout._generate_player_response = realistic_responses
    
    # Load real data from parquet file
    df = pd.read_parquet('test_small.parquet')
    
    # Get 5 unique games
    num_games = 5
    unique_games = df['game_id'].unique()[:num_games]
    
    # Create input for both players of 5 games (batch_size = 10)
    game_states = []
    player_indices = []
    game_ids = []
    prompts = []
    
    for game_id in unique_games:
        game_data = df[df['game_id'] == game_id].iloc[0]
        # Convert game state format
        dialop_game_state = convert_game_state_format(game_data['game_state'])
        dialop_game_state_str = json.dumps(dialop_game_state)
        
        # Add both players for each game
        for player_idx in [0, 1]:
            game_states.append(dialop_game_state_str)
            player_indices.append(player_idx)
            game_ids.append(int(game_id))
            prompts.append(game_data['prompt'])
    
    batch_size = num_games * 2  # 10 total (2 players per game)
    input_data = DataProto(
        batch=TensorDict({
            "dummy": torch.zeros(batch_size, 1)
        }, batch_size=[batch_size]),
        non_tensor_batch={
            "game_state": np.array(game_states),
            "player_index": np.array(player_indices),
            "game_id": np.array(game_ids),
            "prompt": np.array(prompts)
        }
    )
    
    # Run generate_sequences
    print(f"Generating {num_games} example games...")
    output = rollout.generate_sequences(input_data)
    
    # Process each game separately
    all_examples = []
    
    for game_idx in range(num_games):
        # Get indices for both players of this game
        p1_idx = game_idx * 2
        p2_idx = game_idx * 2 + 1
        
        # Extract conversation log for this game
        game_conversation = [log for log in conversation_log if 
                           log.get('game_idx', 0) == game_idx]
        
        # Create formatted output for this game
        example_data = {
            "timestamp": datetime.now().isoformat(),
            "model": "gpt2 (tokenizer only - responses are scripted)",
            "game_id": int(game_ids[p1_idx]),
            "game_conversation": game_conversation,
            "player_perspectives": []
        }
        
        # Format each player's perspective
        for player_offset in [0, 1]:
            i = p1_idx + player_offset
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
            "player_index": player_offset,
            "player_name": f"player-{player_offset+1}",
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
        
        all_examples.append(example_data)
    
    # Save each game to a separate file
    for idx, example_data in enumerate(all_examples):
        output_file = f"realistic_dialop_example_{idx+1}.json"
        with open(output_file, "w") as f:
            json.dump(example_data, f, indent=2)
        print(f"\nExample {idx+1} saved to: {output_file}")
    
    # Also create readable text versions
    for idx, example_data in enumerate(all_examples):
        text_output = []
        text_output.append("=" * 80)
        text_output.append(f"DIALOP SELF-PLAY EXAMPLE DATAPOINT - GAME {idx+1}")
        text_output.append("=" * 80)
        text_output.append(f"Generated: {example_data['timestamp']}")
        text_output.append(f"Game ID: {example_data['game_id']}")
        text_output.append("")
        
        # Show the actual game conversation
        text_output.append("COMPLETE GAME CONVERSATION:")
        text_output.append("-" * 40)
        for turn in example_data['game_conversation']:
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
        
        if idx == 0:  # Only add observations to first file
            text_output.append("\n" + "=" * 80)
            text_output.append("KEY OBSERVATIONS:")
            text_output.append("=" * 80)
            text_output.append("1. Both players see the same game but from different perspectives")
            text_output.append("2. Player 0 sees their messages as 'Assistant:' and other as 'User:'")
            text_output.append("3. Player 1 sees their messages as 'Assistant:' and other as 'User:'")
            text_output.append("4. Both players receive the same reward (cooperative game)")
            text_output.append("5. Rewards are placed on the last token of each sequence")
        
        # Save text version
        text_file = f"realistic_dialop_example_{idx+1}_readable.txt"
        with open(text_file, "w") as f:
            f.write("\n".join(text_output))
        
        print(f"Readable version saved to: {text_file}")
    
    return all_examples


if __name__ == "__main__":
    print("Generating multiple DIALOP self-play datapoints...")
    print("-" * 60)
    
    # Run the generation
    examples = generate_example_games()
    
    print("\nGeneration complete!")
    print(f"\nCreated {len(examples)} games with files:")
    for i in range(len(examples)):
        print(f"{i+1}. realistic_dialop_example_{i+1}.json - Full data in JSON format")
        print(f"   realistic_dialop_example_{i+1}_readable.txt - Human-readable version")
    
    # Quick summary
    print("\nQuick Summary:")
    for i, example in enumerate(examples):
        print(f"\nGame {i+1} (ID: {example.get('game_id', 'N/A')}):")
        if len(example['player_perspectives']) > 0:
            print(f"- Game completed: {example['player_perspectives'][0]['game_completed']}")
            print(f"- Final reward: {example['player_perspectives'][0]['reward']['normalized_value']:.4f}")
            print(f"- Player 0 sequence: {example['player_perspectives'][0]['sequence_length']} tokens")
            if len(example['player_perspectives']) > 1:
                print(f"- Player 1 sequence: {example['player_perspectives'][1]['sequence_length']} tokens")
        else:
            print("- Game failed to generate properly")