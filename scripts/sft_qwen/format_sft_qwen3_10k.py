import os
import sys
import json

# Ensure this directory is on the import path so we can import local modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Add path to where optimization_data.py lives
FINETUNE_DIR = "/home/nickatomlin/georgiazhou/dialop_all/new_dialop/dialop/finetune"
if FINETUNE_DIR not in sys.path:
    sys.path.append(FINETUNE_DIR)

from optimization_data import (
    format_for_agent_model as format_for_agent_model_optimization,
    format_for_user_model as format_for_user_model_optimization,
)


def format_optimization_data(conversations_files, output_dir, max_length):
    os.makedirs(output_dir, exist_ok=True)

    # Parse conversations from all files
    all_conversations = []
    for file_path in conversations_files:
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    conversation_data = json.loads(line)
                    all_conversations.append(conversation_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line in {file_path}")

    print(f"Found {len(all_conversations)} total conversations to format")

    # Format for agent model (match whole_pipeline behavior)
    agent_file = os.path.join(output_dir, "agent_total.jsonl")
    agent_count = format_for_agent_model_optimization(
        all_conversations,
        agent_file,
        complex_model=True,
        max_length=max_length,
    )

    # Format for user model (match whole_pipeline behavior)
    user_file = os.path.join(output_dir, "user_total.jsonl")
    user_count = format_for_user_model_optimization(
        all_conversations,
        user_file,
        complex_model=True,
        max_length=max_length,
    )

    print(
        f"Formatting complete. Created {agent_count} agent examples and {user_count} user examples"
    )
    return agent_file, user_file


def main():
    # Input files provided by the user (absolute paths)
    conversations_files = [
        "/home/nickatomlin/georgiazhou/dialop_all/new_dialop/output_<class 'dialop.envs.optimization.OptimizationEnv'>_sep3-mass-gen-for-sft-10k.jsonl",
        "/home/nickatomlin/georgiazhou/dialop_all/new_dialop/output_<class 'dialop.envs.optimization.OptimizationEnv'>_sep3-mass-gen-for-sft-10k-next-5k.jsonl",
    ]

    # Output directory (absolute path)
    output_dir = "/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k"

    # Use a high max count to include all conversations from both files
    max_examples = 1_000_000

    print("Starting formatting...")
    print(f"Merging and formatting {len(conversations_files)} files into: {output_dir}")

    agent_file, user_file = format_optimization_data(
        conversations_files=conversations_files,
        output_dir=output_dir,
        max_length=max_examples,
    )

    print("Formatting complete.")
    print(f"Agent formatted file: {agent_file}")
    print(f"User formatted file:  {user_file}")


if __name__ == "__main__":
    main() 