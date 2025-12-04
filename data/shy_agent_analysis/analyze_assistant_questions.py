#!/usr/bin/env python3
"""
Analyze question marks in assistant (trainee) agent messages across training rounds.

This script processes train.parquet files from offline GRPO training runs and counts
the number of question marks in the public (non-thinking) messages of the assistant
agent (the agent being trained, identified via loss_mask).

For shy-setup runs: The assistant is the non-shy agent learning to interact with a shy partner.
For symmetric runs: The assistant is whichever player appears in the <|im_start|>assistant section.

Usage Examples:
    # Analyze a specific run
    python analyze_assistant_questions.py logs/offline_grpo/20251110_214435

    # Specify output location
    python analyze_assistant_questions.py logs/offline_grpo/20251107_002451 \\
        --output data/my_analysis.csv

    # Use different tokenizer
    python analyze_assistant_questions.py logs/offline_grpo/20251110_214435 \\
        --model Qwen/Qwen2.5-14B-Instruct

Output:
    - CSV file with columns: round, round_num, mean_questions_per_conversation, num_conversations
    - Console summary table showing question usage trends across rounds
"""

import argparse
import pandas as pd
import json
import re
from pathlib import Path
from transformers import AutoTokenizer


def identify_assistant_player(input_ids, conversation, tokenizer):
    """
    Identify which player is the assistant (agent being trained).
    Returns player name ('player-1' or 'player-2') or None.
    """
    try:
        # Decode with special tokens
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Find first assistant section
        assistant_start = full_text.find("<|im_start|>assistant\n")
        if assistant_start == -1:
            return None

        assistant_end = full_text.find("<|im_end|>", assistant_start)
        if assistant_end == -1:
            assistant_end = len(full_text)

        assistant_text = full_text[assistant_start:assistant_end]

        # Find which player this corresponds to
        for msg in conversation:
            msg_text = msg['message'].strip()
            # Look for the part after [message]/[propose]/[accept]/[reject]
            match = re.search(r'\[(message|propose|accept|reject)\](.*)', msg_text, re.DOTALL)
            if match:
                public_part = match.group(2).strip()[:150]
                if len(public_part) > 20 and public_part in assistant_text:
                    return msg['player']

        return None
    except Exception:
        return None


def extract_public_text(message):
    """
    Extract the public (non-thinking) part of a message.
    This is the part after [message]/[propose]/[accept]/[reject] tag,
    with <think>...</think> sections removed.
    """
    # Find the tag and extract everything after it
    match = re.search(r'\[(message|propose|accept|reject)\](.*)', message, re.DOTALL)
    if not match:
        return ""

    public_text = match.group(2)

    # Remove <think>...</think> sections (including nested ones)
    while '<think>' in public_text.lower():
        public_text = re.sub(r'<think>.*?</think>', '', public_text, flags=re.DOTALL | re.IGNORECASE)

    return public_text


def count_questions_in_conversation(row, tokenizer):
    """
    Count total question marks in non-thinking parts of assistant's messages
    in a single conversation.
    """
    try:
        input_ids = row['input_ids']
        conversation = json.loads(row['full_conversation'])

        # Identify which player is the assistant (being trained)
        assistant_player = identify_assistant_player(input_ids, conversation, tokenizer)
        if not assistant_player:
            return None

        # Count question marks in this player's public messages
        total_questions = 0
        for msg in conversation:
            if msg['player'] == assistant_player:
                public_text = extract_public_text(msg['message'])
                total_questions += public_text.count('?')

        return total_questions
    except Exception:
        return None


def analyze_round(parquet_path, tokenizer):
    """
    Analyze a single round's train.parquet file.
    Returns (mean_questions, num_conversations).
    """
    df = pd.read_parquet(parquet_path)

    # Count questions for each conversation
    question_counts = []

    for idx, row in df.iterrows():
        count = count_questions_in_conversation(row, tokenizer)
        if count is not None:
            question_counts.append(count)

    if not question_counts:
        return None, 0

    mean_questions = sum(question_counts) / len(question_counts)
    return mean_questions, len(question_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze question usage by assistant agent across training rounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s logs/offline_grpo/20251110_214435
  %(prog)s logs/offline_grpo/20251107_002451 -o my_results.csv
  %(prog)s logs/offline_grpo/20251110_214435 --model Qwen/Qwen2.5-14B-Instruct
        """
    )
    parser.add_argument(
        'run_dir',
        type=Path,
        help='Path to run directory (e.g., logs/offline_grpo/20251110_214435)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output CSV path (default: data/shy_agent_analysis/questions_<run_name>.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Tokenizer model name (default: Qwen/Qwen2.5-7B-Instruct)'
    )

    args = parser.parse_args()

    # Validate run directory
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    run_name = run_dir.name

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = Path(f"data/shy_agent_analysis/questions_{run_name}.csv")

    print("Analyzing assistant agent question usage across rounds")
    print(f"Run: {run_name}")
    print("=" * 80)

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Find all round directories
    round_dirs = sorted(run_dir.glob("round_*"))

    if not round_dirs:
        print(f"Error: No round directories found in {run_dir}")
        return 1

    print(f"Found {len(round_dirs)} rounds to analyze\n")

    results = []

    for round_dir in round_dirs:
        round_name = round_dir.name
        parquet_path = round_dir / "train.parquet"

        if not parquet_path.exists():
            print(f"{round_name}: train.parquet not found, skipping")
            continue

        print(f"Analyzing {round_name}...", end=" ")
        mean_questions, num_conversations = analyze_round(parquet_path, tokenizer)

        if mean_questions is not None:
            results.append({
                'round': round_name,
                'round_num': int(round_name.split('_')[1]),
                'mean_questions_per_conversation': mean_questions,
                'num_conversations': num_conversations
            })
            print(f"{mean_questions:.3f} questions/conv ({num_conversations} convs)")
        else:
            print("Failed to analyze")

    if not results:
        print("\nError: No rounds were successfully analyzed")
        return 1

    # Print summary table
    print("\n" + "=" * 80)
    print(f"SUMMARY - {run_name}")
    print("=" * 80)
    print(f"{'Round':<15} {'Mean Questions':<20} {'N Conversations':<20}")
    print("-" * 80)

    for result in results:
        print(f"{result['round']:<15} {result['mean_questions_per_conversation']:<20.3f} {result['num_conversations']:<20}")

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print key statistics
    print("\n" + "=" * 80)
    print("KEY STATISTICS")
    print("=" * 80)
    first_round = results[0]['mean_questions_per_conversation']
    last_round = results[-1]['mean_questions_per_conversation']
    percent_change = ((last_round - first_round) / first_round) * 100

    print(f"First round (round {results[0]['round_num']}): {first_round:.3f} questions/conv")
    print(f"Last round (round {results[-1]['round_num']}): {last_round:.3f} questions/conv")
    print(f"Total change: {percent_change:+.1f}%")

    return 0


if __name__ == "__main__":
    exit(main())
