#!/usr/bin/env python3
"""
Extract detailed question-asking metadata from train.parquet files.

Creates two datasets:
1. questions.parquet - One row per question with full context
2. conversations.parquet - One row per conversation with aggregated stats

This supports both granular analyses and correlation studies.
"""

import pandas as pd
import json
import re
from pathlib import Path
from transformers import AutoTokenizer
from typing import Optional, List, Dict, Any
import argparse


def identify_assistant_player(input_ids, conversation, tokenizer):
    """
    Identify which player is the assistant (non-shy agent being trained).
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


def extract_public_text(message: str) -> str:
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

    # Remove <think>...</think> sections
    while '<think>' in public_text.lower():
        public_text = re.sub(r'<think>.*?</think>', '', public_text, flags=re.DOTALL | re.IGNORECASE)

    return public_text.strip()


def extract_questions_from_text(text: str) -> List[str]:
    """
    Extract individual questions from text.
    Simply counts question marks - each '?' represents one question.
    This matches the original counting methodology.
    """
    if not text:
        return []

    # Count question marks - each represents a question
    # Split on '?' to find question fragments
    question_count = text.count('?')

    if question_count == 0:
        return []

    # Split on '?' and reconstruct questions
    parts = text.split('?')
    questions = []

    for i in range(len(parts) - 1):  # Last part has no '?' after it
        # Take the part and add back the '?'
        question_text = parts[i].strip()
        if question_text:
            # Try to get just the sentence/fragment by taking from last period/newline
            lines = question_text.split('\n')
            last_line = lines[-1]

            # Find the last sentence start
            last_period = max(last_line.rfind('. '), last_line.rfind('! '))
            if last_period >= 0:
                question_text = last_line[last_period + 2:]
            else:
                question_text = last_line

            question_text = question_text.strip() + '?'
            questions.append(question_text[:500])  # Truncate long questions

    return questions


def extract_game_metadata(row: pd.Series) -> Dict[str, Any]:
    """Extract game-level metadata from a row."""
    try:
        game_info = json.loads(row['game_info']) if isinstance(row['game_info'], str) else row['game_info']

        return {
            'game_id': row.get('game_id'),
            'game_reward': row.get('game_normalized_reward', game_info.get('game_normalized_reward', 0)),
            'grpo_weight': row.get('sample_weight', 0),
            'conversation_length': game_info.get('turn_count', 0),
            'completed': game_info.get('completed', False),
        }
    except Exception as e:
        return {
            'game_id': row.get('game_id', -1),
            'game_reward': 0,
            'grpo_weight': 0,
            'conversation_length': 0,
            'completed': False,
        }


def analyze_conversation(row: pd.Series, tokenizer, run_id: str, round_num: int) -> tuple[List[Dict], Dict]:
    """
    Analyze a single conversation and extract question metadata.

    Returns:
        (question_records, conversation_record)
    """
    try:
        input_ids = row['input_ids']
        conversation = json.loads(row['full_conversation'])

        # Get game metadata
        game_meta = extract_game_metadata(row)

        # Identify shy vs non-shy player
        nonshy_player = identify_assistant_player(input_ids, conversation, tokenizer)
        if not nonshy_player:
            return [], None

        shy_player = 'player-1' if nonshy_player == 'player-2' else 'player-2'

        # Track questions
        question_records = []
        shy_questions_by_turn = []
        nonshy_questions_by_turn = []

        for msg in conversation:
            turn = msg.get('turn', -1)
            player = msg.get('player')

            # Skip error messages
            if player == 'error':
                continue

            # Extract public text
            public_text = extract_public_text(msg.get('message', ''))

            # Extract questions
            questions = extract_questions_from_text(public_text)

            is_shy = (player == shy_player)

            # Record each question
            for question_text in questions:
                question_record = {
                    'run_id': run_id,
                    'round': round_num,
                    'game_id': game_meta['game_id'],
                    'turn': turn,
                    'player': player,
                    'is_shy': is_shy,
                    'question_text': question_text[:500],  # Truncate long questions
                    'conversation_length': game_meta['conversation_length'],
                    'game_reward': game_meta['game_reward'],
                    'grpo_weight': game_meta['grpo_weight'],
                }
                question_records.append(question_record)

                # Track turn numbers for aggregation
                if is_shy:
                    shy_questions_by_turn.append(turn)
                else:
                    nonshy_questions_by_turn.append(turn)

        # Create conversation-level record
        conversation_record = {
            'run_id': run_id,
            'round': round_num,
            'game_id': game_meta['game_id'],
            'conversation_length': game_meta['conversation_length'],
            'game_reward': game_meta['game_reward'],
            'grpo_weight': game_meta['grpo_weight'],
            'completed': game_meta['completed'],
            'shy_player': shy_player,
            'nonshy_player': nonshy_player,
            'total_questions': len(question_records),
            'shy_questions_count': len(shy_questions_by_turn),
            'nonshy_questions_count': len(nonshy_questions_by_turn),
            'shy_questions_by_turn': json.dumps(shy_questions_by_turn),
            'nonshy_questions_by_turn': json.dumps(nonshy_questions_by_turn),
        }

        return question_records, conversation_record

    except Exception as e:
        print(f"Error analyzing conversation: {e}")
        return [], None


def process_round(parquet_path: Path, tokenizer, run_id: str, round_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single round's train.parquet file.

    Returns:
        (questions_df, conversations_df)
    """
    print(f"  Loading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)

    all_questions = []
    all_conversations = []

    print(f"  Analyzing {len(df)} rows...")
    for idx, row in df.iterrows():
        questions, conversation = analyze_conversation(row, tokenizer, run_id, round_num)

        if questions:
            all_questions.extend(questions)
        if conversation:
            all_conversations.append(conversation)

        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} rows...")

    questions_df = pd.DataFrame(all_questions) if all_questions else pd.DataFrame()
    conversations_df = pd.DataFrame(all_conversations) if all_conversations else pd.DataFrame()

    return questions_df, conversations_df


def process_run(run_dir: Path, output_dir: Path, run_id: str, tokenizer):
    """
    Process all rounds in a run and save combined datasets.
    """
    print(f"\nProcessing run: {run_id}")
    print("=" * 80)

    round_dirs = sorted(run_dir.glob("round_*"))
    print(f"Found {len(round_dirs)} rounds\n")

    all_questions_dfs = []
    all_conversations_dfs = []

    for round_dir in round_dirs:
        round_name = round_dir.name
        round_num = int(round_name.split('_')[1])
        parquet_path = round_dir / "train.parquet"

        if not parquet_path.exists():
            print(f"Skipping {round_name} (no train.parquet)")
            continue

        print(f"Round {round_num:03d}:")
        questions_df, conversations_df = process_round(parquet_path, tokenizer, run_id, round_num)

        if not questions_df.empty:
            all_questions_dfs.append(questions_df)
            print(f"  Extracted {len(questions_df)} questions from {len(conversations_df)} conversations")
        else:
            print(f"  No questions found")

        if not conversations_df.empty:
            all_conversations_dfs.append(conversations_df)

    # Combine all rounds
    if all_questions_dfs:
        print("\nCombining data from all rounds...")
        combined_questions = pd.concat(all_questions_dfs, ignore_index=True)
        combined_conversations = pd.concat(all_conversations_dfs, ignore_index=True)

        # Save to parquet
        output_dir.mkdir(parents=True, exist_ok=True)

        questions_path = output_dir / f"questions_{run_id}.parquet"
        conversations_path = output_dir / f"conversations_{run_id}.parquet"

        print(f"\nSaving datasets:")
        print(f"  Questions: {questions_path}")
        combined_questions.to_parquet(questions_path, index=False)
        print(f"    {len(combined_questions)} total questions")

        print(f"  Conversations: {conversations_path}")
        combined_conversations.to_parquet(conversations_path, index=False)
        print(f"    {len(combined_conversations)} total conversations")

        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nQuestions by round:")
        print(combined_questions.groupby('round').size())

        print(f"\nQuestions by agent type:")
        shy_q = combined_questions['is_shy'].sum()
        nonshy_q = len(combined_questions) - shy_q
        print(f"  Shy agent: {shy_q}")
        print(f"  Non-shy agent: {nonshy_q}")

        print(f"\nConversation statistics:")
        print(f"  Mean conversation length: {combined_conversations['conversation_length'].mean():.2f}")
        print(f"  Mean questions per conversation: {combined_conversations['total_questions'].mean():.2f}")
        print(f"  Correlation (length vs questions): {combined_conversations['conversation_length'].corr(combined_conversations['total_questions']):.3f}")

    else:
        print("\nNo data extracted!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract detailed question metadata from train.parquet files"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to run directory (e.g., logs/offline_grpo/20251110_214435)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (default: inferred from directory name)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/shy_agent_analysis"),
        help="Output directory for datasets (default: data/shy_agent_analysis)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name for tokenizer"
    )

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1

    # Infer run_id from directory name if not provided
    run_id = args.run_id or run_dir.name

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Process the run
    process_run(run_dir, args.output_dir, run_id, tokenizer)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
