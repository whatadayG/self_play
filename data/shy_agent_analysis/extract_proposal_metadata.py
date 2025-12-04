#!/usr/bin/env python3
"""
Extract proposal metadata from conversations and analyze by conversation length.
"""

import pandas as pd
import json
import re
from pathlib import Path
import argparse
from transformers import AutoTokenizer


def identify_assistant_player(input_ids, conversation, tokenizer):
    """Identify which player is the assistant (non-shy agent being trained)."""
    try:
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        assistant_start = full_text.find("<|im_start|>assistant\n")
        if assistant_start == -1:
            return None

        assistant_end = full_text.find("<|im_end|>", assistant_start)
        if assistant_end == -1:
            assistant_end = len(full_text)

        assistant_text = full_text[assistant_start:assistant_end]

        for msg in conversation:
            msg_text = msg['message'].strip()
            match = re.search(r'\[(message|propose|accept|reject)\](.*)', msg_text, re.DOTALL)
            if match:
                public_part = match.group(2).strip()[:150]
                if len(public_part) > 20 and public_part in assistant_text:
                    return msg['player']
        return None
    except Exception:
        return None


def analyze_conversation_proposals(row, tokenizer, run_id, round_num):
    """Analyze proposal/accept/reject patterns in a conversation."""
    try:
        input_ids = row['input_ids']
        conversation = json.loads(row['full_conversation'])

        # Get game metadata
        game_info = json.loads(row['game_info']) if isinstance(row['game_info'], str) else row['game_info']
        game_id = row.get('game_id')
        game_reward = row.get('game_normalized_reward', game_info.get('game_normalized_reward', 0))
        conv_length = game_info.get('turn_count', 0)

        # Identify players
        nonshy_player = identify_assistant_player(input_ids, conversation, tokenizer)
        if not nonshy_player:
            return None

        shy_player = 'player-1' if nonshy_player == 'player-2' else 'player-2'

        # Count actions by each player
        counts = {
            'shy_messages': 0,
            'shy_proposals': 0,
            'shy_accepts': 0,
            'shy_rejects': 0,
            'nonshy_messages': 0,
            'nonshy_proposals': 0,
            'nonshy_accepts': 0,
            'nonshy_rejects': 0,
        }

        for msg in conversation:
            player = msg.get('player')
            if player == 'error':
                continue

            text = msg.get('message', '')
            is_shy = (player == shy_player)
            prefix = 'shy_' if is_shy else 'nonshy_'

            if '[propose]' in text:
                counts[prefix + 'proposals'] += 1
            elif '[accept]' in text:
                counts[prefix + 'accepts'] += 1
            elif '[reject]' in text:
                counts[prefix + 'rejects'] += 1
            elif '[message]' in text:
                counts[prefix + 'messages'] += 1

        return {
            'run_id': run_id,
            'round': round_num,
            'game_id': game_id,
            'conversation_length': conv_length,
            'game_reward': game_reward,
            'shy_player': shy_player,
            'nonshy_player': nonshy_player,
            **counts
        }

    except Exception as e:
        print(f"Error analyzing conversation: {e}")
        return None


def process_round(parquet_path, tokenizer, run_id, round_num):
    """Process a single round's train.parquet file."""
    print(f"  Loading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)

    print(f"  Analyzing {len(df)} rows...")
    results = []

    for idx, row in df.iterrows():
        result = analyze_conversation_proposals(row, tokenizer, run_id, round_num)
        if result:
            results.append(result)

        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} rows...")

    return pd.DataFrame(results) if results else pd.DataFrame()


def process_run(run_dir, output_dir, run_id, tokenizer):
    """Process all rounds in a run and save combined dataset."""
    print(f"\nProcessing run: {run_id}")
    print("=" * 80)

    round_dirs = sorted(run_dir.glob("round_*"))
    print(f"Found {len(round_dirs)} rounds\n")

    all_dfs = []

    for round_dir in round_dirs:
        round_name = round_dir.name
        round_num = int(round_name.split('_')[1])
        parquet_path = round_dir / "train.parquet"

        if not parquet_path.exists():
            print(f"Skipping {round_name} (no train.parquet)")
            continue

        print(f"Round {round_num:03d}:")
        df = process_round(parquet_path, tokenizer, run_id, round_num)

        if not df.empty:
            all_dfs.append(df)
            print(f"  Extracted data from {len(df)} conversations")

    if all_dfs:
        print("\nCombining data from all rounds...")
        combined = pd.concat(all_dfs, ignore_index=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"proposals_{run_id}.parquet"

        print(f"\nSaving dataset: {output_path}")
        combined.to_parquet(output_path, index=False)
        print(f"  {len(combined)} total conversations")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        print("\nProposal counts:")
        print(f"  Total shy proposals: {combined['shy_proposals'].sum()}")
        print(f"  Total non-shy proposals: {combined['nonshy_proposals'].sum()}")
        print(f"  Mean shy proposals/conversation: {combined['shy_proposals'].mean():.3f}")
        print(f"  Mean non-shy proposals/conversation: {combined['nonshy_proposals'].mean():.3f}")

        print("\nCorrelation with conversation length:")
        print(f"  Shy proposals vs length: {combined['conversation_length'].corr(combined['shy_proposals']):.3f}")
        print(f"  Non-shy proposals vs length: {combined['conversation_length'].corr(combined['nonshy_proposals']):.3f}")
    else:
        print("\nNo data extracted!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract proposal metadata from train.parquet files"
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

    run_id = args.run_id or run_dir.name

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    process_run(run_dir, args.output_dir, run_id, tokenizer)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
