#!/usr/bin/env python3
"""
Measure information shared in first messages.

Simple metric: Count mentions of reviewer and paper names.
- Each reviewer name mention = +1
- Each paper name mention = +1
- Total = data score for that message
"""

import pandas as pd
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse
from transformers import AutoTokenizer


def extract_entities_from_game(row: pd.Series) -> Tuple[List[str], List[str]]:
    """
    Extract reviewer and paper names from the game context.

    Returns:
        (reviewer_names, paper_names)
    """
    try:
        conversation = json.loads(row['full_conversation']) if isinstance(row['full_conversation'], str) else row['full_conversation']

        if not conversation:
            return [], []

        # Get the first message which typically contains the table
        first_message = conversation[0]['message']

        # Extract reviewer names (capitalized first and last name)
        reviewer_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        reviewers = list(set(re.findall(reviewer_pattern, first_message)))

        # Extract paper names (typically have colons or are all caps/mixed case acronyms)
        # Look in the table header row
        paper_pattern = r'\b([A-Z][A-Za-z]*(?:[A-Z][a-z]*)*):?'
        potential_papers = set(re.findall(paper_pattern, first_message))

        # Filter out reviewer first/last names from paper list
        reviewer_words = set()
        for r in reviewers:
            reviewer_words.update(r.split())

        papers = [p for p in potential_papers if p not in reviewer_words and len(p) > 2]

        return reviewers, papers
    except Exception as e:
        return [], []


def extract_public_text(message: str) -> str:
    """Extract the public (non-thinking) part of a message."""
    match = re.search(r'\[(message|propose|accept|reject)\](.*)', message, re.DOTALL)
    if not match:
        return ""

    public_text = match.group(2)

    # Remove <think>...</think> sections
    while '<think>' in public_text.lower():
        public_text = re.sub(r'<think>.*?</think>', '', public_text, flags=re.DOTALL | re.IGNORECASE)

    return public_text.strip()


def count_name_mentions(text: str, reviewers: List[str], papers: List[str]) -> Dict[str, int]:
    """
    Count how many times reviewer and paper names appear in text.

    Returns:
        {
            'reviewer_mentions': count,
            'paper_mentions': count,
            'total_mentions': count
        }
    """
    text_lower = text.lower()

    reviewer_count = 0
    for reviewer in reviewers:
        # Count all occurrences of this reviewer name
        reviewer_lower = reviewer.lower()
        reviewer_count += text_lower.count(reviewer_lower)

    paper_count = 0
    for paper in papers:
        # Count all occurrences of this paper name
        paper_lower = paper.lower()
        paper_count += text_lower.count(paper_lower)

    return {
        'reviewer_mentions': reviewer_count,
        'paper_mentions': paper_count,
        'total_mentions': reviewer_count + paper_count,
    }


def analyze_first_messages(row: pd.Series, tokenizer) -> Dict[str, Any]:
    """
    Analyze first messages from both shy and non-shy agents.
    """
    try:
        conversation = json.loads(row['full_conversation'])

        # Extract entities
        reviewers, papers = extract_entities_from_game(row)

        if not reviewers or not papers:
            return None

        # Identify shy/non-shy players
        input_ids = row['input_ids']
        full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        assistant_start = full_text.find("<|im_start|>assistant\n")

        if assistant_start == -1:
            return None

        # Find which player is assistant (non-shy)
        assistant_end = full_text.find("<|im_end|>", assistant_start)
        assistant_text = full_text[assistant_start:assistant_end] if assistant_end != -1 else full_text[assistant_start:]

        nonshy_player = None
        for msg in conversation:
            msg_text = msg['message'].strip()
            match = re.search(r'\[(message|propose|accept|reject)\](.*)', msg_text, re.DOTALL)
            if match:
                public_part = match.group(2).strip()[:150]
                if len(public_part) > 20 and public_part in assistant_text:
                    nonshy_player = msg['player']
                    break

        if not nonshy_player:
            return None

        shy_player = 'player-1' if nonshy_player == 'player-2' else 'player-2'

        # Find first messages from each agent
        nonshy_first_message = None
        shy_first_message = None

        for msg in conversation:
            if msg.get('player') == 'error':
                continue

            if msg['player'] == nonshy_player and nonshy_first_message is None:
                nonshy_first_message = extract_public_text(msg['message'])

            if msg['player'] == shy_player and shy_first_message is None:
                shy_first_message = extract_public_text(msg['message'])

            if nonshy_first_message and shy_first_message:
                break

        # Count mentions in each first message
        nonshy_counts = count_name_mentions(nonshy_first_message or "", reviewers, papers)
        shy_counts = count_name_mentions(shy_first_message or "", reviewers, papers)

        return {
            'game_id': row['game_id'],
            'conversation_length': len(conversation),
            'game_reward': row.get('game_normalized_reward', 0),
            'nonshy_player': nonshy_player,
            'shy_player': shy_player,

            # Non-shy first message
            'nonshy_first_reviewers': nonshy_counts['reviewer_mentions'],
            'nonshy_first_papers': nonshy_counts['paper_mentions'],
            'nonshy_first_total': nonshy_counts['total_mentions'],

            # Shy first message
            'shy_first_reviewers': shy_counts['reviewer_mentions'],
            'shy_first_papers': shy_counts['paper_mentions'],
            'shy_first_total': shy_counts['total_mentions'],

            # Metadata
            'num_reviewers_in_game': len(reviewers),
            'num_papers_in_game': len(papers),
        }

    except Exception as e:
        print(f"Error analyzing conversation: {e}")
        return None


def process_round(parquet_path: Path, tokenizer, round_num: int) -> pd.DataFrame:
    """Process a single round."""
    print(f"  Loading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)

    print(f"  Analyzing {len(df)} conversations...")
    results = []

    for idx, row in df.iterrows():
        result = analyze_first_messages(row, tokenizer)
        if result:
            result['round'] = round_num
            results.append(result)

        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} rows...")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Measure first-turn information sharing")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier")
    parser.add_argument("--output-dir", type=Path, default=Path("data/shy_agent_analysis"),
                       help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model for tokenizer")

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    run_id = args.run_id or run_dir.name

    print(f"Measuring first-turn information for run: {run_id}")
    print("=" * 80)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Find rounds
    round_dirs = sorted(run_dir.glob("round_*"))
    print(f"Found {len(round_dirs)} rounds\n")

    all_data = []

    for round_dir in round_dirs:
        round_name = round_dir.name
        round_num = int(round_name.split('_')[1])
        parquet_path = round_dir / "train.parquet"

        if not parquet_path.exists():
            print(f"Skipping {round_name} (no train.parquet)")
            continue

        print(f"Round {round_num:03d}:")
        round_df = process_round(parquet_path, tokenizer, round_num)
        all_data.append(round_df)
        print()

    # Combine and save
    print("Combining results...")
    combined_df = pd.concat(all_data, ignore_index=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"first_turn_info_{run_id}.parquet"
    combined_df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY BY ROUND")
    print("=" * 80)

    summary = combined_df.groupby('round').agg({
        'nonshy_first_total': ['mean', 'std'],
        'shy_first_total': ['mean', 'std'],
        'nonshy_first_reviewers': 'mean',
        'nonshy_first_papers': 'mean',
        'shy_first_reviewers': 'mean',
        'shy_first_papers': 'mean',
        'game_id': 'count',
    })

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'game_id_count': 'n_conversations'})

    print("\n", summary.round(2))

    # Save summary
    summary_path = output_dir / f"first_turn_info_summary_{run_id}.csv"
    summary.to_csv(summary_path, float_format='%.3f')
    print(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
