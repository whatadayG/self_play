#!/usr/bin/env python3
"""
Measure mode collapse in dialogue using multiple complementary metrics.

Focus areas:
1. Information content stereotyping (how many facts shared, especially in first turn)
2. Text diversity (self-BLEU, distinct-N, vocabulary richness)
3. Strategic diversity (message patterns, proposal timing, turn structure)
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any
import argparse
from transformers import AutoTokenizer


def extract_entities_from_game(row: pd.Series) -> Tuple[List[str], List[str]]:
    """
    Extract reviewer and paper names from the game context.

    Returns:
        (reviewer_names, paper_names)
    """
    # Try to extract from full_conversation
    try:
        conversation = json.loads(row['full_conversation']) if isinstance(row['full_conversation'], str) else row['full_conversation']

        # Get the first message which typically contains the table
        first_message = conversation[0]['message'] if conversation else ""

        # Extract reviewer names (capitalize first letter of each word)
        # Common patterns: "Daniel Nguyen", "Sofia Patel", etc.
        reviewer_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        reviewers = list(set(re.findall(reviewer_pattern, first_message)))

        # Extract paper names (typically have colons or are all caps acronyms)
        # Patterns: "BLEU:", "GloVe:", "RoBERTa:", etc.
        paper_pattern = r'\b([A-Z][A-Za-z]*(?:[A-Z][a-z]*)*):?'
        potential_papers = set(re.findall(paper_pattern, first_message))

        # Filter out reviewer names from paper list
        reviewer_first_names = {r.split()[0] for r in reviewers}
        papers = [p for p in potential_papers if p not in reviewer_first_names and len(p) > 2]

        return reviewers, papers
    except Exception as e:
        return [], []


def extract_information_units(message: str, reviewers: List[str], papers: List[str]) -> Dict[str, Any]:
    """
    Extract information units from a message using permissive co-occurrence matching.

    Args:
        message: The message text
        reviewers: List of reviewer names in this game
        papers: List of paper names in this game

    Returns:
        Dictionary with extracted information units
    """
    # Normalize message
    message_lower = message.lower()

    # Find reviewer-paper pair mentions (permissive: any proximity, any order)
    pairs_mentioned = set()
    qualitative_statements = []
    numbers_mentioned = []

    # For each reviewer and paper, check if they appear near each other
    for reviewer in reviewers:
        reviewer_lower = reviewer.lower()
        if reviewer_lower not in message_lower:
            continue

        # Find all positions of this reviewer in the message
        reviewer_positions = [m.start() for m in re.finditer(re.escape(reviewer_lower), message_lower)]

        for paper in papers:
            paper_lower = paper.lower()
            if paper_lower not in message_lower:
                continue

            # Find all positions of this paper
            paper_positions = [m.start() for m in re.finditer(re.escape(paper_lower), message_lower)]

            # Check if any positions are within 100 characters of each other
            for rev_pos in reviewer_positions:
                for paper_pos in paper_positions:
                    if abs(rev_pos - paper_pos) < 100:
                        pairs_mentioned.add((reviewer, paper))

                        # Extract nearby qualitative words
                        start = min(rev_pos, paper_pos)
                        end = max(rev_pos, paper_pos) + max(len(reviewer_lower), len(paper_lower))
                        snippet = message_lower[max(0, start-50):min(len(message_lower), end+50)]

                        # Look for qualitative descriptors
                        qualifiers = ['best', 'highest', 'strong', 'top', 'excellent', 'good', 'great',
                                    'weak', 'low', 'poor', 'decent', 'solid', 'very', 'high', 'clear']
                        for qual in qualifiers:
                            if qual in snippet:
                                qualitative_statements.append(f"{reviewer}-{paper}: {qual}")

                        break

    # Extract numbers (scores, rankings)
    number_pattern = r'\b\d+\b'
    numbers_mentioned = re.findall(number_pattern, message)

    # Extract ranking statements
    ranking_pattern = r'\b(first|second|third|fourth|fifth|top|primary|main)\b'
    rankings = re.findall(ranking_pattern, message_lower)

    return {
        'pairs_count': len(pairs_mentioned),
        'pairs': list(pairs_mentioned),
        'qualitative_count': len(set(qualitative_statements)),
        'numbers_count': len(set(numbers_mentioned)),
        'ranking_count': len(rankings),
        'total_info_units': len(pairs_mentioned) + len(set(qualitative_statements)) + len(set(numbers_mentioned)),
    }


def calculate_self_bleu(texts: List[str], n: int = 2, sample_size: int = 500) -> float:
    """
    Calculate self-BLEU score for a list of texts.
    Measures how similar texts are to each other.

    Lower scores indicate more diversity.

    Simplified implementation without NLTK dependency.
    """
    import random

    if len(texts) < 2:
        return 0.0

    # Sample for efficiency
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    scores = []

    for i, hypothesis_text in enumerate(texts):
        # Tokenize
        hypothesis = hypothesis_text.split()

        # Use other texts as references
        references = [t.split() for j, t in enumerate(texts) if j != i]

        if not references or len(hypothesis) < n:
            continue

        # Calculate n-gram precision
        hyp_ngrams = Counter()
        for j in range(len(hypothesis) - n + 1):
            ngram = tuple(hypothesis[j:j+n])
            hyp_ngrams[ngram] += 1

        # Max count across references
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter()
            for j in range(len(ref) - n + 1):
                ngram = tuple(ref[j:j+n])
                ref_ngrams[ngram] += 1

            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

        # Calculate clipped counts
        clipped_counts = sum(min(count, max_ref_counts.get(ngram, 0))
                           for ngram, count in hyp_ngrams.items())
        total_counts = sum(hyp_ngrams.values())

        precision = clipped_counts / total_counts if total_counts > 0 else 0
        scores.append(precision)

    return np.mean(scores) if scores else 0.0


def calculate_distinct_n(texts: List[str], n: int) -> float:
    """
    Calculate distinct-n metric: ratio of unique n-grams to total n-grams.

    Higher scores indicate more diversity.
    """
    all_ngrams = []

    for text in texts:
        words = text.split()
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique = len(set(all_ngrams))
    total = len(all_ngrams)

    return unique / total if total > 0 else 0.0


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


def analyze_conversation_metrics(row: pd.Series, tokenizer) -> Dict[str, Any]:
    """
    Analyze a single conversation for mode collapse metrics.
    """
    try:
        conversation = json.loads(row['full_conversation'])

        # Extract entities
        reviewers, papers = extract_entities_from_game(row)

        # Determine shy/non-shy players
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

        # Analyze messages
        nonshy_messages = []
        shy_messages = []
        first_turn_info = None
        proposal_turn = None

        for msg in conversation:
            if msg.get('player') not in ['player-1', 'player-2']:
                continue

            public_text = extract_public_text(msg['message'])
            if not public_text:
                continue

            # Extract information units
            info_units = extract_information_units(public_text, reviewers, papers)

            # Track first turn from non-shy agent
            if msg['player'] == nonshy_player and first_turn_info is None:
                first_turn_info = info_units

            # Track proposal timing
            if '[propose]' in msg['message'] and proposal_turn is None:
                proposal_turn = msg['turn']

            # Collect messages by agent
            if msg['player'] == nonshy_player:
                nonshy_messages.append(public_text)
            else:
                shy_messages.append(public_text)

        return {
            'game_id': row['game_id'],
            'nonshy_player': nonshy_player,
            'shy_player': shy_player,
            'conversation_length': len(conversation),
            'game_reward': row.get('game_normalized_reward', 0),

            # First-turn information
            'first_turn_pairs': first_turn_info['pairs_count'] if first_turn_info else 0,
            'first_turn_qualitative': first_turn_info['qualitative_count'] if first_turn_info else 0,
            'first_turn_numbers': first_turn_info['numbers_count'] if first_turn_info else 0,
            'first_turn_total_info': first_turn_info['total_info_units'] if first_turn_info else 0,

            # Strategic patterns
            'proposal_turn': proposal_turn if proposal_turn is not None else -1,
            'nonshy_message_count': len(nonshy_messages),
            'shy_message_count': len(shy_messages),

            # Message texts for diversity analysis later
            'nonshy_messages': nonshy_messages,
            'shy_messages': shy_messages,
        }

    except Exception as e:
        print(f"Error analyzing conversation: {e}")
        return None


def process_round(parquet_path: Path, tokenizer, round_num: int) -> pd.DataFrame:
    """Process a single round and extract mode collapse metrics."""
    print(f"  Loading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)

    print(f"  Analyzing {len(df)} conversations...")
    results = []

    for idx, row in df.iterrows():
        result = analyze_conversation_metrics(row, tokenizer)
        if result:
            result['round'] = round_num
            results.append(result)

        if (idx + 1) % 1000 == 0:
            print(f"    Processed {idx + 1}/{len(df)} rows...")

    return pd.DataFrame(results)


def calculate_diversity_metrics(df: pd.DataFrame, round_num: int) -> Dict[str, Any]:
    """Calculate text diversity metrics for a round."""
    print(f"  Calculating diversity metrics for round {round_num}...")

    # Collect all non-shy messages
    all_nonshy = []
    for messages in df['nonshy_messages']:
        all_nonshy.extend(messages)

    if not all_nonshy:
        return {}

    # Self-BLEU (sample for efficiency)
    self_bleu = calculate_self_bleu(all_nonshy, n=2, sample_size=500)

    # Distinct-N
    distinct_1 = calculate_distinct_n(all_nonshy, 1)
    distinct_2 = calculate_distinct_n(all_nonshy, 2)
    distinct_3 = calculate_distinct_n(all_nonshy, 3)

    # Vocabulary diversity
    all_words = ' '.join(all_nonshy).split()
    type_token_ratio = len(set(all_words)) / len(all_words) if all_words else 0

    # Message length stats
    message_lengths = [len(msg.split()) for msg in all_nonshy]

    return {
        'round': round_num,
        'self_bleu': self_bleu,
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'distinct_3': distinct_3,
        'type_token_ratio': type_token_ratio,
        'vocab_size': len(set(all_words)),
        'message_length_mean': np.mean(message_lengths) if message_lengths else 0,
        'message_length_std': np.std(message_lengths) if message_lengths else 0,
        'message_length_entropy': -sum((np.histogram(message_lengths, bins=20)[0] / len(message_lengths)) *
                                      np.log(np.histogram(message_lengths, bins=20)[0] / len(message_lengths) + 1e-10))
                                  if message_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure mode collapse in dialogues")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier")
    parser.add_argument("--output-dir", type=Path, default=Path("data/shy_agent_analysis"),
                       help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model for tokenizer")

    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    run_id = args.run_id or run_dir.name

    print(f"Measuring mode collapse for run: {run_id}")
    print("=" * 80)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Find rounds
    round_dirs = sorted(run_dir.glob("round_*"))
    print(f"Found {len(round_dirs)} rounds\n")

    all_conversation_data = []
    all_diversity_metrics = []

    for round_dir in round_dirs:
        round_name = round_dir.name
        round_num = int(round_name.split('_')[1])
        parquet_path = round_dir / "train.parquet"

        if not parquet_path.exists():
            print(f"Skipping {round_name} (no train.parquet)")
            continue

        print(f"Round {round_num:03d}:")

        # Process conversations
        round_df = process_round(parquet_path, tokenizer, round_num)
        all_conversation_data.append(round_df)

        # Calculate diversity metrics
        diversity_metrics = calculate_diversity_metrics(round_df, round_num)
        if diversity_metrics:
            all_diversity_metrics.append(diversity_metrics)

        print()

    # Combine all data
    print("Combining results...")
    conversations_df = pd.concat(all_conversation_data, ignore_index=True)

    # Drop message lists for saving (too large)
    conversations_save = conversations_df.drop(columns=['nonshy_messages', 'shy_messages'])

    # Save datasets
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    conversations_path = output_dir / f"mode_collapse_conversations_{run_id}.parquet"
    conversations_save.to_parquet(conversations_path, index=False)
    print(f"Saved conversation analysis: {conversations_path}")

    diversity_df = pd.DataFrame(all_diversity_metrics)
    diversity_path = output_dir / f"mode_collapse_diversity_{run_id}.csv"
    diversity_df.to_csv(diversity_path, index=False, float_format='%.4f')
    print(f"Saved diversity metrics: {diversity_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nFirst-turn information by round:")
    first_turn_summary = conversations_df.groupby('round').agg({
        'first_turn_total_info': ['mean', 'std'],
        'first_turn_pairs': 'mean',
    }).round(3)
    print(first_turn_summary)

    print("\nDiversity metrics by round:")
    print(diversity_df.round(4))

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
