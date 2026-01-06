#!/usr/bin/env python3
"""Generate training data for paper-reviewer matching game.

This script generates game instances in JSONL format for SLIME training.

Usage:
    python generate_data.py --num_games 1000 --output data/games.jsonl
"""
import sys
from pathlib import Path

# Add src to path
SRC_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from data_generator import generate_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate paper-reviewer matching game instances"
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=1000,
        help="Number of games to generate (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: data/games.jsonl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter for hard games"
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = str(Path(__file__).parent.parent / "data" / "games.jsonl")

    print(f"Generating {args.num_games} games...")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Filter hard games: {not args.no_filter}")

    generate_dataset(
        num_games=args.num_games,
        output_path=args.output,
        base_seed=args.seed,
        filter_hard=not args.no_filter,
    )


if __name__ == "__main__":
    main()
