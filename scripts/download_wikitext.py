#!/usr/bin/env python3
"""
Download and prepare a sample of wikitext-2-raw for evaluation.
Extracts approximately 100k tokens to data/eval/wikitext_sample.txt
"""

import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def download_wikitext_sample(output_dir: str = "data/eval", target_tokens: int = 100_000, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Download wikitext-2-raw and extract a sample with target number of tokens.

    Args:
        output_dir: Directory to save the sample
        target_tokens: Approximate number of tokens to extract
        model_name: Model name for tokenizer (to count tokens accurately)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading wikitext-2-raw dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Collect text until we reach target tokens
    collected_text = []
    total_tokens = 0

    print(f"Extracting ~{target_tokens} tokens...")
    for i, example in enumerate(dataset):
        text = example["text"].strip()
        if not text:  # Skip empty lines
            continue

        # Count tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)

        collected_text.append(text)
        total_tokens += token_count

        if total_tokens >= target_tokens:
            print(f"Reached {total_tokens} tokens after {i+1} examples")
            break

    # Join text with newlines
    sample_text = "\n".join(collected_text)

    # Save to file
    output_file = output_path / "wikitext_sample.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    print(f"✓ Saved {total_tokens} tokens to {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"  Number of examples: {len(collected_text)}")

    return str(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download wikitext-2-raw sample for evaluation")
    parser.add_argument("--output-dir", type=str, default="data/eval", help="Output directory")
    parser.add_argument("--target-tokens", type=int, default=100_000, help="Target number of tokens")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name for tokenizer")

    args = parser.parse_args()

    download_wikitext_sample(
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        model_name=args.model_name
    )
