#!/usr/bin/env python3
"""
Test wikitext evaluation with memory profiling for Qwen3-8B.
Tests different batch sizes and sequence lengths to find optimal settings.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from verl.utils.wikitext_eval import compute_wikitext_loss


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0


def test_wikitext_eval(
    model_path: str,
    wikitext_path: str = "data/eval/wikitext_sample.txt",
    max_seq_length: int = 2048,
    batch_size: int = 8,
    device: str = "cuda:0",
):
    """Test wikitext evaluation with memory profiling.

    Args:
        model_path: Path to model checkpoint
        wikitext_path: Path to wikitext sample
        max_seq_length: Max sequence length for chunks
        batch_size: Batch size for evaluation
        device: Device to use
    """
    print(f"=" * 80)
    print(f"Testing Wikitext Evaluation")
    print(f"=" * 80)
    print(f"Model: {model_path}")
    print(f"Wikitext: {wikitext_path}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"=" * 80)

    # Check if wikitext file exists
    if not os.path.exists(wikitext_path):
        print(f"ERROR: Wikitext file not found at {wikitext_path}")
        print("Run: python scripts/download_wikitext.py")
        return

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"  ✓ Tokenizer loaded")

    # Load model
    print("\n[2/4] Loading model...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    load_time = time.time() - start_time
    model_memory = get_gpu_memory()
    print(f"  ✓ Model loaded in {load_time:.2f}s")
    print(f"  ✓ Model memory: {model_memory:.2f} GB")

    # Run evaluation
    print("\n[3/4] Running wikitext evaluation...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        metrics = compute_wikitext_loss(
            model=model,
            tokenizer=tokenizer,
            wikitext_path=wikitext_path,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            device=device,
        )

        eval_time = time.time() - start_time
        peak_memory = get_gpu_memory()

        print(f"  ✓ Evaluation completed in {eval_time:.2f}s")
        print(f"  ✓ Peak memory: {peak_memory:.2f} GB")

        # Print results
        print("\n[4/4] Results:")
        print(f"  Loss: {metrics['wikitext/loss']:.4f}")
        print(f"  Perplexity: {metrics['wikitext/perplexity']:.4f}")
        print(f"  Tokens evaluated: {metrics['wikitext/num_tokens']}")
        print(f"  Tokens/sec: {metrics['wikitext/num_tokens'] / eval_time:.2f}")

        return metrics

    except torch.cuda.OutOfMemoryError as e:
        print(f"  ✗ OUT OF MEMORY!")
        print(f"  Error: {e}")
        print(f"  Try reducing batch_size or max_seq_length")
        return None


def benchmark_configurations(model_path: str, wikitext_path: str, device: str = "cuda:0"):
    """Benchmark different configurations to find optimal settings."""
    print("\n" + "=" * 80)
    print("BENCHMARKING DIFFERENT CONFIGURATIONS")
    print("=" * 80)

    configs = [
        # (max_seq_length, batch_size)
        (1024, 16),
        (1024, 8),
        (2048, 8),
        (2048, 4),
        (4096, 4),
        (4096, 2),
    ]

    results = []

    for max_seq_len, batch_sz in configs:
        print(f"\n--- Testing: max_seq_length={max_seq_len}, batch_size={batch_sz} ---")
        torch.cuda.empty_cache()

        try:
            metrics = test_wikitext_eval(
                model_path=model_path,
                wikitext_path=wikitext_path,
                max_seq_length=max_seq_len,
                batch_size=batch_sz,
                device=device,
            )

            if metrics is not None:
                results.append({
                    "max_seq_length": max_seq_len,
                    "batch_size": batch_sz,
                    "loss": metrics["wikitext/loss"],
                    "perplexity": metrics["wikitext/perplexity"],
                    "peak_memory_gb": get_gpu_memory(),
                })
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Max Seq Len':<15} {'Batch Size':<12} {'Loss':<10} {'Perplexity':<12} {'Peak Mem (GB)':<15}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['max_seq_length']:<15} {r['batch_size']:<12} {r['loss']:<10.4f} "
            f"{r['perplexity']:<12.4f} {r['peak_memory_gb']:<15.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test wikitext evaluation with memory profiling")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Path to model checkpoint or HF model name"
    )
    parser.add_argument(
        "--wikitext-path",
        type=str,
        default="data/eval/wikitext_sample.txt",
        help="Path to wikitext sample"
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on multiple configurations")

    args = parser.parse_args()

    if args.benchmark:
        benchmark_configurations(
            model_path=args.model_path,
            wikitext_path=args.wikitext_path,
            device=args.device,
        )
    else:
        test_wikitext_eval(
            model_path=args.model_path,
            wikitext_path=args.wikitext_path,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            device=args.device,
        )
