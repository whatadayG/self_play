"""
Tests for replaying captured mask/logprob errors.

This test file allows you to replay errors captured during production runs
to debug and fix mask building issues without needing a live SGLang server.

Usage:
1. Run your training/generation with the debug dumps enabled (automatic now)
2. Errors will be saved to debug_dumps/*.json
3. Run: pytest tests/test_replay_captured_errors.py -v
4. Examine the captured state and fix the bugs
5. Re-run tests to verify fixes

The test will PASS if the bug has been fixed (no error raised).
The test will FAIL if the bug still exists (error still raised).
"""

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dialop.sglang_model_player import SGLangModelPlayer, SGLangConfig


def get_debug_dump_files():
    """Find all debug dump JSON files."""
    debug_dir = PROJECT_ROOT / "debug_dumps"
    if not debug_dir.exists():
        return []
    return sorted(glob.glob(str(debug_dir / "*.json")))


@pytest.mark.parametrize("debug_dump_file", get_debug_dump_files())
def test_replay_captured_error(debug_dump_file):
    """Replay a captured error to reproduce and debug it.

    This test loads the saved state from a debug dump and attempts to
    rebuild the mask and logprobs. If the bug has been fixed, the test
    will pass. If the bug still exists, it will fail with the same error.
    """
    print(f"\n{'='*80}")
    print(f"Replaying: {os.path.basename(debug_dump_file)}")
    print(f"{'='*80}\n")

    # Load the debug dump
    with open(debug_dump_file) as f:
        debug_data = json.load(f)

    print(f"Error type: {debug_data['error_type']}")
    print(f"Error details: {debug_data['error_details']}")
    print(f"Role: {debug_data['role']}")
    print(f"Num messages: {debug_data['num_messages']}")
    print(f"Num assistant messages: {debug_data['num_assistant_messages']}")
    print(f"Logprob sublists: {debug_data['num_sublists']} with lengths {debug_data['sublist_lengths']}")
    print(f"Full sequence length: {debug_data['full_tokens_length']}")
    print()

    # Create a mock console
    console = type("_Console", (), {
        "print": lambda *args, **kwargs: None,
        "rule": lambda *args, **kwargs: None
    })()

    # Recreate the player configuration
    config_data = debug_data.get("config", {})
    config = SGLangConfig(
        temperature=config_data.get("temperature", 0.7),
        max_tokens=config_data.get("max_tokens", 4096),
        top_p=config_data.get("top_p", 0.9),
        server_url=config_data.get("server_url", "http://localhost:30000"),
    )

    # Create a player instance
    player = SGLangModelPlayer(
        system_prompt="Replaying captured error",  # Doesn't matter for replay
        role=debug_data["role"],
        console=console,
        model_path=debug_data["model_path"],
        config=config,
    )

    # Inject the saved state
    player.messages = debug_data["messages"]
    player.all_generated_logprobs = debug_data["all_generated_logprobs"]
    # Inject token strings if available (needed for fixed tokenization logic)
    if "all_generated_tokens" in debug_data:
        player.all_generated_tokens = debug_data["all_generated_tokens"]

    # Try to build the mask - this will either succeed (bug fixed) or fail (bug still exists)
    try:
        mask = player.get_assistant_mask()
        logprob_tensor = player.get_generated_logprob_tensor()

        # If we get here, the bug has been fixed!
        print(f"✓ BUG FIXED! The error no longer reproduces.")
        print(f"  Mask length: {len(mask)}")
        print(f"  Logprob tensor length: {len(logprob_tensor)}")
        print(f"  Masked positions: {sum(mask)}")

        # Verify basic alignment
        assert len(mask) == len(logprob_tensor), "Mask and logprob tensor lengths don't match"
        assert len(mask) == debug_data['full_tokens_length'], "Mask length doesn't match input sequence"

        # The test passes - bug is fixed!

    except RuntimeError as e:
        # The bug still exists - test fails
        error_str = str(e)
        print(f"\n✗ BUG STILL EXISTS:")
        print(f"  {error_str[:200]}...")
        print(f"\nFull error details saved in: {debug_dump_file}")

        # Add some analysis to help debug
        print(f"\nDebugging hints:")
        print(f"  - Check if logprob collection is working correctly")
        print(f"  - Verify SGLang is returning the expected number of logprobs")
        print(f"  - Look at the message content for patterns (thinking content, special tokens, etc.)")
        print(f"  - Compare sublist lengths {debug_data['sublist_lengths']} with expected token counts")

        # Re-raise to fail the test
        raise


def test_debug_dumps_exist():
    """Meta-test to check if debug dumps directory exists and has files.

    This test helps users understand how to use the replay feature.
    """
    debug_dir = PROJECT_ROOT / "debug_dumps"
    dump_files = get_debug_dump_files()

    if not debug_dir.exists():
        pytest.skip(
            f"Debug dumps directory doesn't exist yet: {debug_dir}\n"
            f"Run your training/generation to create error dumps automatically."
        )

    if len(dump_files) == 0:
        pytest.skip(
            f"No debug dump files found in {debug_dir}\n"
            f"Errors will be automatically dumped there when they occur."
        )

    print(f"\nFound {len(dump_files)} debug dump files:")
    for f in dump_files:
        print(f"  - {os.path.basename(f)}")


def test_analyze_error_patterns():
    """Analyze patterns across all captured errors.

    This helps identify systematic issues (e.g., always off by 4 tokens).
    """
    dump_files = get_debug_dump_files()

    if len(dump_files) == 0:
        pytest.skip("No debug dumps to analyze")

    print(f"\nAnalyzing {len(dump_files)} error dumps:\n")

    error_types = {}
    logprob_discrepancies = []

    for dump_file in dump_files:
        with open(dump_file) as f:
            data = json.load(f)

        error_type = data['error_type']
        error_types[error_type] = error_types.get(error_type, 0) + 1

        # Extract discrepancy info if available
        details = data.get('error_details', {})
        if 'unclaimed_logprobs' in details:
            logprob_discrepancies.append(details['unclaimed_logprobs'])

    print("Error type distribution:")
    for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")

    if logprob_discrepancies:
        print(f"\nLogprob discrepancies (unclaimed logprobs):")
        print(f"  Count: {len(logprob_discrepancies)}")
        print(f"  Min: {min(logprob_discrepancies)}")
        print(f"  Max: {max(logprob_discrepancies)}")
        print(f"  Mean: {sum(logprob_discrepancies) / len(logprob_discrepancies):.2f}")
        print(f"  Values: {sorted(logprob_discrepancies)}")

        # Check if there's a consistent pattern
        if len(set(logprob_discrepancies)) == 1:
            print(f"\n  ⚠ CONSISTENT PATTERN: All discrepancies are {logprob_discrepancies[0]}!")
            print(f"     This suggests a systematic offset in logprob collection.")
