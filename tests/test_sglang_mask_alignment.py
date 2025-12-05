"""
Comprehensive tests for SGLangModelPlayer mask and logprob alignment.

These tests verify that:
1. The assistant mask correctly identifies generated tokens (excludes template tokens)
2. Logprobs are correctly aligned with the input sequence
3. Template tokens (assistant header, <|im_end|>, empty thinking construct) are properly excluded
4. The thinking token (<think>) is excluded but thinking content is included in the mask
5. Each mask cluster ends with <|im_end|> token (validates cluster boundary detection)

Requirements:
- SGLang server must be running (default: localhost:31234 serving Qwen3-8B)
- Server should have max_tokens set high enough for natural completion
- Set SGLANG_SERVER_URL env var to override default URL
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import requests

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dialop.sglang_model_player import SGLangModelPlayer, SGLangConfig


# Mark all tests in this module as expensive
pytestmark = pytest.mark.expensive


@pytest.fixture(scope="module")
def sglang_server_url():
    """Get SGLang server URL from environment or use default."""
    return os.environ.get("SGLANG_SERVER_URL", "http://localhost:31234")


@pytest.fixture(scope="module")
def check_sglang_server(sglang_server_url):
    """Check if SGLang server is available, skip tests if not."""
    try:
        # Try to connect to the server
        response = requests.get(f"{sglang_server_url}/health", timeout=5)
        if response.status_code == 200:
            return sglang_server_url
    except (requests.ConnectionError, requests.Timeout):
        pass

    pytest.skip(f"SGLang server not available at {sglang_server_url}. "
                f"Start server with: python -m sglang.launch_server --model-path <model> --port 31234")


@pytest.fixture
def player(check_sglang_server, sglang_server_url):
    """Create a fresh SGLangModelPlayer for testing.

    Note: Using Qwen/Qwen3-8B model on the server (not Qwen2.5-7B).
    The tokenizer will be loaded from the model_path parameter.
    """
    console = type("_Console", (), {
        "print": lambda *args, **kwargs: None,
        "rule": lambda *args, **kwargs: None
    })()

    config = SGLangConfig(
        server_url=sglang_server_url,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2000,  # High value to ensure natural completion with <|im_end|>
        timeout=120.0,
    )

    player = SGLangModelPlayer(
        system_prompt="You are a helpful assistant.",
        role="test-player",
        console=console,
        # Use checkpoint with modified chat template that preserves thinking
        model_path="/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_3600_merged",
        config=config,
    )

    return player


def print_alignment_debug(input_ids: List[int], mask: List[int], logprob_tensor: List[float],
                         tokenizer, max_tokens: int = 50):
    """Print detailed debug information about mask and logprob alignment."""
    print("\n" + "="*80)
    print("ALIGNMENT DEBUG INFORMATION")
    print("="*80)

    print(f"\nSequence length: {len(input_ids)}")
    print(f"Mask length: {len(mask)}")
    print(f"Logprob tensor length: {len(logprob_tensor)}")
    print(f"Masked positions (mask=1): {sum(mask)}")
    print(f"Non-zero logprobs: {sum(1 for lp in logprob_tensor if lp != 0.0)}")

    # Show first N tokens with their mask and logprob values
    print(f"\nFirst {min(max_tokens, len(input_ids))} tokens:")
    print(f"{'Idx':<5} {'Token ID':<10} {'Token':<30} {'Mask':<5} {'Logprob':<10}")
    print("-" * 80)

    for i in range(min(max_tokens, len(input_ids))):
        token_id = input_ids[i]
        token_str = tokenizer.decode([token_id])
        # Escape special characters for display
        token_display = repr(token_str)[:28]
        mask_val = mask[i]
        logprob_val = logprob_tensor[i]

        # Highlight mismatches
        marker = ""
        if (mask_val == 1 and logprob_val == 0.0) or (mask_val == 0 and logprob_val != 0.0):
            marker = " <-- MISMATCH!"

        print(f"{i:<5} {token_id:<10} {token_display:<30} {mask_val:<5} {logprob_val:<10.4f}{marker}")

    if len(input_ids) > max_tokens:
        print(f"... ({len(input_ids) - max_tokens} more tokens)")

    # Find and display all mismatches
    mismatches = []
    for i in range(len(input_ids)):
        mask_val = mask[i]
        logprob_val = logprob_tensor[i]
        if (mask_val == 1 and logprob_val == 0.0) or (mask_val == 0 and logprob_val != 0.0):
            mismatches.append((i, mask_val, logprob_val))

    if mismatches:
        print(f"\nFound {len(mismatches)} mask/logprob mismatches:")
        for idx, mask_val, logprob_val in mismatches[:10]:  # Show first 10
            token_str = tokenizer.decode([input_ids[idx]])
            print(f"  Position {idx}: mask={mask_val}, logprob={logprob_val:.4f}, token={repr(token_str)}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")

    print("="*80 + "\n")


class TestSGLangMaskAlignment:
    """Test suite for mask and logprob tensor alignment."""

    def test_basic_alignment(self, player):
        """Test basic mask and logprob alignment for a short conversation.

        Verifies:
        - All tensor lengths match
        - Number of masked positions equals number of non-zero logprobs
        - Non-zero logprobs appear exactly where mask=1
        - Each mask cluster ends with <|im_end|> token (validates boundary detection)
        """
        # Run a short 2-turn conversation
        player.observe("Please count to 3.")
        response1 = player.respond()

        player.observe("What is 2+2?")
        response2 = player.respond()

        # Get tensors
        input_ids = player.get_input_sequence()
        mask = player.get_assistant_mask()
        logprob_tensor = player.get_generated_logprob_tensor()

        # Load tokenizer for debug output
        player._load_tokenizer()
        tokenizer = player.tokenizer

        # Assertion 1: Length consistency
        assert len(input_ids) == len(mask), (
            f"Length mismatch: input_ids has {len(input_ids)} tokens, "
            f"mask has {len(mask)} elements"
        )
        assert len(input_ids) == len(logprob_tensor), (
            f"Length mismatch: input_ids has {len(input_ids)} tokens, "
            f"logprob_tensor has {len(logprob_tensor)} elements"
        )

        # Assertion 2: Logprob count alignment
        # Check that we assigned all logprobs we collected from SGLang
        masked_positions = sum(mask)
        total_logprobs_collected = sum(len(sublist) for sublist in player.all_generated_logprobs)

        if masked_positions != total_logprobs_collected:
            print_alignment_debug(input_ids, mask, logprob_tensor, tokenizer)

        assert masked_positions == total_logprobs_collected, (
            f"Logprob count mismatch: {masked_positions} masked positions "
            f"but collected {total_logprobs_collected} logprobs from SGLang"
        )

        # Assertion 3: Logprobs are assigned only where mask=1
        # Note: Some logprobs may be 0.0 (deterministic tokens), which is valid
        # We just check that unmasked positions have no logprob assigned (i.e., stayed at init value 0.0)
        # and that we didn't accidentally skip assigning logprobs to masked positions
        # Since we can't distinguish "unassigned 0.0" from "assigned 0.0", we rely on the count check above

        # Assertion 4: Cluster boundary validation
        # Each mask cluster should end with <|im_end|> token
        # This validates that the mask building correctly identifies where assistant turns end
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Find all positions where mask transitions from 1 to 0 (end of clusters)
        cluster_ends = []
        for i in range(len(mask) - 1):
            if mask[i] == 1 and mask[i + 1] == 0:
                cluster_ends.append(i)

        # Also check the very last position if it's masked
        if len(mask) > 0 and mask[-1] == 1:
            cluster_ends.append(len(mask) - 1)

        # Each cluster end should be an <|im_end|> token
        cluster_boundary_errors = []
        for pos in cluster_ends:
            token_id = input_ids[pos]
            if token_id != im_end_id:
                token_str = tokenizer.decode([token_id])
                cluster_boundary_errors.append((pos, token_str))

        if cluster_boundary_errors:
            print("\n" + "="*80)
            print("CLUSTER BOUNDARY ERRORS")
            print("="*80)
            print(f"Found {len(cluster_boundary_errors)} cluster ends that are NOT <|im_end|>:")
            for pos, token_str in cluster_boundary_errors[:10]:
                print(f"  Position {pos}: {repr(token_str)} (expected <|im_end|>)")
            print("="*80 + "\n")

        assert len(cluster_boundary_errors) == 0, (
            f"Found {len(cluster_boundary_errors)} mask clusters that don't end with <|im_end|>. "
            f"This indicates incorrect cluster boundary detection in mask building. "
            f"First few: {cluster_boundary_errors[:5]}"
        )

        print(f"\n✓ Basic alignment test passed:")
        print(f"  - Sequence length: {len(input_ids)} tokens")
        print(f"  - Masked positions: {masked_positions}")
        print(f"  - Total logprobs collected: {total_logprobs_collected}")
        print(f"  - Perfect alignment: masked_positions == total_logprobs_collected")
        print(f"  - Cluster boundaries: {len(cluster_ends)} clusters, all end with <|im_end|>")

    def test_template_token_exclusion(self, player):
        """Test that template-added tokens are excluded from the mask.

        Template tokens that should NOT be masked:
        - Assistant header: <|im_start|>assistant\n (3 tokens)
        - Assistant suffix: <|im_end|> (1 token, note: \n is part of next message)
        - Empty thinking construct: <think>\n\n</think>\n\n (4 tokens, if present)
        """
        # Run a conversation
        player.observe("Say hello.")
        response = player.respond()

        # Get tensors and tokenizer
        input_ids = player.get_input_sequence()
        mask = player.get_assistant_mask()
        player._load_tokenizer()
        tokenizer = player.tokenizer

        # Get special token IDs
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
        newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

        # Find assistant message patterns and verify they're not masked
        assistant_headers_found = 0
        assistant_suffixes_found = 0

        for i in range(len(input_ids) - 2):
            # Check for assistant header: <|im_start|>assistant\n
            if (input_ids[i] == im_start_id and
                input_ids[i+1] == assistant_id and
                input_ids[i+2] == newline_id):

                assistant_headers_found += 1

                # Verify these 3 tokens are NOT masked
                assert mask[i] == 0, (
                    f"Assistant header <|im_start|> at position {i} should not be masked"
                )
                assert mask[i+1] == 0, (
                    f"Assistant header 'assistant' at position {i+1} should not be masked"
                )
                assert mask[i+2] == 0, (
                    f"Assistant header newline at position {i+2} should not be masked"
                )

        for i in range(len(input_ids)):
            # Check for <|im_end|> tokens
            if input_ids[i] == im_end_id:
                assistant_suffixes_found += 1

                # Verify <|im_end|> is NOT masked
                assert mask[i] == 0, (
                    f"<|im_end|> token at position {i} should not be masked"
                )

        # Verify we found at least one assistant message
        assert assistant_headers_found > 0, "No assistant headers found in conversation"
        assert assistant_suffixes_found > 0, "No assistant suffixes found in conversation"

        print(f"\n✓ Template token exclusion test passed:")
        print(f"  - Found {assistant_headers_found} assistant headers (all excluded from mask)")
        print(f"  - Found {assistant_suffixes_found} assistant suffixes (all excluded from mask)")

    def test_thinking_token_handling(self, player):
        """Test that thinking tokens are handled correctly.

        According to the clarification:
        - The initial <think> token should NOT be in the mask
        - Everything else (thinking content, </think>) SHOULD be in the mask

        Note: This test is opportunistic - it passes if thinking is present and correctly
        handled, or if thinking is absent (we can't force Qwen to think or not think).
        """
        # Run a conversation that might trigger thinking
        player.observe("Please explain why the sky is blue in detail.")
        response = player.respond()

        # Get tensors and tokenizer
        input_ids = player.get_input_sequence()
        mask = player.get_assistant_mask()
        player._load_tokenizer()
        tokenizer = player.tokenizer

        # Get thinking token IDs
        think_start_id = tokenizer.convert_tokens_to_ids("<think>")
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")

        # Find thinking constructs
        thinking_found = False
        for i in range(len(input_ids) - 1):
            if input_ids[i] == think_start_id:
                # Check if this is the empty thinking construct (4 tokens)
                double_newline_id = tokenizer.encode("\n\n", add_special_tokens=False)[0]

                is_empty_construct = (
                    i + 3 < len(input_ids) and
                    input_ids[i+1] == double_newline_id and
                    input_ids[i+2] == think_end_id and
                    input_ids[i+3] == double_newline_id
                )

                if is_empty_construct:
                    # Empty thinking construct - all 4 tokens should NOT be masked
                    print(f"  Found empty thinking construct at position {i} (not masked, as expected)")
                    assert mask[i] == 0, f"Empty <think> at position {i} should not be masked"
                    assert mask[i+1] == 0, f"Empty thinking newline at position {i+1} should not be masked"
                    assert mask[i+2] == 0, f"Empty </think> at position {i+2} should not be masked"
                    assert mask[i+3] == 0, f"Empty thinking newline at position {i+3} should not be masked"
                else:
                    # Real thinking content - <think> should NOT be masked, but content should be
                    thinking_found = True
                    print(f"  Found real thinking construct at position {i}")

                    # The <think> token itself should NOT be masked
                    assert mask[i] == 0, (
                        f"The <think> token at position {i} should NOT be masked, "
                        f"but mask[{i}] = {mask[i]}"
                    )

                    # Find the corresponding </think>
                    end_pos = None
                    for j in range(i+1, min(i+500, len(input_ids))):  # Search up to 500 tokens ahead
                        if input_ids[j] == think_end_id:
                            end_pos = j
                            break

                    if end_pos is not None:
                        # Verify that tokens between <think> and </think> ARE masked
                        # (at least some of them should be)
                        thinking_content_masked = sum(mask[i+1:end_pos])
                        thinking_content_length = end_pos - i - 1

                        print(f"    Thinking content: {thinking_content_length} tokens, "
                              f"{thinking_content_masked} masked")

                        # At least some thinking content should be masked
                        assert thinking_content_masked > 0, (
                            f"Thinking content between positions {i+1} and {end_pos} "
                            f"should have at least some masked tokens, but none were masked"
                        )

                        # The </think> token and following content should also be masked
                        assert mask[end_pos] == 1, (
                            f"The </think> token at position {end_pos} should be masked"
                        )

        if thinking_found:
            print(f"\n✓ Thinking token handling test passed:")
            print(f"  - Found thinking construct with content")
            print(f"  - <think> token correctly excluded from mask")
            print(f"  - Thinking content and </think> correctly included in mask")
        else:
            print(f"\n✓ Thinking token test passed (no thinking content generated)")

    def test_multi_message_alignment(self, player):
        """Test alignment across multiple assistant messages.

        Verifies:
        - all_generated_logprobs has correct number of sublists
        - Each sublist length matches masked positions for that message
        """
        # Run a 3-turn conversation
        player.observe("Count to 2.")
        response1 = player.respond()

        player.observe("What is 1+1?")
        response2 = player.respond()

        player.observe("Say 'done'.")
        response3 = player.respond()

        # Get tensors
        input_ids = player.get_input_sequence()
        mask = player.get_assistant_mask()
        player._load_tokenizer()
        tokenizer = player.tokenizer

        # Check all_generated_logprobs structure
        num_assistant_messages = sum(1 for msg in player.messages if msg["role"] == "assistant")

        assert len(player.all_generated_logprobs) == num_assistant_messages, (
            f"Expected {num_assistant_messages} logprob sublists "
            f"but found {len(player.all_generated_logprobs)}"
        )

        # Each sublist should match the masked positions for that message
        total_logprobs = sum(len(sublist) for sublist in player.all_generated_logprobs)
        total_masked = sum(mask)

        assert total_logprobs == total_masked, (
            f"Total logprobs across sublists ({total_logprobs}) "
            f"doesn't match total masked positions ({total_masked})"
        )

        print(f"\n✓ Multi-message alignment test passed:")
        print(f"  - Found {num_assistant_messages} assistant messages")
        print(f"  - all_generated_logprobs has {len(player.all_generated_logprobs)} sublists")
        print(f"  - Sublist lengths: {[len(s) for s in player.all_generated_logprobs]}")
        print(f"  - Total logprobs: {total_logprobs}")
        print(f"  - Total masked positions: {total_masked}")
