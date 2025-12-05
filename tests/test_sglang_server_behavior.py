"""
Test SGLang server behavior with Qwen3-8B.

This test suite documents the expected behavior of the SGLang server when serving
Qwen3-8B with thinking mode enabled. It serves as a specification for the logprob
and tokenization behavior that our GRPO implementation depends on.

IMPORTANT: This test requires a running SGLang server on port 31234 serving Qwen3-8B.
The model served by SGLang MUST match the tokenizer model name used in this test.

To run these tests:
    pytest tests/test_sglang_server_behavior.py -m expensive

These tests are marked "expensive" because they require external infrastructure.
"""

import json
import pytest
import requests
from transformers import AutoTokenizer


# Configuration - MUST match the SGLang server configuration
SGLANG_SERVER_URL = "http://localhost:31234/v1"
MODEL_NAME = "Qwen/Qwen3-8B"  # Must match what SGLang is serving


@pytest.fixture(scope="module")
def tokenizer():
    """Load the Qwen3-8B tokenizer.

    This MUST be the same model that the SGLang server is serving,
    otherwise tokenization will not match.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.mark.expensive
def test_single_turn_logprob_count():
    """
    BEHAVIOR: SGLang returns exactly completion_tokens logprobs.

    For any generation, the number of logprob entries returned in
    choices[0]['logprobs']['content'] must equal the completion_tokens
    value in the usage field.
    """
    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
            "temperature": 0.7,
            "logprobs": True,
        }
    ).json()

    logprobs_list = response["choices"][0]["logprobs"]["content"]
    completion_tokens = response["usage"]["completion_tokens"]

    # CRITICAL: These must always match
    assert len(logprobs_list) == completion_tokens, (
        f"Logprob count {len(logprobs_list)} != completion_tokens {completion_tokens}"
    )


@pytest.mark.expensive
def test_thinking_mode_always_enabled():
    """
    BEHAVIOR: Qwen3 thinking mode is always enabled, every turn starts with <think>.

    SGLang serves Qwen3 with thinking mode enabled by default. Every assistant
    response will begin with the <think> token (logprob ≈ 0.0), followed by
    reasoning content, then </think> and the final response.

    This behavior is CONSISTENT across all turns in a conversation.
    """
    messages = [{"role": "user", "content": "Count to 3"}]

    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.7,
            "logprobs": True,
        }
    ).json()

    logprobs = response["choices"][0]["logprobs"]["content"]
    content = response["choices"][0]["message"]["content"]

    # First token is always <think>
    assert logprobs[0]["token"] == "<think>", (
        f"Expected first token to be '<think>', got {repr(logprobs[0]['token'])}"
    )

    # The logprob is near-zero (deterministic/forced)
    assert abs(logprobs[0]["logprob"]) < 0.001, (
        f"Expected <think> to have near-zero logprob, got {logprobs[0]['logprob']}"
    )

    # Content starts with <think>
    assert content.startswith("<think>"), (
        f"Expected content to start with '<think>', got {repr(content[:20])}"
    )


@pytest.mark.expensive
def test_multi_turn_thinking_consistency():
    """
    BEHAVIOR: Every turn in a multi-turn conversation starts with <think>.

    Thinking mode is not just a first-turn behavior - EVERY assistant response
    in a conversation will start with <think>, regardless of conversation history.

    This is important for mask building in multi-turn dialogues.
    """
    messages = []

    # Simulate a 3-turn conversation
    turns = [
        "What is 5 times 2?",
        "Now add 3 to that.",
        "What's the final answer?"
    ]

    for user_message in turns:
        messages.append({"role": "user", "content": user_message})

        response = requests.post(
            f"{SGLANG_SERVER_URL}/chat/completions",
            json={
                "model": "default",
                "messages": messages,
                "max_tokens": 60,
                "temperature": 0.7,
                "logprobs": True,
            }
        ).json()

        assistant_content = response["choices"][0]["message"]["content"]
        logprobs = response["choices"][0]["logprobs"]["content"]

        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": assistant_content})

        # EVERY turn must start with <think>
        assert logprobs[0]["token"] == "<think>", (
            f"Turn {len(messages)//2}: Expected <think>, got {repr(logprobs[0]['token'])}"
        )
        assert abs(logprobs[0]["logprob"]) < 0.001, (
            f"Turn {len(messages)//2}: <think> logprob not near-zero: {logprobs[0]['logprob']}"
        )


@pytest.mark.expensive
def test_token_reconstruction(tokenizer):
    """
    BEHAVIOR: Content can be perfectly reconstructed from logprob tokens.

    The token strings in choices[0]['logprobs']['content'][i]['token'] can be
    concatenated to exactly reproduce choices[0]['message']['content'].

    This means no tokens are missing or added in the logprobs structure.
    """
    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 40,
            "temperature": 0.0,  # Deterministic
            "logprobs": True,
        }
    ).json()

    content = response["choices"][0]["message"]["content"]
    logprobs_list = response["choices"][0]["logprobs"]["content"]

    # Reconstruct content from token strings
    reconstructed = "".join([entry["token"] for entry in logprobs_list])

    assert content == reconstructed, (
        f"Content reconstruction failed:\n"
        f"  Original:      {repr(content[:100])}\n"
        f"  Reconstructed: {repr(reconstructed[:100])}"
    )


@pytest.mark.expensive
def test_local_tokenization_matches_sglang(tokenizer):
    """
    BEHAVIOR: Local Qwen3 tokenization exactly matches SGLang tokenization.

    When we tokenize the generated content using AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"),
    the resulting tokens match SGLang's tokens EXACTLY, including:
    - Token count
    - Individual token strings
    - Treatment of <think> as a SINGLE token

    This is critical because we need to align logprobs with locally-computed token IDs
    for training data preparation.

    NOTE: This assumes SGLang is serving the SAME model that we load the tokenizer from.
    """
    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "max_tokens": 50,
            "temperature": 0.0,  # Deterministic for reproducibility
            "logprobs": True,
        }
    ).json()

    content = response["choices"][0]["message"]["content"]
    logprobs_list = response["choices"][0]["logprobs"]["content"]
    sglang_tokens = [entry["token"] for entry in logprobs_list]

    # Tokenize locally
    token_ids = tokenizer.encode(content, add_special_tokens=False)
    local_tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # CRITICAL: Counts must match
    assert len(local_tokens) == len(sglang_tokens), (
        f"Token count mismatch: local={len(local_tokens)}, sglang={len(sglang_tokens)}"
    )

    # CRITICAL: Every token must match
    for i, (local_tok, sglang_tok) in enumerate(zip(local_tokens, sglang_tokens)):
        assert local_tok == sglang_tok, (
            f"Token mismatch at position {i}:\n"
            f"  Local:  {repr(local_tok)}\n"
            f"  SGLang: {repr(sglang_tok)}"
        )


@pytest.mark.expensive
def test_think_token_is_single_token(tokenizer):
    """
    BEHAVIOR: <think> is a SINGLE token in Qwen3 vocabulary.

    Unlike some models where special tokens like <think> might be split into
    multiple subword tokens, Qwen3 treats <think> as a single atomic token.

    This is why local tokenization matches SGLang - both use the same vocab.
    """
    # Encode just the <think> token
    token_ids = tokenizer.encode("<think>", add_special_tokens=False)

    assert len(token_ids) == 1, (
        f"Expected <think> to be a single token, got {len(token_ids)} tokens: "
        f"{[tokenizer.decode([tid]) for tid in token_ids]}"
    )

    # Verify it's in the vocabulary (not unknown)
    think_token_id = tokenizer.convert_tokens_to_ids("<think>")
    unk_token_id = tokenizer.unk_token_id

    assert think_token_id != unk_token_id, (
        f"<think> maps to unknown token ID {think_token_id}"
    )


@pytest.mark.expensive
def test_template_tokens_not_in_logprobs(tokenizer):
    """
    BEHAVIOR: Chat template tokens (<|im_start|>, <|im_end|>) are NOT in logprobs.

    When using /v1/chat/completions, SGLang applies the chat template to create
    the prompt:
        <|im_start|>user
        {user message}<|im_end|>
        <|im_start|>assistant
        [GENERATION STARTS HERE]

    The logprobs start from the FIRST GENERATED TOKEN, not from the template tokens.
    Template tokens are part of the prompt, not the completion.
    """
    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
            "temperature": 0.0,
            "logprobs": True,
        }
    ).json()

    logprobs_list = response["choices"][0]["logprobs"]["content"]
    all_tokens = [entry["token"] for entry in logprobs_list]

    # Template special tokens
    template_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    for template_token in template_tokens:
        assert template_token not in all_tokens, (
            f"Found template token {repr(template_token)} in logprobs, "
            f"but template tokens should be part of the prompt, not the completion"
        )


@pytest.mark.expensive
def test_logprob_values_are_valid():
    """
    BEHAVIOR: Logprob values are valid log probabilities.

    - All logprobs should be <= 0.0 (since log(p) <= 0 for p <= 1)
    - Some tokens may have logprob ≈ 0.0 (high confidence / forced tokens)
    - Typical logprobs are negative
    """
    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is 10-7?"}],
            "max_tokens": 60,
            "temperature": 0.7,
            "logprobs": True,
        }
    ).json()

    logprobs_list = response["choices"][0]["logprobs"]["content"]
    logprob_values = [entry["logprob"] for entry in logprobs_list]

    # All logprobs should be <= 0.0 (with small numerical tolerance)
    for i, lp in enumerate(logprob_values):
        assert lp <= 0.001, (
            f"Token {i} has invalid logprob {lp} > 0"
        )

    # Should have at least some non-trivial negative logprobs
    significantly_negative = [lp for lp in logprob_values if lp < -0.1]
    assert len(significantly_negative) > 0, (
        "Expected at least some tokens with logprob < -0.1, "
        "but all logprobs are near-zero"
    )


@pytest.mark.expensive
def test_logprob_storage_format():
    """
    BEHAVIOR: Logprobs can be stored as JSON list of floats.

    For GRPO training, we store logprobs in parquet as JSON strings.
    This test verifies the format we'll use for storage.
    """
    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 30,
            "temperature": 0.7,
            "logprobs": True,
        }
    ).json()

    logprobs_list = response["choices"][0]["logprobs"]["content"]
    completion_tokens = response["usage"]["completion_tokens"]

    # Extract just the logprob values
    logprob_values = [entry["logprob"] for entry in logprobs_list]

    # Convert to JSON (for parquet storage)
    json_str = json.dumps(logprob_values)

    # Verify round-trip
    recovered = json.loads(json_str)

    assert len(recovered) == completion_tokens
    assert len(recovered) == len(logprob_values)
    assert all(isinstance(x, float) for x in recovered)
    assert recovered == logprob_values


@pytest.mark.expensive
def test_im_end_token_in_logprobs():
    """
    BEHAVIOR: <|im_end|> is in logprobs when finish_reason="stop", not in content.

    When generation completes naturally (not cut off by max_tokens):
    - finish_reason is "stop"
    - The LAST token in logprobs is <|im_end|>
    - <|im_end|> is NOT in the content string
    - <|im_end|> DOES have a logprob value

    This is critical for mask building: we need to include <|im_end|> in the
    assistant mask because it has a logprob from SGLang.

    When generation is cut off by max_tokens:
    - finish_reason is "length"
    - <|im_end|> is NOT in logprobs (generation was interrupted)
    """
    # Test 1: Natural completion (finish_reason="stop")
    response_stop = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Say just: hello"}],
            "max_tokens": 200,  # Plenty of room
            "temperature": 0.0,
            "logprobs": True,
        }
    ).json()

    finish_reason = response_stop["choices"][0]["finish_reason"]
    content = response_stop["choices"][0]["message"]["content"]
    logprobs_list = response_stop["choices"][0]["logprobs"]["content"]

    assert finish_reason == "stop", "Expected natural completion"

    # Last token in logprobs should be <|im_end|>
    last_token = logprobs_list[-1]["token"]
    assert last_token == "<|im_end|>", (
        f"Expected last token to be <|im_end|>, got {repr(last_token)}"
    )

    # <|im_end|> should NOT be in content
    assert "<|im_end|>" not in content, (
        "Expected <|im_end|> to not appear in content field"
    )

    # <|im_end|> should have a valid logprob
    im_end_logprob = logprobs_list[-1]["logprob"]
    assert isinstance(im_end_logprob, float), (
        f"Expected <|im_end|> logprob to be float, got {type(im_end_logprob)}"
    )
    assert im_end_logprob <= 0.001, (
        f"Expected <|im_end|> logprob <= 0, got {im_end_logprob}"
    )

    # Test 2: Length cutoff (finish_reason="length")
    response_length = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Count to 100"}],
            "max_tokens": 15,  # Force cutoff
            "temperature": 0.0,
            "logprobs": True,
        }
    ).json()

    finish_reason_length = response_length["choices"][0]["finish_reason"]
    logprobs_list_length = response_length["choices"][0]["logprobs"]["content"]

    assert finish_reason_length == "length", "Expected length cutoff"

    # Last token should NOT be <|im_end|> when cut off
    last_token_length = logprobs_list_length[-1]["token"]
    assert last_token_length != "<|im_end|>", (
        f"When cut off by max_tokens, last token should not be <|im_end|>, "
        f"but got {repr(last_token_length)}"
    )


@pytest.mark.expensive
def test_im_start_assistant_not_in_logprobs():
    """
    BEHAVIOR: <|im_start|>assistant\n are NOT in logprobs (they're prompt).

    The chat template adds:
        <|im_start|>assistant\n

    before generation starts. These tokens are part of the PROMPT, not the
    completion, so they do NOT appear in logprobs.

    The first logprob token is whatever the model generates first (usually <think>).
    """
    response = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 50,
            "temperature": 0.0,
            "logprobs": True,
        }
    ).json()

    logprobs_list = response["choices"][0]["logprobs"]["content"]
    all_tokens = [entry["token"] for entry in logprobs_list]

    # These tokens should NOT be in logprobs
    assert "<|im_start|>" not in all_tokens
    # Note: "assistant" by itself might appear in the response content,
    # so we just check that <|im_start|> is not there

    # First token should be generated content (usually <think> for Qwen3)
    first_token = logprobs_list[0]["token"]
    assert first_token == "<think>", (
        f"Expected first generated token to be <think>, got {repr(first_token)}"
    )


@pytest.mark.expensive
def test_finish_reason_variants():
    """
    BEHAVIOR: Different finish reasons affect whether <|im_end|> is present.

    finish_reason values:
    - "stop": Natural completion OR custom stop sequence hit
    - "length": Hit max_tokens limit

    <|im_end|> token presence:
    - Present ONLY when model naturally completes (generates EOS)
    - NOT present when hitting max_tokens
    - NOT present when hitting custom stop sequences

    PRODUCTION NOTE:
    In our actual GRPO training runs, max_tokens is set very high, so
    finish_reason="stop" is almost always the case. This means we can rely on
    <|im_end|> being present for validation.

    However, the mask building logic should still be robust to both cases:
    - With <|im_end|>: mask includes it (it has a logprob)
    - Without <|im_end|>: mask ends at last available logprob
    """
    # Test 1: Natural stop -> has <|im_end|>
    response_natural = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 200,
            "temperature": 0.0,
            "logprobs": True,
        }
    ).json()

    assert response_natural["choices"][0]["finish_reason"] == "stop"
    natural_last_token = response_natural["choices"][0]["logprobs"]["content"][-1]["token"]
    assert natural_last_token == "<|im_end|>", (
        "Natural completion should end with <|im_end|>"
    )

    # Test 2: Length cutoff -> NO <|im_end|>
    response_length = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Count to 50"}],
            "max_tokens": 15,
            "temperature": 0.0,
            "logprobs": True,
        }
    ).json()

    assert response_length["choices"][0]["finish_reason"] == "length"
    length_last_token = response_length["choices"][0]["logprobs"]["content"][-1]["token"]
    assert length_last_token != "<|im_end|>", (
        "Length cutoff should NOT end with <|im_end|>"
    )

    # Test 3: Custom stop sequence -> NO <|im_end|>
    response_custom_stop = requests.post(
        f"{SGLANG_SERVER_URL}/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "Count: 1, 2, 3, 4, 5"}],
            "max_tokens": 200,
            "temperature": 0.0,
            "logprobs": True,
            "stop": ["5"],
        }
    ).json()

    # Custom stop also uses "stop" as finish_reason
    assert response_custom_stop["choices"][0]["finish_reason"] == "stop"
    custom_last_token = response_custom_stop["choices"][0]["logprobs"]["content"][-1]["token"]
    # Last token should be the stop sequence, not <|im_end|>
    assert custom_last_token != "<|im_end|>", (
        "Custom stop sequence should NOT end with <|im_end|>"
    )
    assert "5" in custom_last_token, (
        f"Expected stop sequence '5' in last token, got {repr(custom_last_token)}"
    )


@pytest.mark.expensive
def test_multi_turn_logprob_clusters_end_with_im_end():
    """
    VALIDATION TEST: Check that logprob clusters align with <|im_end|> tokens.

    PRODUCTION PATTERN:
    In our GRPO training, we set max_tokens very high to ensure natural completion.
    This means finish_reason is almost always "stop", and each assistant turn ends
    with <|im_end|>.

    This test validates the assistant mask building logic by checking that:
    1. Each assistant turn's logprobs ends with <|im_end|>
    2. The number of logprobs per turn matches what we expect

    This serves as a sanity check that our mask building correctly identifies
    where each assistant turn begins and ends.
    """
    messages = []
    turns = [
        "What is 3 + 4?",
        "Now subtract 2.",
        "What's the result?"
    ]

    all_responses = []

    for turn_msg in turns:
        messages.append({"role": "user", "content": turn_msg})

        response = requests.post(
            f"{SGLANG_SERVER_URL}/chat/completions",
            json={
                "model": "default",
                "messages": messages,
                "max_tokens": 2000,  # VERY HIGH max_tokens to ensure natural completion
                "temperature": 0.7,
                "logprobs": True,
            }
        ).json()

        all_responses.append(response)

        # Add assistant response to conversation
        assistant_content = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_content})

    # Validate each turn
    for i, response in enumerate(all_responses, 1):
        finish_reason = response["choices"][0]["finish_reason"]
        logprobs_list = response["choices"][0]["logprobs"]["content"]

        # With high max_tokens, should always complete naturally
        assert finish_reason == "stop", (
            f"Turn {i}: Expected finish_reason='stop' with high max_tokens, "
            f"got {repr(finish_reason)}"
        )

        # Last logprob should be <|im_end|>
        last_token = logprobs_list[-1]["token"]
        assert last_token == "<|im_end|>", (
            f"Turn {i}: Expected last logprob to be <|im_end|>, got {repr(last_token)}\n"
            f"This indicates a mismatch in how logprob clusters are identified."
        )

        # First logprob should be <think> (consistent thinking mode)
        first_token = logprobs_list[0]["token"]
        assert first_token == "<think>", (
            f"Turn {i}: Expected first logprob to be <think>, got {repr(first_token)}"
        )

        # Logprobs should form a complete sequence
        completion_tokens = response["usage"]["completion_tokens"]
        assert len(logprobs_list) == completion_tokens, (
            f"Turn {i}: Logprob count mismatch"
        )

    print(f"\n✓ All {len(all_responses)} turns have properly aligned logprob clusters")
    print(f"  Each cluster: <think> ... <|im_end|>")
    print(f"  This validates that mask building can use <|im_end|> as cluster boundaries")
