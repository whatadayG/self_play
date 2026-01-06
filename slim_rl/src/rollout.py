"""Custom multi-turn rollout for Qwen+GPT paper-reviewer matching.

This module implements a custom rollout function for SLIME that orchestrates
multi-turn dialogues between:
- Qwen (trainable agent) - generates via SGLang, collects logprobs
- GPT-4.1 (fixed partner) - generates via OpenAI API, no training

Based on SLIME's examples/geo3k_vlm_multi_turn/rollout.py pattern.

Key features:
- loss_mask = 1 for Qwen tokens (train on these)
- loss_mask = 0 for GPT tokens (don't train on these)
- Terminal reward from DialOp environment
- Question counting for hypothesis validation
"""
import os
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field

from .env_matching import MatchingEnv
from .gpt_partner import ShyGPTPartner, MockShyPartner
from .question_logger import count_questions


@dataclass
class RolloutSample:
    """Sample from a multi-turn rollout.

    This matches SLIME's expected Sample structure.
    """
    input_ids: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    reward: float = 0.0
    extra_info: dict = field(default_factory=dict)


async def generate(
    args,
    sample: dict,
    sampling_params: dict,
    evaluation: bool = False,
) -> list[RolloutSample]:
    """Multi-turn rollout with Qwen (trainable) and GPT (fixed partner).

    This follows SLIME's multi-turn pattern:
    1. Initialize environment with game state
    2. Loop: Qwen generates -> GPT responds -> env.step()
    3. Build loss_mask: 1 for Qwen tokens, 0 for GPT/env tokens
    4. Return samples with terminal reward

    Args:
        args: SLIME args object with tokenizer, max_turns, etc.
        sample: Dict with "prompt", "game_state"/"ground_truth"
        sampling_params: SGLang sampling parameters
        evaluation: Whether this is an evaluation rollout

    Returns:
        List containing single RolloutSample with full dialogue
    """
    # Get configuration from args
    max_turns = getattr(args, 'max_turns', 20)
    gpt_model = getattr(args, 'gpt_model', 'gpt-4.1')
    use_mock_partner = getattr(args, 'use_mock_partner', False)

    # Initialize environment
    env = MatchingEnv(max_turns=max_turns)
    game_state = sample.get("game_state", sample.get("ground_truth", {}))
    obs, info = env.reset(game_state)

    # Initialize GPT partner
    if use_mock_partner:
        gpt_partner = MockShyPartner(seed=sample.get("game_id", 42))
    else:
        gpt_partner = ShyGPTPartner(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=gpt_model,
        )

    # Token tracking for the final sample
    all_input_ids: list[int] = []
    all_logprobs: list[float] = []
    all_loss_mask: list[int] = []

    # Dialogue history (for context in generation)
    dialogue: list[dict] = [
        {"role": "system", "content": sample["prompt"]},
        {"role": "user", "content": obs["obs_str"]},
    ]

    # Metrics for logging
    question_count = 0
    qwen_turns = 0
    turn_count = 0
    done = False
    reward = 0.0

    # Get tokenizer from args
    tokenizer = args.tokenizer

    # Add initial prompt tokens (with loss_mask = 0, not trained)
    initial_prompt = _build_prompt(dialogue, tokenizer)
    initial_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)
    all_input_ids.extend(initial_ids)
    all_logprobs.extend([float("-inf")] * len(initial_ids))
    all_loss_mask.extend([0] * len(initial_ids))

    while not done and turn_count < max_turns:
        if env.is_qwen_turn():
            # === QWEN'S TURN (trainable) ===
            qwen_turns += 1

            # Build prompt from dialogue history
            qwen_prompt = _build_prompt(dialogue, tokenizer)

            # Generate with SGLang (collects logprobs)
            # This calls SLIME's internal generation function
            response = await _generate_with_sglang(
                args, qwen_prompt, sampling_params, evaluation
            )

            response_text = response["text"]
            response_ids = response["input_ids"]
            response_logprobs = response["logprobs"]

            # Track Qwen tokens with loss_mask = 1 (TRAIN on these)
            all_input_ids.extend(response_ids)
            all_logprobs.extend(response_logprobs)
            all_loss_mask.extend([1] * len(response_ids))

            # Count questions for hypothesis tracking
            question_count += count_questions(response_text)

            # Add to dialogue history
            dialogue.append({"role": "assistant", "content": response_text})

            # Step environment with Qwen's response
            obs, done, step_info = env.step(response_text)

        else:
            # === GPT'S TURN (fixed partner, not trained) ===
            partner_view = env.get_partner_view()
            gpt_response = await gpt_partner.respond(dialogue, partner_view)

            # Tokenize GPT response (for sequence continuity)
            # Add as "user" message format
            gpt_formatted = f"\n{gpt_response}"
            gpt_ids = tokenizer.encode(gpt_formatted, add_special_tokens=False)

            # Track GPT tokens with loss_mask = 0 (DON'T train on these)
            all_input_ids.extend(gpt_ids)
            all_logprobs.extend([float("-inf")] * len(gpt_ids))
            all_loss_mask.extend([0] * len(gpt_ids))

            # Add to dialogue history
            dialogue.append({"role": "user", "content": gpt_response})

            # Step environment with GPT's response
            obs, done, step_info = env.step(gpt_response)

        turn_count += 1

        if done:
            reward = step_info.get("reward", 0.0)

    # If we hit max turns without finishing, use current reward
    if not done:
        reward = env.get_normalized_reward()

    # Build final sample
    final_sample = RolloutSample(
        input_ids=all_input_ids,
        rollout_log_probs=all_logprobs,
        loss_mask=all_loss_mask,
        reward=reward,
        extra_info={
            "question_count": question_count,
            "turn_count": turn_count,
            "qwen_turns": qwen_turns,
            "game_id": sample.get("game_id", 0),
            "optimal_score": sample.get("optimal_score", info.get("best_score", 1.0)),
            "reward": reward,
        }
    )

    return [final_sample]


def _build_prompt(dialogue: list[dict], tokenizer) -> str:
    """Convert dialogue history to prompt string using chat template.

    Args:
        dialogue: List of {"role": ..., "content": ...} messages
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        Formatted prompt string
    """
    try:
        return tokenizer.apply_chat_template(
            dialogue,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback for tokenizers without chat template
        prompt_parts = []
        for msg in dialogue:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")
        prompt_parts.append("<|assistant|>\n")
        return "\n".join(prompt_parts)


async def _generate_with_sglang(
    args,
    prompt: str,
    sampling_params: dict,
    evaluation: bool = False,
) -> dict:
    """Generate response using SGLang.

    This function interfaces with SLIME's SGLang generation.
    The exact implementation depends on SLIME's internal API.

    Args:
        args: SLIME args with SGLang configuration
        prompt: Input prompt string
        sampling_params: Sampling parameters
        evaluation: Whether in eval mode

    Returns:
        Dict with "text", "input_ids", "logprobs"
    """
    # Import SLIME's generation function
    # This will need to be adjusted based on actual SLIME API
    try:
        from slime.rollout.sglang_rollout import GenerateState

        # Use SLIME's generation infrastructure
        state = GenerateState.get_instance()

        # Tokenize prompt
        tokenizer = args.tokenizer
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Generate
        # Note: Actual SLIME API may differ - adjust as needed
        result = await state.generate(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

        response_ids = result.get("output_ids", [])
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        logprobs = result.get("logprobs", [0.0] * len(response_ids))

        return {
            "text": response_text,
            "input_ids": response_ids,
            "logprobs": logprobs,
        }

    except ImportError:
        # Fallback for testing without SLIME
        return await _mock_generate(args, prompt, sampling_params)


async def _mock_generate(
    args,
    prompt: str,
    sampling_params: dict,
) -> dict:
    """Mock generation for testing without SGLang.

    Args:
        args: Args with tokenizer
        prompt: Input prompt
        sampling_params: Sampling params (unused)

    Returns:
        Mock response dict
    """
    import random

    # Generate a simple mock response
    responses = [
        "What is the score for Ava Li reviewing BLEU?",
        "Can you tell me Sofia Patel's score for GloVe?",
        "I'd like to know about Daniel Nguyen and Electra.",
        "[propose]\n- Ava Li: BLEU\n- Daniel Nguyen: Electra\n- Sofia Patel: GloVe\n- Andrei Petrov: GLUE\n- Morgan Reed: LLaMA\n- Joseph Santos: RoBERTa\n- Ethan Smith: QuAC\n- Noah Wilson: SWAG\n[/propose]",
    ]

    response_text = random.choice(responses)
    tokenizer = args.tokenizer
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    # Mock logprobs (uniform)
    logprobs = [-1.0] * len(response_ids)

    return {
        "text": response_text,
        "input_ids": response_ids,
        "logprobs": logprobs,
    }


# Synchronous wrapper for testing
def generate_sync(
    args,
    sample: dict,
    sampling_params: dict,
    evaluation: bool = False,
) -> list[RolloutSample]:
    """Synchronous version of generate for testing.

    Args:
        args: SLIME args
        sample: Sample dict
        sampling_params: Sampling params
        evaluation: Eval mode flag

    Returns:
        List of RolloutSample
    """
    return asyncio.run(generate(args, sample, sampling_params, evaluation))
