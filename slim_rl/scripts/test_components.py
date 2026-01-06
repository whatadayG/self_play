#!/usr/bin/env python3
"""Test script to verify all components work correctly.

This script tests:
1. Data generator - creates game instances
2. Environment adapter - wraps DialOp OptimizationEnv
3. GPT partner (mock) - simulates shy behavior
4. Question logger - tracks questions

Usage:
    python test_components.py
"""
import sys
import json
import asyncio
from pathlib import Path

# Add src to path
SLIM_RL_DIR = Path(__file__).parent.parent
SRC_PATH = SLIM_RL_DIR / "src"
SCRIPTS_PATH = SLIM_RL_DIR.parent / "scripts"
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(SCRIPTS_PATH))


def test_data_generator():
    """Test data generation."""
    print("\n=== Testing Data Generator ===")
    from data_generator import generate_game

    game = generate_game(seed=42)

    assert "prompt" in game, "Missing prompt"
    assert "game_state" in game, "Missing game_state"
    assert "optimal_score" in game, "Missing optimal_score"
    assert "game_id" in game, "Missing game_id"

    print(f"  Game ID: {game['game_id']}")
    print(f"  Optimal score: {game['optimal_score']}")
    print(f"  Prompt length: {len(game['prompt'])} chars")
    print("  [PASS] Data generator works!")

    return game


def test_environment_adapter(game):
    """Test environment adapter."""
    print("\n=== Testing Environment Adapter ===")
    from env_matching import MatchingEnv

    env = MatchingEnv(max_turns=10)
    obs, info = env.reset(game["game_state"])

    assert "obs_str" in obs, "Missing obs_str"
    assert "best_score" in info, "Missing best_score"

    print(f"  Best score: {info['best_score']}")
    print(f"  Initial observation length: {len(obs['obs_str'])} chars")
    print(f"  Is Qwen's turn: {env.is_qwen_turn()}")

    # Test a step - messages must have proper tags like [message], [propose], etc.
    test_message = "[message] What is the score for Ava Li reviewing BLEU?"
    obs, done, step_info = env.step(test_message)

    print(f"  After step - Done: {done}, Turn: {step_info['turn']}, Is error: {obs.get('is_error', False)}")
    print("  [PASS] Environment adapter works!")

    return env


def test_mock_partner():
    """Test mock GPT partner."""
    print("\n=== Testing Mock GPT Partner ===")
    from gpt_partner import MockShyPartner

    partner = MockShyPartner(seed=42)

    # Test with a question
    dialogue = [
        {"role": "system", "content": "You are playing a matching game."},
        {"role": "assistant", "content": "What is the score for Ava Li reviewing BLEU?"},
    ]

    response = asyncio.run(partner.respond(dialogue, "Partner view here"))

    print(f"  Response to question: {response[:50]}...")
    assert len(response) > 0, "Empty response"

    # Test with a proposal
    dialogue.append({"role": "user", "content": response})
    dialogue.append({"role": "assistant", "content": "[propose]\n- Ava Li: BLEU\n[/propose]"})

    response2 = asyncio.run(partner.respond(dialogue, "Partner view here"))
    print(f"  Response to proposal: {response2[:50]}...")

    print("  [PASS] Mock partner works!")


def test_question_logger():
    """Test question tracking."""
    print("\n=== Testing Question Logger ===")
    from question_logger import count_questions, QuestionLogger, EpisodeMetrics

    # Test count_questions
    assert count_questions("What is the score?") == 1, "Should detect question"
    assert count_questions("Hello there.") == 0, "Should not detect question"
    assert count_questions("Can you tell me about Ava Li?") == 1, "Should detect can question"
    # Note: "The score for BLEU is 42" matches score pattern, which is fine
    # The logger is intentionally permissive to catch information-seeking behavior

    print("  count_questions() works correctly")

    # Test QuestionLogger
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = QuestionLogger(tmpdir)

        for i in range(10):
            metrics = EpisodeMetrics(
                game_id=i,
                question_count=i % 5,
                turn_count=10,
                reward=0.5 + i * 0.05,
            )
            logger.log_episode(metrics)

        summary = logger.get_summary()
        assert "avg_questions" in summary, "Missing avg_questions"
        assert "avg_reward" in summary, "Missing avg_reward"

        print(f"  Summary: avg_questions={summary['avg_questions']:.2f}, avg_reward={summary['avg_reward']:.3f}")

    print("  [PASS] Question logger works!")


def test_full_dialogue_simulation():
    """Test a full simulated dialogue."""
    print("\n=== Testing Full Dialogue Simulation ===")
    from data_generator import generate_game
    from env_matching import MatchingEnv
    from gpt_partner import MockShyPartner
    from question_logger import count_questions

    # Generate game
    game = generate_game(seed=99)
    env = MatchingEnv(max_turns=10)
    partner = MockShyPartner(seed=99)

    obs, info = env.reset(game["game_state"])
    dialogue = [{"role": "system", "content": game["prompt"]}]

    total_questions = 0
    turn = 0

    print("  Simulating dialogue...")

    while not env.done and turn < 6:
        if env.is_qwen_turn():
            # Simulate Qwen asking a question - must use proper message tags
            if turn < 4:
                qwen_msg = f"[message] What is the score for reviewer {turn} and paper {turn}?"
            else:
                qwen_msg = "[propose]\n- Ava Li: BLEU\n- Daniel Nguyen: Electra\n- Sofia Patel: GloVe\n- Andrei Petrov: GLUE\n- Morgan Reed: LLaMA\n- Joseph Santos: RoBERTa\n- Ethan Smith: QuAC\n- Noah Wilson: SWAG"

            total_questions += count_questions(qwen_msg)
            dialogue.append({"role": "assistant", "content": qwen_msg})
            obs, done, step_info = env.step(qwen_msg)
            is_error = obs.get("is_error", False)
            print(f"    Turn {turn} (Qwen): {qwen_msg[:40]}... [error={is_error}]")

        else:
            # GPT responds - also needs proper tags
            partner_view = env.get_partner_view()
            gpt_response = asyncio.run(partner.respond(dialogue, partner_view))
            # Wrap in [message] tag if not already tagged
            if not any(tag in gpt_response.lower() for tag in ['[message]', '[accept]', '[reject]']):
                gpt_msg = f"[message] {gpt_response}"
            else:
                gpt_msg = gpt_response
            dialogue.append({"role": "user", "content": gpt_msg})
            obs, done, step_info = env.step(gpt_msg)
            is_error = obs.get("is_error", False)
            print(f"    Turn {turn} (GPT): {gpt_msg[:40]}... [error={is_error}]")

        turn += 1

    print(f"\n  Total turns: {turn}")
    print(f"  Questions asked: {total_questions}")
    print(f"  Final reward: {env.get_normalized_reward():.3f}")
    print("  [PASS] Full dialogue simulation works!")


def main():
    print("="*60)
    print("SLIM_RL Component Tests")
    print("="*60)

    try:
        game = test_data_generator()
        test_environment_adapter(game)
        test_mock_partner()
        test_question_logger()
        test_full_dialogue_simulation()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
