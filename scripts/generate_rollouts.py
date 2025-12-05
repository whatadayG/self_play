#!/usr/bin/env python3
import os
import json
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Manager, Process
import queue

import numpy as np
import pandas as pd

# Make project root importable when running from scripts/
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Local imports (now importable as top-level 'dialop')
from dialop.sglang_model_player import SGLangModelPlayer, SGLangConfig
from dialop.openai_model_player import OpenAIModelPlayer
from dialop.base_player import OpenAIConfig
from dialop.envs.optimization import OptimizationEnv
from rich.console import Console


def run_one_game(
    player_cfg: dict,
    model_id: str,
    instructions: str,
    game_id: int = 0,
    group_size: int = 8,
    base_seed: int = 42,
    player_type: str = "sglang"
) -> Dict[str, Any]:
    """Run a single game between two players.

    Args:
        player_cfg: Player configuration dict
        model_id: Model identifier
        instructions: Game instructions
        game_id: Unique game ID (0 to total_games-1)
        group_size: Number of replays per unique initial state
        base_seed: Base random seed for game generation
        player_type: Type of player ("sglang" or "openai")

    Returns:
        Dict containing game results, players, rewards, etc.
    """
    # Build two players
    console = Console() if player_type == "openai" else type("_C", (), {"print": lambda *args, **kwargs: None, "rule": lambda *args, **kwargs: None})()

    # Resolve the {unknown_value} placeholder
    instr_p1 = instructions.replace("{unknown_value}", "50")
    instr_p2 = instructions.replace("{unknown_value}", "50")

    if player_type == "sglang":
        cfg = SGLangConfig(
            temperature=player_cfg["temperature"],
            top_p=player_cfg["top_p"],
            max_tokens=player_cfg["max_new_tokens"],
            timeout=player_cfg.get("timeout", 120.0),
        )
        cfg.server_url = player_cfg["server_url"].rstrip("/")
        if not cfg.server_url.endswith("/v1"):
            cfg.server_url += "/v1"

        # Metadata for cache tracing
        p1_metadata = {"game_id": game_id, "turn": 0}
        p2_metadata = {"game_id": game_id, "turn": 0}

        p1 = SGLangModelPlayer(
            system_prompt=instr_p1,
            role="player-1",
            console=console,
            model_path=model_id,
            config=cfg,
            optional=p1_metadata,
        )
        p2 = SGLangModelPlayer(
            system_prompt=instr_p2,
            role="player-2",
            console=console,
            model_path=model_id,
            config=cfg,
            optional=p2_metadata,
        )
    elif player_type == "openai":
        cfg = OpenAIConfig(
            model=model_id,
            temperature=player_cfg["temperature"],
            max_tokens=player_cfg["max_new_tokens"],
            top_p=player_cfg["top_p"],
            api_key_path=player_cfg.get("api_key_path", OpenAIConfig.api_key_path),
            organization=player_cfg.get("organization", None),
        )

        p1 = OpenAIModelPlayer(
            system_prompt=instr_p1,
            role="player-1",
            console=console,
            model_path=model_id,
            config=cfg,
        )
        p2 = OpenAIModelPlayer(
            system_prompt=instr_p2,
            role="player-2",
            console=console,
            model_path=model_id,
            config=cfg,
        )
    else:
        raise ValueError(f"Unknown player_type: {player_type}")

    # Pass game limits to environment so agents can see them
    env = OptimizationEnv(
        max_turns=player_cfg.get("max_turns", 30),
        max_retries_per_turn=player_cfg.get("max_retries_per_turn", 8)
    )

    # GRPO grouping: compute unique game seed based on group membership
    # Games 0-7 (group 0) get same seed, games 8-15 (group 1) get different seed, etc.
    unique_game_id = game_id // group_size
    game_seed = base_seed + unique_game_id * 10000  # Large offset to avoid collisions

    obs = env.reset(game_state=None, seed=game_seed)
    p1.observe(obs["player-1"])  # give initial observation
    p2.observe(obs["player-2"])  # give initial observation

    clean_conversation = []
    full_conversation = []
    done = False
    turn = 0
    max_turns = player_cfg.get("max_turns", 30)
    max_retries = player_cfg.get("max_retries_per_turn", 8)

    # Note: Logprobs are now collected automatically by the player class
    # We'll extract them at the end using player.get_generated_logprob_tensor()

    while not done and turn < max_turns:
        current = obs["turn_player"]
        player = p1 if current == "player-1" else p2

        # Update player metadata with current turn number for cache tracing
        player.optional["turn"] = turn

        retries = 0
        while True:
            try:
                response = player.respond()
                full_conversation.append({"turn": turn, "player": current, "message": response, "retry": retries})

                # Logprobs are automatically accumulated by the player class

            except Exception as e:
                # Treat generation failure as terminal
                done = True
                break

            obs, error = env.step(response)
            if error:
                retries += 1
                full_conversation.append({"turn": turn, "player": "error", "message": obs[current], "retry": retries})
                player.observe(obs[current])
                if retries >= max_retries:
                    done = True
                    break
                continue

            # valid move
            clean_conversation.append({"turn": turn, "player": current, "message": response})
            # broadcast observation to other player
            other_name = "player-2" if current == "player-1" else "player-1"
            other_player = p2 if current == "player-1" else p1
            if obs[other_name]:
                other_player.observe(obs[other_name])

            done = obs["done"]
            turn += 1
            break

    # Build two sequences (one per player's perspective)
    result = {
        "game_id": game_id,
        "players": {"player-1": p1, "player-2": p2},
        "reward": obs.get("info", {}).get("score", 0.0) if done else 0.0,
        "normalized_reward": obs.get("info", {}).get("score_norm", 0.0) if done else 0.0,
        "game_info": {
            "num_messages": obs.get("info", {}).get("num_msgs", turn),
            "completed": done,
            "turn_count": turn,
            "game_normalized_reward": obs.get("info", {}).get("score_norm", 0.0) if done else 0.0,
        },
        "clean_conversation": clean_conversation,
        "full_conversation": full_conversation,
    }
    return result


def create_player_with_prompt(
    role: str,
    base_instructions: str,
    unknown_value: str,
    model_id: str,
    server_config,  # Can be SGLangConfig or OpenAIConfig
    game_id: int,
    player_type: str = "sglang"
):
    """Shared logic for creating a player with customized prompt."""
    console = Console() if player_type == "openai" else type("_C", (), {"print": lambda *args, **kwargs: None, "rule": lambda *args, **kwargs: None})()
    system_prompt = base_instructions.replace("{unknown_value}", str(unknown_value))

    if player_type == "sglang":
        metadata = {"game_id": game_id, "turn": 0}
        return SGLangModelPlayer(
            system_prompt=system_prompt,
            role=role,
            console=console,
            model_path=model_id,
            config=server_config,
            optional=metadata,
        )
    elif player_type == "openai":
        return OpenAIModelPlayer(
            system_prompt=system_prompt,
            role=role,
            console=console,
            model_path=model_id,
            config=server_config,
        )
    else:
        raise ValueError(f"Unknown player_type: {player_type}")


def run_one_game_vs_opponent(
    trainee_cfg: dict,
    opponent_cfg: dict,
    game_id: int = 0,
    group_size: int = 8,
    base_seed: int = 42,
    trainee_is_p1: bool = True,
    trainee_player_type: str = "sglang",
    opponent_player_type: str = "sglang"
) -> Dict[str, Any]:
    """
    Asymmetric mode: trainee vs fixed opponent with different prompts.
    Only returns trainee player sequences for training.

    Args:
        trainee_cfg: Trainee player configuration dict
        opponent_cfg: Opponent player configuration dict
        game_id: Unique game ID (0 to total_games-1)
        group_size: Number of replays per unique initial state
        base_seed: Base random seed for game generation
        trainee_is_p1: Whether trainee plays as player 1
        trainee_player_type: Type of trainee player ("sglang" or "openai")
        opponent_player_type: Type of opponent player ("sglang" or "openai")

    Returns:
        Dict containing game results with only trainee player data
    """
    # Create trainee config
    if trainee_player_type == "sglang":
        trainee_server_cfg = SGLangConfig(
            temperature=trainee_cfg["temperature"],
            top_p=trainee_cfg["top_p"],
            max_tokens=trainee_cfg["max_new_tokens"],
            timeout=trainee_cfg.get("timeout", 120.0),
        )
        trainee_server_cfg.server_url = trainee_cfg["server_url"].rstrip("/")
        if not trainee_server_cfg.server_url.endswith("/v1"):
            trainee_server_cfg.server_url += "/v1"
    else:  # openai
        trainee_server_cfg = OpenAIConfig(
            model=trainee_cfg["model_id"],
            temperature=trainee_cfg["temperature"],
            max_tokens=trainee_cfg["max_new_tokens"],
            top_p=trainee_cfg["top_p"],
            api_key_path=trainee_cfg.get("api_key_path", OpenAIConfig.api_key_path),
            organization=trainee_cfg.get("organization", None),
        )

    # Create opponent config
    if opponent_player_type == "sglang":
        opponent_server_cfg = SGLangConfig(
            temperature=opponent_cfg["temperature"],
            top_p=opponent_cfg["top_p"],
            max_tokens=opponent_cfg["max_new_tokens"],
            timeout=opponent_cfg.get("timeout", 120.0),
        )
        opponent_server_cfg.server_url = opponent_cfg["server_url"].rstrip("/")
        if not opponent_server_cfg.server_url.endswith("/v1"):
            opponent_server_cfg.server_url += "/v1"
    else:  # openai
        opponent_server_cfg = OpenAIConfig(
            model=opponent_cfg["model_id"],
            temperature=opponent_cfg["temperature"],
            max_tokens=opponent_cfg["max_new_tokens"],
            top_p=opponent_cfg["top_p"],
            api_key_path=opponent_cfg.get("api_key_path", OpenAIConfig.api_key_path),
            organization=opponent_cfg.get("organization", None),
        )

    # Create trainee and opponent players
    trainee = create_player_with_prompt(
        role="player-1" if trainee_is_p1 else "player-2",
        base_instructions=trainee_cfg["instructions"],
        unknown_value="50",
        model_id=trainee_cfg["model_id"],
        server_config=trainee_server_cfg,
        game_id=game_id,
        player_type=trainee_player_type
    )

    opponent = create_player_with_prompt(
        role="player-2" if trainee_is_p1 else "player-1",
        base_instructions=opponent_cfg["instructions"],
        unknown_value="50",
        model_id=opponent_cfg["model_id"],
        server_config=opponent_server_cfg,
        game_id=game_id,
        player_type=opponent_player_type
    )

    # Assign to p1/p2 based on trainee position
    p1 = trainee if trainee_is_p1 else opponent
    p2 = opponent if trainee_is_p1 else trainee

    # Pass game limits to environment
    env = OptimizationEnv(
        max_turns=trainee_cfg.get("max_turns", 30),
        max_retries_per_turn=trainee_cfg.get("max_retries_per_turn", 8)
    )

    # GRPO grouping: compute unique game seed based on group membership
    unique_game_id = game_id // group_size
    game_seed = base_seed + unique_game_id * 10000  # Large offset to avoid collisions

    obs = env.reset(game_state=None, seed=game_seed)
    p1.observe(obs["player-1"])
    p2.observe(obs["player-2"])

    clean_conversation = []
    full_conversation = []
    done = False
    turn = 0
    max_turns = trainee_cfg.get("max_turns", 30)
    max_retries = trainee_cfg.get("max_retries_per_turn", 8)

    # Collect logprobs only for trainee
    trainee_all_logprobs = []

    while not done and turn < max_turns:
        current = obs["turn_player"]
        player = p1 if current == "player-1" else p2
        is_trainee_turn = (current == trainee.role)

        # Update player metadata with current turn number for cache tracing
        player.optional["turn"] = turn

        retries = 0
        while True:
            try:
                response = player.respond()
                full_conversation.append({"turn": turn, "player": current, "message": response, "retry": retries})

                # Collect logprobs only for trainee
                if is_trainee_turn:
                    logprobs = player.get_last_generation_logprobs()
                    if logprobs:
                        trainee_all_logprobs.extend(logprobs)

            except Exception as e:
                # Treat generation failure as terminal
                done = True
                break

            obs, error = env.step(response)
            if error:
                retries += 1
                full_conversation.append({"turn": turn, "player": "error", "message": obs[current], "retry": retries})
                player.observe(obs[current])
                if retries >= max_retries:
                    done = True
                    break
                continue

            # valid move
            clean_conversation.append({"turn": turn, "player": current, "message": response})
            # broadcast observation to other player
            other_name = "player-2" if current == "player-1" else "player-1"
            other_player = p2 if current == "player-1" else p1
            if obs[other_name]:
                other_player.observe(obs[other_name])

            done = obs["done"]
            turn += 1
            break

    # Return result with ONLY trainee player
    trainee_role = "player-1" if trainee_is_p1 else "player-2"
    result = {
        "game_id": game_id,
        "players": {trainee_role: trainee},  # Only trainee
        "reward": obs.get("info", {}).get("score", 0.0) if done else 0.0,
        "normalized_reward": obs.get("info", {}).get("score_norm", 0.0) if done else 0.0,
        "game_info": {
            "num_messages": obs.get("info", {}).get("num_msgs", turn),
            "completed": done,
            "turn_count": turn,
            "game_normalized_reward": obs.get("info", {}).get("score_norm", 0.0) if done else 0.0,
        },
        "clean_conversation": clean_conversation,
        "full_conversation": full_conversation,
        "policy_logprobs": {
            trainee_role: trainee_all_logprobs,
        },
    }
    return result


def tensorize_player(player, max_model_len: int):
    input_ids = player.get_input_sequence()
    if len(input_ids) > max_model_len:
        input_ids = input_ids[:max_model_len]
    attention_mask = [1] * len(input_ids)
    # assistant mask for loss
    response_loss_mask = player.get_assistant_mask()
    if len(response_loss_mask) > len(input_ids):
        response_loss_mask = response_loss_mask[: len(input_ids)]
    elif len(response_loss_mask) < len(input_ids):
        response_loss_mask = response_loss_mask + [0] * (len(input_ids) - len(response_loss_mask))
    # position ids as simple range (single-head)
    position_ids = list(range(len(input_ids)))
    return input_ids, attention_mask, position_ids, response_loss_mask


def process_game_result(result: Dict[str, Any], max_model_len: int) -> List[Dict[str, Any]]:
    """Process a single game result into training samples.

    Handles both self-play (2 players) and asymmetric (1 player) modes.
    """
    rows = []
    # Iterate over whatever players exist in the result
    for player_name in result["players"].keys():
        p = result["players"][player_name]
        input_ids, attention_mask, position_ids, response_loss_mask = tensorize_player(p, max_model_len)
        # Use absolute normalized reward as weight (will be converted to relative in GRPO grouping)
        w = float(result["normalized_reward"])

        # Get policy logprobs tensor (aligned with input_ids and loss_mask)
        policy_logprobs = p.get_generated_logprob_tensor()

        rows.append(
            {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "attention_mask": np.array(attention_mask, dtype=np.int64),
                "position_ids": np.array(position_ids, dtype=np.int64),
                "loss_mask": np.array(response_loss_mask, dtype=np.int64),
                "sample_weight": w,
                # Store game-normalized reward explicitly
                "game_normalized_reward": float(
                    result.get("game_info", {}).get("game_normalized_reward", result.get("normalized_reward", 0.0))
                ),
                "game_info": json.dumps(result["game_info"]),
                "game_id": result["game_id"],  # Add game_id for grouping
                # Store full conversation including errors for debugging
                "full_conversation": json.dumps(result.get("full_conversation", [])),
                # Store policy logprobs for KL divergence computation
                "policy_logprobs": json.dumps(policy_logprobs),
            }
        )
    return rows


def process_game_result_messages(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process game result into messages format (for SFT without pre-tokenization).

    Extracts each player's full conversation history as they experienced it,
    including system prompts, all exchanges, and any error messages visible
    only to them.

    Args:
        result: Game result dict with "players" containing player objects

    Returns:
        List of dicts with "messages" field containing ChatML-format conversation
    """
    rows = []
    for player_name in result["players"].keys():
        player = result["players"][player_name]

        # Get player's full conversation history (system + all exchanges)
        # This is maintained by BaseModelPlayer in self.messages
        conversation_history = player.get_conversation_history()

        rows.append({
            "messages": conversation_history,  # List[Dict[str, str]] with role/content
            "player_name": player_name,
            "game_id": result["game_id"],
            "sample_weight": float(result["normalized_reward"]),
            "game_normalized_reward": float(
                result.get("game_info", {}).get("game_normalized_reward", result.get("normalized_reward", 0.0))
            ),
            "game_info": json.dumps(result["game_info"]),
            "full_conversation": json.dumps(result.get("full_conversation", [])),  # For debugging
        })
    return rows


def worker_process(
    task_queue,
    result_queue,
    threads_per_proc: int,
    player_cfg: Dict[str, Any],
    model_id: str,
    instructions: str,
    max_model_len: int,
    group_size: int,
    seed: int,
    process_id: int,
    player_type: str = "sglang",
    output_format: str = "pretokenized",
):
    """Worker process for self-play mode that runs a thread pool pulling games from shared queue.

    Args:
        task_queue: Shared queue containing game_ids to process
        result_queue: Shared queue to put results into
        threads_per_proc: Number of threads to run in this process
        player_cfg: Player configuration dict
        model_id: Model identifier
        instructions: Game instructions
        max_model_len: Maximum model sequence length
        group_size: Number of replays per unique initial state
        seed: Random seed base
        process_id: ID of this process for seeding
        player_type: Type of player ("sglang" or "openai")
        output_format: Output format ("pretokenized" or "messages")
    """
    # Set unique random seed for this process
    np.random.seed(seed + process_id)

    def thread_worker():
        """Thread worker that pulls game_ids from queue and processes them."""
        while True:
            try:
                game_id = task_queue.get_nowait()
            except queue.Empty:
                return  # No more work

            try:
                result = run_one_game(player_cfg, model_id, instructions, game_id, group_size, seed, player_type)
                if output_format == "messages":
                    rows = process_game_result_messages(result)
                else:
                    rows = process_game_result(result, max_model_len)
                result_queue.put(rows)
            except Exception as e:
                # Log failure but don't crash the worker
                print(f"[Process {process_id}] Game {game_id} failed: {e}")
                # Put empty result to maintain progress tracking
                result_queue.put([])

    # Run thread pool for this process
    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = [executor.submit(thread_worker) for _ in range(threads_per_proc)]
        # Wait for all threads to finish
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[Process {process_id}] Thread crashed: {e}")


def worker_process_asymmetric(
    task_queue,
    result_queue,
    threads_per_proc: int,
    trainee_cfg: Dict[str, Any],
    opponent_cfg: Dict[str, Any],
    max_model_len: int,
    group_size: int,
    seed: int,
    process_id: int,
    trainee_player_type: str = "sglang",
    opponent_player_type: str = "sglang",
    output_format: str = "pretokenized",
):
    """Worker process for asymmetric mode (trainee vs opponent).

    Args:
        task_queue: Shared queue containing game_ids to process
        result_queue: Shared queue to put results into
        threads_per_proc: Number of threads to run in this process
        trainee_cfg: Trainee player configuration (including instructions, model_id, server_url)
        opponent_cfg: Opponent player configuration
        max_model_len: Maximum model sequence length
        group_size: Number of replays per unique initial state
        seed: Random seed base
        process_id: ID of this process for seeding
        trainee_player_type: Type of trainee player ("sglang" or "openai")
        opponent_player_type: Type of opponent player ("sglang" or "openai")
        output_format: Output format ("pretokenized" or "messages")
    """
    # Set unique random seed for this process
    np.random.seed(seed + process_id)

    def thread_worker():
        """Thread worker that pulls game_ids from queue and processes them."""
        while True:
            try:
                game_id = task_queue.get_nowait()
            except queue.Empty:
                return  # No more work

            try:
                # Randomize trainee position for balanced data
                trainee_is_p1 = (game_id % 2 == 0)
                result = run_one_game_vs_opponent(
                    trainee_cfg, opponent_cfg, game_id, group_size, seed, trainee_is_p1,
                    trainee_player_type, opponent_player_type
                )
                if output_format == "messages":
                    rows = process_game_result_messages(result)
                else:
                    rows = process_game_result(result, max_model_len)
                result_queue.put(rows)
            except Exception as e:
                # Log failure but don't crash the worker
                print(f"[Process {process_id}] Game {game_id} failed: {e}")
                # Put empty result to maintain progress tracking
                result_queue.put([])

    # Run thread pool for this process
    with ThreadPoolExecutor(max_workers=threads_per_proc) as executor:
        futures = [executor.submit(thread_worker) for _ in range(threads_per_proc)]
        # Wait for all threads to finish
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[Process {process_id}] Thread crashed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:30001", help="URL of the SGLang server (default: port 30001)")
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-games", type=int, default=256, help="Number of unique games to play (each will be played 8 times)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    # Use a large default; external server's KV cache is governed by mem fraction, not this cap
    ap.add_argument("--max-model-len", type=int, default=32768)
    # GRPO grouping size (k). Rewards will be transformed to relative rewards within each group.
    ap.add_argument("--group-size", type=int, default=8, help="Number of times to play each game (creates groups of 2*k sequences)")
    ap.add_argument("--seed", type=int, default=42)
    # Parallelization settings (PER SERVER/ENGINE)
    ap.add_argument("--num-procs", type=int, default=32, help="Number of worker processes (per server)")
    ap.add_argument("--threads-per-proc", type=int, default=8, help="Threads per process for issuing requests")
    ap.add_argument("--progress-interval", type=int, default=32, help="Print progress every N completed games")
    # Game termination settings
    ap.add_argument("--max-turns", type=int, default=10, help="Maximum number of turns per game (default: 10)")
    ap.add_argument("--max-retries-per-turn", type=int, default=2, help="Maximum retries per turn before terminating (default: 2)")

    # Asymmetric mode arguments (trainee vs opponent)
    ap.add_argument("--mode", type=str, choices=["selfplay", "asymmetric"], default="selfplay", help="Training mode: selfplay or asymmetric")
    ap.add_argument("--opponent-model-id", type=str, help="Opponent model ID (required for asymmetric mode)")
    ap.add_argument("--opponent-server-url", type=str, help="Opponent SGLang server URL (required for asymmetric mode)")
    ap.add_argument("--opponent-instructions", type=str, help="Path to opponent instructions file (required for asymmetric mode)")

    # Player type and output format (for SFT data generation with OpenAI)
    ap.add_argument("--player-type", type=str, choices=["sglang", "openai"], default="sglang",
                    help="Type of player to use: sglang (local model via SGLang server) or openai (OpenAI API)")
    ap.add_argument("--output-format", type=str, choices=["pretokenized", "messages"], default="pretokenized",
                    help="Output format: pretokenized (for PPO/GRPO training) or messages (for SFT training)")
    ap.add_argument("--openai-api-key-path", type=str, help="Path to OpenAI API key JSON file (optional, defaults to ~/.api_key)")
    ap.add_argument("--openai-organization", type=str, help="OpenAI organization ID (optional)")

    # Asymmetric player types (for mixed setups)
    ap.add_argument("--trainee-player-type", type=str, choices=["sglang", "openai"],
                    help="Player type for trainee in asymmetric mode (defaults to --player-type)")
    ap.add_argument("--opponent-player-type", type=str, choices=["sglang", "openai"],
                    help="Player type for opponent in asymmetric mode (defaults to --player-type)")

    args = ap.parse_args()

    np.random.seed(args.seed)

    # Set log directory for generation failures
    out_path = Path(args.out)
    log_dir = str(out_path.parent.resolve())
    os.environ["ROLLOUT_LOG_DIR"] = log_dir
    print(f"Generation failure logs will be written to: {log_dir}/generation_failures.log")

    # Validate asymmetric mode arguments
    if args.mode == "asymmetric":
        if not args.opponent_model_id or not args.opponent_server_url or not args.opponent_instructions:
            raise ValueError("Asymmetric mode requires --opponent-model-id, --opponent-server-url, and --opponent-instructions")

    # Determine player types for asymmetric mode
    trainee_player_type = args.trainee_player_type if args.trainee_player_type else args.player_type
    opponent_player_type = args.opponent_player_type if args.opponent_player_type else args.player_type

    # Load instructions
    if args.mode == "selfplay":
        instructions_path = Path(__file__).parent / "dialop" / "envs" / "data" / "optimization.txt"
        instructions = instructions_path.read_text().strip()

        player_cfg = {
            "server_url": args.server_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_turns": getattr(args, 'max_turns', 10),
            "max_retries_per_turn": getattr(args, 'max_retries_per_turn', 2),
        }
        # Add OpenAI-specific fields if using OpenAI player
        if args.player_type == "openai":
            if args.openai_api_key_path:
                player_cfg["api_key_path"] = args.openai_api_key_path
            if args.openai_organization:
                player_cfg["organization"] = args.openai_organization
    else:  # asymmetric mode
        # Load trainee instructions (default)
        instructions_path = Path(__file__).parent / "dialop" / "envs" / "data" / "optimization.txt"
        trainee_instructions = instructions_path.read_text().strip()

        # Load opponent instructions
        opponent_instructions_path = Path(args.opponent_instructions)
        opponent_instructions = opponent_instructions_path.read_text().strip()

        # Create separate configs for trainee and opponent
        trainee_cfg = {
            "model_id": args.model_id,
            "server_url": args.server_url,
            "instructions": trainee_instructions,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_turns": getattr(args, 'max_turns', 10),
            "max_retries_per_turn": getattr(args, 'max_retries_per_turn', 2),
        }
        # Add OpenAI-specific fields for trainee if using OpenAI
        if trainee_player_type == "openai":
            if args.openai_api_key_path:
                trainee_cfg["api_key_path"] = args.openai_api_key_path
            if args.openai_organization:
                trainee_cfg["organization"] = args.openai_organization

        opponent_cfg = {
            "model_id": args.opponent_model_id,
            "server_url": args.opponent_server_url,
            "instructions": opponent_instructions,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "max_turns": getattr(args, 'max_turns', 10),
            "max_retries_per_turn": getattr(args, 'max_retries_per_turn', 2),
        }
        # Add OpenAI-specific fields for opponent if using OpenAI
        if opponent_player_type == "openai":
            if args.openai_api_key_path:
                opponent_cfg["api_key_path"] = args.openai_api_key_path
            if args.openai_organization:
                opponent_cfg["organization"] = args.openai_organization

    # Calculate total games to play: num_games * group_size
    total_games = args.num_games * args.group_size
    total_workers = args.num_procs * args.threads_per_proc
    print(f"Mode: {args.mode}")
    print(f"GRPO Grouping Strategy:")
    print(f"  - {args.num_games} unique initial states (deterministic from seed)")
    print(f"  - {args.group_size} rollouts per initial state (sampling randomness)")
    print(f"  - {total_games} total rollouts")
    print(f"  - Each group of {args.group_size} rollouts shares the same game setup")
    print(
        f"Worker configuration: {args.num_procs} processes × {args.threads_per_proc} threads = {total_workers} concurrent workers"
    )
    if args.mode == "selfplay":
        print(f"Server: {args.server_url}")
    else:
        print(f"Trainee server: {args.server_url}")
        print(f"Opponent server: {args.opponent_server_url}")
    start_time = time.time()

    # Create shared queues using Manager
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # Populate task queue with all game IDs
    for game_id in range(total_games):
        task_queue.put(game_id)

    print(f"Populated task queue with {total_games} games")

    # Spawn worker processes based on mode
    processes = []
    for proc_id in range(args.num_procs):
        if args.mode == "selfplay":
            p = Process(
                target=worker_process,
                args=(
                    task_queue,
                    result_queue,
                    args.threads_per_proc,
                    player_cfg,
                    args.model_id,
                    instructions,
                    args.max_model_len,
                    args.group_size,
                    args.seed,
                    proc_id,
                    args.player_type,
                    args.output_format,
                )
            )
        else:  # asymmetric
            p = Process(
                target=worker_process_asymmetric,
                args=(
                    task_queue,
                    result_queue,
                    args.threads_per_proc,
                    trainee_cfg,
                    opponent_cfg,
                    args.max_model_len,
                    args.group_size,
                    args.seed,
                    proc_id,
                    trainee_player_type,
                    opponent_player_type,
                    args.output_format,
                )
            )
        p.start()
        processes.append(p)

    print(f"Spawned {len(processes)} worker processes")

    # Collect results as they arrive
    all_rows: List[Dict[str, Any]] = []
    completed_games = 0
    last_progress_time = start_time
    last_progress_games = 0

    print(f"Starting result collection (updating every {args.progress_interval} games)...\n")

    while completed_games < total_games:
        try:
            # Non-blocking check for results with short timeout
            rows = result_queue.get(timeout=1.0)
            all_rows.extend(rows)
            # Progress tracking based on mode
            if args.mode == "selfplay":
                completed_games += len(rows) // 2  # Each game yields 2 rows
            else:  # asymmetric
                completed_games += len(rows)  # Each game yields 1 row

        except queue.Empty:
            # Check if all processes are still alive
            alive = [p.is_alive() for p in processes]
            if not any(alive) and completed_games < total_games:
                print(f"\nWarning: All processes terminated but only {completed_games}/{total_games} games completed")
                break

        # Periodic status update (every 30s, regardless of results)
        elapsed = time.time() - start_time
        if elapsed - (last_progress_time - start_time) > 30:  # Every 30s
            alive = sum(1 for p in processes if p.is_alive())
            rate = completed_games / elapsed if elapsed > 0 else 0
            interval_elapsed = time.time() - last_progress_time
            games_in_interval = completed_games - last_progress_games
            interval_rate = games_in_interval / interval_elapsed if interval_elapsed > 0 else 0
            pct = 100.0 * completed_games / total_games if total_games > 0 else 0
            print(
                f"Status: {completed_games}/{total_games} games ({pct:.1f}%) | "
                f"{alive}/{len(processes)} processes alive | "
                f"Avg: {rate:.1f} games/s | Recent: {interval_rate:.1f} games/s | "
                f"Elapsed: {elapsed:.0f}s"
            )
            last_progress_time = time.time()
            last_progress_games = completed_games

    # Wait for all processes to finish
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            print(f"Warning: Process {p.pid} still alive, terminating...")
            p.terminate()
            p.join()

    total_elapsed = time.time() - start_time
    final_rate = completed_games / total_elapsed if total_elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"Generation completed!")
    print(f"  Time: {total_elapsed:.1f}s")
    print(f"  Games: {completed_games}/{total_games} ({100.0 * completed_games / total_games:.1f}%)")
    if args.mode == "selfplay":
        print(f"  Rows: {len(all_rows)} ({len(all_rows) // 2} games × 2 players)")
    else:
        print(f"  Rows: {len(all_rows)} ({len(all_rows)} games × 1 trainee)")
    print(f"  Rate: {final_rate:.1f} games/s")
    print(f"{'='*60}\n")

    # Group sequences by unique game sets for GRPO normalization
    print(f"Grouping sequences for GRPO normalization...")

    # Sort rows by unique_game_id to ensure proper grouping
    all_rows.sort(key=lambda x: x["game_id"] // args.group_size)

    # Apply GRPO normalization within each group
    if args.mode == "selfplay":
        # Each unique game was played group_size times, creating 2*group_size sequences per unique game
        sequences_per_group = 2 * args.group_size  # 2 players × group_size plays
    else:  # asymmetric
        # Each unique game was played group_size times, creating group_size sequences per unique game (trainee only)
        sequences_per_group = args.group_size  # 1 trainee × group_size plays
    total = len(all_rows)
    full_groups = total // sequences_per_group

    if full_groups == 0:
        print(f"Warning: not enough samples ({total}) to form a full group of size {sequences_per_group}; leaving weights unchanged.")
    else:
        remainder = total - full_groups * sequences_per_group
        if remainder > 0:
            print(f"Dropping {remainder} leftover samples to form full groups of size {sequences_per_group} for GRPO.")
        # Only keep full groups
        all_rows = all_rows[: full_groups * sequences_per_group]

        for g in range(full_groups):
            start = g * sequences_per_group
            end = start + sequences_per_group
            group_rewards = [float(all_rows[j]["sample_weight"]) for j in range(start, end)]
            baseline = float(np.mean(group_rewards))
            for j in range(start, end):
                all_rows[j]["sample_weight"] = float(group_rewards[j - start] - baseline)

        print(f"Applied GRPO normalization to {full_groups} groups of {sequences_per_group} sequences each")

    # Save parquet
    df = pd.DataFrame(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out_path))
    print(f"Wrote {len(df)} samples to {out_path}")


if __name__ == "__main__":
    main()
