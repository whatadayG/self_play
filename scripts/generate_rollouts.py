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
from dialop.envs.optimization import OptimizationEnv

# Import branched game manager
from branched_game_manager import (
    BranchedGameManager,
    extract_training_data,
    apply_grpo_normalization as apply_branched_grpo_normalization
)


def run_one_game(player_cfg: dict, model_id: str, instructions: str, game_id: int = 0) -> Dict[str, Any]:
    # Build two players sharing the same server config
    console = type("_C", (), {"print": lambda *args, **kwargs: None, "rule": lambda *args, **kwargs: None})()
    cfg = SGLangConfig(
        temperature=player_cfg["temperature"],
        top_p=player_cfg["top_p"],
        max_tokens=player_cfg["max_new_tokens"],
        timeout=player_cfg.get("timeout", 120.0),
    )
    cfg.server_url = player_cfg["server_url"].rstrip("/")
    if not cfg.server_url.endswith("/v1"):
        cfg.server_url += "/v1"

    # Resolve the {unknown_value} placeholder as in the reference rollout
    instr_p1 = instructions.replace("{unknown_value}", "50")
    instr_p2 = instructions.replace("{unknown_value}", "50")

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

    # Pass game limits to environment so agents can see them
    env = OptimizationEnv(
        max_turns=player_cfg.get("max_turns", 30),
        max_retries_per_turn=player_cfg.get("max_retries_per_turn", 8)
    )
    obs = env.reset(game_state=None)
    p1.observe(obs["player-1"])  # give initial observation
    p2.observe(obs["player-2"])  # give initial observation

    clean_conversation = []
    full_conversation = []
    done = False
    turn = 0
    max_turns = player_cfg.get("max_turns", 30)
    max_retries = player_cfg.get("max_retries_per_turn", 8)

    # Collect logprobs DURING the game loop (not after)
    p1_all_logprobs = []
    p2_all_logprobs = []

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

                # Collect logprobs immediately after successful generation
                logprobs = player.get_last_generation_logprobs()
                if logprobs:
                    if current == "player-1":
                        p1_all_logprobs.extend(logprobs)
                    else:
                        p2_all_logprobs.extend(logprobs)

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
        "policy_logprobs": {
            "player-1": p1_all_logprobs,
            "player-2": p2_all_logprobs,
        },
    }
    return result


def run_branched_game_tree(
    player_cfg: dict,
    model_id: str,
    instructions: str,
    game_id: int,
    max_model_len: int,
    branch_points: List[int],
    branch_factor: int,
    executor=None,
    session=None,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """Run a branched game tree and return training data rows.

    Args:
        player_cfg: Player configuration dict
        model_id: Model identifier
        instructions: Game instructions
        game_id: Base game ID
        max_model_len: Maximum model sequence length
        branch_points: List of turn numbers to branch at
        branch_factor: Number of branches per point
        executor: Optional ThreadPoolExecutor for parallel branch execution
        session: Optional requests.Session for connection reuse
        progress_callback: Optional callback for progress events

    Returns:
        List of training data dicts (already tensorized and GRPO-grouped)
    """
    # Create manager
    manager = BranchedGameManager(
        branch_points=branch_points,
        branch_factor=branch_factor
    )

    # Generate tree
    tree = manager.generate_game_tree(
        base_game_id=game_id,
        player_cfg=player_cfg,
        model_id=model_id,
        instructions=instructions,
        max_model_len=max_model_len,
        executor=executor,
        session=session,
        progress_callback=progress_callback,
    )

    # Extract training data
    rows = extract_training_data(tree, max_model_len)

    # Apply GRPO normalization (within this tree)
    rows = apply_branched_grpo_normalization(rows)

    return rows


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
    """Process a single game result into training samples for both players."""
    rows = []
    # two sequences (player-1 and player-2)
    for player_name in ["player-1", "player-2"]:
        p = result["players"][player_name]
        input_ids, attention_mask, position_ids, response_loss_mask = tensorize_player(p, max_model_len)
        # Use absolute normalized reward as weight (will be converted to relative in GRPO grouping)
        w = float(result["normalized_reward"])

        # Get policy logprobs for this player
        policy_logprobs = result.get("policy_logprobs", {}).get(player_name, [])

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


def smart_queue_feeder(
    task_queue,
    total_games: int,
    branch_executor,
    initial_batch: int = 4,
    initial_delay: float = 20.0,
    check_interval: float = 15.0,
):
    """Feed games to queue with throttling to maximize cache hits.

    Strategy:
    1. Submit first 4 games immediately
    2. Wait 20s for them to branch out and populate the thread pool
    3. Then check every 15s: if pool < 50% utilized, submit ONE tree and wait another 15s
    4. This ensures existing trees' branches/leaves get priority

    Args:
        task_queue: Queue to feed game IDs into
        total_games: Total number of games to submit
        branch_executor: TrackedThreadPool to monitor for availability
        initial_batch: Number of games to submit upfront (default: 4)
        initial_delay: Seconds to wait after initial batch (default: 20)
        check_interval: Seconds between availability checks (default: 15)
    """
    games_submitted = 0

    # Submit first batch games immediately
    for i in range(min(initial_batch, total_games)):
        task_queue.put(games_submitted)
        print(f"[QueueFeeder] Submitted initial tree {games_submitted}")
        games_submitted += 1

    # Wait for initial games to branch out
    print(f"[QueueFeeder] Waiting {initial_delay}s for initial trees to populate pool...")
    time.sleep(initial_delay)

    # Throttle subsequent submissions based on pool availability
    while games_submitted < total_games:
        available = branch_executor.available_workers()

        # Only submit ONE tree if more than 1 thread is idle
        if available > 1:
            task_queue.put(games_submitted)
            print(f"[QueueFeeder] Submitted tree {games_submitted} ({available} threads available)")
            games_submitted += 1
            # Wait another check_interval after submitting
            time.sleep(check_interval)
        else:
            # Log when throttling (occasionally)
            if games_submitted == min(initial_batch, total_games):  # Log first throttle
                print(f"[QueueFeeder] Pool saturated ({available} threads available), waiting {check_interval}s...")
            time.sleep(check_interval)

    print(f"[QueueFeeder] All {total_games} trees submitted")


def progress_aggregator(
    progress_queue,
    total_games: int,
    branch_factor: int,
    start_time: float,
):
    """Aggregate and display fine-grained progress updates.

    Args:
        progress_queue: Queue receiving progress events from workers
        total_games: Total number of game trees to generate
        branch_factor: Branching factor (e.g., 8)
        start_time: Time when generation started
    """
    total_leaves = total_games * branch_factor ** 2
    leaves_completed = 0
    trees_completed = 0

    # Track per-tree progress
    tree_stats = {}  # {tree_id: {branches_done: int, leaves_done: int}}

    last_log_time = time.time()
    leaf_completion_timestamps = []  # Track wall-clock timestamps when leaves complete

    while trees_completed < total_games:
        try:
            event = progress_queue.get(timeout=1.0)

            if event['event'] == 'tree_started':
                # Initialize tree stats immediately when tree starts
                tree_id = event['tree_id']
                tree_stats[tree_id] = {'branches_done': 0, 'leaves_done': 0}
                print(f"[Tree] Tree {tree_id:3d} STARTED")

            elif event['event'] == 'leaf_complete':
                leaves_completed += 1
                leaf_completion_timestamps.append(time.time())  # Store wall-clock time, not duration

                tree_id = event['tree_id']
                if tree_id not in tree_stats:
                    tree_stats[tree_id] = {'branches_done': 0, 'leaves_done': 0}
                tree_stats[tree_id]['leaves_done'] += 1

                # Verbose logging: every leaf
                print(
                    f"[Leaf] Tree {tree_id:3d}, Branch {event['branch_id']}, "
                    f"Leaf {event['leaf_id']} complete "
                    f"(reward: {event['reward']:.3f}, {event['elapsed']:.1f}s)"
                )

            elif event['event'] == 'branch_complete':
                tree_id = event['tree_id']
                if tree_id not in tree_stats:
                    tree_stats[tree_id] = {'branches_done': 0, 'leaves_done': 0}
                tree_stats[tree_id]['branches_done'] += 1

                print(f"[Branch] Tree {tree_id:3d}, Branch {event['branch_id']} complete")

            elif event['event'] == 'tree_complete':
                trees_completed += 1
                tree_id = event['tree_id']
                print(f"[Tree] Tree {tree_id:3d} COMPLETE ({trees_completed}/{total_games})")

        except queue.Empty:
            pass

        # Status update every 30s
        elapsed = time.time() - start_time
        if elapsed > 0 and time.time() - last_log_time > 30:
            leaves_rate = leaves_completed / elapsed if elapsed > 0 else 0

            # Recent rate (last 100 leaves) - use wall-clock time span
            recent_timestamps = leaf_completion_timestamps[-100:]
            if len(recent_timestamps) >= 2:
                recent_span = recent_timestamps[-1] - recent_timestamps[0]
                recent_rate = (len(recent_timestamps) - 1) / recent_span if recent_span > 0 else 0
            else:
                recent_rate = 0

            active_trees = [
                tid for tid, stats in tree_stats.items()
                if stats['leaves_done'] < branch_factor ** 2
            ]

            print(f"\n{'='*80}")
            print(f"Status Update (t={elapsed:.0f}s):")
            print(f"  Trees: {trees_completed}/{total_games} complete ({100*trees_completed/total_games:.1f}%)")
            print(f"  Leaves: {leaves_completed}/{total_leaves} complete ({100*leaves_completed/total_leaves:.1f}%)")
            print(f"  Rates: {leaves_rate:.2f} leaves/s avg | {recent_rate:.2f} leaves/s recent")
            print(f"  Active trees: {len(active_trees)}")
            if active_trees[:10]:
                print(f"    IDs: {active_trees[:10]}{'...' if len(active_trees) > 10 else ''}")
            print(f"{'='*80}\n")

            last_log_time = time.time()

    print(f"\n[ProgressAggregator] All trees complete!")


def worker_process(
    task_queue,
    result_queue,
    threads_per_proc: int,
    player_cfg: Dict[str, Any],
    model_id: str,
    instructions: str,
    max_model_len: int,
    seed: int,
    process_id: int,
    total_games: int,
    branching_mode: bool = False,
    branch_points: List[int] = [1, 2],
    branch_factor: int = 8,
    progress_queue = None,
):
    """Worker process that runs a thread pool pulling games from shared queue.

    Args:
        task_queue: Shared queue containing game_ids to process
        result_queue: Shared queue to put results into
        threads_per_proc: Number of threads to run in this process
        player_cfg: Player configuration dict
        model_id: Model identifier
        instructions: Game instructions
        max_model_len: Maximum model sequence length
        seed: Random seed base
        process_id: ID of this process for seeding
        total_games: Total number of games to generate
        branching_mode: Whether to use branching (default: False)
        branch_points: Turn numbers to branch at (default: [1, 2])
        branch_factor: Number of branches per point (default: 8)
        progress_queue: Optional queue for progress events
    """
    # Set unique random seed for this process
    np.random.seed(seed + process_id)

    # Create shared HTTP session for this process (for connection reuse)
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except Exception:
        Retry = None

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=256)
    if Retry is not None:
        retry = Retry(total=0, backoff_factor=0)
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=256, max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    })

    # Create tracked executor for parallel branch execution within trees
    # This limits concurrent HTTP requests from this process
    from branched_game_manager import TrackedThreadPool
    branch_executor = TrackedThreadPool(max_workers=threads_per_proc)

    # Start queue feeder thread (for branching mode)
    if branching_mode:
        import threading
        feeder_thread = threading.Thread(
            target=smart_queue_feeder,
            args=(task_queue, total_games, branch_executor),
            daemon=True
        )
        feeder_thread.start()

    def thread_worker():
        """Thread worker that pulls game_ids from queue and processes them."""
        while True:
            try:
                game_id = task_queue.get_nowait()
            except queue.Empty:
                return  # No more work

            try:
                if branching_mode:
                    # Progress callback
                    def progress_callback(event):
                        if progress_queue:
                            progress_queue.put(event)

                    # Generate branched tree (returns pre-normalized rows)
                    # Pass shared executor and session for parallel execution and connection reuse
                    rows = run_branched_game_tree(
                        player_cfg, model_id, instructions, game_id,
                        max_model_len, branch_points, branch_factor,
                        executor=branch_executor,
                        session=session,
                        progress_callback=progress_callback,
                    )
                else:
                    # Generate single game
                    result = run_one_game(player_cfg, model_id, instructions, game_id)
                    rows = process_game_result(result, max_model_len)
                result_queue.put(rows)
            except Exception as e:
                # Log failure but don't crash the worker
                print(f"[Process {process_id}] Game {game_id} failed: {e}")
                import traceback
                traceback.print_exc()
                # Put empty result to maintain progress tracking
                result_queue.put([])

    # Run thread pool for this process (for pulling games from queue)
    with ThreadPoolExecutor(max_workers=threads_per_proc) as game_puller_executor:
        futures = [game_puller_executor.submit(thread_worker) for _ in range(threads_per_proc)]
        # Wait for all threads to finish
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[Process {process_id}] Thread crashed: {e}")

    # Clean up branch executor
    branch_executor.shutdown(wait=True)


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
    ap.add_argument("--num-procs", type=int, default=4, help="Number of worker processes (per server)")
    ap.add_argument("--threads-per-proc", type=int, default=64, help="Threads per process for issuing requests")
    ap.add_argument("--progress-interval", type=int, default=32, help="Print progress every N completed games")
    # Game termination settings
    ap.add_argument("--max-turns", type=int, default=10, help="Maximum number of turns per game (default: 10)")
    ap.add_argument("--max-retries-per-turn", type=int, default=2, help="Maximum retries per turn before terminating (default: 2)")
    # Branching settings for credit assignment
    ap.add_argument("--branching-mode", action="store_true", help="Enable branching mode for credit assignment (default: False)")
    ap.add_argument("--branch-points", type=int, nargs=2, default=[1, 2], help="Turn numbers to branch at (default: 1 2)")
    ap.add_argument("--branch-factor", type=int, default=8, help="Number of branches per branch point (default: 8)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Set log directory for generation failures
    out_path = Path(args.out)
    log_dir = str(out_path.parent.resolve())
    os.environ["ROLLOUT_LOG_DIR"] = log_dir
    print(f"Generation failure logs will be written to: {log_dir}/generation_failures.log")

    # Load instructions used by DialopSelfPlayRollout
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

    # Calculate total games to play
    if args.branching_mode:
        # In branching mode, each "game" is a full tree
        # Each tree produces (branch_factor^2 + branch_factor) × 2 sequences
        # = (64 + 8) × 2 = 144 sequences per tree
        total_games = args.num_games  # User specifies number of trees
        seqs_per_tree = (args.branch_factor ** 2 + args.branch_factor) * 2
        print(f"Branching mode enabled: branching at turns {args.branch_points} with factor {args.branch_factor}")
        print(f"Each game tree produces {seqs_per_tree} training sequences")
        print(f"Total trees to generate: {total_games}")
    else:
        # Standard mode: num_games * group_size
        total_games = args.num_games * args.group_size
        print(
            f"Starting generation of {total_games} total games "
            f"({args.num_games} unique games × {args.group_size} plays each)"
        )

    # HARDCODED: 1 process × 200 threads for now
    num_procs = 1
    threads_per_proc = 200

    total_workers = num_procs * threads_per_proc
    print(
        f"Worker configuration: {num_procs} process × {threads_per_proc} threads = {total_workers} concurrent workers"
    )
    print(f"Server: {args.server_url}")
    start_time = time.time()

    # Create shared queues using Manager
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()

    # Start progress aggregator thread (for branching mode)
    if args.branching_mode:
        import threading
        progress_thread = threading.Thread(
            target=progress_aggregator,
            args=(progress_queue, total_games, args.branch_factor, start_time),
            daemon=True
        )
        progress_thread.start()
        print(f"Started progress aggregator thread")

    # DON'T populate task queue upfront in branching mode
    # The queue feeder in worker_process will handle it
    if not args.branching_mode:
        # Standard mode: populate all games upfront
        for game_id in range(total_games):
            task_queue.put(game_id)
        print(f"Populated task queue with {total_games} games")
    else:
        print(f"Branching mode: queue will be fed dynamically by worker")

    # Spawn worker processes
    processes = []
    for proc_id in range(num_procs):
        p = Process(
            target=worker_process,
            args=(
                task_queue,
                result_queue,
                threads_per_proc,
                player_cfg,
                args.model_id,
                instructions,
                args.max_model_len,
                args.seed,
                proc_id,
                total_games,
                args.branching_mode,
                args.branch_points,
                args.branch_factor,
                progress_queue,
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

    # Calculate rows per game for progress tracking
    if args.branching_mode:
        rows_per_game = (args.branch_factor ** 2 + args.branch_factor) * 2  # 144 for default 8x8
    else:
        rows_per_game = 2  # Standard mode: 2 players

    while completed_games < total_games:
        try:
            # Non-blocking check for results with short timeout
            rows = result_queue.get(timeout=1.0)
            all_rows.extend(rows)
            completed_games += len(rows) // rows_per_game  # Calculate based on mode

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
    print(f"  Rows: {len(all_rows)} ({len(all_rows) // 2} games × 2 players)")
    print(f"  Rate: {final_rate:.1f} games/s")
    print(f"{'='*60}\n")

    # Apply GRPO normalization (only for standard mode; branching mode already normalized)
    if args.branching_mode:
        print(f"Branching mode: GRPO normalization already applied per-tree")
        print(f"Total sequences collected: {len(all_rows)}")
    else:
        # Group sequences by unique game sets for GRPO normalization
        # Each unique game was played group_size times, creating 2*group_size sequences per unique game
        print(f"Grouping sequences for GRPO normalization...")

        # Sort rows by unique_game_id to ensure proper grouping
        all_rows.sort(key=lambda x: x["game_id"] // args.group_size)

        # Apply GRPO normalization within each group of 2*group_size sequences
        sequences_per_group = 2 * args.group_size  # 2 players × group_size plays
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
