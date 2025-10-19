#!/usr/bin/env python3
"""
Branched Game Manager for tree-based rollout generation.

Manages branching game trees for precise credit assignment in GRPO training.
Each base game branches at specific turns (default: turns 1 and 2) to create
a tree structure where early vs. late decisions can be evaluated separately.
"""

import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from dialop.sglang_model_player import SGLangModelPlayer, SGLangConfig
from dialop.envs.optimization import OptimizationEnv


class TrackedThreadPool:
    """Thread pool with visibility into active worker count.

    Allows monitoring of thread pool occupancy to implement smart
    queue feeding for cache optimization.
    """

    def __init__(self, max_workers: int):
        """Initialize tracked thread pool.

        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers
        self.active_count = 0
        self.lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        """Submit a callable to the thread pool with tracking.

        Args:
            fn: Callable to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Future representing the pending execution
        """
        with self.lock:
            self.active_count += 1

        def wrapper():
            try:
                return fn(*args, **kwargs)
            finally:
                with self.lock:
                    self.active_count -= 1

        return self.executor.submit(wrapper)

    def available_workers(self) -> int:
        """Get number of currently available (idle) workers.

        Returns:
            Number of workers not currently executing tasks
        """
        with self.lock:
            return self.max_workers - self.active_count

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool.

        Args:
            wait: If True, wait for pending futures to complete
        """
        self.executor.shutdown(wait=wait)


@dataclass
class BranchNode:
    """Represents one node in the branched game tree."""

    node_id: str  # e.g., "0", "0-3", "0-3-7"
    parent_id: Optional[str]
    depth: int  # 0=root, 1=first branch level, 2=leaf level

    # Player references (will be cloned for branches)
    player_1: SGLangModelPlayer
    player_2: SGLangModelPlayer

    # Environment state (serialized for checkpointing)
    env_state: Dict[str, Any]

    # Turn information
    turn_number: int  # Turn at which this branch was created

    # Results (filled when game completes)
    final_reward: Optional[float] = None
    normalized_reward: Optional[float] = None
    conversation: Optional[List[Dict]] = None
    game_info: Optional[Dict] = None
    is_complete: bool = False


class BranchedGameManager:
    """Manages branching game tree generation for credit assignment."""

    def __init__(
        self,
        branch_points: List[int] = [1, 2],
        branch_factor: int = 8,
    ):
        """Initialize the branched game manager.

        Args:
            branch_points: List of turn numbers to branch at (default: [1, 2])
            branch_factor: Number of branches per point (default: 8)
        """
        self.branch_points = sorted(branch_points)
        self.branch_factor = branch_factor

        # Validate branch points
        if len(self.branch_points) != 2:
            raise ValueError(f"Expected exactly 2 branch points, got {len(self.branch_points)}")
        if self.branch_points[0] >= self.branch_points[1]:
            raise ValueError(f"Branch points must be strictly increasing: {self.branch_points}")

    def generate_game_tree(
        self,
        base_game_id: int,
        player_cfg: Dict[str, Any],
        model_id: str,
        instructions: str,
        max_model_len: int,
        session: Optional[Any] = None,
        executor: Optional[Any] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Generate a complete branched game tree with optional parallel execution.

        Args:
            base_game_id: Unique identifier for this base game
            player_cfg: Configuration dict for players (server_url, temperature, etc.)
            model_id: Model identifier
            instructions: Game instructions template
            max_model_len: Maximum model sequence length
            session: Optional requests.Session for connection reuse
            executor: Optional ThreadPoolExecutor for parallel branch execution
            progress_callback: Optional callback for progress events

        Returns:
            Dict containing:
                - base_game_id: int
                - nodes: Dict[node_id, BranchNode]
                - leaf_ids: List[str] (64 leaf node IDs)
                - branch_ids: List[str] (8 first-level branch IDs)
        """
        # Report tree start immediately
        if progress_callback:
            progress_callback({
                'event': 'tree_started',
                'tree_id': base_game_id,
            })

        # 1. Start root game
        root = self._initialize_root(base_game_id, player_cfg, model_id, instructions, session=session)

        # 2. Play root game to first branch point
        self._play_until_turn(root, target_turn=self.branch_points[0])

        # 3. Branch at first branch point with eager leaf submission
        if executor:
            # Parallel execution: submit all first-level branches AT ONCE
            first_futures = {
                executor.submit(
                    self._create_and_play_branch,
                    root, i, self.branch_points[1], base_game_id, progress_callback
                ): i
                for i in range(self.branch_factor)
            }

            # 4. As EACH branch completes, IMMEDIATELY submit its 8 leaves
            first_branches = [None] * self.branch_factor
            leaf_futures = []

            for future in as_completed(first_futures):
                branch_idx = first_futures[future]
                branch = future.result()
                first_branches[branch_idx] = branch

                # Report branch completion
                if progress_callback:
                    progress_callback({
                        'event': 'branch_complete',
                        'tree_id': base_game_id,
                        'branch_id': branch_idx,
                    })

                # IMMEDIATELY submit 8 leaves (while branch prefix is hot in cache)
                for leaf_idx in range(self.branch_factor):
                    future = executor.submit(
                        self._create_and_play_leaf_with_tracking,
                        branch, leaf_idx, base_game_id, branch_idx, progress_callback
                    )
                    leaf_futures.append(future)

            # 5. Collect all leaves as they complete
            leaf_nodes = [f.result() for f in leaf_futures]

        else:
            # Sequential fallback
            first_branches = []
            for i in range(self.branch_factor):
                branch = self._create_branch(root, branch_index=i)
                self._play_until_turn(branch, target_turn=self.branch_points[1])
                first_branches.append(branch)

            leaf_nodes = []
            for first_branch in first_branches:
                for i in range(self.branch_factor):
                    leaf = self._create_branch(first_branch, branch_index=i)
                    self._play_to_completion(leaf)
                    leaf_nodes.append(leaf)

        # 6. Build result structure
        all_nodes = {root.node_id: root}
        all_nodes.update({n.node_id: n for n in first_branches})
        all_nodes.update({n.node_id: n for n in leaf_nodes})

        # Report tree completion
        if progress_callback:
            progress_callback({
                'event': 'tree_complete',
                'tree_id': base_game_id,
            })

        return {
            "base_game_id": base_game_id,
            "nodes": all_nodes,
            "leaf_ids": [n.node_id for n in leaf_nodes],
            "branch_ids": [n.node_id for n in first_branches],
        }

    def _initialize_root(
        self,
        game_id: int,
        player_cfg: Dict[str, Any],
        model_id: str,
        instructions: str,
        session: Optional[Any] = None,
    ) -> BranchNode:
        """Initialize the root node of the game tree.

        Args:
            game_id: Unique game identifier
            player_cfg: Player configuration dict
            model_id: Model identifier
            instructions: Game instructions
            session: Optional requests.Session for connection reuse (both players will share)
        """
        # Create console for players
        from rich.console import Console
        console = type("_C", (), {"print": lambda *args, **kwargs: None, "rule": lambda *args, **kwargs: None})()

        # Create SGLang config
        cfg = SGLangConfig(
            temperature=player_cfg["temperature"],
            top_p=player_cfg["top_p"],
            max_tokens=player_cfg["max_new_tokens"],
            timeout=player_cfg.get("timeout", 120.0),
        )
        cfg.server_url = player_cfg["server_url"].rstrip("/")
        if not cfg.server_url.endswith("/v1"):
            cfg.server_url += "/v1"

        # Resolve the {unknown_value} placeholder
        instr_p1 = instructions.replace("{unknown_value}", "50")
        instr_p2 = instructions.replace("{unknown_value}", "50")

        # Create players with metadata
        p1_metadata = {"game_id": game_id, "turn": 0, "current_turn": 0}
        p2_metadata = {"game_id": game_id, "turn": 0, "current_turn": 0}

        p1 = SGLangModelPlayer(
            system_prompt=instr_p1,
            role="player-1",
            console=console,
            model_path=model_id,
            config=cfg,
            optional=p1_metadata,
            session=session,  # Share session for connection reuse
        )
        p2 = SGLangModelPlayer(
            system_prompt=instr_p2,
            role="player-2",
            console=console,
            model_path=model_id,
            config=cfg,
            optional=p2_metadata,
            session=session,  # Share session for connection reuse
        )

        # Create environment
        env = OptimizationEnv(
            max_turns=player_cfg.get("max_turns", 30),
            max_retries_per_turn=player_cfg.get("max_retries_per_turn", 8)
        )
        obs = env.reset(game_state=None)

        # Give initial observations
        p1.observe(obs["player-1"])
        p2.observe(obs["player-2"])

        # Serialize environment state
        env_state = self._serialize_env_state(env, obs)

        # Create root node
        return BranchNode(
            node_id=str(game_id),
            parent_id=None,
            depth=0,
            player_1=p1,
            player_2=p2,
            env_state=env_state,
            turn_number=0,
        )

    def _create_branch(self, parent: BranchNode, branch_index: int) -> BranchNode:
        """Create a new branch by cloning parent state."""
        # Create new node ID
        node_id = f"{parent.node_id}-{branch_index}"

        # Deep copy players (this copies message history and turn markers)
        p1_clone = self._clone_player(parent.player_1)
        p2_clone = self._clone_player(parent.player_2)

        # Clone environment state
        env_state_clone = copy.deepcopy(parent.env_state)

        # Create new node
        return BranchNode(
            node_id=node_id,
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            player_1=p1_clone,
            player_2=p2_clone,
            env_state=env_state_clone,
            turn_number=parent.turn_number,
        )

    def _create_and_play_branch(
        self,
        parent: BranchNode,
        branch_index: int,
        target_turn: int,
        tree_id: int,
        progress_callback: Optional[Callable],
    ) -> BranchNode:
        """Create a branch and play it to a target turn (for parallel execution).

        Args:
            parent: Parent node to branch from
            branch_index: Index of this branch
            target_turn: Turn to play until
            tree_id: ID of the tree this branch belongs to
            progress_callback: Optional callback for progress events

        Returns:
            Completed branch node
        """
        branch = self._create_branch(parent, branch_index)
        self._play_until_turn(branch, target_turn=target_turn)
        return branch

    def _create_and_play_leaf_with_tracking(
        self,
        parent: BranchNode,
        leaf_idx: int,
        tree_id: int,
        branch_id: int,
        progress_callback: Optional[Callable],
    ) -> BranchNode:
        """Create a leaf and play it to completion with progress tracking.

        Args:
            parent: Parent node to branch from
            leaf_idx: Index of this leaf
            tree_id: ID of the tree this leaf belongs to
            branch_id: ID of the parent branch
            progress_callback: Optional callback for progress events

        Returns:
            Completed leaf node
        """
        start_time = time.time()

        leaf = self._create_branch(parent, leaf_idx)
        self._play_to_completion(leaf)

        if progress_callback:
            progress_callback({
                'event': 'leaf_complete',
                'tree_id': tree_id,
                'branch_id': branch_id,
                'leaf_id': leaf_idx,
                'elapsed': time.time() - start_time,
                'reward': leaf.normalized_reward if leaf.normalized_reward is not None else 0.0,
            })

        return leaf

    def _clone_player(self, player: SGLangModelPlayer) -> SGLangModelPlayer:
        """Deep copy a player, preserving message history and turn markers.

        Shares the HTTP session from the parent player for connection reuse.
        """
        # Create new player with same config
        console = type("_C", (), {"print": lambda *args, **kwargs: None, "rule": lambda *args, **kwargs: None})()

        # Get the system prompt from first message
        system_prompt = player.messages[0]["content"] if player.messages and player.messages[0]["role"] == "system" else ""

        new_player = SGLangModelPlayer(
            system_prompt=system_prompt,
            role=player.role,
            console=console,
            model_path=player.model_path,
            config=player.config,
            optional=copy.deepcopy(player.optional),
            session=player.session,  # Share the session for connection reuse
        )

        # Copy message history
        new_player.messages = copy.deepcopy(player.messages)

        # Copy turn markers
        new_player.turn_markers = copy.deepcopy(player.turn_markers)
        new_player.current_turn = player.current_turn

        # Copy tokenizer reference (don't deep copy the tokenizer itself)
        new_player.tokenizer = player.tokenizer

        return new_player

    def _reconstruct_full_conversation(self, player_1: SGLangModelPlayer, player_2: SGLangModelPlayer) -> List[Dict]:
        """Reconstruct the full conversation from both players' message histories.

        Uses turn_markers from both players to properly attribute messages to turns,
        including error messages and retry attempts.

        Args:
            player_1: First player with complete message history
            player_2: Second player with complete message history

        Returns:
            List of conversation entries with fields:
                - turn: Turn number (int)
                - player: Which player acted ("player-1" or "player-2")
                - message: The message content
                - retry: Retry attempt number (0 for first attempt, 1+ for retries)
        """
        conversation = []

        # Process each player's messages with turn tracking
        for player in [player_1, player_2]:
            player_name = player.role  # "player-1" or "player-2"

            # Skip system message, process user and assistant messages
            assistant_idx = 0
            for i, msg in enumerate(player.messages):
                if msg["role"] == "system":
                    continue

                elif msg["role"] == "assistant":
                    # Find which turn this assistant message belongs to
                    turn_num = None
                    for turn, msg_idx in player.turn_markers:
                        if msg_idx == assistant_idx:
                            turn_num = turn
                            break

                    if turn_num is not None:
                        # Count how many assistant messages we've seen for this turn
                        retry = sum(1 for t, idx in player.turn_markers[:assistant_idx+1] if t == turn_num) - 1

                        conversation.append({
                            "turn": turn_num,
                            "player": player_name,
                            "message": msg["content"],
                            "retry": retry,
                            "msg_type": "assistant",
                        })

                    assistant_idx += 1

                elif msg["role"] == "user":
                    # User messages to the active player are either:
                    # 1. Error messages (during retries)
                    # 2. Game state updates (after successful moves)
                    # We can identify errors by looking at the context:
                    # - Errors come between assistant messages with the same turn number
                    # - Game updates come after turn increment

                    # For now, mark as error if it contains "Error:" or comes between retries
                    # We'll infer the turn number from the surrounding assistant messages
                    content = msg["content"]

                    # Try to determine turn number by looking at previous assistant message
                    prev_turn = None
                    if assistant_idx > 0:
                        # Look at the turn of the most recent assistant message
                        for turn, msg_idx in reversed(player.turn_markers):
                            if msg_idx == assistant_idx - 1:
                                prev_turn = turn
                                break

                    # Errors typically contain "Error:" string
                    if prev_turn is not None and ("Error:" in content or "error" in content.lower()):
                        # The error is feedback for the previous assistant attempt
                        # Count how many attempts we've seen for this turn up to previous assistant
                        retry = sum(1 for t, idx in player.turn_markers[:assistant_idx] if t == prev_turn) - 1

                        conversation.append({
                            "turn": prev_turn,
                            "player": "error",
                            "message": content,
                            "retry": retry,
                            "msg_type": "error",
                        })

        # Sort by (turn, retry, msg_type) to get chronological order
        # Within each turn and retry number:
        # - assistant message comes first (the attempt)
        # - error message comes second (the feedback)
        def sort_key(entry):
            # Primary: turn number
            # Secondary: retry number
            # Tertiary: message type (assistant=0, error=1)
            type_order = {"assistant": 0, "error": 1}
            return (entry["turn"], entry["retry"], type_order.get(entry["msg_type"], 2))

        conversation.sort(key=sort_key)

        # Clean up msg_type field (not needed in output)
        for entry in conversation:
            entry.pop("msg_type", None)

        return conversation

    def _serialize_env_state(self, env: OptimizationEnv, obs: Dict) -> Dict[str, Any]:
        """Serialize environment state for checkpointing."""
        return {
            "game_state": env.game.get_game_info(),
            "num_msgs": env.num_msgs,
            "current_turn": env.current_turn,
            "current_retry": env.current_retry,
            "observation": obs,
        }

    def _deserialize_env_state(self, env_state: Dict[str, Any], player_cfg: Dict[str, Any]) -> Tuple[OptimizationEnv, Dict]:
        """Deserialize environment state from checkpoint."""
        # Create new environment
        env = OptimizationEnv(
            max_turns=player_cfg.get("max_turns", 30),
            max_retries_per_turn=player_cfg.get("max_retries_per_turn", 8)
        )

        # Restore game state
        obs = env.reset(game_state=env_state["game_state"])
        env.num_msgs = env_state["num_msgs"]
        env.current_turn = env_state["current_turn"]
        env.current_retry = env_state["current_retry"]

        return env, env_state["observation"]

    def _play_until_turn(self, node: BranchNode, target_turn: int):
        """Play the game until reaching a specific turn (for branching)."""
        # Deserialize environment
        player_cfg = {
            "max_turns": 30,
            "max_retries_per_turn": 8,
        }
        env, obs = self._deserialize_env_state(node.env_state, player_cfg)

        # Continue playing until target turn
        done = obs.get("done", False)
        turn = node.turn_number
        max_turns = 30
        max_retries = 8

        while not done and turn < target_turn:
            current = obs["turn_player"]
            player = node.player_1 if current == "player-1" else node.player_2

            # Update player metadata with current turn
            player.optional["turn"] = turn
            player.optional["current_turn"] = turn

            retries = 0
            while True:
                try:
                    response = player.respond()
                except Exception as e:
                    # Generation failed - terminate
                    done = True
                    break

                obs, error = env.step(response)
                if error:
                    retries += 1
                    player.observe(obs[current])
                    if retries >= max_retries:
                        done = True
                        break
                    continue

                # Valid move - broadcast to other player
                other_name = "player-2" if current == "player-1" else "player-1"
                other_player = node.player_2 if current == "player-1" else node.player_1
                if obs[other_name]:
                    other_player.observe(obs[other_name])

                done = obs["done"]
                turn += 1
                break

        # Update node state
        node.turn_number = turn
        node.env_state = self._serialize_env_state(env, obs)

    def _play_to_completion(self, node: BranchNode):
        """Play a game to completion (for leaf nodes)."""
        # Deserialize environment
        player_cfg = {
            "max_turns": 30,
            "max_retries_per_turn": 8,
        }
        env, obs = self._deserialize_env_state(node.env_state, player_cfg)

        # Continue playing until done
        done = obs.get("done", False)
        turn = node.turn_number
        max_turns = 30
        max_retries = 8

        while not done and turn < max_turns:
            current = obs["turn_player"]
            player = node.player_1 if current == "player-1" else node.player_2

            # Update player metadata
            player.optional["turn"] = turn
            player.optional["current_turn"] = turn

            retries = 0
            while True:
                try:
                    response = player.respond()
                except Exception as e:
                    # Generation failed - terminate
                    done = True
                    break

                obs, error = env.step(response)
                if error:
                    retries += 1
                    player.observe(obs[current])
                    if retries >= max_retries:
                        done = True
                        break
                    continue

                # Valid move - broadcast to other player
                other_name = "player-2" if current == "player-1" else "player-1"
                other_player = node.player_2 if current == "player-1" else node.player_1
                if obs[other_name]:
                    other_player.observe(obs[other_name])

                done = obs["done"]
                turn += 1
                break

        # Store results
        node.final_reward = obs.get("info", {}).get("score", 0.0) if done else 0.0
        node.normalized_reward = obs.get("info", {}).get("score_norm", 0.0) if done else 0.0

        # Reconstruct full conversation from both players' message histories
        # This includes all turns (0 onwards) with proper turn tracking and retry attribution
        node.conversation = self._reconstruct_full_conversation(node.player_1, node.player_2)

        node.game_info = {
            "num_messages": obs.get("info", {}).get("num_msgs", turn),
            "completed": done,
            "turn_count": turn,
            "game_normalized_reward": node.normalized_reward,
        }
        node.is_complete = True
        node.turn_number = turn


def extract_training_data(tree: Dict[str, Any], max_model_len: int) -> List[Dict[str, Any]]:
    """Extract training data from a branched game tree.

    Args:
        tree: Game tree dict with nodes, leaf_ids, branch_ids
        max_model_len: Maximum model sequence length

    Returns:
        List of training data rows (dicts) ready for DataFrame conversion
    """
    all_rows = []

    # 1. Process 64 leaf games (train on turns >= 2)
    for leaf_id in tree["leaf_ids"]:
        leaf_node = tree["nodes"][leaf_id]

        # Process both players
        for player_name, player in [("player-1", leaf_node.player_1), ("player-2", leaf_node.player_2)]:
            row = _tensorize_player_with_turn_mask(
                player=player,
                max_model_len=max_model_len,
                turn_mask_min=2,  # Only train on turns >= 2
                turn_mask_max=None,  # No upper bound
                reward=leaf_node.normalized_reward,
                game_id=tree["base_game_id"],
                metadata={
                    "node_id": leaf_id,
                    "parent_id": leaf_node.parent_id,
                    "node_type": "leaf",
                    "depth": leaf_node.depth,
                },
                game_info=leaf_node.game_info,
                full_conversation=leaf_node.conversation,
            )
            all_rows.append(row)

    # 2. Process 8 first-level branches (train on turns 0-1 only)
    for branch_id in tree["branch_ids"]:
        branch_node = tree["nodes"][branch_id]

        # Compute mean reward of 8 children
        child_ids = [nid for nid in tree["leaf_ids"] if nid.startswith(branch_id + "-")]
        child_rewards = [tree["nodes"][cid].normalized_reward for cid in child_ids]
        mean_reward = float(np.mean(child_rewards))

        # Process both players (with truncated sequences)
        for player_name, player in [("player-1", branch_node.player_1), ("player-2", branch_node.player_2)]:
            row = _tensorize_player_with_turn_mask(
                player=player,
                max_model_len=max_model_len,
                turn_mask_min=None,  # No lower bound
                turn_mask_max=1,  # Only train on turns <= 1
                reward=mean_reward,
                game_id=tree["base_game_id"],
                metadata={
                    "node_id": branch_id,
                    "parent_id": branch_node.parent_id,
                    "node_type": "branch",
                    "depth": branch_node.depth,
                    "num_children": len(child_ids),
                    "mean_of_children": mean_reward,
                },
                game_info={"turn_count": branch_node.turn_number, "completed": False},
                full_conversation=[],  # Branches don't have full conversations
            )
            all_rows.append(row)

    return all_rows


def _tensorize_player_with_turn_mask(
    player: SGLangModelPlayer,
    max_model_len: int,
    turn_mask_min: Optional[int],
    turn_mask_max: Optional[int],
    reward: float,
    game_id: int,
    metadata: Dict[str, Any],
    game_info: Dict[str, Any],
    full_conversation: List[Dict],
) -> Dict[str, Any]:
    """Convert player state to training data dict with turn-based gradient masking.

    Args:
        player: The SGLangModelPlayer instance
        max_model_len: Maximum sequence length
        turn_mask_min: Minimum turn for gradient (None = no lower bound)
        turn_mask_max: Maximum turn for gradient (None = no upper bound)
        reward: Reward value (game-normalized)
        game_id: Base game ID
        metadata: Branch metadata dict
        game_info: Game info dict
        full_conversation: Full conversation history

    Returns:
        Dict with all fields needed for training
    """
    # Get full input sequence
    input_ids = player.get_input_sequence()

    # Truncate if we have turn_mask_max (for branch nodes training on early turns)
    if turn_mask_max is not None:
        # Find token index where turn_mask_max ends
        truncation_idx = player.get_turn_end_token_index(turn_mask_max)
        input_ids = input_ids[:truncation_idx]

    # Clip to max length
    if len(input_ids) > max_model_len:
        input_ids = input_ids[:max_model_len]

    # Basic masks
    attention_mask = [1] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    # Get assistant mask filtered by turn range (turn filtering happens here)
    assistant_mask = player.get_assistant_mask(turn_min=turn_mask_min, turn_max=turn_mask_max)

    # Clip to match input_ids length
    assistant_mask = assistant_mask[:len(input_ids)]

    # Pad assistant_mask if it's shorter
    if len(assistant_mask) < len(input_ids):
        assistant_mask = assistant_mask + [0] * (len(input_ids) - len(assistant_mask))

    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64),
        "position_ids": np.array(position_ids, dtype=np.int64),
        "loss_mask": np.array(assistant_mask, dtype=np.int64),
        "sample_weight": float(reward),
        "game_normalized_reward": float(reward),
        "game_info": json.dumps(game_info),
        "game_id": game_id,
        "full_conversation": json.dumps(full_conversation),
        "branch_metadata": json.dumps(metadata),
        "turn_mask_min": turn_mask_min,
        "turn_mask_max": turn_mask_max,
        # Note: policy_logprobs would need to be collected during generation
        # For now, set empty list
        "policy_logprobs": json.dumps([]),
    }


def apply_grpo_normalization(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply GRPO normalization to training data from branched trees.

    Groups sequences by tree structure:
    - Leaf sequences: Group by parent_id (8 groups of 16 sequences each)
    - Branch sequences: All together (1 group of 16 sequences)

    Within each group, normalizes rewards by subtracting the group mean.

    Args:
        rows: List of training data dicts from extract_training_data

    Returns:
        Same list with sample_weight updated to GRPO-normalized values
    """
    # Parse metadata for grouping
    for row in rows:
        metadata = json.loads(row["branch_metadata"])
        row["_node_type"] = metadata.get("node_type", "unknown")
        row["_parent_id"] = metadata.get("parent_id", None)

    # Separate leaf and branch sequences
    leaf_rows = [r for r in rows if r["_node_type"] == "leaf"]
    branch_rows = [r for r in rows if r["_node_type"] == "branch"]

    # Group leaf sequences by parent and normalize
    leaf_parents = {}
    for row in leaf_rows:
        parent = row["_parent_id"]
        if parent not in leaf_parents:
            leaf_parents[parent] = []
        leaf_parents[parent].append(row)

    for parent_id, group in leaf_parents.items():
        # Compute baseline (mean) for this group
        rewards = [r["sample_weight"] for r in group]
        baseline = float(np.mean(rewards))

        # Normalize
        for row in group:
            row["sample_weight"] = float(row["sample_weight"] - baseline)

    # Normalize all branch sequences as one group
    if branch_rows:
        rewards = [r["sample_weight"] for r in branch_rows]
        baseline = float(np.mean(rewards))

        for row in branch_rows:
            row["sample_weight"] = float(row["sample_weight"] - baseline)

    # Remove temporary fields
    for row in rows:
        row.pop("_node_type", None)
        row.pop("_parent_id", None)

    return rows
