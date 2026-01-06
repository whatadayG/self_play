# SLIME RL Training Pipeline for Paper-Reviewer Matching

## Overview

This plan leverages SLIME's existing multi-turn rollout pattern (`examples/geo3k_vlm_multi_turn`) to train Qwen on the paper-reviewer matching game with GPT-4.1 as a fixed "shy" partner.

**Key insight**: SLIME already provides exactly what we need:
- `geo3k_vlm_multi_turn/rollout.py` - Multi-turn rollout with loss_mask and logprob tracking
- `geo3k_vlm_multi_turn/env_geo3k.py` - Environment interface pattern (reset, step, format_observation)
- We just need to adapt these patterns to our Qwen↔GPT dialogue setup

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SLIME Training Loop                              │
│                                                                         │
│  ┌──────────────┐     ┌─────────────────────────────────────────────┐  │
│  │  game.jsonl  │────▶│  Custom Multi-Turn Rollout (rollout.py)     │  │
│  │  (prompts +  │     │                                             │  │
│  │  game_state) │     │   Turn Loop (max_turns):                    │  │
│  └──────────────┘     │   ┌─────────────────────────────────────┐   │  │
│                       │   │ if qwen_turn:                       │   │  │
│                       │   │   response = sglang.generate()      │   │  │
│                       │   │   loss_mask[tokens] = 1  ← TRAIN    │   │  │
│                       │   │   collect logprobs                  │   │  │
│                       │   │                                     │   │  │
│                       │   │ else (gpt_turn):                    │   │  │
│                       │   │   response = openai.chat()          │   │  │
│                       │   │   loss_mask[tokens] = 0  ← NO TRAIN │   │  │
│                       │   │                                     │   │  │
│                       │   │ obs, done, reward = env.step()      │   │  │
│                       │   └─────────────────────────────────────┘   │  │
│                       │                    │                        │  │
│                       │                    ▼                        │  │
│                       │            Return samples with:             │  │
│                       │            - input_ids (full dialogue)      │  │
│                       │            - loss_mask (Qwen tokens only)   │  │
│                       │            - logprobs (Qwen tokens only)    │  │
│                       │            - reward (from DialOp env)       │  │
│                       └─────────────────────────────────────────────┘  │
│                                            │                            │
│                                            ▼                            │
│                       ┌─────────────────────────────────────────────┐  │
│                       │     GRPO Training (Megatron + SGLang)       │  │
│                       └─────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
slim_rl/
├── PLAN.md                     # This document
├── configs/
│   └── matching_config.yaml    # SLIME config (like geo3k_vlm_multi_turn_config.yaml)
├── src/
│   ├── __init__.py
│   ├── env_matching.py         # Environment adapter (like env_geo3k.py)
│   ├── rollout.py              # Custom rollout (like geo3k_vlm_multi_turn/rollout.py)
│   ├── gpt_partner.py          # GPT-4.1 "shy" partner client
│   ├── data_generator.py       # Generate game instances as JSONL
│   └── question_logger.py      # Track questions per dialogue
├── scripts/
│   ├── generate_data.py        # CLI to generate training data
│   └── run_matching.sh         # Training launch script
└── data/
    └── games.jsonl             # Generated game instances
```

---

## Component 1: Environment Adapter (`env_matching.py`)

Adapts existing `OptimizationEnv` to SLIME's expected interface.

```python
"""Environment adapter for paper-reviewer matching game.

Follows SLIME's env pattern from examples/geo3k_vlm_multi_turn/env_geo3k.py
"""
import sys
sys.path.insert(0, "/home/nickatomlin/georgiazhou/self_play/scripts")

from dialop.envs.optimization import OptimizationEnv
from dialop.games.optimization import OptimizationGame


class MatchingEnv:
    """SLIME-compatible wrapper for DialOp OptimizationEnv."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.env = OptimizationEnv(max_turns=max_turns)
        self.turn_count = 0
        self.current_player = 0  # 0 = Qwen (agent), 1 = GPT (partner)

    def reset(self, game_state: dict) -> tuple[dict, dict]:
        """Reset environment with a game state.

        Returns:
            obs: dict with "obs_str" for initial observation
            info: dict with metadata
        """
        self.turn_count = 0
        self.current_player = 0

        obs = self.env.reset(game_state=game_state)

        return {
            "obs_str": obs["player-1"],  # Qwen sees player-1 view
            "role": "user",
        }, {
            "best_score": self.env.best_score,
            "game_state": game_state,
        }

    def step(self, response_text: str) -> tuple[dict, bool, dict]:
        """Process a response and return next observation.

        Args:
            response_text: The response from current player

        Returns:
            obs: dict with "obs_str" for next observation
            done: bool indicating if game ended
            info: dict with reward and metadata
        """
        self.turn_count += 1

        # Step the underlying environment
        result = self.env.step(response_text)
        done = result.get("done", False)

        # Extract reward if game ended
        reward = 0.0
        if done and "reward" in result:
            reward = result["reward"]  # Normalized 0-1

        # Alternate players
        self.current_player = 1 - self.current_player

        # Get observation for next player
        next_player = "player-1" if self.current_player == 0 else "player-2"
        obs_str = result.get(next_player, "")

        return {
            "obs_str": obs_str,
            "role": "user",
        }, done, {
            "reward": reward,
            "turn": self.turn_count,
            "current_player": self.current_player,
        }

    def get_partner_view(self) -> str:
        """Get the partner's (GPT's) view of the game state."""
        # Return player-2's observation from the game
        return self.env.game.get_obs(1)  # Player 1 (0-indexed) is GPT

    def is_qwen_turn(self) -> bool:
        """Check if it's Qwen's turn to respond."""
        return self.current_player == 0


def build_env(max_turns: int = 20) -> MatchingEnv:
    """Factory function for SLIME config."""
    return MatchingEnv(max_turns=max_turns)
```

---

## Component 2: Custom Rollout (`rollout.py`)

Follows SLIME's `geo3k_vlm_multi_turn/rollout.py` pattern but alternates between Qwen (SGLang) and GPT (OpenAI API).

```python
"""Custom multi-turn rollout for Qwen↔GPT paper-reviewer matching.

Based on SLIME's examples/geo3k_vlm_multi_turn/rollout.py pattern.
"""
import os
import asyncio
from typing import Any
from openai import AsyncOpenAI

from slime.rollout.sglang_rollout import generate as sglang_generate
from slime.rollout.base_types import Sample

from .env_matching import MatchingEnv
from .gpt_partner import ShyGPTPartner
from .question_logger import count_questions


async def generate(
    args,
    sample: dict,
    sampling_params: dict,
    evaluation: bool = False,
) -> list[Sample]:
    """Multi-turn rollout with Qwen (trainable) and GPT (fixed partner).

    This follows SLIME's multi-turn pattern:
    1. Initialize environment with game state
    2. Loop: Qwen generates → GPT responds → env.step()
    3. Build loss_mask: 1 for Qwen tokens, 0 for GPT/env tokens
    4. Return samples with terminal reward
    """
    # Initialize environment
    env = MatchingEnv(max_turns=args.max_turns)
    game_state = sample.get("game_state", sample.get("ground_truth", {}))
    obs, info = env.reset(game_state)

    # Initialize GPT partner
    gpt_partner = ShyGPTPartner(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=getattr(args, "gpt_model", "gpt-4.1"),
    )

    # Token tracking
    all_input_ids = []
    all_logprobs = []
    all_loss_mask = []

    # Dialogue history for context
    dialogue = [{"role": "system", "content": sample["prompt"]}]
    dialogue.append({"role": "user", "content": obs["obs_str"]})

    # Metrics
    question_count = 0
    turn_count = 0
    done = False
    reward = 0.0

    while not done and turn_count < args.max_turns:
        if env.is_qwen_turn():
            # === QWEN'S TURN (trainable) ===
            # Build prompt from dialogue history
            qwen_prompt = _build_prompt(dialogue, args.tokenizer)

            # Generate with SGLang (collects logprobs)
            response = await sglang_generate(
                args,
                {"prompt": qwen_prompt},
                sampling_params,
                evaluation=evaluation
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

        else:
            # === GPT'S TURN (fixed partner, not trained) ===
            partner_view = env.get_partner_view()
            gpt_response = await gpt_partner.respond(dialogue, partner_view)

            # Tokenize GPT response (for sequence continuity)
            gpt_ids = args.tokenizer.encode(gpt_response, add_special_tokens=False)

            # Track GPT tokens with loss_mask = 0 (DON'T train on these)
            all_input_ids.extend(gpt_ids)
            all_logprobs.extend([float("-inf")] * len(gpt_ids))  # No valid logprobs
            all_loss_mask.extend([0] * len(gpt_ids))

            # Add to dialogue history
            dialogue.append({"role": "user", "content": gpt_response})
            response_text = gpt_response

        # Step environment
        obs, done, info = env.step(response_text)
        turn_count += 1

        if done:
            reward = info.get("reward", 0.0)

    # Build final sample
    final_sample = Sample(
        input_ids=all_input_ids,
        rollout_log_probs=all_logprobs,
        loss_mask=all_loss_mask,
        reward=reward,
        # Custom metrics for logging
        extra_info={
            "question_count": question_count,
            "turn_count": turn_count,
            "game_id": sample.get("game_id", 0),
        }
    )

    return [final_sample]


def _build_prompt(dialogue: list[dict], tokenizer) -> str:
    """Convert dialogue history to prompt string."""
    return tokenizer.apply_chat_template(
        dialogue,
        tokenize=False,
        add_generation_prompt=True
    )
```

---

## Component 3: GPT Partner (`gpt_partner.py`)

```python
"""Shy GPT-4.1 partner that only shares information when directly asked."""
import os
from openai import AsyncOpenAI

SHY_PARTNER_SYSTEM_PROMPT = """You are a reviewer coordinator in a paper-reviewer matching game.
You can see which reviewers are available and their expertise/affinity scores for certain papers.

IMPORTANT BEHAVIORAL RULES - You are SHY:
1. NEVER volunteer information proactively
2. Only share a specific score when DIRECTLY asked about that exact reviewer-paper pair
3. When asked vague questions, give brief non-committal responses like "I have some information about that"
4. Never reveal all your scores at once - make the agent work for each piece of information
5. When the agent proposes an assignment, evaluate it honestly and accept/reject based on the total score

Your goal is to reach a good assignment, but the agent must ask the right questions to get your information.

YOUR VIEW OF THE GAME:
{partner_view}
"""


class ShyGPTPartner:
    """GPT-4.1 partner with shy behavior for paper-reviewer matching."""

    def __init__(self, api_key: str = None, model: str = "gpt-4.1"):
        self.client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    async def respond(self, dialogue: list[dict], partner_view: str) -> str:
        """Generate a shy response given dialogue history and partner's view.

        Args:
            dialogue: List of {"role": ..., "content": ...} messages
            partner_view: String describing what GPT can see (scores, reviewers)

        Returns:
            GPT's response string
        """
        # Build messages with partner's view injected into system prompt
        system_prompt = SHY_PARTNER_SYSTEM_PROMPT.format(partner_view=partner_view)

        messages = [{"role": "system", "content": system_prompt}]

        # Add dialogue history (skip original system prompt)
        for msg in dialogue:
            if msg["role"] != "system":
                # Flip roles: Qwen's "assistant" becomes GPT's "user" input
                role = "user" if msg["role"] == "assistant" else "assistant"
                messages.append({"role": role, "content": msg["content"]})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )

        return response.choices[0].message.content
```

---

## Component 4: Data Generator (`data_generator.py`)

```python
"""Generate game instances in SLIME's expected JSONL format."""
import json
import sys
sys.path.insert(0, "/home/nickatomlin/georgiazhou/self_play/scripts")

from dialop.games.optimization import OptimizationGame

AGENT_SYSTEM_PROMPT = """You are playing a paper-reviewer matching game. Your goal is to assign 8 reviewers to 8 papers to maximize the total affinity score.

You can see SOME of the affinity scores between reviewers and papers. Your partner can see OTHER scores that you cannot see.

To win, you need to:
1. ASK your partner about scores you don't know
2. Share information when asked
3. Propose assignments using [propose] format
4. Accept/reject partner's proposals

Format for proposals:
[propose]
- Reviewer1: Paper1
- Reviewer2: Paper2
...
[/propose]

YOUR VISIBLE SCORES:
{agent_view}
"""


def generate_game(seed: int) -> dict:
    """Generate a single game instance.

    Returns:
        dict with: prompt, game_state, optimal_score, game_id
    """
    game = OptimizationGame({}, one_player=False)
    game.reset(seed=seed)

    # Get agent's (player 0) view
    agent_view = game.get_obs(0)

    # Build the prompt with agent's view
    prompt = AGENT_SYSTEM_PROMPT.format(agent_view=agent_view)

    # Serialize game state for environment reset
    game_state = {
        "table_values": game.table.values.tolist(),
        "player0_mask": game.table.player0_mask.tolist() if hasattr(game.table, 'player0_mask') else None,
        "player1_mask": game.table.player1_mask.tolist() if hasattr(game.table, 'player1_mask') else None,
        "best_assignment_reward": float(game.best_assignment_reward),
        "action_log": [],
    }

    return {
        "prompt": prompt,
        "game_state": game_state,
        "ground_truth": game_state,  # SLIME sometimes uses this key
        "optimal_score": float(game.best_assignment_reward),
        "game_id": seed,
    }


def generate_dataset(num_games: int, output_path: str, base_seed: int = 42):
    """Generate a dataset of game instances."""
    with open(output_path, "w") as f:
        for i in range(num_games):
            game = generate_game(base_seed + i)
            f.write(json.dumps(game) + "\n")
    print(f"Generated {num_games} games to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/games.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.num_games, args.output, args.seed)
```

---

## Component 5: Question Logger (`question_logger.py`)

```python
"""Track questions per dialogue to validate hypothesis."""
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import json

QUESTION_PATTERNS = [
    r'\?',  # Contains question mark
    r'^(what|who|where|when|why|how|which|can|could|would|is|are|do|does|tell me)',
    r'score.*(for|of|between)',
    r'(affinity|rating|preference).*(for|of|between)',
]


def count_questions(text: str) -> int:
    """Count question-like patterns in text."""
    text_lower = text.lower().strip()
    count = 0
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, text_lower):
            count += 1
            break  # Count as 1 even if multiple patterns match
    return count


@dataclass
class EpisodeMetrics:
    game_id: int
    question_count: int
    turn_count: int
    reward: float

    @property
    def questions_per_turn(self) -> float:
        return self.question_count / max(1, self.turn_count)


class QuestionLogger:
    """Log and analyze question-asking patterns across training."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = []

    def log_episode(self, metrics: EpisodeMetrics):
        self.episodes.append(asdict(metrics))

    def save(self, step: int):
        path = self.log_dir / f"questions_step{step}.json"
        with open(path, "w") as f:
            json.dump(self.episodes, f)

    def get_summary(self, window: int = 100) -> dict:
        recent = self.episodes[-window:] if len(self.episodes) >= window else self.episodes
        if not recent:
            return {}
        return {
            "avg_questions": sum(e["question_count"] for e in recent) / len(recent),
            "avg_reward": sum(e["reward"] for e in recent) / len(recent),
            "avg_turns": sum(e["turn_count"] for e in recent) / len(recent),
        }
```

---

## Component 6: SLIME Config (`configs/matching_config.yaml`)

```yaml
# Based on examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml

# Multi-turn settings
max_turns: 20
rollout_interaction_env_path: slim_rl.src.env_matching

# Custom rollout function
custom_generate_function_path: slim_rl.src.rollout:generate

# GPT partner settings (accessed via args in rollout)
gpt_model: gpt-4.1
```

---

## Component 7: Run Script (`scripts/run_matching.sh`)

```bash
#!/bin/bash
# Based on SLIME's example run scripts

# Set paths
export PYTHONPATH="${PYTHONPATH}:/home/nickatomlin/georgiazhou/self_play/slim_rl"
export OPENAI_API_KEY="${OPENAI_API_KEY}"

# Model settings
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR="./checkpoints"

# Training settings
python -m slime.train \
    --model-path $MODEL_PATH \
    --train-data ./data/games.jsonl \
    --checkpoint-dir $CHECKPOINT_DIR \
    --custom-generate-function-path slim_rl.src.rollout:generate \
    --rollout-interaction-env-path slim_rl.src.env_matching \
    --rollout-max-context-len 4096 \
    --max-turns 20 \
    --algorithm grpo \
    --group-size 8 \
    --global-batch-size 64 \
    --rollout-batch-size 8 \
    --learning-rate 1e-5 \
    --num-train-epochs 3 \
    --tensor-model-parallel-size 1 \
    --sglang-mem-fraction-static 0.8
```

---

## Implementation Timeline

| Day | Task | Details |
|-----|------|---------|
| 1 | Setup + Data Generator | Create file structure, implement data_generator.py, generate 1000 games |
| 2 | Environment Adapter | Implement env_matching.py, test with existing OptimizationEnv |
| 3 | GPT Partner | Implement gpt_partner.py, test API calls with shy behavior |
| 4 | Custom Rollout | Implement rollout.py following SLIME's geo3k pattern |
| 5 | Integration + Config | Create config, run script, test end-to-end with small batch |
| 6 | Full Training | Run training, monitor question metrics |

---

## Key Reuse from self_play/

| What | Where | How |
|------|-------|-----|
| Game logic | `scripts/dialop/games/optimization.py` | Import `OptimizationGame` directly |
| Environment | `scripts/dialop/envs/optimization.py` | Wrap `OptimizationEnv` in our adapter |
| Reward computation | Built into OptimizationEnv | `reward = proposal_score / best_score` |
| Proposal parsing | `OptimizationEnv._parse_proposal()` | Already handles `[propose]...[/propose]` format |

---

## Why This Works

1. **SLIME's multi-turn pattern is exactly what we need**: The `geo3k_vlm_multi_turn` example shows how to:
   - Run a turn loop
   - Build loss_mask (1 for model tokens, 0 for env tokens)
   - Track logprobs only for model generations
   - Apply terminal reward

2. **We just swap components**:
   - `env_geo3k.py` → `env_matching.py` (our game environment)
   - Math tool calls → GPT-4.1 partner responses
   - Math scoring → Paper-reviewer assignment scoring

3. **Existing DialOp code handles the hard parts**:
   - Game generation with proper difficulty
   - Partial information for each player
   - Proposal parsing and validation
   - Optimal assignment computation (Hungarian algorithm)
   - Reward normalization

---

## Next Steps

1. Confirm SLIME is installed and working
2. Verify OpenAI API access for GPT-4.1
3. Run `generate_data.py` to create training dataset
4. Test rollout with a single game before full training
