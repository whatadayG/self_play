# CLAUDE.md - Dialop Self-Play GRPO Implementation

The current plan can be found at PLAN.md:

@PLAN.md

## Project Overview

This project implements GRPO (Group Relative Policy Optimization) training for LLMs using self-play on dialop conversation games. Two instances of the same model engage in cooperative dialogue games, and both receive gradients based on their joint performance.

## Repository Structure and Key Insights

### dialop/ Repository

**Purpose**: Implements cooperative dialogue games where two agents work together to achieve a goal.

**Key Components**:
- `dialop/envs/optimization.py` - The main environment we're using
- `dialop/games/optimization.py` - Game logic for paper-reviewer matching
- `dialop/scripts/templates.py` - Jinja2 templates for formatting game state

**How OptimizationEnv Works**:
1. Two players see partial, overlapping information about a matching problem
2. They communicate via structured messages: `[message]`, `[propose]`, `[accept]`/`[reject]`
3. Game ends when a proposal is accepted
4. Reward = actual score / best possible score (normalized)

**Critical Discovery**: The template imports were incorrect (`dialop.templates` → `dialop.scripts.templates`)

### verl/ Repository

**Purpose**: Implements various RL algorithms for LLM training, including GRPO.

**Key Components**:
- `verl/utils/dataset/rl_dataset.py` - RLHFDataset that loads training data
- `verl/trainer/ppo/ray_trainer.py` - Main training loop
- `verl/workers/rollout/` - Handles model inference during training
- `verl/interactions/` - System for simulating user interactions

**GRPO Training Loop** (from our analysis):
```python
for epoch in range(total_epochs):
    for batch in train_dataloader:  # Loads from RLHFDataset
        # 1. Generate rollouts (inference)
        gen_output = actor_rollout_wg.generate_sequences(batch)
        # 2. Compute rewards
        rewards = reward_fn(batch)
        # 3. Compute advantages
        advantages = compute_advantage(batch)
        # 4. Update model
        actor_rollout_wg.apply_grads(batch)
```

## Key Architectural Insights Discovered

### 1. Interaction System Mismatch

**Initial Understanding**: We thought the interaction system could handle self-play directly.

**Reality Discovered**: verl's interaction system assumes:
- Model generates "assistant" messages
- Interaction generates "user" messages
- This simulates a user, not another agent

**Implication**: True self-play (both agents are the model) doesn't fit this paradigm.

### 2. Data Flow Understanding

**Initial Assumption**: We'd need to modify verl's rollout worker for self-play.

**Better Approach Discovered**: Generate complete self-play games offline, then load them as if they were pre-collected data. This bypasses the complex rollout system entirely.

**Key Insight**: `RLHFDataset` expects:
```python
{
    "data_source": str,
    "messages": [  # Chat format messages
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "reward_model": {
        "normalized_reward": float,  # Must be normalized!
        "ground_truth": None,  # Not used for self-play
    }
}
```

### 3. Reward Structure

**Critical Point**: Dialop computes `reward / best_possible_reward` as the normalized score. This must be preserved exactly.

### 4. Training Data Perspective

**Important**: Each game generates TWO training examples:
- One from player 1's perspective
- One from player 2's perspective

Both receive the same reward (cooperative game).

## Implementation Approach

After extensive exploration, we settled on a **standalone self-play data generator** approach:

### Why This Approach?

1. **Simplicity**: No need to modify verl's complex rollout infrastructure
2. **Clarity**: Self-play logic is isolated and testable
3. **Compatibility**: Generates data in exact format verl expects

### What We Built (But Didn't Fully Integrate)

1. **DialopSelfplayInteraction** - Manages dialop environment within verl's framework
2. **DialopSelfPlayRewardManager** - Assigns shared rewards to both agents
3. **Data preprocessing script** - Converts dialop games to verl format
4. **Monkey patch system** - Would modify SGLangRollout for true self-play

### What We Recommend Instead

Build a standalone script that:
1. Loads the current model checkpoint
2. Runs complete self-play games using SGLang for efficiency
3. Saves games in parquet format that `RLHFDataset` can load
4. Skips verl's rollout generation entirely during training

## Technical Details

### Message Format Translation

Dialop uses: `[message] content`, `[propose] content`, `[accept]`
Verl expects: Standard OpenAI chat format

### Masking Requirements

- **Attention mask**: All tokens attended to
- **Label mask**: Only compute loss on assistant responses
- **Reward mask**: Reward only on last assistant token

### Utilities to Reuse from verl

- `get_response_mask()` - Identifies response tokens
- `compute_position_id_with_mask()` - Position embeddings
- `postprocess_data()` - Tokenization utilities
- SGLang integration for efficient generation

## Unresolved Challenges

1. **True Self-Play in Rollout**: Modifying verl's rollout worker to have both agents use the same policy is complex due to the user-simulation assumption.

2. **Online vs Offline Generation**: We provided both options but recommend offline for simplicity.

3. **Evaluation Strategy**: How to measure progress beyond just reward metrics.

## Summary

The project revealed that integrating true self-play into verl's existing infrastructure is challenging due to architectural assumptions about user simulation. The recommended approach generates self-play data offline and feeds it to verl's standard training pipeline, avoiding these complications while achieving the desired training objective.
