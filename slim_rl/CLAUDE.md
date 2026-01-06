# SLIM_RL Project Notes

## Project Goal

Train a **Qwen model** to play the paper-reviewer matching game from DialOp using **SLIME RL framework** with **GRPO** (Group Relative Policy Optimization). The setup:

- **Qwen** = trainable agent (sees papers, trained via GRPO)
- **GPT-4.1** = fixed "shy" partner (sees reviewers, only shares info when asked)
- **Reward** = assignment quality from DialOp environment (normalized 0-1)

**Hypothesis to validate**: Qwen will learn to ask more questions as training progresses.

---

## Current State: Implementation Complete, Integration Pending

All components have been implemented and tested. What remains is:
1. Installing SLIME and verifying the integration
2. Adjusting `rollout.py` to match SLIME's exact API
3. Running actual training

---

## File Structure

```
slim_rl/
├── claude.md                        # THIS FILE - context for continuation
├── PLAN.md                          # Detailed implementation plan
├── configs/
│   └── matching_config.yaml         # SLIME configuration
├── data/
│   └── test_games.jsonl             # Test data (5 games generated)
├── scripts/
│   ├── run_matching.sh              # Training launch script (executable)
│   ├── generate_data.py             # CLI for data generation
│   ├── analyze_questions.py         # Post-training question analysis
│   └── test_components.py           # Component tests (ALL PASS)
└── src/
    ├── __init__.py                  # Package exports
    ├── data_generator.py            # Game instance JSONL generator
    ├── env_matching.py              # SLIME-compatible environment adapter
    ├── gpt_partner.py               # GPT-4.1 shy partner + MockShyPartner
    ├── question_logger.py           # Question tracking metrics
    └── rollout.py                   # Custom multi-turn rollout for SLIME
```

---

## Key Components Explained

### 1. Data Generator (`src/data_generator.py`)

Generates game instances in SLIME's expected JSONL format.

**Usage:**
```bash
/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python scripts/generate_data.py --num_games 1000 --output data/games.jsonl
```

**Output format:**
```json
{
  "prompt": "You are playing a paper-reviewer matching game...<agent's visible scores>",
  "game_state": {"table": [[...]], "mask1": [[...]], "mask2": [[...]], "scale1": 3.2, "scale2": 7.1, "best_assignment_reward": 637},
  "ground_truth": <same as game_state>,
  "optimal_score": 637.0,
  "game_id": 42
}
```

**Reuses**: `dialop.games.optimization.OptimizationGame` from `self_play/scripts/dialop/`

### 2. Environment Adapter (`src/env_matching.py`)

Wraps DialOp's `OptimizationEnv` to match SLIME's expected interface.

**SLIME interface pattern** (from `examples/geo3k_vlm_multi_turn/env_geo3k.py`):
```python
env = MatchingEnv(max_turns=20)
obs, info = env.reset(game_state)  # Returns (obs_dict, info_dict)
obs, done, info = env.step(response)  # Returns (obs_dict, bool, info_dict)
```

**Important**: The underlying `OptimizationEnv.step()` returns a tuple `(result_dict, is_error)`.

**Message format requirement**: All messages must include tags: `[message]`, `[propose]`, `[accept]`, or `[reject]`.

### 3. GPT Partner (`src/gpt_partner.py`)

Two implementations:
- `ShyGPTPartner` - Real GPT-4.1 API calls (requires `OPENAI_API_KEY`)
- `MockShyPartner` - For testing without API calls

**Shy behavior rules** (in system prompt):
1. Never volunteer information proactively
2. Only share specific score when directly asked about that reviewer-paper pair
3. Give vague responses to general questions
4. Evaluate proposals honestly

### 4. Custom Rollout (`src/rollout.py`)

**This is the core SLIME integration point.**

The `generate()` function implements multi-turn dialogue:
```python
async def generate(args, sample, sampling_params, evaluation=False) -> list[RolloutSample]:
    # 1. Initialize environment with game_state
    # 2. Loop until done or max_turns:
    #    - If Qwen's turn: generate via SGLang, loss_mask=1
    #    - If GPT's turn: call OpenAI API, loss_mask=0
    #    - env.step(response)
    # 3. Return RolloutSample with terminal reward
```

**Key design decisions:**
- `loss_mask = 1` for Qwen tokens (train on these)
- `loss_mask = 0` for GPT tokens (don't train on these)
- Logprobs only collected for Qwen generations
- Terminal reward from `env.get_normalized_reward()`

**TODO**: The `_generate_with_sglang()` function has a mock implementation. Need to adjust to match SLIME's actual SGLang API once SLIME is installed.

### 5. Question Logger (`src/question_logger.py`)

Tracks questions per dialogue to validate the hypothesis.

**Question patterns detected:**
- `?` (question mark)
- Question words (what, who, how, can, could, etc.)
- Score-seeking patterns ("score for", "affinity between")

**Metrics tracked:**
- `question_count` per episode
- `reward` per episode
- `turn_count` per episode
- Correlation between questions and reward

---

## How to Test

Run all component tests:
```bash
cd /home/nickatomlin/georgiazhou/self_play/slim_rl
/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python scripts/test_components.py
```

Expected output: `ALL TESTS PASSED!`

---

## Integration with SLIME

### SLIME's Multi-Turn Pattern

Based on `examples/geo3k_vlm_multi_turn/`:

1. **Config** specifies:
   ```yaml
   max_turns: 20
   rollout_interaction_env_path: slim_rl.src.env_matching
   custom_generate_function_path: slim_rl.src.rollout:generate
   ```

2. **Rollout function signature**:
   ```python
   async def generate(args, sample, sampling_params, evaluation=False) -> list[Sample]
   ```

3. **Sample structure** (from `slime/rollout/base_types.py`):
   ```python
   @dataclass
   class Sample:
       input_ids: list[int]
       rollout_log_probs: list[float]
       loss_mask: list[int]
       reward: float
   ```

### What Needs Adjustment

1. **`_generate_with_sglang()` in `rollout.py`**: Currently has mock implementation. Need to:
   - Import actual SLIME generation function
   - Match SLIME's SGLang API for token generation with logprob collection
   - Check `slime/rollout/sglang_rollout.py` for exact interface

2. **Return type**: Verify `RolloutSample` matches SLIME's expected `Sample` dataclass

3. **Config flags**: May need additional SLIME-specific args in `matching_config.yaml`

---

## Key Reusable Code from self_play/

| What | Location | Notes |
|------|----------|-------|
| Game logic | `scripts/dialop/games/optimization.py` | `OptimizationGame` class |
| Environment | `scripts/dialop/envs/optimization.py` | `OptimizationEnv` class |
| Player implementations | `scripts/dialop/sglang_model_player.py` | Reference for logprob collection |
| Reward functions | `scripts/verl_hooks/matching_reward_fn.py` | `compute_score_matching()` |
| Existing GRPO loop | `scripts/offline_grpo_loop.py` | Reference implementation with verl |

---

## Running Training (After SLIME Setup)

1. **Generate training data:**
   ```bash
   cd /home/nickatomlin/georgiazhou/self_play/slim_rl
   /home/nickatomlin/georgiazhou/self_play/test_venv/bin/python scripts/generate_data.py --num_games 1000
   ```

2. **Set environment:**
   ```bash
   export OPENAI_API_KEY=your-key-here
   export PYTHONPATH="${PYTHONPATH}:/home/nickatomlin/georgiazhou/self_play/slim_rl:/home/nickatomlin/georgiazhou/self_play/scripts"
   ```

3. **Run training:**
   ```bash
   ./scripts/run_matching.sh
   ```

4. **Analyze results:**
   ```bash
   /home/nickatomlin/georgiazhou/self_play/test_venv/bin/python scripts/analyze_questions.py --log_dir logs/ --output_dir plots/
   ```

---

## Known Issues / TODOs

1. **SLIME SGLang integration**: `rollout.py:_generate_with_sglang()` needs real implementation
2. **Proposal parsing errors**: In test, proposals sometimes fail because exact paper names are required (e.g., "BLEU: a Method for Automatic Evaluation of MT" not just "BLEU")
3. **Mock partner**: Currently returns random accept/reject. Real GPT-4.1 will be smarter.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLIME Training Loop                          │
│                                                                 │
│  games.jsonl ──► Custom Rollout (rollout.py:generate)          │
│                         │                                       │
│                         ▼                                       │
│                  ┌─────────────────────────────────────┐       │
│                  │ Turn Loop (max_turns=20):           │       │
│                  │                                     │       │
│                  │ if qwen_turn:                       │       │
│                  │   response = SGLang.generate()      │       │
│                  │   loss_mask[tokens] = 1  ◄─ TRAIN   │       │
│                  │   collect logprobs                  │       │
│                  │                                     │       │
│                  │ else (gpt_turn):                    │       │
│                  │   response = OpenAI.chat()          │       │
│                  │   loss_mask[tokens] = 0  ◄─ NO TRAIN│       │
│                  │                                     │       │
│                  │ obs, done, info = env.step()        │       │
│                  └─────────────────────────────────────┘       │
│                         │                                       │
│                         ▼                                       │
│                  Return RolloutSample with:                     │
│                  - input_ids (full dialogue)                    │
│                  - loss_mask (1 for Qwen, 0 for GPT)           │
│                  - logprobs (Qwen only)                        │
│                  - reward (terminal, from DialOp)              │
│                         │                                       │
│                         ▼                                       │
│                  GRPO Training (Megatron)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Python Environment

Use this Python for all commands:
```
/home/nickatomlin/georgiazhou/self_play/test_venv/bin/python
```

This venv has torch, transformers, and other dependencies installed.

---

## Contact / Resources

- **SLIME repo**: https://github.com/THUDM/slime
- **DialOp repo**: https://github.com/jlin816/dialop
- **SLIME multi-turn example**: `examples/geo3k_vlm_multi_turn/`
- **SLIME multi-agent example**: `examples/multi_agent/`
