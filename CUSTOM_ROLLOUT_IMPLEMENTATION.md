# Custom Rollout Implementation for Dialop Self-Play

## Overview

This document describes the new custom rollout worker implementation for dialop self-play GRPO training. This approach replaces the previous monkey-patching solution with a cleaner, more maintainable architecture.

## Key Changes from Previous Approach

### 1. **Custom Rollout Worker** (`verl/verl/workers/rollout/dialop_selfplay_rollout.py`)
- Inherits directly from `SGLangRollout`
- Implements `generate_sequences` to run complete self-play games
- Uses current model for BOTH players (true self-play)
- Returns full conversations with proper rewards

### 2. **Minimal Data Generation** (`verl/examples/data_preprocess/dialop_game_init.py`)
- Generates only game initializations, not full conversations
- Each entry contains:
  - `game_state`: Serialized game state (JSON string for parquet compatibility)
  - `player_index`: Which player's perspective (0 or 1)
  - `game_id`: Unique identifier for the game
  - `prompt`: Initial observation for the player
- Much more efficient than pre-generating full games

### 3. **New Configuration** (`verl/examples/sglang_multiturn/config/dialop_selfplay_custom_grpo.yaml`)
- Uses `_target_` to specify custom rollout worker
- No need for `VERL_APPLY_SELFPLAY_PATCH` environment variable
- Cleaner configuration structure

## Architecture Flow

```
1. Data Generation Phase:
   dialop_game_init.py → Minimal game initializations → train/test.parquet

2. Training Phase (per GRPO iteration):
   DataLoader → Game init → DialopSelfPlayRollout → Complete game → Training data
                                    ↓
                             Current policy generates
                              moves for both players
```

## Usage Instructions

### 1. Generate Training Data

```bash
cd /home/nickatomlin/georgiazhou/self_play
source test_venv/bin/activate  # or your venv

python verl/examples/data_preprocess/dialop_game_init.py \
  --output_dir ~/data/dialop_selfplay_init \
  --num_train 1000 \
  --num_test 100
```

### 2. Run Training (Requires GPUs)

```bash
bash verl/examples/sglang_multiturn/dialop_optimization/run_dialop_selfplay_custom.sh
```

Or manually:

```bash
python -m verl.trainer.main_ppo \
  --config-path="verl/examples/sglang_multiturn/config" \
  --config-name="dialop_selfplay_custom_grpo" \
  data.train_files='["~/data/dialop_selfplay_init/train.parquet"]' \
  data.val_files='["~/data/dialop_selfplay_init/test.parquet"]'
```

## Key Implementation Details

### Game State Serialization
- Game states are JSON-serialized for parquet storage
- Automatically deserialized in the rollout worker
- Preserves all game information (tables, assignments, etc.)

### Reward Assignment
- Rewards placed on last token of conversation
- GRPO automatically spreads reward across all response tokens
- Both players receive same normalized reward (cooperative game)

### Memory Efficiency
- Only game initializations stored on disk
- Full conversations generated on-demand during training
- Supports batched game generation via SGLang

## Testing

All components have been tested:

1. **Data Generation**: ✓ Creates valid parquet files with game initializations
2. **Rollout Import**: ✓ Custom rollout worker imports correctly
3. **Unit Tests**: ✓ All unit tests pass
4. **Config Validation**: ✓ Configuration properly structured
5. **Dialop Integration**: ✓ Environment functions correctly

Run tests with:
```bash
source test_venv/bin/activate
python test_dialop_selfplay_integration.py
```

## Advantages Over Monkey Patching

1. **Cleaner Architecture**: No runtime code modification
2. **Better Maintainability**: All logic in one dedicated class
3. **Easier Testing**: Can test rollout worker in isolation
4. **More Flexible**: Easy to extend or modify behavior
5. **Proper Integration**: Works naturally with verl's systems

## Next Steps

1. Run full-scale training with GPUs
2. Monitor convergence and reward progression
3. Evaluate learned policies on held-out games
4. Consider extensions:
   - Different dialop environments (mediation, planning)
   - Alternative reward structures
   - Multi-agent scenarios (>2 players)