# Dialop Self-Play GRPO Implementation Summary

## Updated Architecture (Custom Rollout Worker)

**Note**: This implementation has been updated to use a custom rollout worker instead of monkey patching. See CUSTOM_ROLLOUT_IMPLEMENTATION.md for details.

## Completed Components

### 1. **Custom Rollout Worker** (`verl/verl/workers/rollout/dialop_selfplay_rollout.py`)
- NEW: Inherits from SGLangRollout
- Generates complete self-play games on-demand
- Both agents use the current policy being trained
- Returns full conversations with normalized rewards

### 2. **Core Self-Play Interaction** (`verl/verl/interactions/dialop_selfplay_interaction.py`)
- Wraps dialop's OptimizationEnv
- Manages game state and turn-taking
- Converts between dialop format (`[message]`, `[propose]`, `[accept]`) and chat format
- Returns normalized rewards (exactly as in dialop)

### 3. **Reward Manager** (`verl/verl/workers/reward_manager/dialop_selfplay.py`)
- Assigns shared rewards to both agents (cooperative game)
- Places reward on the last token of the conversation
- Compatible with verl's DataProto format

### 4. **Data Generation** (`verl/examples/data_preprocess/dialop_game_init.py`)
- NEW: Generates minimal game initializations (not full conversations)
- Creates parquet files with game states and player perspectives
- Much more efficient than pre-generating full games

### 5. **Configuration & Launch**
- NEW Config: `verl/examples/sglang_multiturn/config/dialop_selfplay_custom_grpo.yaml`
- NEW Launch script: `.../dialop_optimization/run_dialop_selfplay_custom.sh`
- No longer requires `VERL_APPLY_SELFPLAY_PATCH` environment variable

## How Self-Play Works

1. **Game Initialization**: Each rollout starts a new dialop game
2. **Turn Taking**: 
   - Interaction provides observation for current player
   - Model generates response (e.g., `[message] Let's discuss...`)
   - Environment processes action and switches turns
3. **Termination**: Game ends when proposal accepted or max turns reached
4. **Reward**: Both agents receive the same normalized reward

## Key Design Decisions

1. **True Self-Play**: Both agents are the current policy (not user simulation)
2. **Preserved Rewards**: Using dialop's exact reward structure
3. **Monkey Patching**: Minimal changes to existing verl code
4. **Flexible Data**: Support for both pre-generated and online data

## Usage

```bash
# 1. Generate training data
cd /home/nickatomlin/georgiazhou/self_play
python verl/examples/data_preprocess/dialop_optimization.py \
  --output_dir ~/data/dialop_optimization_data \
  --num_train 1000 --num_test 100

# 2. Run training (requires GPUs)
bash verl/examples/sglang_multiturn/dialop_optimization/run_qwen2.5-0.5b_dialop_optimization_selfplay.sh
```

## Test Results

✓ Dialop environment working correctly
✓ Components properly structured
✓ Configuration files in place
✓ Monkey patch mechanism ready

## Next Steps

The implementation is ready for GPU training. The system will:
1. Load pre-generated games or generate online
2. Run self-play rollouts with both agents using the current policy
3. Compute shared rewards based on final outcome
4. Update the model using GRPO

## Technical Notes

- The `VERL_APPLY_SELFPLAY_PATCH=1` environment variable activates self-play mode
- The system gracefully falls back to standard interaction if not using dialop
- All rewards are normalized by best_possible_reward as in original dialop