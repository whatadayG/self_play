# Dialop Self-Play Implementation Summary

## Overview
I've implemented the core components for GRPO self-play training where two instances of the same LLM play the dialop optimization game cooperatively.

## Components Implemented

### 1. **DialopSelfplayInteraction** (`verl/verl/interactions/dialop_selfplay_interaction.py`)
- Manages dialop environment as a multi-turn interaction
- Converts between dialop's message format (`[message]`, `[propose]`, `[accept]`/`[reject]`) and standard chat format
- Tracks game state and determines when to terminate
- Returns normalized rewards (proposal_reward / best_assignment_reward) exactly as defined in dialop

### 2. **DialopSelfPlayRewardManager** (`verl/verl/workers/reward_manager/dialop_selfplay.py`)
- Assigns shared rewards to both agents (cooperative game)
- Places reward on the last token of the conversation
- Preserves dialop's reward structure without modification

### 3. **Data Generation Options**

#### Option A: Pre-generated Data Files (implemented)
- Script: `verl/examples/data_preprocess/dialop_optimization.py`
- Generates train/test parquet files with game instances
- Usage: 
  ```bash
  python verl/examples/data_preprocess/dialop_optimization.py \
    --output_dir ~/data/dialop_optimization_data \
    --num_train 1000 --num_test 100
  ```

#### Option B: Online Generation (alternative)
- Class: `verl/verl/utils/dataset/dialop_online_dataset.py`
- Generates games on-the-fly during training
- No preprocessing needed, but harder to ensure reproducible evaluation

### 4. **Configuration Files**
- Main config: `verl/examples/sglang_multiturn/config/dialop_optimization_selfplay_grpo.yaml`
- Interaction config: `verl/examples/sglang_multiturn/config/interaction_config/dialop_selfplay_interaction_config.yaml`

### 5. **Training Script**
- `verl/examples/sglang_multiturn/dialop_optimization/run_qwen2.5-0.5b_dialop_optimization_selfplay.sh`
- Configured for 8xH100 or similar setup
- Uses SGLang for multi-turn rollouts

## Example Dialop Game Output

```
Initial game state:
- Current player: player-1
- Best possible reward: 670

Player-1 sees:
Reviewer Paper Similarity Scores:
[Table with reviewer-paper matching scores]

Player-2 sees:
[Different view of the same table with partial information]

Example conversation:
Player 1: [message] Hello partner, I see we have a paper-reviewer matching task.
Player 2: [message] Yes, let me share what I see in my table.
```

## Key Design Decisions

1. **Reward Structure**: Kept exactly as in dialop - normalized reward when proposal accepted
2. **Both Agents Same Model**: True self-play with single policy
3. **Data Generation**: Provided both offline (parquet files) and online options
4. **Turn Management**: Handled by dialop environment's existing turn-taking logic

## Next Steps

1. **For Pre-generated Data Approach**:
   ```bash
   # Generate data
   python verl/examples/data_preprocess/dialop_optimization.py \
     --output_dir ~/data/dialop_optimization_data
   
   # Run training
   bash verl/examples/sglang_multiturn/dialop_optimization/run_qwen2.5-0.5b_dialop_optimization_selfplay.sh
   ```

2. **For Online Generation**: Modify the training script to use DialopOnlineDataset instead of loading parquet files

## Questions for Consideration

1. **Data Generation**: Should we use pre-generated files (reproducible, faster) or online generation (no preprocessing, infinite variety)?

2. **Evaluation Strategy**: How should we evaluate progress? Options:
   - Average normalized reward over evaluation games
   - Conversation length (shorter = more efficient)
   - Proposal acceptance rate

3. **Extensions**: The framework can easily extend to other dialop environments (mediation, planning) by changing the environment class.