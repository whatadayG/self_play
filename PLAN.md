# Updated Self-Play Data Generation Plan

Based on your clarifying questions, here's the refined approach:

## Architecture Overview

The proposal creates a **standalone self-play data generator** that produces complete games offline. This data is then loaded by verl's standard `RLHFDataset` during training, effectively replacing the rollout generation phase.

### How it fits in GRPO:
- **Standard GRPO**: Load prompts → Generate rollouts → Compute rewards → Update model
- **Our approach**: Load complete games (with rewards) → Skip generation → Update model

## Implementation Plan

### 1. **Core Self-Play Generator** (`generate_selfplay_games.py`)
Will reuse verl utilities where possible:
- Use SGLang for efficient batch generation
- Reuse `verl.utils.torch_functional` for masking utilities
- Leverage `compute_position_id_with_mask` from verl
- Use verl's tokenization patterns from `RLHFDataset`

Key features:
- Load model checkpoint into SGLang server
- Generate complete dialop games
- Output format compatible with `RLHFDataset`

## Note: Replacement
This design was superseded by that in CUSTOM_ROLLOUT_IMPLEMENTATION.md, which is more integrated:
@CUSTOM_ROLLOUT_IMPLEMENTATION.md

### 2. **Data Processing Pipeline**
Will create proper token sequences with:
- **Input IDs**: Full conversation from one player's perspective
- **Attention mask**: Standard attention for all tokens
- **Label mask**: Only compute loss on assistant responses (using `get_response_mask`)
- **Reward mask**: Place reward only on last assistant token
- **Position IDs**: Using verl's `compute_position_id_with_mask`

### 3. **Integration Points**
The generated data will have the same schema as verl expects:
```python
{
    "data_source": str,
    "messages": list[dict],  # For RLHFDataset to tokenize
    "reward_model": {
        "normalized_reward": float,  # Key point: normalized!
        "game_state": dict,
    }
}
```

### 4. **Testing Strategy**
Will create comprehensive tests with logging:
- Unit tests for game generation
- Tokenization consistency tests  
- Mask correctness verification
- End-to-end data loading test
- All results logged to `.txt` file for review

## Advantages of This Approach

1. **Simplicity**: No fighting with verl's user-simulation assumptions
2. **Testability**: Can verify correctness before training
3. **Efficiency**: Can use SGLang's batching directly
4. **Compatibility**: Produces exact format verl expects

## Key Implementation Notes

1. **Normalized rewards**: Will ensure rewards are normalized by best_possible_reward
2. **Self-play attribution**: Both players' perspectives saved as separate training examples
3. **Efficient generation**: Will use SGLang with appropriate batch sizes for 2-4 GPUs
4. **Proper masking**: Only train on assistant tokens, reward on last token

This approach avoids complex modifications to verl's rollout system while leveraging its training infrastructure effectively.
