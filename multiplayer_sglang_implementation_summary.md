# Multi-Player SGLang Implementation Summary

## Overview

This document summarizes the implementation of multi-player support in SGLang for self-play training with information asymmetry, based on the analysis of evaluate_opt.py's player management approach.

## Key Components Created

### 1. MatchingGameMultiplayerInteraction (`verl/verl/interactions/matching_game_multiplayer_interaction.py`)

A specialized interaction class that:
- Maintains separate conversation histories for each player
- Shows different information (masks) to each player based on their perspective
- Alternates between player perspectives on each turn
- Implements turn control logic similar to evaluate_opt.py

Key methods:
- `start_interaction()`: Initializes game state with player tracking
- `generate_response()`: Processes turns and switches perspectives
- `get_player_context()`: Returns player-specific conversation history

### 2. MultiplayerSGLangExtension (`verl/verl/workers/rollout/sglang_rollout/multiplayer_extension.py`)

Extension module providing:
- `MultiplayerRequestWrapper`: Manages multi-player state around AsyncRolloutRequest
- `MultiplayerSGLangExtension`: Static methods for multi-player game flow

Key features:
- Tracks current player (0 or 1) and switches after each turn
- Maintains separate message histories per player
- Creates multiple training instances (one per player) with same batch_id for GRPO

### 3. Test Implementation (`test_multiplayer_standalone.py`)

Comprehensive test demonstrating:
- Information asymmetry preservation (different masks per player)
- Turn-based conversation flow
- Training instance generation with proper GRPO grouping
- Verification of key properties

## How It Works

### 1. Player Management
- Uses `current_player` variable (0 or 1) to track active player
- Switches via: `current_player = (current_player + 1) % num_players`
- Each player maintains independent conversation history

### 2. Information Asymmetry
- Player 1 sees `mask1` and `scale1`
- Player 2 sees `mask2` and `scale2`
- Different table views shown based on masks (hidden values shown as "??")

### 3. Message Handling
When Player A speaks:
- Added to Player A's history as "assistant" message
- Added to Player B's history as "user" message: "The other player said: [message]"

### 4. GRPO Compatibility
- Both player perspectives get same `batch_data_id`
- GRPO groups them together and computes advantages correctly
- Both players receive the same final reward (collaborative game)

### 5. Training Data Generation
At game end:
- Create 2 training instances (one per player perspective)
- Each shows the full conversation from that player's viewpoint
- Same reward and batch_id for proper GRPO grouping

## Integration with SGLang

To integrate with SGLang's rollout system, modify `_async_rollout_a_request` in `sglang_rollout.py`:

```python
if self.config.multi_turn.multiplayer_mode:
    # Initialize multiplayer wrapper
    mp_wrapper = MultiplayerSGLangExtension.initialize_multiplayer_state(_req)
    
    # During generation loop:
    while _req.state == AsyncRolloutRequestStateEnum.RUNNING:
        # Prepare prompt for current player
        MultiplayerSGLangExtension.prepare_player_prompt(
            mp_wrapper, interaction, self.processing_class
        )
        
        # Update base request with current player's view
        mp_wrapper.update_base_request_messages()
        
        # Generate response
        output = await self._handle_engine_call(_req, sampling_params)
        
        # Handle the response and switch players
        MultiplayerSGLangExtension.handle_player_response(
            mp_wrapper, output["text"]
        )
        
    # After game ends, create training instances
    player_requests = MultiplayerSGLangExtension.finalize_multiplayer_rollout(
        mp_wrapper, final_reward
    )
    
    return player_requests  # Return list instead of single request
```

## Benefits

1. **True Self-Play**: Single model learns both roles by seeing both perspectives
2. **Information Asymmetry**: Preserves partial observability from original game
3. **GRPO Compatible**: Proper grouping for advantage computation
4. **Modular Design**: Easy to extend to other multi-player games
5. **Consistent with evaluate_opt.py**: Uses same turn management approach

## Test Results

The standalone test successfully demonstrated:
- ✓ Separate conversation histories per player
- ✓ Information asymmetry (different masks)
- ✓ Turn alternation between players
- ✓ Grouped training instances (same batch_id)
- ✓ Same reward for both players
- ✓ Both player perspectives in training data

## Next Steps

1. Test the pipeline with GPT-3.5 first
2. Switch to Qwen2.5-7B for actual self-play training
3. Run full iterative training loop with weight updates
4. Monitor convergence and negotiation quality