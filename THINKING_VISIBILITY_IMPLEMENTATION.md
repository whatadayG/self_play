# Chain-of-Thought Visibility Implementation

## Overview

We've implemented proper handling of "let's think step by step" chain-of-thought blocks in our dialop self-play system, ensuring they are not visible to the other agent.

## Key Implementation Details

### 1. **Separate Message Histories** (`dialop_selfplay_rollout.py`)

Each player maintains their own message history with different visibility:

```python
player_messages = {
    "player-1": [],  # Player 1's view of the conversation
    "player-2": []   # Player 2's view of the conversation
}
```

### 2. **Message Visibility Rules**

When a player makes a move:
- **Current player** sees their full message (including thinking)
- **Other player** sees only from the `[tag]` onwards (no thinking)

Example:
- Player 1 sends: `"Let's think step by step. I see patterns. [message] Hello partner"`
- Player 1 sees: The full message
- Player 2 sees: `"Partner: [message] Hello partner"`

### 3. **Helper Function**

```python
def get_visible_part(message):
    """Extract the part visible to partner (from tag onwards)."""
    tag_match = re.search(r"\[(message|propose|accept|reject)\]", message, re.IGNORECASE)
    if tag_match:
        return message[tag_match.start():]
    return message
```

### 4. **Message History Management**

After a successful move:
```python
# Current player sees their full message
player_messages[current_player].append({
    "role": "assistant",
    "content": response  # Full message with thinking
})

# Other player sees only the visible part
visible_response = get_visible_part(response)
player_messages[other_player].append({
    "role": "user", 
    "content": f"Partner: {visible_response}"  # No thinking
})
```

### 5. **Training Data**

The full conversation (including thinking) is preserved for training:
- Each player's training data shows their own thinking
- But hides their partner's thinking
- This teaches the model to think before responding

## Compatibility with OptimizationEnv

Our implementation matches the behavior of the updated OptimizationEnv:
- ✅ Thinking blocks are private to each player
- ✅ The `observe()` behavior is correctly implemented
- ✅ Player 2 must include "let's think step by step" (enforced by env)
- ✅ Error messages preserve thinking visibility rules

## Testing

Comprehensive tests verify:
1. Basic thinking visibility in OptimizationEnv
2. Message extraction logic
3. Separate message histories per player
4. Correct training data format
5. Error handling preserves visibility rules

Run tests with:
```bash
python test_thinking_visibility.py
python test_rollout_thinking.py
```

## Benefits

1. **Privacy**: Each agent's reasoning is private
2. **Realism**: Mimics human collaboration where internal thoughts aren't shared
3. **Better Training**: Models learn to reason internally before communicating
4. **Fair Play**: Neither agent has an unfair information advantage