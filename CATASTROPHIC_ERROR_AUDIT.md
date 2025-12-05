# Catastrophic Error Handling Audit

## Overview
This document identifies all locations where the codebase has fallbacks for catastrophic errors (uncaught exceptions, generation failures) that should instead fail the game entirely.

## Distinction
- **Normal errors** (GameError): Expected errors like invalid proposals, misformatted messages. These should allow retries and NOT fail the game.
- **Catastrophic errors** (unexpected Exception): Bugs, invalid state, generation failures. These should set `done=True` to terminate the game and prevent bad data from entering training.

## Issues Found

### 1. optimization.py:231-238 - General Exception Handler
**Location**: `scripts/dialop/envs/optimization.py:235`

**Current behavior**:
```python
except Exception as e:
    print(f"!!! {traceback.format_exc()}")
    return {
        **{p: "error" for p in self.players},
        "done": False,  # ← WRONG: allows game to continue
        "reward": 0,
        "turn_player": self.players[self.game.turn_player]
    }, True
```

**Problem**: Sets `done=False`, allowing the game to continue in an invalid state.

**Fix**: Change to `"done": True`

**Rationale**: This handler catches ANY exception that isn't a GameError (line 219 raises ValueError for unknown message types). These are unexpected errors that indicate the game is in an invalid state.

---

### 2. base_player.py:133-139 - Generation Failure Fallback
**Location**: `scripts/dialop/base_player.py:135-139`

**Current behavior**:
```python
try:
    response_text, input_tokens, output_tokens = self._generate_text(self.messages, **gen_kwargs)
except Exception as e:
    self.console.print(f"[red]Generation error: {e}[/red]")
    response_text = "I need to think about this."  # ← WRONG: fallback response
    input_tokens = 0
    output_tokens = len(response_text.split())
```

**Problem**: Catches generation exceptions and returns a fallback response instead of propagating the exception.

**Fix**: Re-raise the exception

**Rationale**:
- The game loop in `generate_rollouts.py:174-177` already handles exceptions from `respond()` and treats them as terminal
- SGLang player correctly re-raises exceptions after retries
- The fallback "I need to think about this." gets filtered during data processing but still wastes compute
- Generation failures indicate server issues or OOM, not recoverable errors

---

### 3. wrappers.py:141-143 - Silent Exception Swallowing
**Location**: `scripts/dialop/envs/wrappers.py:141-143`

**Current behavior**:
```python
try:
    turn_player = obss["turn_player"]
    if turn_player in self.players and turn_player in obss:
        obss[turn_player] = self._insert_word_limit(obss[turn_player])
except Exception:
    # Defensive — never allow the wrapper to crash; fall back to raw obs
    pass  # ← WRONG: silently swallows errors
```

**Problem**: Silently swallows all exceptions when inserting word limit message.

**Fix**: At minimum, log the exception. Optionally fail the game if the error indicates invalid state.

**Rationale**:
- Silent failures mask bugs
- If `_insert_word_limit()` fails, it likely indicates malformed observation data
- Better to fail loudly than continue with potentially corrupted state

---

## Implementation Plan

1. Fix optimization.py general exception handler
2. Fix base_player.py generation fallback
3. Fix wrappers.py silent exception handling
4. Add tests to verify catastrophic errors terminate games
