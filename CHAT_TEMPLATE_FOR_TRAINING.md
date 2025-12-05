# Chat Template Modification for Training with Thinking Models

## Problem

When training Qwen3 models (or other models with thinking mode) using GRPO/PPO, there's a critical issue with how chat templates handle thinking content (`<think>...</think>`):

**What happens during generation:**
1. SGLang generates: `<think>\nreasoning...\n</think>\n\nactual response<|im_end|>`
2. SGLang returns logprobs for **ALL** tokens, including thinking content
3. The response gets stored in `messages` with full thinking content

**What happens during training data preparation:**
1. We use `tokenizer.apply_chat_template()` to reconstruct the token sequence
2. **The default Qwen3 template STRIPS thinking content from non-last turns!**
3. The reconstructed sequence is missing tokens that have logprobs
4. Alignment fails: we can't match logprobs to token IDs

## Why Templates Strip Thinking

The default Qwen3 template intentionally removes thinking content when reconstructing conversation context. This makes sense for inference (the model doesn't need to see its previous reasoning), but breaks training because:

- Training requires exact token-level alignment between sequences and logprobs
- If tokens are missing from the sequence, we can't assign their logprobs
- Mask building fails because we can't identify which tokens were actually generated

## Solution

Modify the chat template to **always preserve thinking content** for all assistant turns.

### Step 1: Run the modification script

```bash
python modify_chat_template_for_training.py checkpoints/sft_qwen3_8b/global_step_3600_merged
```

This script:
- Backs up the original template to `chat_template.jinja.original`
- Modifies lines 43-48 to remove conditional thinking stripping
- Preserves thinking content for ALL assistant messages, not just the last one

### Step 2: Restart inference servers

**IMPORTANT:** Any SGLang or vLLM servers must be restarted to load the modified template:

```bash
# Stop existing server
pkill -f sglang

# Start with modified checkpoint
python -m sglang.launch_server \
    --model-path checkpoints/sft_qwen3_8b/global_step_3600_merged \
    --port 31234 \
    --tp 2
```

### Step 3: Verify alignment

Run the test suite to verify logprob/token alignment:

```bash
pytest tests/test_sglang_mask_alignment.py -m expensive -v
```

## What Gets Modified

### Original Template (lines 43-50)
```jinja
{%- if loop.index0 > ns.last_query_index %}
    {%- if loop.last or (not loop.last and reasoning_content) %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
    {%- endif %}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

**Problem:** Only preserves thinking if `loop.index0 > ns.last_query_index` (after last user query)

### Modified Template (lines 43-48)
```jinja
{# MODIFIED FOR TRAINING: Always include thinking content for all assistant turns #}
{%- if reasoning_content %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

**Fix:** Always includes thinking if present, regardless of position in conversation

## Error Messages

If you see this error:

```
Logprob/token alignment failure for assistant message #2:
  Consumed: 7 tokens from reconstructed sequence
  Expected: 149 logprobs from SGLang
  Difference: 142 logprobs unmatched

LIKELY CAUSE: Chat template is stripping thinking content!
```

**Solution:** Run the template modification script and restart servers.

## Deployment vs Training

- **Training data preparation:** Use modified template (preserves thinking)
- **Deployment/inference:** Use original template (strips thinking for efficiency)

The backup file (`chat_template.jinja.original`) preserves the original template for deployment.

## Testing

The test suite in `tests/test_sglang_mask_alignment.py` validates:

1. ✅ Token sequence length matches mask length
2. ✅ Total masked positions equals total logprobs collected
3. ✅ Each assistant turn cluster ends with `<|im_end|>`
4. ✅ Multi-turn conversations preserve thinking in all turns

Run with: `pytest tests/test_sglang_mask_alignment.py -m expensive`

## Files Changed

- `/home/nickatomlin/georgiazhou/self_play/modify_chat_template_for_training.py` - Modification script
- `checkpoints/*/chat_template.jinja` - Modified template (in checkpoint dirs)
- `checkpoints/*/chat_template.jinja.original` - Backup of original
- `scripts/dialop/sglang_model_player.py` - Enhanced error messages for alignment failures
- `tests/test_sglang_server_behavior.py` - SGLang behavior documentation tests
- `tests/test_sglang_mask_alignment.py` - Mask/logprob alignment validation tests
