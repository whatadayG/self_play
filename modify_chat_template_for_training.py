#!/usr/bin/env python3
"""
Modify Qwen3 chat template to preserve thinking content for all assistant turns.

The default Qwen3 chat template strips thinking content (<think>...</think>) from
assistant turns when reconstructing conversation context. This is fine for inference
but breaks training because:

1. SGLang returns logprobs for ALL generated tokens, including thinking content
2. We need to align logprobs with token IDs for GRPO training
3. If the template strips thinking, we can't match logprobs to tokens

This script modifies the chat_template.jinja file in a model checkpoint to always
preserve thinking content for all assistant messages.

Usage:
    python modify_chat_template_for_training.py <checkpoint_path>

Example:
    python modify_chat_template_for_training.py checkpoints/sft_qwen3_8b/global_step_3600_merged
"""

import sys
import os
from pathlib import Path


MODIFIED_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {# MODIFIED FOR TRAINING: Always include thinking content for all assistant turns #}
        {%- if reasoning_content %}
            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}
"""


def modify_chat_template(checkpoint_path: str) -> None:
    """Modify the chat template in a checkpoint directory."""

    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_path}")
        sys.exit(1)

    template_file = checkpoint_dir / "chat_template.jinja"
    if not template_file.exists():
        print(f"Error: chat_template.jinja not found in {checkpoint_path}")
        print(f"This script only works with Qwen3 models that have a chat_template.jinja file")
        sys.exit(1)

    # Backup original
    backup_file = checkpoint_dir / "chat_template.jinja.original"
    if not backup_file.exists():
        print(f"Creating backup: {backup_file}")
        with open(template_file, 'r') as f:
            original_content = f.read()
        with open(backup_file, 'w') as f:
            f.write(original_content)
    else:
        print(f"Backup already exists: {backup_file}")

    # Write modified template
    print(f"Writing modified template to: {template_file}")
    with open(template_file, 'w') as f:
        f.write(MODIFIED_TEMPLATE)

    print("\n✓ Successfully modified chat template")
    print("\nWhat changed:")
    print("  - Lines 43-48: Removed conditional logic that stripped thinking from non-last turns")
    print("  - Now ALL assistant turns preserve <think>...</think> content")
    print("\nIMPORTANT:")
    print("  - You must restart any SGLang/vLLM servers using this checkpoint")
    print("  - The modified template is only for training data preparation")
    print("  - For inference/deployment, use the original template (backed up)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    modify_chat_template(checkpoint_path)
