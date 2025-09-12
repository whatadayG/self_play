# Environment

The environment is managed using `uv`, so any `pip` commands should be `uv pip`. The venv is located at the parent dir of this, `self_play`, so you may either need extra permissions or the user should activate the venv before asking questions.

# Key Concepts

## SGLang vs vLLM
- SGLang and vLLM are alternative inference engines for rollout generation
- You use one or the other, configured via `rollout.name`
- Multi-turn rollouts (with tools/interactions) currently require SGLang

## Architecture
- Ray: Distributed orchestration across nodes/processes (NOT the actual compute)
- FSDP/Megatron: Handle actual training/gradients
- vLLM/SGLang: Handle actual text generation
- `dp_actor` = Data Parallel Actor (FSDP implementation) The venv is located at the parent dir of this, `self_play`, so you may either need extra permissions or the user should activate the venv before asking questions.
