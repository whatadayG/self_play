# Qwen2.5-7B SFT with VERL (FSDP)

This folder provides a minimal SFT pipeline for `Qwen/Qwen2.5-7B-Instruct` using VERL's FSDP SFT trainer, plus a utility to convert the trained checkpoint to HuggingFace format for downstream RL (PPO/GRPO).

## Prerequisites
- Python environment with CUDA + PyTorch.
- Install VERL (already vendored under `self_play/verl`):
```bash
pip install -e /home/nickatomlin/georgiazhou/self_play/verl
# Optional extras
pip install -r /home/nickatomlin/georgiazhou/self_play/verl/requirements.txt
# For conversion
pip install pandas pyarrow
```

## Data format: Multi-turn conversations
VERL's SFT trainer expects Parquet input. For multi-turn dialogue, use a Parquet file with at least a `messages` column where each row is a list of objects like:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
Optional columns: `tools`, `enable_thinking`.

If you have JSONL, convert it first:
```bash
python /home/nickatomlin/georgiazhou/self_play/sft_qwen/convert_jsonl_to_parquet.py \
  /home/nickatomlin/georgiazhou/self_play/sft_qwen/sft_qwen_from_july25_sft_5_proposals_not_weighted.jsonl \
  /home/nickatomlin/georgiazhou/self_play/sft_qwen/sft_qwen_from_july25.parquet \
  --messages_key messages --tools_key tools --thinking_key enable_thinking
```

## 1) Run SFT Training (recommended hyperparameters for ~1k multi-turn dialogues)
```bash
bash /home/nickatomlin/georgiazhou/self_play/sft_qwen/run_sft_qwen.sh \
  -t /home/nickatomlin/georgiazhou/self_play/sft_qwen/sft_qwen_from_july25.parquet \
  -v /home/nickatomlin/georgiazhou/self_play/sft_qwen/sft_qwen_from_july25.parquet \
  -U -K messages \
  -m Qwen/Qwen2.5-7B-Instruct \
  -P qwen2_5_7b_sft \
  -E multiturn_1k_default \
  -e 3 \
  -b 1 \
  -B 64 \
  -g 8 \
  -o /home/nickatomlin/georgiazhou/self_play/verl/checkpoints/sft_qwen2_5_7b
```
This sets:
- Multi-turn mode with `messages` as the key
- `data.max_length=2048`, `model.trust_remote_code=true`, `model.fsdp_config.model_dtype=bf16`
- `optim.lr=1e-5` for full-parameter FSDP; if you add `-L 32` to enable LoRA, LR auto-switches to `2e-4`
- Batch: `micro_batch_size_per_gpu=1`, `train_batch_size=64` (must be divisible by DP size)
- Epochs: 3

Note: If your GPU count differs, the script detects it automatically. Adjust `-B` to keep `train_batch_size % DP == 0`.

## 2) Convert Checkpoint to HuggingFace Format
```bash
bash /home/nickatomlin/georgiazhou/self_play/sft_qwen/merge_checkpoint.sh \
  -o /home/nickatomlin/georgiazhou/self_play/verl/checkpoints/sft_qwen2_5_7b \
  -t /home/nickatomlin/georgiazhou/self_play/verl/checkpoints/sft_qwen2_5_7b_hf
```

## 3) Use the SFT model for RL (PPO/GRPO)
```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  actor_rollout_ref.model.path=/home/nickatomlin/georgiazhou/self_play/verl/checkpoints/sft_qwen2_5_7b_hf \
  critic.model.path=/home/nickatomlin/georgiazhou/self_play/verl/checkpoints/sft_qwen2_5_7b_hf
```
To use SGLang or vLLM for rollouts during RL:
```bash
python3 -m verl.trainer.main_ppo \
  actor_rollout_ref.rollout.name=sglang \
  ...
```

## Tips
- `trainer.resume_mode` defaults to `auto`. For fault tolerance, it resumes from the latest step in the output dir.
- Ensure tokenizer/model compatibility between SFT and RL.
- Store checkpoints in persistent storage if RL runs on different nodes.
- If you see import errors for `verl`, double-check you ran `pip install -e /home/nickatomlin/georgiazhou/self_play/verl` or set `PYTHONPATH` accordingly. 