# Qwen3-8B SFT with VERL (FSDP)

This folder provides a minimal SFT pipeline for `Qwen/Qwen3-8B` using VERL's FSDP SFT trainer, plus a utility to convert the trained checkpoint to HuggingFace format for downstream RL (PPO/GRPO).

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

## 1) Run SFT Training (multi-turn, parameters per README)
Use the VERL multiturn demo shape and pass overrides to match these hyperparameters (epochs, batch sizes, max length, dtype, LR) while switching to Qwen3-8B.
```bash
source /home/nickatomlin/georgiazhou/self_play/test_venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash /home/nickatomlin/georgiazhou/self_play/verl/examples/sft/multiturn/run_qwen_05_sp2.sh \
  4 \
  /home/nickatomlin/georgiazhou/self_play/verl/checkpoints/sft_qwen3_8b \
  data.train_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_train.parquet \
  data.val_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_val.parquet \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.max_length=10000 \so
  data.micro_batch_size=1 \
  data.train_batch_size=32 \
  model.partial_pretrain=Qwen/Qwen3-8B \
  model.trust_remote_code=true \
  model.fsdp_config.model_dtype=bf16 \
  optim.lr=1e-5 \
  trainer.project_name=multiturn-sft \
  trainer.experiment_name=multiturn_qwen3_8b_len10k_b32 \
  trainer.logger=console \
  trainer.max_epochs=3 \
  ulysses_sequence_parallel_size=2 \
  use_remove_padding=true
```