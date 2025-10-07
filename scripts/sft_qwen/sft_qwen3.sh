#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: sft_qwen3.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
# Restrict NCCL P2P to NVLink pairs to avoid cross-bridge P2P stalls on GPUs 4-7
export NCCL_P2P_LEVEL=NVL
# Alternatively, to force socket/shm only, uncomment the next line:
# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --rdzv_endpoint=localhost:29400 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_train.parquet \
    data.val_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_val.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=5000 \
    data.truncation=error \
    data.micro_batch_size_per_gpu=3 \
    data.train_batch_size=12 \
    model.partial_pretrain=Qwen/Qwen3-8B \
    model.trust_remote_code=true \
    model.fsdp_config.model_dtype=bf16 \
    model.use_liger=true \
    optim.lr=1e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn_qwen3_8b_len10k_b32 \
    trainer.logger='["console", "wandb"]' \
    trainer.total_epochs=10 $@ \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    use_remove_padding=true

    # trainer.n_gpus_per_node=2 \

