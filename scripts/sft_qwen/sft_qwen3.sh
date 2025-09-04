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

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_train.parquet \
    data.val_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_val.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=4000 \
    data.micro_batch_size_per_gpu=4 \
    data.train_batch_size=16 \
    model.partial_pretrain=Qwen/Qwen3-8B \
    model.trust_remote_code=true \
    model.fsdp_config.model_dtype=bf16 \
    optim.lr=1e-5 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn_qwen3_8b_len10k_b32 \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=10 $@ \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    use_remove_padding=true
