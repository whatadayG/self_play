#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/home/nickatomlin/georgiazhou/self_play/old/data/multiturn/train.parquet \
    data.val_files=/home/nickatomlin/georgiazhou/self_play/old/data/multiturn/train.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=2000 \
    data.micro_batch_size=1 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=1 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-2.5-0.5b-instruct-sp2 \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1 $@ \
    trainer.total_training_steps=1 $@ \
    use_remove_padding=true
