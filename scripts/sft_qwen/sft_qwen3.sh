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

# Build arguments array for easy commenting/uncommenting
args=(
    # Data configuration
    data.train_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_train.parquet
    data.val_files=/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_val.parquet
    data.multiturn.enable=true
    data.multiturn.messages_key=messages
    data.max_length=4500
    data.truncation=right
    # data.micro_batch_size_per_gpu=2
    # data.train_batch_size=8
    # +data.val_batch_size_per_gpu=2
    data.micro_batch_size_per_gpu=4 # because we are gradient checkpointing, a larger batch size is actively worse
    data.train_batch_size=32
    +data.val_batch_size_per_gpu=4 # i think i didn't filter the val datapoints for length; i don't want to deal with this right now

    # Model configuration
    model.partial_pretrain=Qwen/Qwen3-8B
    model.trust_remote_code=true
    +model.chat_template_path=/home/nickatomlin/georgiazhou/self_play/data/custom_chat_template/qwen3_no_remove_thinking.jinja
    model.fsdp_config.model_dtype=bf16
    model.use_liger=true
    model.enable_gradient_checkpointing=true
    # model.lora_rank=32  # Set to 8, 16, 32, or 64 to enable LoRA (0 = disabled)
    # model.lora_alpha=32  # LoRA alpha scaling factor (typical: 16 or 32)
    # model.target_modules=all-linear  # LoRA target modules

    # Optimizer configuration
    optim.lr=1e-5  # Claude suggests a much higher LR if LoRA is enabled , like 2e-4, but verl/ examples give only 3x as high a learning rate.
    # enable these for initial fine-tunes, not so much for multi-epoch things like the grpo update step
    # optim.lr_scheduler=wsd,
    # +optim.stable_ratio=0.99,
    # +optim.min_lr_ratio=0.1,


    # Trainer configuration
    trainer.default_local_dir=$save_path
    trainer.project_name=perfect-sft
    trainer.experiment_name=trial-1
    "trainer.logger=[\"console\", \"wandb\"]"
    trainer.total_epochs=5
    trainer.save_freq=500
    trainer.test_freq=100
    trainer.entropy_coeff=0
    # +trainer.eval_wikitext=true  # Enable wikitext evaluation
    #+trainer.wikitext_path=data/eval/wikitext_sample.txt
    #+trainer.wikitext_max_seq_length=2048
    #+trainer.wikitext_batch_size=32

    # Other configuration
    use_remove_padding=true
    "+trainer.checkpoint_config.save_contents=[\"model\", \"extra\", \"hf_model\"]"

    # User-provided arguments
    "$@"
)

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node --rdzv_endpoint=localhost:${RDZV_PORT:-29400} \
    -m verl.trainer.fsdp_sft_trainer \
    "${args[@]}"

    # NOTE: PreTokenizedSFTDataset args moved to offline_grpo_loop.py (only needed for GRPO with sample weights)
    # For regular multiturn SFT, the default MultiTurnSFTDataset is used with data.multiturn.enable=true
    # trainer.n_gpus_per_node=2 \

