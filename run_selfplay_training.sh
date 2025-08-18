#!/bin/bash
# Run self-play RL training with verl using generated self-play data
# Ensure you're in the self_play directory when running this

set -x

ulimit -n 65535 2>/dev/null || true

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/verl/examples/sglang_multiturn/config"
DATA_DIR="$PROJECT_DIR/data/matching_game_selfplay"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}

# Check if data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Error: Training data not found at $DATA_DIR/train.parquet"
    echo "Please run ./run_selfplay_generation.sh first"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Make sure we're in the verl directory for the trainer
cd verl

CUDA_VISIBLE_DEVICES=2,3,4,5 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='matching_game_multiturn_grpo_w_interaction' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=$((1024 * 3)) \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='matching_game_selfplay' \
    trainer.experiment_name='qwen2.5-7B-selfplay' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_PATH/interaction_config/matching_game_interaction_config.yaml" \
    trainer.total_epochs=3 $@