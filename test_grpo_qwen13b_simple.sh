#!/bin/bash

# Simple test for dialop self-play GRPO with Qwen2.5-13B fine-tune
# Based on the working minimal config but adapted for larger model

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${PROJECT_DIR}/data"
MODEL_PATH="${PROJECT_DIR}/old/save_points/global_step_1000_merged"

# Use GPUs 4,5,6,7
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/dialop:$PROJECT_DIR/verl:$PYTHONPATH"

# Output directory
OUTPUT_DIR="$PROJECT_DIR/test_output/qwen13b_simple_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Simple Dialop GRPO Test - Qwen2.5-13B Fine-tune"
echo "==============================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

# Check requirements
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Error: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Activate venv
source test_venv/bin/activate

cd verl

# Run with command-line overrides similar to working minimal config
# but with adjustments for 13B model on 4 GPUs
python -m verl.trainer.main_ppo \
    data.train_files="[\"$DATA_DIR/train.parquet\"]" \
    data.val_files="[\"$DATA_DIR/train.parquet\"]" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.return_raw_chat=true \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.project_name="dialop_test" \
    trainer.experiment_name="qwen13b_simple_test" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger='["console"]' \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout._target_="verl.workers.rollout.dialop_selfplay_rollout.DialopSelfPlayRollout" \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.trust_remote_code=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    reward_model.reward_manager=dialop_selfplay \
    critic.model.path="$MODEL_PATH" \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.trust_remote_code=true \
    critic.ppo_micro_batch_size_per_gpu=1 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete. Check $OUTPUT_DIR/training.log for details."

# Quick check for success
if grep -q "Train Step" "$OUTPUT_DIR/training.log"; then
    echo "✓ Successfully completed at least one gradient update"
else
    echo "⚠ No gradient updates found - check log for errors"
fi