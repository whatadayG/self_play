#!/bin/bash

# Minimal test script to verify GRPO integration with custom rollout worker
# Uses minimal parameters for quick feedback

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONFIG_NAME="dialop_selfplay_custom_grpo"
DATA_DIR="${HOME}/data/dialop_selfplay_init"

# Model configuration - use small model for testing
MODEL_PATH="${MODEL_PATH:-gpt2}"  # Small model for quick test
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

# Minimal training configuration
BATCH_SIZE="${BATCH_SIZE:-2}"  # Minimum batch
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"  # Just 1 epoch
GRADIENT_STEPS="${GRADIENT_STEPS:-1}"  # Just 1 update

# Output configuration  
OUTPUT_DIR="/tmp/dialop_grpo_test"
EXPERIMENT_NAME="minimal_integration_test"

echo "=================================================="
echo "Minimal GRPO Integration Test"
echo "=================================================="
echo "Project directory: $PROJECT_DIR"
echo "Config: $CONFIG_NAME"
echo "Model: $MODEL_PATH (small model for testing)"
echo "Batch size: $BATCH_SIZE"
echo "Steps: $GRADIENT_STEPS"
echo "=================================================="

# Check if test data exists
if [ ! -f "$DATA_DIR/train_small.parquet" ]; then
    echo "Error: $DATA_DIR/train_small.parquet not found!"
    echo "Please ensure train_small.parquet and test_small.parquet exist in $DATA_DIR"
    exit 1
fi

# Activate virtual environment
source test_venv/bin/activate

# Change to verl directory
cd "$PROJECT_DIR/verl"

# Run minimal training
echo "Starting minimal GRPO training..."

# Use GPUs 0 and 1
export CUDA_VISIBLE_DEVICES="0,1"

python -m verl.trainer.main_ppo \
    --config-path="${PROJECT_DIR}/verl/examples/sglang_multiturn/config" \
    --config-name="$CONFIG_NAME" \
    data.train_files="[\"$DATA_DIR/train_small.parquet\"]" \
    data.val_files="[\"$DATA_DIR/test_small.parquet\"]" \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$BATCH_SIZE \
    algorithm.adv_estimator=grpo \
    trainer.total_epochs=$EPOCHS \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name="dialop_test" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger='["console"]' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.override_config.max_seq_len=$MAX_SEQ_LEN \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.max_new_tokens=256 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    critic.model.path="$MODEL_PATH" \
    critic.model.override_config.max_seq_len=$MAX_SEQ_LEN \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    trainer.n_gpus_per_node=2 \
    trainer.default_hdfs_dir="$OUTPUT_DIR" 2>&1 | tee grpo_test.log

# Check if it completed
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ GRPO integration test completed successfully!"
    echo "The custom rollout worker is properly integrated."
    echo ""
    echo "Check grpo_test.log for details"
    echo ""
    echo "To run full training, use:"
    echo "  bash verl/examples/sglang_multiturn/dialop_optimization/run_dialop_selfplay_custom.sh"
else
    echo ""
    echo "✗ GRPO integration test failed!"
    echo "Check grpo_test.log for errors"
    exit 1
fi