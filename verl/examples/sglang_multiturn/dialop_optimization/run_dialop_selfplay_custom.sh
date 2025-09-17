#!/bin/bash

# Script to run dialop self-play training with custom rollout worker
# This uses the new DialopSelfPlayRollout class instead of monkey patching

set -e

# Configuration
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
CONFIG_NAME="dialop_selfplay_custom_grpo"
DATA_DIR="${DATA_DIR:-${HOME}/data/dialop_selfplay_init}"

# Model configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"

# Training configuration
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-6}"

# Output configuration  
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/dialop_selfplay_custom}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="dialop_selfplay_custom_${TIMESTAMP}"

echo "=================================================="
echo "Dialop Self-Play Training with Custom Rollout"
echo "=================================================="
echo "Project directory: $PROJECT_DIR"
echo "Config: $CONFIG_NAME"
echo "Data directory: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Experiment name: $EXPERIMENT_NAME"
echo "=================================================="

# Check if data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Training data not found at $DATA_DIR/train.parquet"
    echo ""
    echo "Please generate game initializations first:"
    echo "  cd $PROJECT_DIR"
    echo "  python verl/examples/data_preprocess/dialop_game_init.py \\"
    echo "    --output_dir $DATA_DIR \\"
    echo "    --num_train 1000 --num_test 100"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
VENV_PATH="${PROJECT_DIR}/test_venv"
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Change to verl directory
cd "$PROJECT_DIR/verl"

# Run training
# Note: No need for VERL_APPLY_SELFPLAY_PATCH with custom rollout
echo "Starting training..."

python -m verl.trainer.main_ppo \
    --config-path="${PROJECT_DIR}/verl/examples/sglang_multiturn/config" \
    --config-name="$CONFIG_NAME" \
    data.train_files="[\"$DATA_DIR/train.parquet\"]" \
    data.val_files="[\"$DATA_DIR/test.parquet\"]" \
    data.train_batch_size=$BATCH_SIZE \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.total_epochs=$EPOCHS \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name="dialop_selfplay_custom" \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.override_config.max_seq_len=$MAX_SEQ_LEN \
    actor_rollout_ref.actor.optim.lr=$LR \
    critic.model.path="$MODEL_PATH" \
    critic.model.override_config.max_seq_len=$MAX_SEQ_LEN \
    trainer.default_hdfs_dir="$OUTPUT_DIR/$EXPERIMENT_NAME"

echo ""
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR/$EXPERIMENT_NAME"

# Merge checkpoint to HuggingFace format
echo ""
echo "Merging checkpoint to HuggingFace format..."
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$OUTPUT_DIR/$EXPERIMENT_NAME/actor" \
    --target_dir "$OUTPUT_DIR/$EXPERIMENT_NAME/actor_merged"

echo "Merged model saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/actor_merged"