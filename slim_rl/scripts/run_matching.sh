#!/bin/bash
# SLIME Training Script for Paper-Reviewer Matching Game
#
# This script launches GRPO training with:
# - Qwen as the trainable agent
# - GPT-4.1 as the fixed shy partner
# - DialOp optimization game environment
#
# Prerequisites:
# 1. SLIME installed and configured
# 2. OPENAI_API_KEY set in environment
# 3. Training data generated (run generate_data.py first)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIM_RL_DIR="$(dirname "$SCRIPT_DIR")"
SELF_PLAY_DIR="$(dirname "$SLIM_RL_DIR")"

# Add to Python path
export PYTHONPATH="${PYTHONPATH}:${SLIM_RL_DIR}:${SELF_PLAY_DIR}/scripts"

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Please set it with: export OPENAI_API_KEY=your-key-here"
    exit 1
fi

# Model settings
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SLIM_RL_DIR}/checkpoints}"

# Training data
TRAIN_DATA="${TRAIN_DATA:-${SLIM_RL_DIR}/data/games.jsonl}"

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Training data not found at: $TRAIN_DATA"
    echo "Generating training data..."
    python "${SLIM_RL_DIR}/src/data_generator.py" \
        --num_games 1000 \
        --output "$TRAIN_DATA" \
        --seed 42
fi

# Training hyperparameters
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
GROUP_SIZE="${GROUP_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_TURNS="${MAX_TURNS:-20}"

# GPU settings
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.8}"

echo "=========================================="
echo "SLIME Paper-Reviewer Matching Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Train data: $TRAIN_DATA"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Batch size: $GLOBAL_BATCH_SIZE"
echo "Group size: $GROUP_SIZE"
echo "Max turns: $MAX_TURNS"
echo "=========================================="

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

# Launch training
# Note: Adjust the command based on actual SLIME CLI
python -m slime.train \
    --model-path "$MODEL_PATH" \
    --train-data "$TRAIN_DATA" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --custom-generate-function-path "slim_rl.src.rollout:generate" \
    --rollout-interaction-env-path "slim_rl.src.env_matching" \
    --rollout-max-context-len 4096 \
    --max-turns "$MAX_TURNS" \
    --algorithm grpo \
    --group-size "$GROUP_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --num-train-epochs "$NUM_EPOCHS" \
    --tensor-model-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION" \
    "$@"

echo "Training complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
