#!/bin/bash

# Very simple test for dialop self-play GRPO

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${PROJECT_DIR}/data"

# Use GPUs 2,3
export CUDA_VISIBLE_DEVICES="2,3"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/dialop:$PROJECT_DIR/verl:$PYTHONPATH"
export PROJECT_DIR="$PROJECT_DIR"  # For config file

# Output
OUTPUT_DIR="$PROJECT_DIR/test_output/dialop_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Simple Dialop GRPO Test"
echo "======================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"

# Check data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Error: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi

# Activate venv and run
source test_venv/bin/activate
cd verl

# Run with minimal config
python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR" \
    --config-name="test_grpo_minimal_config" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.default_hdfs_dir=null 2>&1 | tee "$OUTPUT_DIR/log.txt"

echo ""
echo "Training complete! Check $OUTPUT_DIR/log.txt"

# Check results
if grep -q "actor_loss" "$OUTPUT_DIR/log.txt"; then
    echo "✓ Successfully completed gradient update!"
else
    echo "✗ No gradient update found"
    tail -30 "$OUTPUT_DIR/log.txt"
fi