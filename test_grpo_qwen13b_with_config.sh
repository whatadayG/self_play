#!/bin/bash

# Test script using config file for Qwen2.5-13B fine-tune
# Uses the test_grpo_qwen13b_config.yaml configuration

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${PROJECT_DIR}/data"

# Use GPUs 4,5,6,7
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/dialop:$PROJECT_DIR/verl:$PYTHONPATH"

# Output directory
OUTPUT_DIR="$PROJECT_DIR/test_output/qwen13b_config_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Dialop GRPO Test with Config - Qwen2.5-13B"
echo "=========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: test_grpo_qwen13b_config.yaml"
echo "Output: $OUTPUT_DIR"

# Check requirements
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Error: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi

# Copy config to output dir for reference
cp test_grpo_qwen13b_config.yaml "$OUTPUT_DIR/"

# Activate venv
source test_venv/bin/activate

cd verl

# Run with config file
python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_DIR" \
    --config-name="test_grpo_qwen13b_config" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.experiment_name="qwen13b_config_test_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete. Check $OUTPUT_DIR/training.log for details."