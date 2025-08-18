#!/bin/bash
# Minimal test of the self-play pipeline
# This tests data generation -> conversion -> training setup

set -e  # Exit on error
set -x  # Show commands

PROJECT_DIR="$(pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/data/test_minimal"

# Clean up previous test
rm -rf $DATA_DIR
mkdir -p $DATA_DIR

# Step 1: Test with GPT first to ensure pipeline works
echo "=== Step 1: Testing with GPT-3.5 ==="
cd $SCRIPTS_DIR

python evaluate_opt.py \
    --exp-name "test_gpt" \
    --game matching \
    --mode selfplay \
    --agent-model-id "gpt-3.5-turbo" \
    --user-model-id "gpt-3.5-turbo" \
    --end 1 \
    --threshold 0.0

# Check output
OUTPUT_FILE="output_<class 'dialop.envs.optimization.OptimizationEnv'>_test_gpt.jsonl"
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "ERROR: No output file from evaluate_opt.py"
    exit 1
fi

echo "Generated self-play data:"
cat "$OUTPUT_FILE" | jq .

# Step 2: Convert to VERL format
echo ""
echo "=== Step 2: Converting to VERL format ==="
python convert_selfplay_to_verl.py \
    --input "$OUTPUT_FILE" \
    --output "$DATA_DIR/train.parquet"

# Check conversion
python -c "
import pandas as pd
df = pd.read_parquet('$DATA_DIR/train.parquet')
print(f'Successfully converted {len(df)} instances')
print('Sample data:')
print(df.iloc[0])
"

echo ""
echo "=== Pipeline test successful! ==="
echo "Next steps:"
echo "1. Run with local model: MODEL_NAME='Qwen/Qwen2.5-0.5B-Instruct' ./test_local_model_selfplay.sh"
echo "2. Run full pipeline: ./run_selfplay_iteration.sh"