#!/bin/bash
# Test self-play with local model

set -x

PROJECT_DIR="$(pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/data/test_local_model"

# Use a smaller model for testing
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"  # Much smaller for testing
NUM_GAMES=1
EXP_NAME="test_local_model"

# Create output directory
mkdir -p $DATA_DIR

# Change to scripts directory
cd $SCRIPTS_DIR

# Test with local model
echo "Testing self-play with local model..."
python evaluate_opt.py \
    --exp-name "$EXP_NAME" \
    --game matching \
    --mode selfplay \
    --agent-model-id "$MODEL_NAME" \
    --user-model-id "$MODEL_NAME" \
    --end $NUM_GAMES \
    --threshold 0.0

# Check if output was created
OUTPUT_FILE="output_<class 'dialop.envs.optimization.OptimizationEnv'>_${EXP_NAME}.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Success! Output file created: $OUTPUT_FILE"
    echo "File contents:"
    cat "$OUTPUT_FILE" | jq . | head -100
    
    # Try to convert it
    echo ""
    echo "Testing conversion to VERL format..."
    python convert_selfplay_to_verl.py \
        --input "$OUTPUT_FILE" \
        --output "$DATA_DIR/test.parquet"
    
    # Check parquet file
    python -c "
import pandas as pd
df = pd.read_parquet('$DATA_DIR/test.parquet')
print(f'Successfully created {len(df)} training instances')
print('Columns:', list(df.columns))
"
else
    echo "No output file created."
    echo "Checking for error logs..."
    ls -la *.txt 2>/dev/null || echo "No log files found"
fi