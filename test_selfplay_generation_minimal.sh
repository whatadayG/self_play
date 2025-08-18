#!/bin/bash
# Minimal test for self-play generation - just 1 game

set -x

PROJECT_DIR="$(pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/data/test_selfplay"

# Configuration for minimal test
MODEL_NAME="gpt-3.5-turbo"  # Cheaper model for testing
NUM_GAMES=1
EXP_NAME="test_minimal"

# Create output directory
mkdir -p $DATA_DIR

# Activate virtual environment
source .venv/bin/activate

# Change to scripts directory
cd $SCRIPTS_DIR

# Run self-play generation with minimal settings
echo "Running minimal self-play test..."
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
    cat "$OUTPUT_FILE" | jq . | head -50
    
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
print('First instance:')
print(df.iloc[0].to_dict())
"
else
    echo "No output file created. Checking for error files..."
    ls -la *.txt
fi