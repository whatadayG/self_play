#!/bin/bash

# Test script for self-play pipeline with minimal settings
# This runs a quick test to ensure all components work together

set -e  # Exit on error

echo "Testing self-play pipeline..."

# Configuration for testing
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
NUM_GAMES=2  # Just 2 games for testing
OUTPUT_DIR="./test_selfplay"
PROJECT_DIR="/home/nickatomlin/georgiazhou/self_play"

# Clean up previous test
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

echo "Step 1: Testing evaluate_opt.py..."
cd $PROJECT_DIR/scripts

# First, generate some test games using the existing generate_matching_game_data.py
python generate_matching_game_data.py \
    --num-games 5 \
    --output "$OUTPUT_DIR/test_games.parquet"

echo ""
echo "Step 2: Running a single self-play game..."
python evaluate_opt.py \
    --game matching \
    --agent-model "$MODEL_NAME" \
    --user-model "$MODEL_NAME" \
    --num-selfplay-games $NUM_GAMES \
    --model-mode full_conversation_reproposal \
    --exp-name "test_selfplay" \
    --max-length 50 \
    --temperature 0.7 \
    --threshold 0.0 || {
        echo "evaluate_opt.py failed. Check if the model is accessible."
        exit 1
    }

# Check if output was created
SELFPLAY_OUTPUT=$(ls output_*test_selfplay*.jsonl 2>/dev/null | head -n1)
if [ -z "$SELFPLAY_OUTPUT" ]; then
    echo "No output file created by evaluate_opt.py"
    exit 1
fi

echo "Self-play output: $SELFPLAY_OUTPUT"
mv "$SELFPLAY_OUTPUT" "$OUTPUT_DIR/test_games.jsonl"

echo ""
echo "Step 3: Converting to VERL format..."
python convert_selfplay_to_verl.py \
    --input "$OUTPUT_DIR/test_games.jsonl" \
    --output "$OUTPUT_DIR/test_train.parquet"

echo ""
echo "Step 4: Checking converted data..."
python -c "
import pandas as pd
df = pd.read_parquet('$OUTPUT_DIR/test_train.parquet')
print(f'Successfully created {len(df)} training instances')
print(f'Columns: {list(df.columns)}')
print(f'First instance index: {df.iloc[0][\"extra_info\"][\"index\"]}')
print(f'First instance perspective: {df.iloc[0][\"extra_info\"][\"perspective\"]}')
"

echo ""
echo "Pipeline test complete! All components working."
echo ""
echo "To run the full training pipeline, use:"
echo "  ./run_selfplay_pipeline.sh [model_name] [num_games] [num_iterations]"
echo ""
echo "Example:"
echo "  ./run_selfplay_pipeline.sh meta-llama/Llama-3.2-1B-Instruct 50 3"