#!/bin/bash
# Generate self-play data for matching game
# Run from the self_play directory

set -x

PROJECT_DIR="$(pwd)"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/data/matching_game_selfplay"

# Configuration
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
NUM_GAMES=${NUM_GAMES:-100}
MAX_LENGTH=${MAX_LENGTH:-100}
TEMPERATURE=${TEMPERATURE:-0.7}
EXP_NAME=${EXP_NAME:-"selfplay_gen"}

# Create output directory
mkdir -p $DATA_DIR

# Activate virtual environment
source .venv/bin/activate

# Change to scripts directory
cd $SCRIPTS_DIR

# Run self-play generation
python evaluate_opt.py \
    --exp-name "$EXP_NAME" \
    --game matching \
    --mode full_conversation_reproposal \
    --agent-model-id "$MODEL_NAME" \
    --user-model-id "$MODEL_NAME" \
    --end $NUM_GAMES \
    --threshold 0.0

# Move output to data directory
OUTPUT_FILE="output_<class 'dialop.envs.optimization.OptimizationEnv'>_${EXP_NAME}.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    mv "$OUTPUT_FILE" "$DATA_DIR/selfplay_games.jsonl"
    echo "Self-play data saved to $DATA_DIR/selfplay_games.jsonl"
else
    echo "Error: Output file not found: $OUTPUT_FILE"
    exit 1
fi

# Convert to VERL format
echo "Converting to VERL format..."
python convert_selfplay_to_verl.py \
    --input "$DATA_DIR/selfplay_games.jsonl" \
    --output "$DATA_DIR/train.parquet"

# Create a small validation set from the same data (last 10%)
python -c "
import pandas as pd
df = pd.read_parquet('$DATA_DIR/train.parquet')
split_idx = int(len(df) * 0.9)
train_df = df[:split_idx]
val_df = df[split_idx:]
train_df.to_parquet('$DATA_DIR/train_split.parquet', index=False)
val_df.to_parquet('$DATA_DIR/val.parquet', index=False)
print(f'Created train set with {len(train_df)} instances')
print(f'Created val set with {len(val_df)} instances')
"

# Rename train file
mv "$DATA_DIR/train_split.parquet" "$DATA_DIR/train.parquet"

echo "Self-play data generation complete!"
echo "Training data: $DATA_DIR/train.parquet"
echo "Validation data: $DATA_DIR/val.parquet"