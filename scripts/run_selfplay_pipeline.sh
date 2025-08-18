#!/bin/bash

# Self-play training pipeline for matching game using VERL
# This script orchestrates the full pipeline:
# 1. Generate self-play data using evaluate_opt.py
# 2. Convert to VERL format
# 3. Run GRPO training

set -e  # Exit on error

# Configuration
MODEL_NAME=${1:-"meta-llama/Llama-3.2-1B-Instruct"}  # Default model
NUM_GAMES=${2:-100}  # Number of games per iteration
NUM_ITERATIONS=${3:-5}  # Number of self-play iterations
OUTPUT_DIR=${4:-"./selfplay_results"}
EXP_NAME=${5:-"selfplay_exp"}

# VERL configuration
VERL_CONFIG_PATH="/home/nickatomlin/georgiazhou/self_play/verl/examples/sglang_multiturn/config/matching_game_multiturn_grpo_w_interaction.yaml"
PROJECT_DIR="/home/nickatomlin/georgiazhou/self_play"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting self-play pipeline..."
echo "Model: $MODEL_NAME"
echo "Games per iteration: $NUM_GAMES"
echo "Total iterations: $NUM_ITERATIONS"
echo "Output directory: $OUTPUT_DIR"

# Main training loop
for iter in $(seq 1 $NUM_ITERATIONS); do
    echo ""
    echo "=== Iteration $iter/$NUM_ITERATIONS ==="
    
    # Set paths for this iteration
    SELFPLAY_OUTPUT="$OUTPUT_DIR/iter_${iter}_games.jsonl"
    VERL_DATA="$OUTPUT_DIR/iter_${iter}_train.parquet"
    MODEL_CHECKPOINT="$OUTPUT_DIR/iter_${iter}_checkpoint"
    
    # Step 1: Generate self-play data
    echo "Generating self-play data..."
    cd $PROJECT_DIR/scripts
    
    python evaluate_opt.py \
        --game matching \
        --agent-model "$MODEL_NAME" \
        --user-model "$MODEL_NAME" \
        --num-selfplay-games $NUM_GAMES \
        --model-mode full_conversation_reproposal \
        --exp-name "${EXP_NAME}_iter${iter}" \
        --max-length 100 \
        --temperature 0.7 \
        --threshold 0.0
    
    # Move the output file to our directory
    mv "output_<class 'dialop.envs.optimization.OptimizationEnv'>_${EXP_NAME}_iter${iter}.jsonl" "$SELFPLAY_OUTPUT"
    
    # Step 2: Convert to VERL format
    echo "Converting to VERL format..."
    python convert_selfplay_to_verl.py \
        --input "$SELFPLAY_OUTPUT" \
        --output "$VERL_DATA"
    
    # Step 3: Run VERL training
    echo "Running GRPO training..."
    cd $PROJECT_DIR/verl
    
    # Activate the virtual environment if needed
    # source ../venv/bin/activate  # Uncomment if using venv
    
    python -m verl.trainer.main_ppo \
        --config-path="$VERL_CONFIG_PATH" \
        --config-name='matching_game_multiturn_grpo_w_interaction' \
        data.train_files=["$VERL_DATA"] \
        data.train_batch_size=128 \
        data.val_batch_size=128 \
        data.shuffle=True \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=True \
        trainer.total_epochs=3 \
        trainer.save_freq=-1 \
        actor_rollout_ref.model.path="$MODEL_NAME" \
        actor_rollout_ref.model.override_config.max_seq_len=8192 \
        critic.model.path="$MODEL_NAME" \
        critic.model.override_config.max_seq_len=8192 \
        metrics.output_dir="$OUTPUT_DIR/iter_${iter}_metrics" \
        metrics.log_to_console=True \
        trainer.default_hdfs_dir="$MODEL_CHECKPOINT"
    
    # Update model name for next iteration
    MODEL_NAME="$MODEL_CHECKPOINT/actor"
    
    echo "Iteration $iter complete. Model saved to $MODEL_CHECKPOINT"
done

echo ""
echo "Self-play pipeline complete!"
echo "Final model: $MODEL_NAME"
echo "All results saved in: $OUTPUT_DIR"

# Optional: Evaluate final model
echo ""
echo "To evaluate the final model, run:"
echo "python evaluate_opt.py --game matching --agent-model '$MODEL_NAME' --user-model '$MODEL_NAME' --num-selfplay-games 20"