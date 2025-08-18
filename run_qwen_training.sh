#!/bin/bash
# Run self-play training for Qwen2.5-4B-Thinking-2507
# Start with small parameters for testing

set -e  # Exit on error
set -x  # Show commands

# Activate conda environment
echo "Activating conda environment..."
source /home/nickatomlin/sources/conda/etc/profile.d/conda.sh
conda activate py38
source .venv/bin/activate  


# Set project directories
PROJECT_DIR="/home/nickatomlin/georgiazhou/self_play"
CONFIG_PATH="$PROJECT_DIR/verl/examples/sglang_multiturn/config"
BASE_DATA_DIR="$PROJECT_DIR/data/matching_game_selfplay_iterations"
SCRIPTS_DIR="$PROJECT_DIR/scripts"

# Configuration for initial test
INITIAL_MODEL="Qwen/Qwen2.5-32B-Instruct"
NUM_ITERATIONS=1  # Start with just 1 iteration
GAMES_PER_ITER=5  # Fewer games for initial 32B test
TRAIN_BATCH_SIZE=4  # Reduced for 32B model
MICRO_BATCH_SIZE=1  # Reduced for 32B model
EPOCHS_PER_ITER=1  # Just 1 epoch for testing

# Create base directory
mkdir -p $BASE_DATA_DIR

echo "=========================================="
echo "Starting Qwen2.5-32B self-play training"
echo "Model: $INITIAL_MODEL"
echo "Games per iteration: $GAMES_PER_ITER"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "=========================================="

CURRENT_MODEL=$INITIAL_MODEL

for iter in $(seq 1 $NUM_ITERATIONS); do
    echo ""
    echo "=========================================="
    echo "Starting iteration $iter/$NUM_ITERATIONS"
    echo "Current model: $CURRENT_MODEL"
    echo "=========================================="
    
    ITER_DIR="$BASE_DATA_DIR/iter_$iter"
    mkdir -p $ITER_DIR
    
    # Step 1: Generate self-play data with current model
    echo ""
    echo "=== Step 1: Generating self-play data ==="
    cd $SCRIPTS_DIR
    
    # Check if we have local model or need to use API
    if [ "$CURRENT_MODEL" == "Qwen/Qwen2.5-32B" ]; then
        # For the first iteration, we need to use the base model
        echo "Using base Qwen model for generation..."
        # The evaluate_opt.py script should automatically detect and use HFModelPlayer
    fi
    
    python evaluate_opt.py \
        --exp-name "qwen_iter_${iter}" \
        --game matching \
        --mode selfplay \
        --agent-model-id "$CURRENT_MODEL" \
        --user-model-id "$CURRENT_MODEL" \
        --end $GAMES_PER_ITER \
        --samples-per-game 1 \
        --threshold 0.0 \
        --temperature 0.7
    
    # Move and convert data
    OUTPUT_FILE="output_<class 'dialop.envs.optimization.OptimizationEnv'>_qwen_iter_${iter}.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        mv "$OUTPUT_FILE" "$ITER_DIR/selfplay_games.jsonl"
        echo "Generated data saved to $ITER_DIR/selfplay_games.jsonl"
    else
        echo "Error: Output file not found: $OUTPUT_FILE"
        exit 1
    fi
    
    # Step 2: Convert to VERL format
    echo ""
    echo "=== Step 2: Converting to VERL format ==="
    python convert_selfplay_to_verl.py \
        --input "$ITER_DIR/selfplay_games.jsonl" \
        --output "$ITER_DIR/train_full.parquet"
    
    # Create train/val split
    python -c "
import pandas as pd
df = pd.read_parquet('$ITER_DIR/train_full.parquet')
# For small datasets, use 80/20 split
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx]
val_df = df[split_idx:] if split_idx < len(df) else df[-2:]  # Ensure at least 2 val samples
train_df.to_parquet('$ITER_DIR/train.parquet', index=False)
val_df.to_parquet('$ITER_DIR/val.parquet', index=False)
print(f'Iteration $iter: {len(train_df)} train, {len(val_df)} val instances')
print(f'Total instances: {len(df)} (remember: 2 instances per game)')
"
    
    # Step 3: Train on the generated data
    echo ""
    echo "=== Step 3: Training model ==="
    cd $PROJECT_DIR/verl
    
    CHECKPOINT_DIR="$ITER_DIR/checkpoint"
    
    # Check available GPUs
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.free --format=csv
    
    # Use GPUs 2,3,4,5 (adjust based on your system)
    CUDA_VISIBLE_DEVICES=2,3,4,5 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='matching_game_multiturn_grpo_w_interaction' \
        algorithm.adv_estimator=grpo \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.max_prompt_length=1024 \
        data.max_response_length=$((1024 * 3)) \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="$CURRENT_MODEL" \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        +actor_rollout_ref.model.enable_activation_offloading=True \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name='matching_game_selfplay_qwen' \
        trainer.experiment_name="Qwen2.5-32B-selfplay-iter${iter}" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.default_hdfs_dir="$CHECKPOINT_DIR" \
        data.train_files="$ITER_DIR/train.parquet" \
        data.val_files="$ITER_DIR/val.parquet" \
        actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_PATH/interaction_config/matching_game_interaction_config.yaml" \
        trainer.total_epochs=$EPOCHS_PER_ITER
    
    # Update model path for next iteration
    CURRENT_MODEL="$CHECKPOINT_DIR/actor"
    
    echo ""
    echo "Iteration $iter complete. Model saved to $CURRENT_MODEL"
done

echo ""
echo "=========================================="
echo "Self-play training complete!"
echo "Final model: $CURRENT_MODEL"
echo "All data and checkpoints in: $BASE_DATA_DIR"
echo "=========================================="