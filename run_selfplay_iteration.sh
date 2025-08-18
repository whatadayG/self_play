#!/bin/bash
# Run iterative self-play training loop
# Each iteration: generate data → train → use new model for next iteration

set -x

ulimit -n 65535 2>/dev/null || true

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/verl/examples/sglang_multiturn/config"
BASE_DATA_DIR="$PROJECT_DIR/data/matching_game_selfplay_iterations"
SCRIPTS_DIR="$PROJECT_DIR/scripts"

# Configuration
INITIAL_MODEL=${INITIAL_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
NUM_ITERATIONS=${NUM_ITERATIONS:-1}
GAMES_PER_ITER=${GAMES_PER_ITER:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
EPOCHS_PER_ITER=${EPOCHS_PER_ITER:-2}

# Create base directory
mkdir -p $BASE_DATA_DIR

# Activate virtual environment
source .venv/bin/activate  # Commented out - using conda env instead

CURRENT_MODEL=$INITIAL_MODEL

for iter in $(seq 1 $NUM_ITERATIONS); do
    echo "=========================================="
    echo "Starting iteration $iter/$NUM_ITERATIONS"
    echo "Current model: $CURRENT_MODEL"
    echo "=========================================="
    
    ITER_DIR="$BASE_DATA_DIR/iter_$iter"
    mkdir -p $ITER_DIR
    
    # Step 1: Generate self-play data with current model
    echo "Generating self-play data for iteration $iter..."
    cd $SCRIPTS_DIR
    
    python evaluate_opt.py \
        --exp-name "iter_${iter}" \
        --game matching \
        --mode full_conversation_reproposal \
        --agent-model-id "$CURRENT_MODEL" \
        --user-model-id "$CURRENT_MODEL" \
        --end $GAMES_PER_ITER \
        --threshold 0.0
    
    # Move and convert data
    OUTPUT_FILE="output_<class 'dialop.envs.optimization.OptimizationEnv'>_iter_${iter}.jsonl"
    if [ -f "$OUTPUT_FILE" ]; then
        mv "$OUTPUT_FILE" "$ITER_DIR/selfplay_games.jsonl"
    else
        echo "Error: Output file not found: $OUTPUT_FILE"
        exit 1
    fi
    
    # Convert to VERL format
    python convert_selfplay_to_verl.py \
        --input "$ITER_DIR/selfplay_games.jsonl" \
        --output "$ITER_DIR/train_full.parquet"
    
    # Create train/val split
    python -c "
import pandas as pd
df = pd.read_parquet('$ITER_DIR/train_full.parquet')
split_idx = int(len(df) * 0.9)
train_df = df[:split_idx]
val_df = df[split_idx:]
train_df.to_parquet('$ITER_DIR/train.parquet', index=False)
val_df.to_parquet('$ITER_DIR/val.parquet', index=False)
print(f'Iteration $iter: {len(train_df)} train, {len(val_df)} val instances')
"
    
    # Step 2: Train on the generated data
    echo "Training model for iteration $iter..."
    cd $PROJECT_DIR/verl
    
    CHECKPOINT_DIR="$ITER_DIR/checkpoint"
    
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
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name='matching_game_selfplay_iterative' \
        trainer.experiment_name="qwen2.5-7B-selfplay-iter${iter}" \
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
    
    echo "Iteration $iter complete. Model saved to $CURRENT_MODEL"
    echo ""
done

echo "=========================================="
echo "Self-play training complete!"
echo "Final model: $CURRENT_MODEL"
echo "All data and checkpoints in: $BASE_DATA_DIR"
echo "=========================================="