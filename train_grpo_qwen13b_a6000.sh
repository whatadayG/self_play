#!/bin/bash

# Training script for dialop self-play GRPO with Qwen2.5-13B fine-tune
# Optimized for 4x NVIDIA RTX A6000 (48GB each)

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${PROJECT_DIR}/data"
MODEL_PATH="${PROJECT_DIR}/old/save_points/global_step_1000_merged"

# Use GPUs 4,5,6,7
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/dialop:$PROJECT_DIR/verl:$PYTHONPATH"

# Output directory with timestamp
OUTPUT_DIR="$PROJECT_DIR/test_output/qwen13b_grpo_a6000_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Dialop Self-Play GRPO Training - Qwen2.5-13B Fine-tune"
echo "======================================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES (4x NVIDIA RTX A6000)"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_DIR/train.parquet"
echo ""

# Check data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Error: Training data not found at $DATA_DIR/train.parquet"
    exit 1
fi

# Check model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Copy config to output directory for reference
cp "$0" "$OUTPUT_DIR/training_script.sh"

# Activate test venv
source test_venv/bin/activate

cd verl

# Create config file with realistic parameters for 13B model on 4xA6000
CONFIG_PATH="$OUTPUT_DIR/training_config.yaml"
cat > "$CONFIG_PATH" << 'EOF'
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

# Data configuration
data:
  max_prompt_length: 2048  # Dialop prompts can be long
  max_response_length: 1024  # Allow room for complete negotiations
  train_batch_size: 32  # 8 per GPU for 13B model
  val_batch_size: 16
  return_raw_chat: True
  train_files: ["${data.train_path}"]
  val_files: ["${data.val_path}"]
  reward_fn_key: reward_model

# Algorithm configuration
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: True
  use_kl_in_reward: False
  adv_clip: 5.0  # Clip advantages for stability

# Actor/Rollout/Reference configuration
actor_rollout_ref:
  hybrid_engine: True
  
  # Model configuration
  model:
    path: ${model.path}
    use_remove_padding: True
    enable_gradient_checkpointing: True  # Essential for 13B model
    enable_activation_offloading: False  # Keep off for A6000s
    trust_remote_code: True
    
  # Actor training configuration
  actor:
    optim:
      lr: 5e-7  # Conservative LR for fine-tuned model
      weight_decay: 0.01
      warmup_steps: 50
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 2  # Conservative for 13B model
    use_kl_loss: True
    kl_loss_coef: 0.01
    kl_loss_type: low_var_kl
    entropy_coeff: 0.01
    gradient_accumulation_steps: 4  # Effective batch = 2 * 4 * 4 = 32
    fsdp_config:
      param_offload: False  # A6000s have enough memory
      optimizer_offload: False
      model_dtype: bfloat16
      sharding_strategy: FULL_SHARD
  
  # Rollout configuration
  rollout:
    _target_: verl.workers.rollout.dialop_selfplay_rollout.DialopSelfPlayRollout
    name: sglang
    
    # Memory optimization for 13B model
    gpu_memory_utilization: 0.85  # Use most of the 48GB
    tensor_model_parallel_size: 1  # Keep at 1 for FSDP efficiency
    pipeline_model_parallel_size: 1
    log_prob_micro_batch_size_per_gpu: 4
    
    # Generation parameters
    temperature: 0.7
    top_p: 0.9
    max_new_tokens: 512
    repetition_penalty: 1.02  # Slight penalty to reduce repetition
    
    # SGLang server optimization
    mem_fraction_static: 0.85
    dtype: float16  # Use fp16 for inference
    disable_flashinfer: False
    enable_flashinfer: True
    tokenizer_mode: auto
  
  # Reference model configuration
  ref:
    log_prob_micro_batch_size_per_gpu: 4
    fsdp_config:
      param_offload: False
      model_dtype: bfloat16

# Reward configuration
reward_model:
  reward_manager: dialop_selfplay
  enable: False
  micro_batch_size_per_gpu: 4

# Critic configuration
critic:
  model:
    path: ${model.path}
    enable_gradient_checkpointing: True
    trust_remote_code: True
  ppo_micro_batch_size_per_gpu: 2
  optim:
    lr: 1e-6
    weight_decay: 0.01
  fsdp_config:
    param_offload: False
    optimizer_offload: False
    model_dtype: bfloat16

# Trainer configuration
trainer:
  critic_warmup: 100  # Warmup critic for stability
  logger: ["console", "wandb"]
  project_name: 'dialop_selfplay_qwen13b'
  experiment_name: 'qwen13b_grpo_4xa6000'
  n_gpus_per_node: 4
  nnodes: 1
  save_freq: 100  # Save every 100 steps
  test_freq: 50   # Test every 50 steps
  val_before_train: True
  total_epochs: 5  # Reasonable for fine-tuning
  grad_clip: 1.0  # Gradient clipping for stability
  
  # Ray configuration for 4 GPUs
  ray:
    num_gpus: 4
    num_cpus: 32  # Adjust based on system
    object_store_memory: 20_000_000_000  # 20GB object store
EOF

# Run training
echo "Starting training with config at $CONFIG_PATH"
echo ""

python -m verl.trainer.main_ppo \
    --config-path="$OUTPUT_DIR" \
    --config-name="training_config" \
    model.path="$MODEL_PATH" \
    data.train_path="$DATA_DIR/train.parquet" \
    data.val_path="$DATA_DIR/train.parquet" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.experiment_name="qwen13b_grpo_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete!"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $OUTPUT_DIR/training.log"

# Check for successful completion
if grep -q "Training completed" "$OUTPUT_DIR/training.log"; then
    echo "✓ Training completed successfully"
    
    # Show final metrics
    echo ""
    echo "Final metrics:"
    grep -E "(actor_loss|critic_loss|reward_mean|kl_divergence)" "$OUTPUT_DIR/training.log" | tail -20
else
    echo "⚠ Training may not have completed successfully"
    echo "Check the log for errors"
fi