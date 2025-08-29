#!/bin/bash

# Minimal test for dialop self-play GRPO that actually reaches gradient update

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="${HOME}/data/dialop_selfplay_init"

# Use only GPU 3
export CUDA_VISIBLE_DEVICES="3"
export VERL_APPLY_SELFPLAY_PATCH="1"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/dialop:$PROJECT_DIR/verl:$PYTHONPATH"

# Output - use local directory to avoid HDFS issues
OUTPUT_DIR="$PROJECT_DIR/test_output/dialop_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Minimal Dialop GRPO Test"
echo "========================"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output: $OUTPUT_DIR"

# Generate minimal data if needed
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Generating test data..."
    source test_venv/bin/activate
    python verl/examples/data_preprocess/dialop_game_init.py \
        --output_dir "$DATA_DIR" \
        --num_train 2 \
        --num_test 1
fi

# Activate venv
source test_venv/bin/activate

cd verl

# Run with absolutely minimal config to reach gradient update
python -m verl.trainer.main_ppo \
    data.train_files="[\"$DATA_DIR/train.parquet\"]" \
    data.val_files="[\"$DATA_DIR/test.parquet\"]" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.return_raw_chat=true \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=false \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.project_name="dialop_test" \
    trainer.experiment_name="minimal_gradient_test" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger='["console"]' \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
    actor_rollout_ref.hybrid_engine=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/verl/examples/sglang_multiturn/config/interaction_config/dialop_selfplay_interaction_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=dialop_selfplay \
    critic.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
    critic.ppo_micro_batch_size_per_gpu=1 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "Training complete. Check $OUTPUT_DIR/training.log for details."