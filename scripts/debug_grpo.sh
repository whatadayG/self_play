#!/bin/bash
set -x  # Print commands as they execute

# Set environment variables
export PYTHONPATH="/home/nickatomlin/georgiazhou/new_dialop/verl:/home/nickatomlin/georgiazhou/new_dialop/RL-matching"
export CUDA_VISIBLE_DEVICES=4,5

# Run the command
conda run -n verl311 python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files='["/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/matching.parquet"]' \
  data.val_files='["/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/matching.parquet"]' \
  data.prompt_key=messages \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path=/home/nickatomlin/georgiazhou/new_dialop/RL-matching/configs/matching_interaction.yaml \
  actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.response_length=128 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_num_batched_tokens=512 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  critic.model.path=Qwen/Qwen2.5-7B-Instruct \
  critic.model.enable_gradient_checkpointing=true \
  critic.model.use_remove_padding=true \
  critic.ppo_micro_batch_size_per_gpu=1 \
  reward_model.enable=false \
  trainer.project_name=verl_matching \
  trainer.experiment_name=grpo_mem_stage1_flashinfer_glibc_messages \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=0 \
  trainer.test_freq=1 \
  trainer.total_epochs=0 \
  trainer.val_only=true