#!/usr/bin/env bash
set -euo pipefail

cd /home/nickatomlin/georgiazhou/new_dialop

# Make RL-matching modules importable
export PYTHONPATH="${PYTHONPATH:-}":/home/nickatomlin/georgiazhou/new_dialop/verl:/home/nickatomlin/georgiazhou/new_dialop/RL-matching

# Use GPUs 4 and 5
export CUDA_VISIBLE_DEVICES=4,5

# Disable flash attention check
export TRANSFORMERS_VERBOSITY=error

# Models
export POLICY_MODEL=${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}

# Run GRPO training directly with verl.trainer.main_ppo
conda run -n verl311 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    data.train_files='["/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/matching.parquet"]' \
    data.val_files='["/home/nickatomlin/georgiazhou/new_dialop/RL-matching/data/matching.parquet"]' \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$POLICY_MODEL \
    actor_rollout_ref.model.lora_rank=34 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj"]' \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=kl \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=/home/nickatomlin/georgiazhou/new_dialop/RL-matching/configs/matching_interaction.yaml \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    critic.model.path=$POLICY_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=true \
    critic.model.lora_rank=34 \
    critic.model.lora_alpha=64 \
    critic.model.target_modules='["q_proj","k_proj","v_proj","o_proj"]' \
    critic.ppo_micro_batch_size_per_gpu=8 \
    reward_model.enable=false \
    trainer.project_name=verl_matching \
    trainer.experiment_name=grpo_multiturn_lora_gpu45_official \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.critic_warmup=0