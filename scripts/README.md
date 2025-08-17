# RL-matching + VERL PPO Quickstart

## Files
- `train_ppo_verl.py`: minimal launcher that calls `verl.trainer.main_ppo` with sane defaults. You can override any config key via CLI.
- `configs/ppo_verl.yaml`: sample single-GPU config. Edit or pass CLI overrides.
- `verl_hooks/matching_reward_manager.py`: custom reward manager registered as `matching`. It parses the response using your matching env and computes normalized reward.
- `rl/verl_env.py`: utilities to convert your optimization game state to a chat prompt for VERL.
- `dialop/*`: self-contained matching environment and assets.

## Launch (single GPU)
```bash
cd /home/nickatomlin/georgiazhou/new_dialop
# Install VERL requirements and transformers stack (one-time)
pip install -r verl/requirements.txt transformers accelerate datasets peft bitsandbytes omegaconf

# Choose models (small ones to start)
export POLICY_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export REF_MODEL=$POLICY_MODEL
export CRITIC_MODEL=$POLICY_MODEL

# Run with yaml
python RL-matching/train_ppo_verl.py @RL-matching/configs/ppo_verl.yaml \
  data.train_batch_size=8 trainer.total_epochs=1
```

## Decisions to make
- Model choice: start small (0.5B–1.5B) to validate loop; scale later.
- Rollout backend: default is `hf` local sampling. You can switch to `vllm` or `sglang` by changing `actor_rollout_ref.rollout.name`.
- KL control: add `actor_rollout_ref.actor.use_kl_loss=true` and tune `kl_loss_coef`, or set `algorithm.use_kl_in_reward=true` for in-reward penalty.
- Dataset source: current setup expects on-policy prompts. To initialize from recorded games, read `dialop/data/optimization.jsonl` and convert each to chat using `rl/verl_env.py`.
- Compute: 1 GPU for a smoke test; distributed configs follow VERL examples in `examples/ppo_trainer`.

## Notes
- Rewards are computed at the final token position using `score_norm` from the environment.
- Responses must include valid tags like `[message]` or `[propose]` as per your env; invalid parses get 0 reward. 