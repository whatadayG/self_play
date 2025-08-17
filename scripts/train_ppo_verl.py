import os
from omegaconf import OmegaConf

# Entrypoint that defers to VERL's main_ppo with overrides
# Usage example:
#   python -m RL-matching.train_ppo_verl \
#       actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#       critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#       data.train_batch_size=8


def main():
	# Base minimal config for local single-GPU PPO with HF rollout
	cfg = {
		"ray_init": {
			"num_cpus": None,  # use all available CPUs
			"num_gpus": 2,  # we're using 2 GPUs
		},
		"global_profiler": {
			"tool": None,
			"steps": None,
		},
		"algorithm": {
			"adv_estimator": "gae",
			"gamma": 1.0,
			"lam": 0.95,
			"use_kl_in_reward": False,
		},
		"data": {
			# We won't use parquet. We'll synthesize prompts from our env in a custom dataloader built-in VERL.
			# For bootstrapping PPO in VERL, set a tiny batch.
			"train_batch_size": 8,
			"max_prompt_length": 1024,
			"max_response_length": 256,
			"filter_overlong_prompts": False,
			"truncation": "error",
			"return_raw_chat": True,
			"reward_fn_key": "reward_model",  # key to use for storing rewards
			"train_files": None,  # will be set from CLI
			"val_files": [],  # empty list for no validation set
			"sampler": None,  # use default sampler
		},
		"actor_rollout_ref": {
			"model": {
				"path": os.environ.get("POLICY_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
				"enable_gradient_checkpointing": True,
				"use_remove_padding": True,
			},
			"actor": {
				"strategy": "fsdp",  # distributed strategy
				"optim": {"lr": 1e-6},
				"ppo_mini_batch_size": 8,
				"ppo_micro_batch_size_per_gpu": 2,
				"use_kl_loss": False,
			},
			"rollout": {
				"name": "hf",  # use built-in HF sampler
				"mode": "sync",  # synchronous rollout
				"prompt_length": 1024,
				"response_length": 256,
				"temperature": 1.0,
				"top_k": -1,
				"top_p": 1.0,
				"do_sample": True,
				"n": 1,
			},
			"ref": {  # reference policy mirrors actor by default
				"model": {"path": os.environ.get("REF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")},
			},
		},
		"critic": {
			"strategy": "fsdp",  # distributed strategy
			"model": {
				"path": os.environ.get("CRITIC_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
				"enable_gradient_checkpointing": True,
				"use_remove_padding": True,
			},
			"optim": {"lr": 1e-5},
			"ppo_micro_batch_size_per_gpu": 2,
		},
		"reward_model": {
			"enable": False,
			"sandbox_fusion": {
				"url": None,
				"max_concurrent": 64,
				"memory_limit_mb": 1024,
			},
		},
		"trainer": {
			"project_name": "verl_matching",
			"experiment_name": "ppo_local",
			"logger": ["console"],
			"n_gpus_per_node": 1,
			"nnodes": 1,
			"save_freq": 1,
			"test_freq": 1,
			"total_epochs": 1,
			"use_legacy_worker_impl": "auto",
		},
	}

	# Allow user to override via CLI Hydra-style key=val pairs
	import sys
	if len(sys.argv) > 1:
		overrides = OmegaConf.from_cli(sys.argv[1:])
		cfg = OmegaConf.merge(cfg, overrides)
	cfg = OmegaConf.create(cfg)

	# Register custom reward manager by name: matching
	# Users can select it at CLI: reward_manager.name=matching
	cfg.setdefault("reward_manager", {})
	cfg["reward_manager"]["name"] = cfg["reward_manager"].get("name", "matching")
	cfg["reward_manager"]["num_examine"] = cfg["reward_manager"].get("num_examine", 0)

	# Kick off VERL PPO
	from verl.trainer.main_ppo import main as verl_main_ppo
	return verl_main_ppo(cfg)


if __name__ == "__main__":
	main() 