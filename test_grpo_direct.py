#!/usr/bin/env python3
"""
Direct test of GRPO integration bypassing Hydra config issues.
"""

import os
import sys
from pathlib import Path

# Set up paths
sys.path.insert(0, str(Path(__file__).parent / "verl"))
sys.path.insert(0, str(Path(__file__).parent / "dialop"))

# Set GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Import after paths are set
from omegaconf import DictConfig
from verl.trainer.main_ppo import main as ppo_main

def test_minimal_grpo():
    """Run minimal GRPO test with direct config."""
    
    data_dir = os.path.expanduser("~/data/dialop_selfplay_init")
    
    # Create config directly
    cfg = DictConfig({
        "data": {
            "max_prompt_length": 999999,  # Disable filtering for now
            "max_response_length": 512,
            "train_batch_size": 2,
            "val_batch_size": 2,
            "return_raw_chat": True,
            "reward_fn_key": "rewards",
            "prompt_key": "messages",  # Tell it which field contains the chat messages
            "train_files": [f"{data_dir}/train_chat_format.parquet"],
            "val_files": [f"{data_dir}/test_chat_format.parquet"],
        },
        
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
        },
        
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {
                "path": "gpt2",
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "enable_activation_offloading": True,
                "enable_gradient_offloading": False,
                "custom_chat_template": "{% for message in messages %}{{ message.content }}\n{% endfor %}",
            },
            "actor": {
                "strategy": "fsdp",
                "optim": {"lr": 1e-6},
                "ppo_mini_batch_size": 2,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": 0.01,
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                    "model_dtype": "bfloat16",
                },
            },
            "rollout": {
                "_target_": "verl.workers.rollout.dialop_selfplay_rollout.DialopSelfPlayRollout",
                "name": "sglang",  # Required
                "mode": "sync",  # Required
                "gpu_memory_utilization": 0.5,
                "n": 1,
                "tensor_model_parallel_size": 1,
                "log_prob_micro_batch_size_per_gpu": 1,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": -1,
                "prompt_length": 2048,
                "response_length": 256,
                "max_new_tokens": 256,
                "dtype": "bfloat16",
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": False},
            },
        },
        
        "reward_model": {
            "enable": False,  # We use custom reward manager, not model-based
            "reward_manager": "dialop_selfplay", 
            "strategy": "dataproto",
        },
        
        "trainer": {
            "total_epochs": 1,
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "dialop_test",
            "experiment_name": "minimal_integration",
            "n_gpus_per_node": 2,
            "nnodes": 1,
            "save_freq": -1,
            "test_freq": -1,
            "default_hdfs_dir": "/tmp/dialop_test_output",
        },
        
        "critic": {
            "strategy": "fsdp",
            "model": {"path": "gpt2"},
            "optim": {"lr": 1e-6},
            "ppo_micro_batch_size_per_gpu": 1,
            "fsdp_config": {
                "param_offload": False,
                "optimizer_offload": False,
            },
        },
        
        "ray_init": {
            "num_cpus": None,
            "num_gpus": None,
            "address": None,
        },
    })
    
    print("Starting minimal GRPO training...")
    print(f"Train data: {cfg.data.train_files[0]}")
    print(f"Test data: {cfg.data.val_files[0]}")
    print(f"Model: {cfg.actor_rollout_ref.model.path}")
    print(f"Rollout: {cfg.actor_rollout_ref.rollout._target_}")
    print(f"Batch size: {cfg.data.train_batch_size}")
    print(f"GPUs: {cfg.trainer.n_gpus_per_node}")
    
    try:
        ppo_main(cfg)
        print("\n✓ GRPO integration test completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_minimal_grpo()