#!/usr/bin/env python3
"""
Test script to verify GRPO integration with custom rollout worker.
Runs minimal training (1 step) to check data formatting and flow.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent / "verl"))

# Set minimal configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use single GPU if available

def create_minimal_test_data():
    """Create minimal test data for quick verification."""
    import pandas as pd
    import json
    from dialop.envs.optimization import OptimizationEnv
    
    print("Creating minimal test data...")
    
    # Create just 2 game initializations (minimum for a batch)
    game_inits = []
    env = OptimizationEnv()
    
    for game_idx in range(1):  # Just 1 game
        obs = env.reset()
        game_state = {
            "tables": env.game.tables,
            "best_assignment_reward": env.game.best_assignment_reward,
            "seed": game_idx,
        }
        
        # Create entries for both players
        for player_idx in range(2):
            game_init = {
                "prompt": obs[env.players[player_idx]],
                "game_state": json.dumps(game_state),
                "player_index": player_idx,
                "game_id": game_idx,
            }
            game_inits.append(game_init)
    
    # Save to temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dialop_test_")
    df = pd.DataFrame(game_inits)
    
    train_path = Path(temp_dir) / "train.parquet"
    test_path = Path(temp_dir) / "test.parquet"
    
    df.to_parquet(train_path, index=False)
    df.to_parquet(test_path, index=False)  # Same data for test
    
    print(f"Created test data in: {temp_dir}")
    print(f"  Train: {train_path} ({len(df)} entries)")
    print(f"  Test: {test_path}")
    
    return temp_dir


def run_minimal_grpo_test():
    """Run minimal GRPO training to test integration."""
    
    # First create test data
    data_dir = create_minimal_test_data()
    
    # Import after paths are set
    from verl.trainer.main_ppo import main as ppo_main
    from omegaconf import DictConfig
    import hydra
    from hydra import compose, initialize_config_dir
    
    print("\nRunning minimal GRPO test...")
    
    # Create minimal config
    config = {
        # Data configuration
        "data": {
            "max_prompt_length": 2048,
            "max_response_length": 512,
            "train_batch_size": 2,  # Minimum batch size
            "val_batch_size": 2,
            "return_raw_chat": True,
            "train_files": [f"{data_dir}/train.parquet"],
            "val_files": [f"{data_dir}/test.parquet"],
        },
        
        # Algorithm configuration
        "algorithm": {
            "adv_estimator": "grpo",
            "norm_adv_by_std_in_grpo": True,
            "use_kl_in_reward": False,
        },
        
        # Actor/Rollout configuration
        "actor_rollout_ref": {
            "hybrid_engine": False,  # Disable for testing
            
            # Model configuration
            "model": {
                "path": "gpt2",  # Use small model for testing
                "use_remove_padding": False,
                "enable_gradient_checkpointing": False,
                "enable_activation_offloading": False,
                "override_config": {
                    "max_seq_len": 2048,
                }
            },
            
            # Actor configuration
            "actor": {
                "optim": {
                    "lr": 1e-6,
                },
                "ppo_mini_batch_size": 2,
                "ppo_micro_batch_size_per_gpu": 2,
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                }
            },
            
            # Rollout configuration - our custom worker
            "rollout": {
                "_target_": "verl.workers.rollout.dialop_selfplay_rollout.DialopSelfPlayRollout",
                "gpu_memory_utilization": 0.5,
                "n": 1,  # Single rollout
                "tensor_model_parallel_size": 1,
                "log_prob_micro_batch_size_per_gpu": 2,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 256,
                "response_length": 256,
            },
            
            # Reference configuration
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 2,
                "fsdp_config": {
                    "param_offload": False,
                }
            }
        },
        
        # Reward model configuration
        "reward_model": {
            "type": "dialop_selfplay"
        },
        
        # Trainer configuration
        "trainer": {
            "total_epochs": 1,  # Just 1 epoch
            "critic_warmup": 0,
            "logger": ["console"],
            "project_name": "dialop_test",
            "experiment_name": "minimal_integration_test",
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "save_freq": -1,  # Don't save
            "test_freq": -1,  # Don't test
            "default_hdfs_dir": tempfile.mkdtemp(prefix="dialop_output_"),
        },
        
        # Critic configuration (minimal)
        "critic": {
            "model": {
                "path": "gpt2",
                "override_config": {
                    "max_seq_len": 2048,
                }
            },
            "optim": {
                "lr": 1e-6,
            },
            "ppo_micro_batch_size_per_gpu": 2,
            "fsdp_config": {
                "param_offload": False,
                "optimizer_offload": False,
            }
        }
    }
    
    # Convert to DictConfig
    cfg = DictConfig(config)
    
    print("\nConfiguration:")
    print(f"  Model: {cfg.actor_rollout_ref.model.path}")
    print(f"  Batch size: {cfg.data.train_batch_size}")
    print(f"  Epochs: {cfg.trainer.total_epochs}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {cfg.trainer.default_hdfs_dir}")
    
    try:
        # Run the training
        print("\nStarting GRPO training...")
        ppo_main(cfg)
        
        print("\n✓ GRPO integration test completed successfully!")
        print("  The custom rollout worker is properly integrated with verl's GRPO implementation.")
        
    except Exception as e:
        print(f"\n✗ Error during GRPO test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


def test_rollout_data_format():
    """Test the data format produced by custom rollout."""
    print("\n=== Testing Rollout Data Format ===")
    
    try:
        from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout
        from verl.protocol import DataProto
        import torch
        import json
        import asyncio
        
        # Create mock config
        class MockConfig:
            max_model_len = 2048
            response_length = 256
            
        class MockProcessingClass:
            class MockTokenizer:
                def encode(self, text, **kwargs):
                    # Simple mock tokenization
                    return list(range(min(len(text) // 4, 100)))
            tokenizer = MockTokenizer()
        
        # Create rollout instance
        rollout = DialopSelfPlayRollout()
        rollout.config = MockConfig()
        rollout.processing_class = MockProcessingClass()
        rollout.sampling_params = {"temperature": 0.7}
        
        # Mock the engine call
        async def mock_engine_call(request, params):
            return {"text": "[message] Test response"}
        rollout._handle_engine_call = mock_engine_call
        
        # Create test data
        from dialop.envs.optimization import OptimizationEnv
        env = OptimizationEnv()
        obs = env.reset()
        
        test_data = DataProto(
            batch={"dummy": torch.tensor([1])},
            non_tensor_batch={
                "game_state": [json.dumps({
                    "tables": env.game.tables,
                    "best_assignment_reward": env.game.best_assignment_reward,
                    "seed": 0,
                })],
                "player_index": [0],
            }
        )
        
        print("Running rollout.generate_sequences...")
        
        # Test generate_sequences
        async def test_async():
            result = await rollout.generate_sequences(test_data)
            return result
            
        result = asyncio.run(test_async())
        
        print("\nData format verification:")
        print(f"  Result type: {type(result)}")
        print(f"  Has batch: {'batch' in result.__dict__ if hasattr(result, '__dict__') else 'N/A'}")
        
        if hasattr(result, 'batch'):
            print(f"  Batch keys: {list(result.batch.keys())}")
            for key, value in result.batch.items():
                print(f"    {key}: shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
                
        if hasattr(result, 'non_tensor_batch'):
            print(f"  Non-tensor batch keys: {list(result.non_tensor_batch.keys())}")
            
        print("\n✓ Rollout data format test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Rollout data format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing GRPO integration with custom dialop rollout...")
    
    # Add dialop to path
    sys.path.insert(0, str(Path(__file__).parent / "dialop"))
    
    # Test 1: Data format
    format_ok = test_rollout_data_format()
    
    # Test 2: Full integration (only if format test passes)
    if format_ok:
        integration_ok = run_minimal_grpo_test()
    else:
        print("\nSkipping integration test due to format test failure")
        integration_ok = False
        
    if format_ok and integration_ok:
        print("\n✓ All integration tests passed!")
        print("\nYou can now run full training with:")
        print("  bash verl/examples/sglang_multiturn/dialop_optimization/run_dialop_selfplay_custom.sh")
    else:
        print("\n✗ Some integration tests failed. Check the errors above.")