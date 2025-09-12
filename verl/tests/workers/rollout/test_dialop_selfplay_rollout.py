#!/usr/bin/env python3
"""
Unit tests for DialopSelfPlayRollout.

Tests the custom rollout worker independently of the full training pipeline.
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import torch

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
dialop_root = project_root.parent.parent.parent / "dialop"
sys.path.insert(0, str(dialop_root))

from dialop.envs.optimization import OptimizationEnv
from verl.protocol import DataProto
from verl.workers.rollout.dialop_selfplay_rollout import DialopSelfPlayRollout


class MockConfig:
    """Mock configuration for testing."""
    max_model_len = 2048
    response_length = 512
    
    
class MockProcessingClass:
    """Mock processing class with tokenizer."""
    class MockTokenizer:
        def encode(self, text, add_special_tokens=True, truncation=True, max_length=2048):
            # Simple mock: return list of integers based on text length
            return list(range(min(len(text) // 10, max_length)))
            
    tokenizer = MockTokenizer()


class TestDialopSelfPlayRollout(unittest.TestCase):
    """Test cases for DialopSelfPlayRollout."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MockConfig()
        self.sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512
        }
        
    def create_mock_rollout(self):
        """Create a mock DialopSelfPlayRollout instance."""
        # Mock the parent class initialization
        with patch('verl.workers.rollout.dialop_selfplay_rollout.SGLangRollout.__init__'):
            rollout = DialopSelfPlayRollout()
            rollout.config = self.config
            rollout.sampling_params = self.sampling_params
            rollout.processing_class = MockProcessingClass()
            
            # Mock the engine call
            async def mock_engine_call(request, params):
                return {"text": "[message] Mock response from model"}
                
            rollout._handle_engine_call = AsyncMock(side_effect=mock_engine_call)
            
        return rollout
        
    def test_initialization(self):
        """Test rollout initialization."""
        rollout = self.create_mock_rollout()
        self.assertEqual(rollout.env_class, OptimizationEnv)
        
    def test_format_player_perspective(self):
        """Test formatting game results for player perspective."""
        rollout = self.create_mock_rollout()
        
        # Create mock game result
        game_result = {
            "conversation": [
                {"player": 0, "player_name": "player-1", "message": "[message] Hello", "turn": 0},
                {"player": 1, "player_name": "player-2", "message": "[message] Hi there", "turn": 1},
                {"player": 0, "player_name": "player-1", "message": "[propose] My proposal", "turn": 2},
                {"player": 1, "player_name": "player-2", "message": "[accept]", "turn": 3},
            ],
            "reward": 500,
            "normalized_reward": 0.75,
            "game_info": {
                "num_messages": 4,
                "completed": True,
                "best_possible_reward": 667,
                "turn_count": 4
            }
        }
        
        # Format for player 0
        result = rollout._format_player_perspective(game_result, 0)
        
        # Check structure
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertIn("rewards", result)
        self.assertIn("non_tensor_batch", result)
        
        # Check reward assignment (should be on last token)
        rewards = result["rewards"].tolist()
        self.assertEqual(rewards[-1], 0.75)
        self.assertEqual(sum(rewards[:-1]), 0.0)
        
        # Check non-tensor data
        self.assertEqual(result["non_tensor_batch"]["reward_model"]["normalized_reward"], 0.75)
        self.assertEqual(result["non_tensor_batch"]["player_index"], 0)
        
    def test_create_empty_result(self):
        """Test creation of empty result for error cases."""
        rollout = self.create_mock_rollout()
        result = rollout._create_empty_result()
        
        self.assertEqual(len(result["input_ids"]), 0)
        self.assertEqual(len(result["attention_mask"]), 0)
        self.assertEqual(len(result["rewards"]), 0)
        self.assertTrue(result["non_tensor_batch"]["game_info"]["error"])
        
    async def test_run_selfplay_game(self):
        """Test running a complete self-play game."""
        rollout = self.create_mock_rollout()
        
        # Mock the player response generation
        response_sequence = [
            "[message] Hello partner",
            "[message] Hi, let's work together",
            "[propose] Proposal:\nBLEU: Ava Li\nElectra: Daniel Nguyen",
            "[accept]"
        ]
        response_iter = iter(response_sequence)
        
        async def mock_generate_response(messages, obs, player):
            return next(response_iter)
            
        rollout._generate_player_response = mock_generate_response
        
        # Run game
        result = await rollout._run_selfplay_game(None)
        
        # Check result structure
        self.assertIn("conversation", result)
        self.assertIn("reward", result)
        self.assertIn("normalized_reward", result)
        self.assertIn("game_info", result)
        
        # Check conversation
        conv = result["conversation"]
        self.assertGreater(len(conv), 0)
        self.assertEqual(conv[0]["message"], "[message] Hello partner")
        
    async def test_generate_sequences(self):
        """Test the main generate_sequences method."""
        rollout = self.create_mock_rollout()
        
        # Mock run_selfplay_game
        async def mock_run_game(game_state):
            return {
                "conversation": [
                    {"player": 0, "message": "[message] Test", "turn": 0}
                ],
                "reward": 100,
                "normalized_reward": 0.5,
                "game_info": {"completed": True}
            }
            
        rollout._run_selfplay_game = mock_run_game
        
        # Create input data
        input_data = DataProto(
            batch={"dummy": torch.tensor([1, 2])},
            non_tensor_batch={
                "game_state": [None, None],
                "player_index": [0, 1]
            }
        )
        
        # Run generation
        output = await rollout.generate_sequences(input_data)
        
        # Check output
        self.assertIsInstance(output, DataProto)
        self.assertIn("input_ids", output.batch)
        self.assertIn("rewards", output.batch)
        
    def test_messages_to_text(self):
        """Test conversion of messages to text format."""
        rollout = self.create_mock_rollout()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        
        text = rollout._messages_to_text(messages)
        self.assertIn("User: Hello", text)
        self.assertIn("Assistant: Hi there", text)
        

class TestIntegration(unittest.TestCase):
    """Integration tests with actual dialop environment."""
    
    async def test_real_game_flow(self):
        """Test with real dialop environment."""
        # Create actual environment
        env = OptimizationEnv()
        initial_obs = env.reset()
        
        # Verify environment works
        self.assertIn("turn_player", initial_obs)
        self.assertIn("done", initial_obs)
        self.assertFalse(initial_obs["done"])
        
        # Test a simple interaction
        obs, error = env.step("[message] Hello")
        self.assertIsNone(error)
        self.assertIn("turn_player", obs)
        

def run_tests():
    """Run all tests and save results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDialopSelfPlayRollout))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Save results to file
    output_file = Path(__file__).parent / "test_dialop_selfplay_rollout_results.txt"
    with open(output_file, "w") as f:
        f.write(f"Test Results for DialopSelfPlayRollout\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Tests run: {result.testsRun}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Success: {result.wasSuccessful()}\n\n")
        
        if result.failures:
            f.write("Failures:\n")
            for test, trace in result.failures:
                f.write(f"{test}: {trace}\n")
                
        if result.errors:
            f.write("\nErrors:\n")
            for test, trace in result.errors:
                f.write(f"{test}: {trace}\n")
                
    print(f"\nTest results saved to: {output_file}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run async tests
    success = run_tests()
    sys.exit(0 if success else 1)