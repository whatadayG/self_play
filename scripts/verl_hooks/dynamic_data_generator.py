"""Dynamic data generator for VERL that creates new gamestates on-the-fly."""

import json
import numpy as np
from typing import Any, Dict, List, Optional
from dialop.envs.optimization import OptimizationEnv


class MatchingDataGenerator:
    """Generates fresh matching gamestates dynamically during training.
    
    This generator creates new random similarity tables and gamestates
    for each batch, ensuring the model sees infinite diversity.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.env = OptimizationEnv()
        
    def generate_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate a batch of fresh training examples with new gamestates.
        
        Returns:
            List of dictionaries, each containing:
            - messages: Initial conversation setup
            - reward_model: Ground truth gamestate for reward calculation
            - data_source: Dataset identifier
        """
        batch = []
        
        for _ in range(batch_size):
            # Generate a completely new gamestate
            obss = self.env.reset(game_state=None)  # None triggers random generation
            
            # Extract the generated gamestate
            gamestate = {
                "table": self.env.game.table.tolist() if hasattr(self.env.game.table, 'tolist') else self.env.game.table,
                "mask1": self.env.game.masks[0].tolist() if hasattr(self.env.game.masks[0], 'tolist') else self.env.game.masks[0],
                "mask2": self.env.game.masks[1].tolist() if hasattr(self.env.game.masks[1], 'tolist') else self.env.game.masks[1],
                "best_assignment_reward": float(self.env.game.best_assignment_reward),
                "scale1": float(self.env.game.scales[0]),
                "scale2": float(self.env.game.scales[1]),
                "action_log": []
            }
            
            # Create training example
            messages = [
                {"content": "You are a helpful assistant for reviewer-paper matching.", "role": "system"},
                # Add the initial observation as the user message
                {"content": obss["player-1"], "role": "user"}
            ]
            
            entry = {
                "messages": messages,
                "reward_model": {"ground_truth": json.dumps(gamestate)},
                "data_source": "matching"
            }
            
            batch.append(entry)
            
        return batch
    
    def __call__(self, batch_size: int) -> List[Dict[str, Any]]:
        """Make the generator callable for VERL's data pipeline."""
        return self.generate_batch(batch_size)