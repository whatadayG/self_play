"""
Unified player module providing backward-compatible imports.
This module allows gradual migration from old player classes to new unified ones.
"""

# Import all unified players
from .base_player import BaseModelPlayer, ModelConfig, SGLangConfig, VLLMConfig, HFConfig
from .openai_model_player import OpenAIModelPlayer, OpenAICostTracker
from .sglang_model_player_unified import SGLangModelPlayer
from .hf_model_player_unified import HFModelPlayer  
from .vllm_model_player import VLLMModelPlayer

# For backward compatibility, create wrapper classes that match old interfaces
from .players import LLMPlayer, HumanPlayer, DryRunPlayer

class LLMPlayerCompat(OpenAIModelPlayer):
    """Backward-compatible wrapper for LLMPlayer -> OpenAIModelPlayer."""
    
    def __init__(self, prompt, role, console, model_kwargs=None,
                 prefix="\nYou:", optional=None, strategy=None):
        """Initialize with LLMPlayer interface."""
        # Extract model from model_kwargs if provided
        model = "gpt-4.1"  # Default from original
        if model_kwargs and "model" in model_kwargs:
            model = model_kwargs["model"]
        
        # Convert to new interface
        super().__init__(
            prompt=prompt,
            role=role,
            console=console,
            model_path=model,
            prefix=prefix,
            optional=optional,
            enable_cost_tracking=False,  # Original didn't have built-in tracking
            enable_temporal=False,  # Disable by default
            enable_personas=False,  # Disable by default
            **model_kwargs or {}
        )
        
        # Store strategy for compatibility
        self.strategy = strategy
        
        # Compatibility attributes
        self.api_tracker = None  # Original expected external tracker
        self.model = model
        self.model_format = {"system": prompt, "user": "", "assistant": ""}


# Create compatibility mappings
__all__ = [
    # New unified classes
    "BaseModelPlayer",
    "OpenAIModelPlayer", 
    "SGLangModelPlayer",
    "HFModelPlayer",
    "VLLMModelPlayer",
    # Configurations
    "ModelConfig",
    "SGLangConfig",
    "VLLMConfig", 
    "HFConfig",
    "OpenAICostTracker",
    # Backward compatibility
    "LLMPlayer",  # Maps to LLMPlayerCompat
    "HumanPlayer",  # Keep original
    "DryRunPlayer",  # Keep original
]


def get_player(player_type: str, *args, **kwargs):
    """Factory function to get appropriate player instance.
    
    Args:
        player_type: One of 'openai', 'sglang', 'hf', 'vllm', 'human', 'dryrun'
        *args, **kwargs: Arguments to pass to player constructor
        
    Returns:
        Player instance
    """
    player_map = {
        'openai': OpenAIModelPlayer,
        'sglang': SGLangModelPlayer,
        'hf': HFModelPlayer,
        'vllm': VLLMModelPlayer,
        'human': HumanPlayer,
        'dryrun': DryRunPlayer,
        'llm': LLMPlayerCompat,  # For backward compatibility
    }
    
    if player_type not in player_map:
        raise ValueError(f"Unknown player type: {player_type}. Choose from {list(player_map.keys())}")
    
    return player_map[player_type](*args, **kwargs)