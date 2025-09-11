"""
OpenAI model player using the OpenAI API.
Includes internal cost tracking functionality.
"""

import os
import json
import random
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base_player import BaseModelPlayer, ModelConfig
from rich.console import Console


@dataclass
class OpenAIConfig(ModelConfig):
    """Configuration specific to OpenAI model player."""
    model: str = "gpt-4.1"  # Match original LLMPlayer default
    organization: Optional[str] = None
    api_key_path: str = "/home/nickatomlin/georgiazhou/dialop/dialop/.api_key"
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # Override to match original LLMPlayer stop tokens
    stop_sequences: List[str] = field(default_factory=lambda: ["User", "Agent", "You", "\n"])


class OpenAICostTracker:
    """Internal cost tracking for OpenAI API calls."""
    
    # Approximate costs per 1K tokens (as of 2024)
    COST_PER_1K = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4.1": {"input": 0.01, "output": 0.03},  # Custom model name
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(self, log_file: Optional[Path] = None):
        """Initialize cost tracker.
        
        Args:
            log_file: Optional path to log file for API calls
        """
        self.total_cost = 0.0
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.log_file = log_file
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a single API call.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        # Find the base model for cost calculation
        base_model = None
        for key in self.COST_PER_1K:
            if key in model:
                base_model = key
                break
        
        if not base_model:
            # Default to GPT-4 costs if unknown
            base_model = "gpt-4"
        
        input_cost = (input_tokens / 1000) * self.COST_PER_1K[base_model]["input"]
        output_cost = (output_tokens / 1000) * self.COST_PER_1K[base_model]["output"]
        
        return input_cost + output_cost
    
    def log_api_call(self, model: str, input_text: str, output_text: str,
                     input_tokens: int, output_tokens: int, project_id: str) -> None:
        """Log an API call with cost tracking.
        
        Args:
            model: Model used
            input_text: Input prompt
            output_text: Generated response
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            project_id: Project/role identifier
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Update totals
        self.call_count += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        
        # Log to file if configured
        if self.log_file:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "project_id": project_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "total_cost": self.total_cost,
                "call_number": self.call_count,
                "input_text": input_text[:500],  # Truncate for logging
                "output_text": output_text[:500]
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of API usage and costs.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_cost_per_call": round(self.total_cost / max(1, self.call_count), 4)
        }


class OpenAIModelPlayer(BaseModelPlayer):
    """Player that uses OpenAI API for generation."""
    
    def __init__(
        self,
        prompt: str,
        role: str, 
        console: Console,
        model_path: str = "gpt-4-turbo-preview",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        config: Optional[OpenAIConfig] = None,
        enable_cost_tracking: bool = True,
        cost_log_file: Optional[Path] = None,
        **kwargs
    ):
        """Initialize OpenAI model player.
        
        Args:
            prompt: Initial system prompt
            role: Role of the player
            console: Rich console for output
            model_path: OpenAI model name
            prefix: Prefix for responses
            optional: Optional context
            config: OpenAI configuration
            enable_cost_tracking: Whether to track API costs
            cost_log_file: Optional path to cost log file
            **kwargs: Additional arguments
        """
        # Initialize config first
        self.config = config or OpenAIConfig(model=model_path)
        self.client = None
        self.cost_tracker = None
        
        # Initialize cost tracking if enabled
        if enable_cost_tracking:
            self.cost_tracker = OpenAICostTracker(log_file=cost_log_file)
        
        # Support for cleaned-up temporal/persona features
        self.enable_temporal = kwargs.get("enable_temporal", False)
        self.enable_personas = kwargs.get("enable_personas", False)
        
        # Initialize model_format for compatibility with evaluate_opt.py
        # This matches HF/SGLang players, NOT original LLMPlayer
        self.model_format = [{"role": "system", "content": prompt}]
        
        # For backward compatibility detection in evaluate_opt.py
        self.model_kwargs = {"model": self.config.model}
        
        # Call parent init
        super().__init__(prompt, role, console, model_path, prefix, optional, self.config, **kwargs)
    
    def _setup_model(self) -> None:
        """Set up OpenAI client."""
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIModelPlayer")
        
        # Try to load API key from file or environment
        api_key = None
        organization = self.config.organization
        
        # Try loading from file first
        if Path(self.config.api_key_path).exists():
            try:
                with open(self.config.api_key_path) as f:
                    creds = json.load(f)
                    api_key = creds.get("api_key")
                    organization = creds.get("organization", organization)
                self.console.print("[green]Loaded API credentials from file[/green]")
            except Exception as e:
                self.console.print(f"[yellow]Could not load API key file: {e}[/yellow]")
        
        # Fall back to environment variable
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.console.print("[green]Using API key from environment[/green]")
        
        if not api_key:
            raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY or provide .api_key file")
        
        self.client = OpenAI(api_key=api_key, organization=organization)
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count for OpenAI models.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimation: ~4 characters per token
        # For production, use tiktoken library
        return len(text) // 4
    
    def _generate_text(self, prompt: str, **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using OpenAI API.
        
        Args:
            prompt: The prompt to generate from
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Use model_format if it has been updated by client
        # Otherwise fall back to simple prompt
        if hasattr(self, 'model_format') and len(self.model_format) > 1:
            # Client is managing conversation history
            messages = self.model_format
        else:
            # Simple mode - just use prompt
            messages = [{"role": "system", "content": prompt}]
        
        # Build request parameters
        request_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", self.config.temperature),
            "max_tokens": gen_kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": gen_kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stop": self.config.stop_sequences[:-1],  # Exclude \n for OpenAI
        }
        
        # Add seed for variation
        if gen_kwargs.get("temperature", 0.7) > 1.5:  # High temp indicates vary=True
            request_params["seed"] = random.randint(1, 10000)
        
        # Make API call
        response = self.client.chat.completions.create(**request_params)
        
        # Extract response details
        response_text = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # Track costs if enabled
        if self.cost_tracker:
            self.cost_tracker.log_api_call(
                model=self.config.model,
                input_text=prompt,
                output_text=response_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                project_id=self.role
            )
        
        # Check if we hit token limit
        if response.choices[0].finish_reason == "length":
            self.console.print("[yellow]Hit token limit, response may be truncated[/yellow]")
            if self.optional and not self.removed_optional:
                # Could retry with removed optional context
                self._remove_optional_context()
        
        return response_text, input_tokens, output_tokens
    
    def get_cost_summary(self) -> Optional[Dict[str, Any]]:
        """Get cost tracking summary.
        
        Returns:
            Cost summary dict or None if tracking disabled
        """
        if self.cost_tracker:
            return self.cost_tracker.get_summary()
        return None
    
    def respond(self, t: int = 0, max_len: int = 256, vary: bool = False,
                propose: bool = False, temporal_id: Optional[int] = None,
                strategy: Optional[str] = None) -> str:
        """Generate response with OpenAI-specific features.
        
        Maintains compatibility with original LLMPlayer interface while
        using the cleaner base class implementation.
        """
        # For now, ignore temporal features unless explicitly enabled
        if temporal_id is not None and not self.enable_temporal:
            temporal_id = None
        
        # Call parent implementation
        return super().respond(t, max_len, vary, propose, temporal_id, strategy)