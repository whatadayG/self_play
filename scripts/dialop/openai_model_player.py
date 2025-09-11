"""
OpenAI model player using the OpenAI API.
"""

import os
import json
from typing import Optional, Tuple, List, Dict
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base_player import BaseModelPlayer, OpenAIConfig
from rich.console import Console


class OpenAIModelPlayer(BaseModelPlayer):
    """Player that uses OpenAI API for generation."""
    
    def __init__(
        self,
        system_prompt: str,
        role: str,
        console: Console,
        model_path: str = "gpt-4-turbo-preview",
        config: Optional[OpenAIConfig] = None,
    ):
        """Initialize OpenAI model player.
        
        Args:
            system_prompt: Initial system prompt that defines the player
            role: Role of the player
            console: Rich console for output
            model_path: OpenAI model name
            config: OpenAI configuration
        """
        self.config = config or OpenAIConfig()
        if model_path != self.config.model:
            self.config.model = model_path
        self.client = None
        super().__init__(system_prompt, role, console, model_path, self.config)
    
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
    
    def _generate_text(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Build request parameters
        request_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", self.config.temperature),
            "max_tokens": gen_kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": gen_kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }
        
        # Make API call
        response = self.client.chat.completions.create(**request_params)
        
        # Extract response details
        response_text = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        return response_text, input_tokens, output_tokens
    
    def get_input_sequence(self) -> str:
        """Not implemented for OpenAI player. Use SGLangModelPlayer for this functionality."""
        raise NotImplementedError("get_input_sequence is only implemented for SGLangModelPlayer")
    
    def get_assistant_mask(self) -> List[int]:
        """Not implemented for OpenAI player. Use SGLangModelPlayer for this functionality."""
        raise NotImplementedError("get_assistant_mask is only implemented for SGLangModelPlayer")
    
    def get_masked_sequences_pretty(self) -> str:
        """Not implemented for OpenAI player. Use SGLangModelPlayer for this functionality."""
        raise NotImplementedError("get_masked_sequences_pretty is only implemented for SGLangModelPlayer")