"""
SGLang model player using SGLang server HTTP API.
Unified implementation based on BaseModelPlayer.
"""

import os
import requests
from typing import Optional, Tuple, Dict, Any
from rich.console import Console

from .base_player import BaseModelPlayer, SGLangConfig


class SGLangModelPlayer(BaseModelPlayer):
    """Player that queries an SGLang server via HTTP API."""
    
    def __init__(
        self,
        prompt: str,
        role: str,
        console: Console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        config: Optional[SGLangConfig] = None,
        **kwargs
    ):
        """Initialize SGLang model player.
        
        Args:
            prompt: Initial system prompt
            role: Role of the player
            console: Rich console for output
            model_path: Model identifier on the SGLang server
            prefix: Prefix for responses
            optional: Optional context
            config: SGLang configuration
            **kwargs: Additional arguments (e.g., legacy parameters)
        """
        # Handle legacy parameters
        if "sglang_url" in kwargs and config is None:
            config = SGLangConfig(
                server_url=kwargs.pop("sglang_url"),
                temperature=kwargs.pop("temperature", 0.7),
                timeout=kwargs.pop("timeout_s", 120.0)
            )
        
        self.config = config or SGLangConfig()
        self.model_format = [{"role": "system", "content": prompt}]
        
        # Call parent init
        super().__init__(prompt, role, console, model_path, prefix, optional, self.config, **kwargs)
    
    def _setup_model(self) -> None:
        """Set up SGLang server connection."""
        # Ensure base URL has correct format
        base_url = self.config.server_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        self.base_url = base_url
        self.completions_url = f"{self.base_url}/chat/completions"
        
        # Get API key from environment or config
        self.api_key = os.environ.get("SGLANG_API_KEY", self.config.api_key)
        
        self.console.print(f"[green]Using SGLang server at {self.base_url} for {self.role}[/green]")
    
    # Use parent's observe() directly to update self.prompt
    # The client code manages model_format externally
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimation for models like Qwen
        # For production, consider using the model's tokenizer
        return len(text) // 3
    
    def _generate_text(self, prompt: str, **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using SGLang HTTP API.
        
        Args:
            prompt: The prompt to generate from (unused, we use model_format)
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Build request payload
        # Use the externally managed model_format
        payload = {
            "model": self.model_path,
            "messages": self.model_format,
            "temperature": gen_kwargs.get("temperature", self.config.temperature),
            "top_p": gen_kwargs.get("top_p", self.config.top_p),
            "max_tokens": gen_kwargs.get("max_tokens", self.config.max_tokens),
            "n": 1,
            "stop": self.config.stop_sequences,  # Original uses full stop sequences
        }
        
        # Add seed for high temperature (vary=True)
        if gen_kwargs.get("temperature", 0.7) > 1.5:
            import random
            payload["seed"] = random.randint(1, 10000)
        
        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Make request
        try:
            resp = requests.post(
                self.completions_url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"SGLang server timeout after {self.config.timeout}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"SGLang server error: {e}")
        
        # Extract response
        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"Invalid response from SGLang server: {data}")
        
        response_text = data["choices"][0]["message"]["content"]
        
        # Get token counts (if provided by server)
        input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
        
        # Estimate if not provided
        if not input_tokens:
            input_tokens = sum(self._count_tokens(str(msg)) for msg in self.model_format)
        if not output_tokens:
            output_tokens = self._count_tokens(response_text)
        
        # Do NOT update model_format - it's managed externally
        
        return response_text, input_tokens, output_tokens