"""
SGLang model player using SGLang server HTTP API.
"""

import os
import requests
from typing import Optional, Tuple, List, Dict
from rich.console import Console

from .base_player import BaseModelPlayer, SGLangConfig


class SGLangModelPlayer(BaseModelPlayer):
    """Player that queries an SGLang server via HTTP API."""
    
    def __init__(
        self,
        system_prompt: str,
        role: str,
        console: Console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        config: Optional[SGLangConfig] = None,
    ):
        """Initialize SGLang model player.
        
        Args:
            system_prompt: Initial system prompt that defines the player
            role: Role of the player
            console: Rich console for output
            model_path: Model identifier on the SGLang server
            config: SGLang configuration
        """
        self.config = config or SGLangConfig()
        super().__init__(system_prompt, role, console, model_path, self.config)
    
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
    
    def _generate_text(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using SGLang HTTP API.
        
        Args:
            messages: List of message dictionaries
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Build request payload
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", self.config.temperature),
            "top_p": gen_kwargs.get("top_p", self.config.top_p),
            "max_tokens": gen_kwargs.get("max_tokens", self.config.max_tokens),
            "n": 1,
        }
        
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
        
        return response_text, input_tokens, output_tokens


# Backward compatibility alias
SglangModelPlayer = SGLangModelPlayer