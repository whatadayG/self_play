"""
SGLang model player using SGLang server HTTP API.
"""

import os
import requests
import numpy as np
from typing import Optional, Tuple, List, Dict
from rich.console import Console
from transformers import AutoTokenizer

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
        self.tokenizer = None  # Will be loaded on demand
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
    
    def _load_tokenizer(self):
        """Load tokenizer if not already loaded."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_input_string(self) -> str:
        """Get the input string that would be passed to model inference.
        
        Returns:
            The formatted input string with chat template applied
        """
        self._load_tokenizer()
        
        # Apply chat template to get the formatted input
        input_text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return input_text
    
    def get_input_sequence(self) -> List[int]:
        """Get the actual input tensor (token IDs) that would be passed to model inference.
        
        Returns:
            List of token IDs representing the input sequence
        """
        self._load_tokenizer()
        
        # Apply chat template and tokenize
        input_text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_tokens = self.tokenizer.encode(input_text, add_special_tokens=True)
        return input_tokens
    
    def get_assistant_mask(self) -> List[int]:
        """Generate a mask that is 1 for tokens within assistant messages.
        
        Returns:
            List of 0s and 1s, where 1 indicates assistant tokens
        """
        self._load_tokenizer()
        
        # First, tokenize the full conversation
        full_text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        # Create mask initialized to 0
        mask = [0] * len(full_tokens)
        
        # For each assistant message, find where it appears and mark those tokens
        for i, msg in enumerate(self.messages):
            if msg["role"] == "assistant":
                # Get the conversation up to this point (inclusive)
                partial_messages = self.messages[:i+1]
                partial_text = self.tokenizer.apply_chat_template(
                    partial_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                partial_tokens = self.tokenizer.encode(partial_text, add_special_tokens=False)
                
                # Get the conversation up to before this message
                if i > 0:
                    prev_messages = self.messages[:i]
                    prev_text = self.tokenizer.apply_chat_template(
                        prev_messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    prev_tokens = self.tokenizer.encode(prev_text, add_special_tokens=False)
                    start_idx = len(prev_tokens)
                else:
                    start_idx = 0
                
                # Mark the assistant tokens
                end_idx = len(partial_tokens)
                for j in range(start_idx, min(end_idx, len(mask))):
                    mask[j] = 1
        
        return mask
    
    def get_masked_sequences_pretty(self) -> str:
        """Get a pretty-formatted view of the masked assistant sequences.
        
        Returns:
            Formatted string showing assistant token sequences separated by newlines
        """
        self._load_tokenizer()
        
        # Get the full tokenized sequence and mask
        full_text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        mask = self.get_assistant_mask()
        
        # Extract continuous sequences where mask is 1
        sequences = []
        current_sequence = []
        
        for i, (token_id, mask_val) in enumerate(zip(full_tokens, mask)):
            if mask_val == 1:
                current_sequence.append(token_id)
            else:
                if current_sequence:
                    # Decode and add the sequence
                    text = self.tokenizer.decode(current_sequence, skip_special_tokens=False)
                    sequences.append(text)
                    current_sequence = []
        
        # Don't forget the last sequence
        if current_sequence:
            text = self.tokenizer.decode(current_sequence, skip_special_tokens=False)
            sequences.append(text)
        
        # Join with double newlines for readability
        return "\n\n".join(sequences)


# Backward compatibility alias
SglangModelPlayer = SGLangModelPlayer