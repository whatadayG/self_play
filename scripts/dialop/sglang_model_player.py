"""
SGLang model player using SGLang server HTTP API or internal engine.
"""

import os
import asyncio
import requests
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from rich.console import Console
from transformers import AutoTokenizer

from .base_player import BaseModelPlayer, SGLangConfig


class SGLangModelPlayer(BaseModelPlayer):
    """Player that queries an SGLang server via HTTP API or internal engine."""
    
    def __init__(
        self,
        system_prompt: str,
        role: str,
        console: Console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        config: Optional[SGLangConfig] = None,
        # New optional parameters for internal engine mode
        engine: Optional[Any] = None,
        processing_class: Optional[Any] = None,
    ):
        """Initialize SGLang model player.
        
        Args:
            system_prompt: Initial system prompt that defines the player
            role: Role of the player
            console: Rich console for output
            model_path: Model identifier on the SGLang server
            config: SGLang configuration
            engine: Optional SGLang AsyncEngine for internal mode
            processing_class: Optional tokenizer/processor for internal mode
        """
        self.config = config or SGLangConfig()
        self.tokenizer = None  # Will be loaded on demand for external mode
        self.engine = engine
        self.processing_class = processing_class
        
        # Validate configuration: exactly one mode should be configured
        # Design intent:
        # - Internal mode: engine and processing_class are provided (for training within verl)
        # - External mode: neither are provided, uses HTTP API to external SGLang server
        # - Mixed mode is not supported to maintain clear separation of concerns
        
        if (engine is None) != (processing_class is None):
            raise ValueError(
                "Invalid configuration: engine and processing_class must both be provided (internal mode) "
                "or both be None (external mode). Mixed configurations are not supported."
            )
        
        if engine is not None and processing_class is not None:
            # Internal mode - uses direct engine calls
            mode = "internal engine"
        else:
            # External mode - uses HTTP API
            mode = "external HTTP"
            
        # Log the mode once during initialization
        print(f"[SGLangModelPlayer] Initialized in {mode} mode for {role}")
        
        super().__init__(system_prompt, role, console, model_path, self.config)
    
    def _setup_model(self) -> None:
        """Set up SGLang server connection or internal engine."""
        if self.engine is not None:
            # Internal engine mode
            self.console.print(f"[green]Using internal SGLang engine for {self.role}[/green]")
        else:
            # External HTTP mode
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
        """Generate text using SGLang.
        
        Args:
            messages: List of message dictionaries
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        if self.engine is not None:
            # Internal engine mode
            return self._generate_text_internal(messages, **gen_kwargs)
        else:
            # External HTTP mode
            return self._generate_text_external(messages, **gen_kwargs)
    
    def _generate_text_external(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
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
            if self.processing_class is not None:
                # Use the provided processing class (tokenizer) for internal mode
                self.tokenizer = self.processing_class
            else:
                # Load tokenizer for external mode
                try:
                    # First try to load as a local path
                    if os.path.exists(self.model_path):
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
                    else:
                        # Fall back to loading from Hugging Face hub
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                except Exception as e:
                    # If loading fails, try a common base model tokenizer as fallback
                    print(f"Warning: Failed to load tokenizer from {self.model_path}: {e}")
                    print("Falling back to Qwen2.5-7B-Instruct tokenizer")
                    self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
                
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
    
    def _generate_text_internal(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using internal SGLang engine.
        
        Args:
            messages: List of message dictionaries
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Apply chat template to format messages
        text = self.processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize to get input IDs
        prompt_ids = self.processing_class.encode(text)
        
        # Prepare sampling parameters
        sampling_params = {
            "temperature": gen_kwargs.get("temperature", self.config.temperature),
            "top_p": gen_kwargs.get("top_p", self.config.top_p),
            "max_new_tokens": gen_kwargs.get("max_tokens", self.config.max_tokens),
            "n": 1,
        }
        
        # Log generation start
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
        logger.info(f"[SGLangPlayer] Starting internal generation with {len(prompt_ids)} input tokens")
        logger.info(f"[SGLangPlayer] Sampling params: {sampling_params}")
        logger.info(f"[SGLangPlayer] Engine available: {self.engine is not None}")
        logger.info(f"[SGLangPlayer] Processing class available: {self.processing_class is not None}")
        
        try:
            logger.info(f"[SGLangPlayer] Calling engine.async_generate...")
            
            # Get or create event loop (needed in Ray async actors)
            loop = asyncio.get_event_loop()
            
            output = loop.run_until_complete(
                self.engine.async_generate(
                    input_ids=prompt_ids,
                    sampling_params=sampling_params,
                    prompt=None,  # We're providing input_ids directly
                )
            )
            logger.info(f"[SGLangPlayer] Engine call completed")
            
            # Extract response text and calculate tokens
            response_text = output["text"]
            input_tokens = len(prompt_ids)
            # Output tokens is the total generated minus the input
            output_tokens = len(output["output_ids"]) - input_tokens
            
            logger.debug(f"[SGLangPlayer] Generated {output_tokens} tokens")
            return response_text, input_tokens, output_tokens
            
        except Exception as e:
            logger.error(f"[SGLangPlayer] Internal generation error: {e}")
            self.console.print(f"[red]Internal generation error: {e}[/red]")
            # Return a fallback response
            return "I need to think about this.", len(prompt_ids), 7


# Backward compatibility alias
SglangModelPlayer = SGLangModelPlayer
