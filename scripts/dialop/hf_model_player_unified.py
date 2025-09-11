"""
Hugging Face model player for local inference.
Unified implementation based on BaseModelPlayer.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Dict, Any
from rich.console import Console

from .base_player import BaseModelPlayer, HFConfig


class HFModelPlayer(BaseModelPlayer):
    """Player that uses Hugging Face transformers for local inference."""
    
    def __init__(
        self,
        prompt: str,
        role: str,
        console: Console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        config: Optional[HFConfig] = None,
        **kwargs
    ):
        """Initialize Hugging Face model player.
        
        Args:
            prompt: Initial system prompt
            role: Role of the player
            console: Rich console for output
            model_path: Path to the Hugging Face model
            prefix: Prefix for responses
            optional: Optional context
            config: HF configuration
            **kwargs: Additional arguments (e.g., legacy parameters)
        """
        # Handle legacy parameters
        if config is None and any(k in kwargs for k in ["device", "temperature"]):
            config = HFConfig(
                device=kwargs.pop("device", "cuda"),
                temperature=kwargs.pop("temperature", 0.7),
                torch_dtype=kwargs.pop("torch_dtype", "bfloat16")
            )
        
        self.config = config or HFConfig()
        self.model = None
        self.tokenizer = None
        self.model_format = [{"role": "system", "content": prompt}]
        
        # Call parent init
        super().__init__(prompt, role, console, model_path, prefix, optional, self.config, **kwargs)
    
    def _setup_model(self) -> None:
        """Load model and tokenizer."""
        self.console.print(f"[yellow]Loading model {self.model_path} for {self.role}...[/yellow]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="left"
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine torch dtype
        torch_dtype = getattr(torch, self.config.torch_dtype, torch.bfloat16)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()
        
        self.console.print(f"[green]Model loaded for {self.role}[/green]")
    
    def observe(self, obs: str, info: bool = False, force: bool = False,
                ignore_obs: bool = False) -> None:
        """Observe and update prompt for next generation."""
        # Call parent observe to update self.prompt
        super().observe(obs, info, force, ignore_obs)
        
        # Note: The client code (evaluate_opt.py) directly manipulates model_format
        # so we don't update it here - it's managed externally
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _generate_text(self, prompt: str, **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using Hugging Face transformers.
        
        Args:
            prompt: The prompt to generate from (ignored - we use model_format)
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Use the externally managed model_format
        # This matches the original HF player behavior
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            self.model_format,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Check if we need to truncate
        input_length = inputs["input_ids"].shape[1]
        if input_length > 7500:
            # Try removing optional context
            if self.optional and not self.removed_optional:
                self._remove_optional_context()
                # Update the system message in model_format with the truncated prompt
                # This preserves the conversation history while removing optional context
                if self.model_format and self.model_format[0]["role"] == "system":
                    self.model_format[0]["content"] = self.prompt
                
                # Re-apply chat template with updated model_format
                text = self.tokenizer.apply_chat_template(
                    self.model_format,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=8192
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                input_length = inputs["input_ids"].shape[1]
        
        # Set generation parameters
        gen_params = {
            "max_new_tokens": gen_kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": gen_kwargs.get("temperature", self.config.temperature),
            "top_p": gen_kwargs.get("top_p", self.config.top_p),
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Add stop token ids if possible
        stop_ids = []
        for stop_str in self.config.stop_sequences[:-1]:  # Exclude \n
            tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
            if len(tokens) == 1:
                stop_ids.append(tokens[0])
        
        if stop_ids:
            gen_params["eos_token_id"] = stop_ids + [self.tokenizer.eos_token_id]
        
        # Add seed for high temperature (vary=True)
        if gen_kwargs.get("temperature", 0.7) > 1.5:
            import random
            gen_params["seed"] = random.randint(1, 10000)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
        
        # Decode only the generated part
        response_ids = outputs[0][input_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Count tokens
        input_tokens = input_length
        output_tokens = len(response_ids)
        
        # Do NOT update model_format - match original behavior
        
        return response_text, input_tokens, output_tokens