"""
Hugging Face model player for local inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List, Dict
from rich.console import Console

from .base_player import BaseModelPlayer, HFConfig


class HFModelPlayer(BaseModelPlayer):
    """Player that uses Hugging Face transformers for local inference."""
    
    def __init__(
        self,
        system_prompt: str,
        role: str,
        console: Console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        config: Optional[HFConfig] = None,
    ):
        """Initialize Hugging Face model player.
        
        Args:
            system_prompt: Initial system prompt that defines the player
            role: Role of the player
            console: Rich console for output
            model_path: Path to the Hugging Face model
            config: HF configuration
        """
        self.config = config or HFConfig()
        self.model = None
        self.tokenizer = None
        super().__init__(system_prompt, role, console, model_path, self.config)
    
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
    
    def _generate_text(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using Hugging Face transformers.
        
        Args:
            messages: List of message dictionaries
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
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
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
        
        # Decode only the generated part
        response_ids = outputs[0][input_length:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Count tokens
        input_tokens = input_length
        output_tokens = len(response_ids)
        
        return response_text, input_tokens, output_tokens
    
    def get_input_string(self) -> str:
        """Not implemented for HF player. Use SGLangModelPlayer for this functionality."""
        raise NotImplementedError("get_input_string is only implemented for SGLangModelPlayer")
    
    def get_input_sequence(self) -> List[int]:
        """Not implemented for HF player. Use SGLangModelPlayer for this functionality."""
        raise NotImplementedError("get_input_sequence is only implemented for SGLangModelPlayer")
    
    def get_assistant_mask(self) -> List[int]:
        """Not implemented for HF player. Use SGLangModelPlayer for this functionality."""
        raise NotImplementedError("get_assistant_mask is only implemented for SGLangModelPlayer")
    
    def get_masked_sequences_pretty(self) -> str:
        """Not implemented for HF player. Use SGLangModelPlayer for this functionality."""
        raise NotImplementedError("get_masked_sequences_pretty is only implemented for SGLangModelPlayer")