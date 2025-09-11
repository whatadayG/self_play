"""
VLLM model player for efficient local inference.
Unified implementation based on BaseModelPlayer.
"""

from typing import Optional, Tuple, Dict, Any
from rich.console import Console

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

from .base_player import BaseModelPlayer, VLLMConfig


class VLLMModelPlayer(BaseModelPlayer):
    """Player that uses vLLM for efficient local inference."""
    
    def __init__(
        self,
        prompt: str,
        role: str,
        console: Console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        config: Optional[VLLMConfig] = None,
        **kwargs
    ):
        """Initialize vLLM model player.
        
        Args:
            prompt: Initial system prompt
            role: Role of the player
            console: Rich console for output
            model_path: Path to the model
            prefix: Prefix for responses
            optional: Optional context
            config: vLLM configuration
            **kwargs: Additional arguments
        """
        if LLM is None:
            raise ImportError("vllm package is required for VLLMModelPlayer")
        
        self.config = config or VLLMConfig()
        self.llm = None
        self.tokenizer = None
        
        # Call parent init
        super().__init__(prompt, role, console, model_path, prefix, optional, self.config, **kwargs)
    
    def _setup_model(self) -> None:
        """Initialize vLLM model."""
        self.console.print(f"[yellow]Loading vLLM model {self.model_path} for {self.role}...[/yellow]")
        
        # Initialize vLLM
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        
        # Get tokenizer for token counting
        self.tokenizer = self.llm.get_tokenizer()
        
        self.console.print(f"[green]vLLM model loaded for {self.role}[/green]")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation
            return len(text) // 3
    
    def _generate_text(self, prompt: str, **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using vLLM.
        
        Args:
            prompt: The prompt to generate from
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=gen_kwargs.get("temperature", self.config.temperature),
            top_p=gen_kwargs.get("top_p", self.config.top_p),
            max_tokens=gen_kwargs.get("max_tokens", self.config.max_tokens),
            stop=self.config.stop_sequences[:-1],  # Exclude \n
        )
        
        # Add seed for high temperature (vary=True)
        if gen_kwargs.get("temperature", 0.7) > 1.5:
            import random
            sampling_params.seed = random.randint(1, 10000)
        
        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        
        # Extract response
        response_text = outputs[0].outputs[0].text
        
        # Get token counts
        # vLLM provides these in the output
        output = outputs[0]
        input_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else self._count_tokens(prompt)
        output_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else self._count_tokens(response_text)
        
        return response_text, input_tokens, output_tokens
    
    def cleanup(self) -> None:
        """Clean up vLLM resources."""
        if self.llm:
            # vLLM doesn't have explicit cleanup, but we can clear references
            self.llm = None
            self.tokenizer = None
            
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()