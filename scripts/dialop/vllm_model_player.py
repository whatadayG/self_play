"""
VLLM model player for efficient local inference.
"""

from typing import Optional, Tuple, List, Dict
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
        system_prompt: str,
        role: str,
        console: Console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        config: Optional[VLLMConfig] = None,
    ):
        """Initialize vLLM model player.
        
        Args:
            system_prompt: Initial system prompt that defines the player
            role: Role of the player
            console: Rich console for output
            model_path: Path to the model
            config: vLLM configuration
        """
        if LLM is None:
            raise ImportError("vllm package is required for VLLMModelPlayer")
        
        self.config = config or VLLMConfig()
        self.llm = None
        super().__init__(system_prompt, role, console, model_path, self.config)
    
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
        
        self.console.print(f"[green]vLLM model loaded for {self.role}[/green]")
    
    def _generate_text(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using vLLM.
        
        Args:
            messages: List of message dictionaries
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Get tokenizer and apply chat template
        tokenizer = self.llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=gen_kwargs.get("temperature", self.config.temperature),
            top_p=gen_kwargs.get("top_p", self.config.top_p),
            max_tokens=gen_kwargs.get("max_tokens", self.config.max_tokens),
        )
        
        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        
        # Extract response
        output = outputs[0]
        response_text = output.outputs[0].text
        
        # Get token counts
        input_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
        output_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
        
        return response_text, input_tokens, output_tokens