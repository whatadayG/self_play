"""
VLLM model player for efficient local inference.
"""

from typing import Optional, Tuple, List, Dict, Any
from rich.console import Console

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

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
        # Optional parameters for integration with verl
        engine: Optional[Any] = None,  # vLLM LLM instance from rollout worker
        tokenizer: Optional[Any] = None,  # Tokenizer instance from rollout worker
    ):
        """Initialize vLLM model player.
        
        Args:
            system_prompt: Initial system prompt that defines the player
            role: Role of the player
            console: Rich console for output
            model_path: Path to the model
            config: vLLM configuration
            engine: Optional vLLM LLM instance (for internal mode)
            tokenizer: Optional tokenizer instance (for internal mode)
        """
        if LLM is None:
            raise ImportError("vllm package is required for VLLMModelPlayer")
        
        self.config = config or VLLMConfig()
        self.llm = engine  # Use provided engine if available
        self.tokenizer = tokenizer
        self._external_engine = engine is not None  # Track if engine was provided
        super().__init__(system_prompt, role, console, model_path, self.config)
    
    def _setup_model(self) -> None:
        """Initialize vLLM model."""
        if self._external_engine:
            # Engine was provided externally
            self.console.print(f"[green]Using provided vLLM engine for {self.role}[/green]")
        else:
            # Create our own engine
            self.console.print(f"[yellow]Loading vLLM model {self.model_path} for {self.role}...[/yellow]")
            
            # Initialize vLLM
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=self.config.trust_remote_code,
                dtype=self.config.dtype,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
            
            self.console.print(f"[green]vLLM model loaded for {self.role}[/green]")
        
        # Get tokenizer if not provided
        if self.tokenizer is None:
            if hasattr(self.llm, 'get_tokenizer'):
                self.tokenizer = self.llm.get_tokenizer()
            elif hasattr(self.llm, 'tokenizer'):
                self.tokenizer = self.llm.tokenizer
            elif AutoTokenizer is not None:
                # Fallback to loading tokenizer directly
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=self.config.trust_remote_code
                )
    
    def _generate_text(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using vLLM.
        
        Args:
            messages: List of message dictionaries
            **gen_kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        # Get tokenizer
        tokenizer = self.tokenizer or self.llm.get_tokenizer()
        
        # Apply chat template
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
    
    def get_input_string(self) -> str:
        """Get the full input string with chat template applied."""
        tokenizer = self.tokenizer or self.llm.get_tokenizer()
        return tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def get_input_sequence(self) -> List[int]:
        """Get the tokenized input sequence."""
        tokenizer = self.tokenizer or self.llm.get_tokenizer()
        input_text = tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False
        )
        input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
        return input_tokens
    
    def get_assistant_mask(self) -> List[int]:
        """Generate a mask that is 1 for tokens within assistant messages.
        
        Returns:
            List of 0s and 1s, where 1 indicates assistant tokens
        """
        tokenizer = self.tokenizer or self.llm.get_tokenizer()
        
        # First, tokenize the full conversation
        full_text = tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        
        # Create mask initialized to 0
        mask = [0] * len(full_tokens)
        
        # For each assistant message, find where it appears and mark those tokens
        for i, msg in enumerate(self.messages):
            if msg["role"] == "assistant":
                # Get the conversation up to this point (inclusive)
                partial_messages = self.messages[:i+1]
                partial_text = tokenizer.apply_chat_template(
                    partial_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                partial_tokens = tokenizer.encode(partial_text, add_special_tokens=False)
                
                # Get the conversation up to before this message
                if i > 0:
                    prev_messages = self.messages[:i]
                    prev_text = tokenizer.apply_chat_template(
                        prev_messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    prev_tokens = tokenizer.encode(prev_text, add_special_tokens=False)
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
        tokenizer = self.tokenizer or self.llm.get_tokenizer()
        
        # Get the full tokenized sequence and mask
        full_text = tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False
        )
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
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
                    text = tokenizer.decode(current_sequence, skip_special_tokens=False)
                    sequences.append(text)
                    current_sequence = []
        
        # Don't forget the last sequence
        if current_sequence:
            text = tokenizer.decode(current_sequence, skip_special_tokens=False)
            sequences.append(text)
        
        # Join with double newlines for readability
        return "\n\n".join(sequences)