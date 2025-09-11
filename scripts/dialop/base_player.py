"""
Base class for all model players in the dialop framework.
Provides common functionality for dialogue agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from rich.console import Console
from rich.markup import escape


@dataclass
class ModelConfig:
    """Base configuration for all model players."""
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=lambda: ["User:", "Agent:", "You:"])
    # Note: Original LLMPlayer includes "\n", but HF/SGLang/Local do not
    # Subclasses can override as needed for exact compatibility
    

@dataclass
class SGLangConfig(ModelConfig):
    """Configuration specific to SGLang model player."""
    server_url: str = "http://localhost:30000"
    api_key: str = "EMPTY"
    timeout: float = 120.0


@dataclass 
class VLLMConfig(ModelConfig):
    """Configuration specific to VLLM model player."""
    gpu_memory_utilization: float = 0.8
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class HFConfig(ModelConfig):
    """Configuration specific to Hugging Face model player."""
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    

class BaseModelPlayer(ABC):
    """Abstract base class for all model players."""
    
    def __init__(
        self,
        prompt: str,
        role: str,
        console: Console,
        model_path: str,
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs
    ):
        """Initialize the base model player.
        
        Args:
            prompt: Initial system prompt
            role: Role of the player (user, agent, etc.)
            console: Rich console for output
            model_path: Path to the model
            prefix: Prefix for responses
            optional: Optional context that can be removed if needed
            config: Model configuration object
            **kwargs: Additional arguments for backward compatibility
        """
        self.prompt = prompt
        self.role = role
        self.console = console
        self.model_path = model_path
        self.prefix = prefix
        self.optional = optional
        self.removed_optional = False
        self.config = config or ModelConfig()
        
        # Store any extra kwargs for subclass use
        self.extra_kwargs = kwargs
        
        # Initialize model-specific components
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self) -> None:
        """Set up model-specific components (tokenizer, model, client, etc.)"""
        pass
    
    @abstractmethod
    def _generate_text(self, prompt: str, **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using the model.
        
        Args:
            prompt: The prompt to generate from
            **gen_kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        pass
    
    @abstractmethod
    def _count_tokens(self, text: str) -> int:
        """Count tokens in the given text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    def observe(self, obs: str, info: bool = False, force: bool = False, 
                ignore_obs: bool = False) -> None:
        """Observe new information and update conversation state.
        
        Args:
            obs: Observation text
            info: Whether this is informational context
            force: Force observation even if empty
            ignore_obs: Ignore this observation
        """
        # Clean observation
        obs = obs.replace("<PAD>", "").strip()
        if not obs and not force:
            return
            
        if ignore_obs:
            return
            
        if info:
            self.prompt += (obs + "\n" + "Here is the start of your current conversation:\n")
        else:
            if "Partner:" in obs:
                self.prompt += obs
            else:
                self.prompt += (self.prefix + obs)
    
    def _prepare_prompt(self, propose: bool = False, strategy: Optional[str] = None) -> str:
        """Prepare the prompt for generation.
        
        Args:
            propose: Whether to prompt for a proposal
            strategy: Optional strategy to append
            
        Returns:
            Prepared prompt string
        """
        prompt = self.prompt
        
        if strategy:
            prompt = prompt + '\n' + 'Here is your conversational strategy: ' + strategy
        
        if not prompt.endswith(self.prefix):
            if propose:
                prompt += (self.prefix + '[propose]')
            else:
                prompt += self.prefix
                
        return prompt
    
    def _remove_optional_context(self) -> None:
        """Remove optional context to fit within context window."""
        if self.optional and not self.removed_optional:
            self.prompt = self.prompt.replace(self.optional, "")
            self.removed_optional = True
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response.
        
        Args:
            response: Raw response from model
            
        Returns:
            Cleaned response
        """
        response = response.strip()
        
        # Remove any echoed speaker tags at the start
        for stop in self.config.stop_sequences:
            if response.startswith(stop):
                response = response[len(stop):].lstrip()
        
        # Truncate at the first occurrence of any stop tag (but not at start)
        for stop in self.config.stop_sequences:
            idx = response.find(stop)
            if idx > 0:
                response = response[:idx]
                break
                
        return response.strip()
    
    def _log_response(self, response: str, input_tokens: int, output_tokens: int) -> None:
        """Log the response to console.
        
        Args:
            response: Generated response
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.console.rule(f"{self.role}'s turn")
        self.console.print(f"Response: {escape(response)}")
        self.console.print(f"Tokens - Input: {input_tokens}, Output: {output_tokens}")
    
    def respond(self, t: int = 0, max_len: int = 256, vary: bool = False,
                propose: bool = False, temporal_id: Optional[int] = None,
                strategy: Optional[str] = None) -> str:
        """Generate a response from the model.
        
        Args:
            t: Current turn number
            max_len: Maximum conversation length
            vary: Whether to use higher temperature for variation
            propose: Whether the model should make a proposal
            temporal_id: Temporal index (for compatibility)
            strategy: Conversation strategy to append
            
        Returns:
            Generated response string
        """
        # Prepare prompt
        prompt = self._prepare_prompt(propose=propose, strategy=strategy)
        
        # Check context length and handle if needed
        token_count = self._count_tokens(prompt)
        if token_count > 7500:  # Conservative limit
            if self.optional and not self.removed_optional:
                self._remove_optional_context()
                prompt = self._prepare_prompt(propose=propose, strategy=strategy)
        
        # Set generation parameters
        gen_kwargs = {
            "temperature": 1.8 if vary else self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        
        # Generate response
        try:
            response_text, input_tokens, output_tokens = self._generate_text(prompt, **gen_kwargs)
        except Exception as e:
            self.console.print(f"[red]Generation error: {e}[/red]")
            response_text = "[message] I need to think about this."
            input_tokens = token_count
            output_tokens = len(response_text.split())
        
        # Clean response
        response_text = self._clean_response(response_text)
        
        # Log response
        self._log_response(response_text, input_tokens, output_tokens)
        
        return response_text