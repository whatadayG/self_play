"""
Base class for all model players in the dialop framework.
Provides common functionality for dialogue agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from rich.console import Console
from rich.markup import escape


@dataclass
class ModelConfig:
    """Base configuration for all model players."""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9


@dataclass
class SGLangConfig(ModelConfig):
    """Configuration specific to SGLang model player."""
    server_url: str = "http://localhost:30000"
    api_key: str = "EMPTY"
    timeout: float = 120.0
    max_thinking_tokens: Optional[int] = None  # Limit thinking tokens for Qwen3 models


@dataclass 
class VLLMConfig(ModelConfig):
    """Configuration specific to vLLM model player."""
    gpu_memory_utilization: float = 0.8
    dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class HFConfig(ModelConfig):
    """Configuration specific to Hugging Face model player."""
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


@dataclass
class OpenAIConfig(ModelConfig):
    """Configuration specific to OpenAI model player."""
    model: str = "gpt-4-turbo-preview"
    organization: Optional[str] = None
    api_key_path: str = "/home/nickatomlin/georgiazhou/dialop/dialop/.api_key"
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class BaseModelPlayer(ABC):
    """Abstract base class for all model players."""
    
    def __init__(
        self,
        system_prompt: str,
        role: str,
        console: Console,
        model_path: str,
        config: Optional[ModelConfig] = None,
    ):
        """Initialize the base model player.
        
        Args:
            system_prompt: Initial system prompt that defines the player
            role: Role of the player (user, agent, etc.)
            console: Rich console for output
            model_path: Path to the model
            config: Model configuration object
        """
        self.role = role
        self.console = console
        self.model_path = model_path
        self.config = config or ModelConfig()
        
        # Conversation state management
        self.messages = [{"role": "system", "content": system_prompt}]
        
        # Initialize model-specific components
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self) -> None:
        """Set up model-specific components (tokenizer, model, client, etc.)"""
        pass
    
    @abstractmethod
    def _generate_text(self, messages: List[Dict[str, str]], **gen_kwargs) -> Tuple[str, int, int]:
        """Generate text using the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **gen_kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        pass
    
    def observe(self, obs: str) -> None:
        """Observe new information and update conversation state.
        
        Args:
            obs: Observation text (treated as user message)
        """
        # Clean observation
        obs = obs.strip()
        if obs:
            # Add as user message
            self.messages.append({"role": "user", "content": obs})
    
    def respond(self, **kwargs) -> str:
        """Generate a response from the model.
        
        Args:
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response string
        """
        # Set generation parameters
        gen_kwargs = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }
        
        # Generate response
        # NOTE: Generation failures (server errors, OOM, etc.) are catastrophic
        # and should propagate to the game loop. Do NOT catch and return fallback.
        response_text, input_tokens, output_tokens = self._generate_text(self.messages, **gen_kwargs)
        
        # Add assistant response to conversation history
        self.messages.append({"role": "assistant", "content": response_text})
        
        # Log response
        self.console.rule(f"{self.role}'s turn")
        self.console.print(f"Response: {escape(response_text)}")
        self.console.print(f"Tokens - Input: {input_tokens}, Output: {output_tokens}")
        
        return response_text
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages.copy()
    
    def reset_conversation(self, keep_system: bool = True) -> None:
        """Reset the conversation history.
        
        Args:
            keep_system: Whether to keep the system prompt
        """
        if keep_system and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def get_pretty_conversation(self) -> str:
        """Get a pretty-formatted representation of the conversation.
        
        Returns:
            Formatted string with System:, Assistant:, User: labels
        """
        parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System:\n{content}")
            elif role == "assistant":
                parts.append(f"Assistant:\n{content}")
            elif role == "user":
                parts.append(f"User:\n{content}")
        return "\n\n".join(parts)