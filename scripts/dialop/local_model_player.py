"""
Local model player using SGLang for inference.
This allows using local models like Qwen2.5-7B instead of OpenAI API.
"""

import os
import asyncio
import random
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer
import sglang as sgl
from sglang import RuntimeEndpoint
from rich.markup import escape

# Import the SGLang server args for model configuration
from sglang.srt.server_args import ServerArgs


class LocalModelPlayer:
    """Player that uses a local model through SGLang for generation."""
    
    def __init__(
        self, 
        prompt: str, 
        role: str, 
        console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        runtime_endpoint: Optional[RuntimeEndpoint] = None,
        **kwargs
    ):
        """Initialize a local model player.
        
        Args:
            prompt: Initial prompt/system message
            role: Role of the player (user, agent, etc)
            console: Rich console for output
            model_path: Path to the local model
            prefix: Prefix for responses
            optional: Optional context that can be removed if needed
            runtime_endpoint: Optional pre-initialized SGLang runtime
            **kwargs: Additional arguments (for compatibility)
        """
        self.prompt = prompt
        self.role = role
        self.console = console
        self.model_path = model_path
        self.prefix = prefix
        self.optional = optional
        self.removed_optional = False
        
        # For compatibility with cost tracking
        self.api_tracker = None
        
        # Initialize runtime if not provided
        self.runtime = runtime_endpoint
        if self.runtime is None:
            # For now, assume the SGLang server is already running
            # In production, you'd start the server here or connect to existing one
            self.runtime = RuntimeEndpoint("http://localhost:30000")
            
        # Initialize tokenizer for token counting
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
    def observe(self, obs: str, info: bool = False, force: bool = False, 
                ignore_obs: bool = False):
        """Observe new information and update prompt."""
        # Skip special tokens like PAD
        obs = obs.replace("<PAD>", "").strip()
        if not obs and not force:
            return
            
        if ignore_obs:
            pass
        else:
            if info:
                self.prompt += (obs + "\n" + "Here is the start of your current conversation:\n")
            else:
                if "Partner:" in obs:
                    self.prompt += obs
                else:
                    self.prompt += (self.prefix + obs)
    
    def _remove_optional_context(self):
        """Remove optional context to fit within context window."""
        if self.optional and not self.removed_optional:
            self.prompt = self.prompt.replace(self.optional, "")
            self.removed_optional = True
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        return len(self.tokenizer.encode(text))
    
    async def _generate_async(self, prompt: str, **kwargs) -> str:
        """Async generation using SGLang."""
        # Default generation parameters
        gen_params = {
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["User:", "Agent:", "You:", "\n"]
        }
        gen_params.update(kwargs)
        
        # Use SGLang's generation
        @sgl.function
        def generate_response(s):
            s += prompt
            s += sgl.gen("response", **gen_params)
        
        # Run generation
        state = generate_response.run()
        return state["response"].strip()
    
    def respond(self, t: int = 0, max_len: int = 3, vary: bool = False, 
                propose: bool = False, temporal_id: Optional[int] = None, 
                strategy: Optional[str] = None) -> str:
        """Generate a response from the model.
        
        Args:
            t: Current turn number
            max_len: Maximum conversation length
            vary: Whether to use higher temperature for variation
            propose: Whether the model should make a proposal
            temporal_id: Temporal index for specific prompts
            strategy: Conversation strategy to append
            
        Returns:
            Generated response string
        """
        # Prepare the prompt
        selfprompt = self.prompt
        
        if strategy:
            selfprompt = self.prompt + '\n' + 'Here is your conversational strategy: ' + strategy
        
        # Add prefix if needed
        if not selfprompt.endswith(self.prefix):
            if propose:
                selfprompt += (self.prefix + '[propose]')
            else:
                selfprompt += self.prefix
        
        # Check token count and remove optional if needed
        token_count = self._count_tokens(selfprompt)
        max_context = 8192  # Qwen2.5-7B context length
        
        if token_count > max_context - 512:  # Leave room for generation
            if self.optional and not self.removed_optional:
                self._remove_optional_context()
                selfprompt = self.prompt + self.prefix
                token_count = self._count_tokens(selfprompt)
        
        # Set generation parameters
        gen_kwargs = {
            "temperature": 1.8 if vary else 0.7,
            "max_tokens": min(256, max_context - token_count),
        }
        
        if vary:
            gen_kwargs["seed"] = random.randint(1, 10000)
        
        # Generate response (run async in sync context)
        try:
            response_text = asyncio.run(self._generate_async(selfprompt, **gen_kwargs))
        except Exception as e:
            self.console.print(f"[red]Generation error: {e}[/red]")
            response_text = "[message] I need to think about this."
        
        # Print output
        self.console.rule(f"{self.role}'s turn")
        self.console.print(f"Response: {escape(response_text)}")
        
        # Log token usage
        response_tokens = self._count_tokens(response_text)
        self.console.print(f"Tokens - Input: {token_count}, Output: {response_tokens}")
        
        # Cost tracking compatibility (no actual cost for local models)
        if getattr(self, "api_tracker", None) is not None:
            try:
                # Log with zero cost for local models
                self.api_tracker.log_api_call(
                    model=self.model_path,
                    input_text=selfprompt,
                    output_text=response_text,
                    project_id=self.role,
                    input_tokens=token_count,
                    output_tokens=response_tokens,
                    cost=0.0  # No cost for local models
                )
            except Exception as e:
                self.console.print(f"[yellow]Cost tracking error: {e}[/yellow]")
        
        return response_text


class LocalModelPlayerVLLM:
    """Alternative implementation using vLLM directly (simpler for self-contained use)."""
    
    def __init__(
        self,
        prompt: str,
        role: str,
        console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        **kwargs
    ):
        """Initialize with vLLM for simpler setup."""
        from vllm import LLM, SamplingParams
        
        self.prompt = prompt
        self.role = role
        self.console = console
        self.model_path = model_path
        self.prefix = prefix
        self.optional = optional
        self.removed_optional = False
        self.api_tracker = None
        
        # Initialize vLLM model
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        )
        
        # Default sampling params
        self.default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["User:", "Agent:", "You:", "\n"]
        )
    
    def observe(self, obs: str, info: bool = False, force: bool = False, 
                ignore_obs: bool = False):
        """Same as LocalModelPlayer."""
        obs = obs.replace("<PAD>", "").strip()
        if not obs and not force:
            return
            
        if ignore_obs:
            pass
        else:
            if info:
                self.prompt += (obs + "\n" + "Here is the start of your current conversation:\n")
            else:
                if "Partner:" in obs:
                    self.prompt += obs
                else:
                    self.prompt += (self.prefix + obs)
    
    def respond(self, t: int = 0, max_len: int = 3, vary: bool = False,
                propose: bool = False, temporal_id: Optional[int] = None,
                strategy: Optional[str] = None) -> str:
        """Generate response using vLLM."""
        # Prepare prompt
        selfprompt = self.prompt
        
        if strategy:
            selfprompt = self.prompt + '\n' + 'Here is your conversational strategy: ' + strategy
        
        if not selfprompt.endswith(self.prefix):
            if propose:
                selfprompt += (self.prefix + '[propose]')
            else:
                selfprompt += self.prefix
        
        # Update sampling params
        sampling_params = SamplingParams(
            temperature=1.8 if vary else 0.7,
            top_p=0.9,
            max_tokens=256,
            stop=["User:", "Agent:", "You:", "\n"],
            seed=random.randint(1, 10000) if vary else None
        )
        
        # Generate
        outputs = self.llm.generate([selfprompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        
        # Print output
        self.console.rule(f"{self.role}'s turn")
        self.console.print(f"Response: {escape(response_text)}")
        
        return response_text