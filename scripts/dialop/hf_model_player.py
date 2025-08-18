"""
Hugging Face model player for local inference.
Uses transformers library directly for simplicity.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import random
from rich.markup import escape


class HFModelPlayer:
    """Player that uses a Hugging Face model locally."""
    
    def __init__(
        self,
        prompt: str,
        role: str,
        console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        device: str = "cuda",
        temperature: float = 0.7,
        **kwargs
    ):
        """Initialize a Hugging Face model player.
        
        Args:
            prompt: Initial prompt/system message
            role: Role of the player (user, agent, etc)
            console: Rich console for output
            model_path: Path to the Hugging Face model
            prefix: Prefix for responses
            optional: Optional context that can be removed if needed
            device: Device to run the model on
            **kwargs: Additional arguments (for compatibility)
        """
        self.prompt = prompt
        self.role = role
        self.console = console
        self.model_path = model_path
        self.prefix = prefix
        self.optional = optional
        self.removed_optional = False
        self.device = device
        self.temperature = temperature
        
        # For compatibility with cost tracking
        self.api_tracker = None
        
        # Load model and tokenizer
        print(f"Loading model {model_path} for {role}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        print(f"Model loaded for {role}")
        
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
    
    def respond(self, t: int = 0, max_len: int = 1024, vary: bool = False,
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
        
        # Format as chat template
        messages = [
            {"role": "system", "content": selfprompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Check if we need to truncate
        if inputs["input_ids"].shape[1] > 7500:
            if self.optional and not self.removed_optional:
                self._remove_optional_context()
                # Re-prepare prompt
                selfprompt = self.prompt
                if strategy:
                    selfprompt = selfprompt + '\n' + 'Here is your conversational strategy: ' + strategy
                if not selfprompt.endswith(self.prefix):
                    if propose:
                        selfprompt += (self.prefix + '[propose]')
                    else:
                        selfprompt += self.prefix
                messages = [{"role": "system", "content": selfprompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": 256,
            "temperature": 1.8 if vary else self.temperature,
            "top_p": 0.9,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Add stop strings
        stop_strings = ["User:", "Agent:", "You:"]
        if hasattr(self.tokenizer, "encode"):
            # Try to add stop token ids
            stop_ids = []
            for stop_str in stop_strings:
                tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if len(tokens) == 1:
                    stop_ids.append(tokens[0])
            if stop_ids:
                gen_kwargs["eos_token_id"] = stop_ids + [self.tokenizer.eos_token_id]
        
        if vary:
            gen_kwargs["seed"] = random.randint(1, 10000)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode only the generated part
        response_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Clean up response - handle echoed prefixes and stop words safely
        response_text = response_text.strip()
        # Remove any echoed speaker tags at the start
        for stop in stop_strings:
            if response_text.startswith(stop):
                response_text = response_text[len(stop):].lstrip()
        # Truncate at the first occurrence of any stop tag, but ignore index 0
        for stop in stop_strings:
            idx = response_text.find(stop)
            if idx > 0:
                response_text = response_text[:idx]
                break
        response_text = response_text.strip()
        
        # Print output
        self.console.rule(f"{self.role}'s turn")
        self.console.print(f"Response: {escape(response_text)}")
        
        # Log token usage
        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = len(response_ids)
        self.console.print(f"Tokens - Input: {input_tokens}, Output: {output_tokens}")
        
        # Cost tracking compatibility (no actual cost for local models)
        if getattr(self, "api_tracker", None) is not None:
            try:
                # Log with zero cost for local models
                self.api_tracker.log_api_call(
                    model=self.model_path,
                    input_text=selfprompt,
                    output_text=response_text,
                    project_id=self.role,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0  # No cost for local models
                )
            except Exception as e:
                self.console.print(f"[yellow]Cost tracking error: {e}[/yellow]")
        
        return response_text