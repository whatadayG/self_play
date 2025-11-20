"""
SGLang model player using SGLang server HTTP API or internal engine.
"""

import os
import asyncio
import requests
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # urllib3 may vary by environment
    Retry = None  # type: ignore
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
        optional: Optional[Dict[str, Any]] = None,
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
        self.optional = optional or {}
        self.last_generation_logprobs = None  # Store logprobs from last generation
        self.all_generated_logprobs: List[float] = []  # Store ALL logprobs across all generations
        self._assistant_prefix_token_count = None  # Cache for assistant prefix token count
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
            
            # Create a persistent HTTP session for connection reuse
            self.session = requests.Session()
            # Configure adapter with a reasonable pool; retries disabled by default
            adapter = HTTPAdapter(pool_connections=64, pool_maxsize=256)
            if Retry is not None:
                # Optional: very conservative retry policy on transient TCP resets
                retry = Retry(total=0, backoff_factor=0)
                adapter = HTTPAdapter(pool_connections=64, pool_maxsize=256, max_retries=retry)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            # Default headers
            self.session.headers.update(
                {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "Connection": "keep-alive",
                }
            )

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
        # TODO: Consider switching to native /generate endpoint instead of /v1/chat/completions
        # The native endpoint returns token IDs directly (meta_info.output_token_logprobs[i][1]),
        # which would avoid tokenization roundtrip issues when reconstructing sequences.
        # This would make get_generated_logprob_tensor() alignment more robust.
        # See: test_native_sglang_endpoint.py for example usage.

        # Build request ID with game context for cache tracing
        # Format: game_{id}_turn_{n}_player_{role}_msg_{count}
        game_id = self.optional.get("game_id", "unknown")
        turn = self.optional.get("turn", 0)
        msg_count = len([m for m in messages if m["role"] == "assistant"])  # Count assistant messages
        rid = f"game_{game_id}_turn_{turn}_{self.role}_msg_{msg_count}"

        # Build request payload
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", self.config.temperature),
            "top_p": gen_kwargs.get("top_p", self.config.top_p),
            "max_tokens": gen_kwargs.get("max_tokens", self.config.max_tokens),
            "n": 1,
            "logprobs": True,  # Request logprobs for KL divergence computation
            "top_logprobs": 1,  # Only need the selected token's logprob
            "rid": rid,  # Include structured request ID for cache tracing
        }

        # Use longer timeout to account for queueing at high concurrency
        timeout = 900.0
        max_retries = 5
        last_exception = None

        for attempt in range(max_retries):
            try:
                resp = self.session.post(
                    self.completions_url,
                    json=payload,
                    timeout=timeout
                )
                resp.raise_for_status()
                data = resp.json()

                # Extract response
                if "choices" not in data or not data["choices"]:
                    raise RuntimeError(f"Invalid response from SGLang server: {data}")

                response_text = data["choices"][0]["message"]["content"]

                # IMPORTANT: SGLang sometimes includes the role prefix in the response
                # Strip "\nassistant\n" or similar prefixes if present
                if response_text.startswith("\nassistant\n"):
                    response_text = response_text[len("\nassistant\n"):]

                # Get token counts (if provided by server)
                input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                output_tokens = data.get("usage", {}).get("completion_tokens", 0)

                # Extract and store logprobs if available
                logprobs_data = data["choices"][0].get("logprobs", None)
                if logprobs_data and logprobs_data.get("content"):
                    # Extract the logprob values for each token
                    # SGLang returns: {"content": [{"token": "...", "logprob": float, ...}, ...]}
                    all_logprobs = [
                        token_data["logprob"]
                        for token_data in logprobs_data["content"]
                    ]
                    # IMPORTANT: SGLang returns logprobs ONLY for completion tokens (output)
                    # The returned logprobs should exactly match output_tokens count
                    if len(all_logprobs) != output_tokens:
                        raise RuntimeError(
                            f"Logprobs count mismatch: SGLang returned {len(all_logprobs)} logprobs "
                            f"but output_tokens={output_tokens}. SGLang should return exactly one logprob per output token."
                        )
                    self.last_generation_logprobs = all_logprobs
                    # Also accumulate in the full list
                    self.all_generated_logprobs.extend(self.last_generation_logprobs)
                else:
                    self.last_generation_logprobs = None

                return response_text, input_tokens, output_tokens

            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                last_exception = e
                error_type = type(e).__name__
                self.console.print(f"[yellow]Attempt {attempt + 1}/{max_retries} failed ({error_type}): {e}[/yellow]")
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    # Final failure after all retries - log loudly
                    import logging
                    logger = logging.getLogger(__name__)
                    error_msg = f"GENERATION FAILED AFTER {max_retries} RETRIES: {error_type} - {e}"
                    logger.error(error_msg)
                    self.console.print(f"[red bold]{error_msg}[/red bold]")

                    # Write to disk in log directory if available
                    log_dir = os.environ.get("ROLLOUT_LOG_DIR", None)
                    if log_dir:
                        import time
                        log_file = os.path.join(log_dir, "generation_failures.log")
                        try:
                            with open(log_file, "a") as f:
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                f.write(f"[{timestamp}] {error_msg}\n")
                                f.write(f"  Payload: {payload}\n\n")
                        except Exception as write_error:
                            logger.error(f"Failed to write error log: {write_error}")

                    # Re-raise the original exception with full details
                    raise last_exception
    
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

    def _compute_assistant_prefix_token_count(self) -> int:
        """Compute the number of tokens in the assistant turn prefix.

        For Qwen chat template, this is '<|im_start|>assistant\\n' which tokenizes to 3 tokens.
        This is computed once and cached since it's model-specific and doesn't change.

        Returns:
            Number of tokens in the assistant prefix that the model doesn't generate
        """
        # Return cached value if already computed
        if self._assistant_prefix_token_count is not None:
            return self._assistant_prefix_token_count

        self._load_tokenizer()

        # Create a minimal message pair to extract the assistant prefix
        test_messages = [
            {"role": "user", "content": "test"}
        ]

        # Get text without generation prompt (stops after user message)
        without_gen_prompt = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Get text with generation prompt (adds assistant prefix)
        with_gen_prompt = self.tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # The difference is the assistant prefix (e.g., '<|im_start|>assistant\n')
        # This is what's in the prompt but not generated by the model
        if not with_gen_prompt.startswith(without_gen_prompt):
            raise RuntimeError(
                "Unexpected chat template structure: text with generation prompt "
                "doesn't start with text without generation prompt"
            )

        assistant_prefix_text = with_gen_prompt[len(without_gen_prompt):]

        # Tokenize the prefix to count tokens
        # Note: We don't add special tokens here because they're already in the text
        prefix_tokens = self.tokenizer.encode(assistant_prefix_text, add_special_tokens=False)

        # Cache the result
        self._assistant_prefix_token_count = len(prefix_tokens)

        return self._assistant_prefix_token_count

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
        """Generate a mask that is 1 for tokens actually generated by the model.

        Excludes the chat template prefix ('<|im_start|>assistant\n') since the model
        doesn't generate those tokens - they're part of the prompt.

        Returns:
            List of 0s and 1s, where 1 indicates tokens the model generated
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

        # The assistant prefix that's in the prompt (not generated by the model)
        # For Qwen: '<|im_start|>assistant\n'
        assistant_prefix_tokens = self._compute_assistant_prefix_token_count()

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

                # Mark the assistant tokens, but SKIP the first 3 tokens (the prefix)
                # The model only generates: content + <|im_end|>\n
                # It does NOT generate: <|im_start|>assistant\n
                end_idx = len(partial_tokens)
                actual_start_idx = start_idx + assistant_prefix_tokens

                for j in range(actual_start_idx, min(end_idx, len(mask))):
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

    def get_last_generation_logprobs(self) -> Optional[List[float]]:
        """Get the logprobs from the last generation.

        Returns:
            List of logprob values (one per generated token), or None if not available
        """
        return self.last_generation_logprobs

    def get_generated_logprob_tensor(self) -> List[float]:
        """Create a logprob tensor aligned with input_ids and loss_mask.

        Returns a tensor where:
        - Positions with loss_mask=1 (generated tokens) have their actual logprobs
        - Positions with loss_mask=0 (prompt tokens) have 0.0

        The length matches get_input_sequence() and get_assistant_mask().

        Returns:
            List of logprob values, same length as input_ids
        """
        self._load_tokenizer()

        # Get the full input sequence length
        input_ids = self.get_input_sequence()

        # Initialize tensor with zeros
        logprob_tensor = [0.0] * len(input_ids)

        # Get assistant mask to know where generated tokens are
        assistant_mask = self.get_assistant_mask()

        # The all_generated_logprobs contains logprobs for all generated tokens in order
        # We need to place them at positions where assistant_mask=1
        generated_positions = [i for i, mask_val in enumerate(assistant_mask) if mask_val == 1]

        if len(self.all_generated_logprobs) != len(generated_positions):
            # Debug: print details about each assistant message
            print(f"\nDEBUG: Logprob count mismatch")
            print(f"  Collected logprobs: {len(self.all_generated_logprobs)}")
            print(f"  Assistant mask positions: {len(generated_positions)}")
            print(f"  Total messages: {len(self.messages)}")

            # Show each assistant message and how many tokens it should have
            assistant_msg_count = 0
            for i, msg in enumerate(self.messages):
                if msg["role"] == "assistant":
                    assistant_msg_count += 1
                    content = msg["content"]
                    print(f"  Assistant message {assistant_msg_count}:")
                    print(f"    Content (first 200 chars): {repr(content[:200])}")
                    # Tokenize just this message content
                    content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
                    # The actual response includes <|im_end|>\n (2 tokens)
                    expected_gen_tokens = len(content_tokens) + 2
                    print(f"    Content tokens: {len(content_tokens)}")
                    print(f"    Expected generated (content + <|im_end|>\\n): {expected_gen_tokens}")

            # Show actual logprob collection history
            print(f"\n  Total logprobs collected across all generations: {len(self.all_generated_logprobs)}")

            raise RuntimeError(
                f"Logprob count mismatch: collected {len(self.all_generated_logprobs)} logprobs "
                f"but assistant_mask has {len(generated_positions)} generated positions"
            )

        # Fill in the logprobs at generated positions
        for pos, logprob in zip(generated_positions, self.all_generated_logprobs):
            logprob_tensor[pos] = logprob

        return logprob_tensor
    
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
            error_msg = f"INTERNAL ENGINE GENERATION FAILED: {type(e).__name__} - {e}"
            logger.error(error_msg)
            self.console.print(f"[red bold]{error_msg}[/red bold]")

            # Write to disk in log directory if available
            log_dir = os.environ.get("ROLLOUT_LOG_DIR", None)
            if log_dir:
                import time
                log_file = os.path.join(log_dir, "generation_failures.log")
                try:
                    with open(log_file, "a") as f:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{timestamp}] {error_msg}\n")
                        f.write(f"  Input tokens: {len(prompt_ids)}\n\n")
                except Exception as write_error:
                    logger.error(f"Failed to write error log: {write_error}")

            # Re-raise the exception to provide full details to outer handler
            raise e


# Backward compatibility alias
SglangModelPlayer = SGLangModelPlayer
