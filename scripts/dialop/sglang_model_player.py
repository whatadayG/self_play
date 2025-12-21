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


class MaxTokensExceededError(Exception):
    """Raised when a response hits the max_tokens limit, indicating degenerate output."""
    pass


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
        self.all_generated_logprobs: List[List[float]] = []  # Store logprobs per assistant message (one sublist per message)
        self.all_generated_tokens: List[List[str]] = []  # Store token strings from SGLang (for debugging)
        self._assistant_prefix_token_count = None  # Cache for assistant prefix token count
        self._cached_mask_and_logprobs = None  # Cache for (mask, logprob_tensor) - only computed after conversation ends
        self._cached_input_sequence = None  # Cache for token sequence built in _build_mask_and_logprobs
        self._in_error_dump = False  # Recursion guard for error dumping
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

        # Add max_thinking_tokens if configured (for Qwen3 thinking budget limitation)
        # NOTE: This requires the SGLang thinking budget patch to be applied.
        # If the patch is not applied, SGLang will silently ignore this parameter.
        max_thinking_tokens = gen_kwargs.get("max_thinking_tokens", getattr(self.config, "max_thinking_tokens", None))
        if max_thinking_tokens is not None:
            payload["max_thinking_tokens"] = max_thinking_tokens
            # Store for validation after response
            self._expected_max_thinking_tokens = max_thinking_tokens
        else:
            self._expected_max_thinking_tokens = None

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

                # Check if generation was truncated due to max_tokens limit
                # This indicates degenerate output (e.g., repetition loops) and should fail the game
                finish_reason = data["choices"][0].get("finish_reason", "stop")

                # Log non-stop finish reasons for debugging truncation issues
                if finish_reason != "stop":
                    print(
                        f"[SGLangPlayer WARNING] finish_reason='{finish_reason}' "
                        f"after {output_tokens} tokens (max_tokens={gen_kwargs.get('max_tokens', self.config.max_tokens)})"
                    )

                if finish_reason == "length":
                    self._dump_max_tokens_exceeded(
                        response_text=response_text,
                        output_tokens=output_tokens,
                        max_tokens=gen_kwargs.get("max_tokens", self.config.max_tokens),
                    )
                    raise MaxTokensExceededError(
                        f"Response hit max_tokens limit ({output_tokens} tokens). "
                        f"This indicates degenerate output (e.g., repetition loop). "
                        f"Debug info saved to max_tokens_exceeded_dumps/"
                    )

                # Extract and store logprobs if available
                logprobs_data = data["choices"][0].get("logprobs", None)
                if logprobs_data and logprobs_data.get("content"):
                    # Extract the logprob values AND token strings for each token
                    # SGLang returns: {"content": [{"token": "...", "logprob": float, ...}, ...]}
                    all_logprobs = [
                        token_data["logprob"]
                        for token_data in logprobs_data["content"]
                    ]
                    all_tokens = [
                        token_data["token"]
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
                    # Append as a new sublist (one per assistant message)
                    # Use .copy() to ensure independent sublists (defensive against future modifications)
                    self.all_generated_logprobs.append(self.last_generation_logprobs.copy())
                    self.all_generated_tokens.append(all_tokens.copy())
                    # Clear cache since we added new data
                    self._cached_mask_and_logprobs = None
                    self._cached_input_sequence = None
                else:
                    self.last_generation_logprobs = None

                # Validate max_thinking_tokens was respected (detect if patch not applied)
                self._validate_thinking_budget(response_text, output_tokens)

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
                # Load tokenizer from model_path - fail loudly if it doesn't work
                if os.path.exists(self.model_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
                else:
                    # Fall back to loading from Hugging Face hub
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                
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
        # Use the cached sequence from _build_mask_and_logprobs if available
        # This ensures get_input_sequence() and get_assistant_mask() use the same tokenization
        if self._cached_input_sequence is not None:
            return self._cached_input_sequence

        # If not cached, build it (which will cache it)
        self._build_mask_and_logprobs()
        return self._cached_input_sequence

    def _build_mask_and_logprobs(self) -> Tuple[List[int], List[float]]:
        """Build both assistant mask and logprob tensor by scanning token sequence.

        IMPORTANT: Only call this after the conversation is complete. The result is cached.

        This method scans through the tokenized conversation looking for assistant message
        patterns and aligning them with collected logprobs from SGLang.

        For Qwen3-8B with thinking mode:
        - Template tokens (NOT generated): <|im_start|>assistant\n
        - Generated tokens (have logprobs): <think> ... </think> ... response ... <|im_end|>

        CRITICAL: The chat template MUST preserve thinking content for all assistant turns.
        The default Qwen3 template strips thinking from non-last turns, which breaks alignment.
        Use modify_chat_template_for_training.py to fix this before training.
        See CHAT_TEMPLATE_FOR_TRAINING.md for details.

        Returns:
            Tuple of (mask, logprob_tensor) where:
            - mask: List of 0/1 indicating which tokens were actually generated
            - logprob_tensor: List of logprobs aligned with tokens (0.0 for non-generated)
        """
        # Return cached result if available
        if self._cached_mask_and_logprobs is not None:
            return self._cached_mask_and_logprobs

        self._load_tokenizer()

        # Build token sequence from actual generated tokens (not re-tokenization)
        # For assistant messages, use SGLang's actual tokens to avoid tokenization mismatches
        full_tokens = []

        # Get special token IDs
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]

        # Track assistant message index for accessing all_generated_tokens
        assistant_msg_count = 0

        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            # Add message header: <|im_start|>{role}\n
            full_tokens.append(im_start_id)
            role_tokens = self.tokenizer.encode(role, add_special_tokens=False)
            full_tokens.extend(role_tokens)
            full_tokens.append(newline_id)

            # Add message content
            if role == "assistant":
                # Use actual generated tokens from SGLang (avoid re-tokenization mismatch)
                if assistant_msg_count < len(self.all_generated_tokens):
                    sglang_tokens = self.all_generated_tokens[assistant_msg_count]
                    # Convert token strings to IDs INCLUDING <|im_end|> if SGLang generated it
                    # SGLang's tokens include <|im_end|> with its own logprob
                    sglang_includes_im_end = sglang_tokens and sglang_tokens[-1] == "<|im_end|>"

                    for token_str in sglang_tokens:
                        # Convert token string to ID
                        token_id = self.tokenizer.encode(token_str, add_special_tokens=False)[0]
                        full_tokens.append(token_id)

                    # Add newline after message (this is template-added, not generated)
                    full_tokens.append(newline_id)

                    assistant_msg_count += 1
                    continue  # Skip the template <|im_end|> addition below since SGLang included it
                else:
                    # Fallback: tokenize normally if we don't have stored tokens
                    content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
                    full_tokens.extend(content_tokens)
                    assistant_msg_count += 1
            else:
                # For system/user messages, tokenize normally
                content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
                full_tokens.extend(content_tokens)

            # Add message suffix: <|im_end|>\n (for non-assistant or fallback)
            full_tokens.append(im_end_id)
            full_tokens.append(newline_id)

        # Initialize mask and logprob tensor
        mask = [0] * len(full_tokens)
        logprob_tensor = [0.0] * len(full_tokens)

        # Get assistant token ID (for scanning)
        assistant_id = self.tokenizer.encode("assistant", add_special_tokens=False)[0]

        # Count expected assistant messages
        num_assistant_messages = sum(1 for msg in self.messages if msg["role"] == "assistant")

        # Validate we have logprobs for all assistant messages
        if len(self.all_generated_logprobs) != num_assistant_messages:
            self._dump_debug_info_on_error(
                error_type="logprob_sublist_count_mismatch",
                num_assistant_messages=num_assistant_messages,
                num_logprob_sublists=len(self.all_generated_logprobs),
            )
            raise RuntimeError(
                f"Mismatch between assistant messages ({num_assistant_messages}) "
                f"and collected logprob sublists ({len(self.all_generated_logprobs)}). "
                f"Each assistant message should have a corresponding logprob sublist."
            )

        # Track which assistant message we're processing
        assistant_msg_idx = 0

        # Scan through tokens
        i = 0
        while i < len(full_tokens):
            # Look for assistant message start: <|im_start|>assistant\n
            if (i + 2 < len(full_tokens) and
                full_tokens[i] == im_start_id and
                full_tokens[i + 1] == assistant_id and
                full_tokens[i + 2] == newline_id):

                # Skip the assistant header (3 tokens)
                # These are part of the template, not generated
                i += 3

                # Get logprobs for this assistant message
                if assistant_msg_idx >= len(self.all_generated_logprobs):
                    self._dump_debug_info_on_error(
                        error_type="assistant_message_without_logprobs",
                        assistant_msg_idx=assistant_msg_idx,
                        num_logprob_sublists=len(self.all_generated_logprobs),
                        position=i,
                    )
                    raise RuntimeError(
                        f"Found more assistant messages in token sequence than collected logprob sublists. "
                        f"Processing assistant message #{assistant_msg_idx + 1} but only have {len(self.all_generated_logprobs)} sublists."
                    )

                current_message_logprobs = self.all_generated_logprobs[assistant_msg_idx]
                msg_logprob_idx = 0

                # Now we're at the start of actually generated tokens
                # Mark tokens and fill in logprobs, INCLUDING <|im_end|>
                # SGLang includes <|im_end|> in the logprobs when finish_reason="stop"
                start_pos = i
                while i < len(full_tokens):
                    # Mark this token as generated and assign its logprob
                    mask[i] = 1
                    logprob_tensor[i] = current_message_logprobs[msg_logprob_idx]
                    msg_logprob_idx += 1

                    # If this was <|im_end|>, we should have consumed all logprobs
                    if full_tokens[i] == im_end_id:
                        if msg_logprob_idx != len(current_message_logprobs):
                            self._dump_debug_info_on_error(
                                error_type="logprob_mismatch_at_im_end",
                                assistant_msg_idx=assistant_msg_idx + 1,
                                position=i,
                                consumed_logprobs=msg_logprob_idx,
                                expected_logprobs=len(current_message_logprobs),
                                unclaimed_logprobs=len(current_message_logprobs) - msg_logprob_idx,
                            )
                            raise RuntimeError(
                                f"Found <|im_end|> at position {i} for assistant message #{assistant_msg_idx + 1}, "
                                f"but only consumed {msg_logprob_idx}/{len(current_message_logprobs)} logprobs. "
                                f"Expected <|im_end|> to be the LAST token with logprob (finish_reason='stop'). "
                                f"This indicates unexpected SGLang behavior or a bug in mask building logic."
                            )
                        i += 1
                        break

                    # Check if we've consumed all logprobs without finding <|im_end|>
                    if msg_logprob_idx >= len(current_message_logprobs):
                        # We've consumed all logprobs but haven't hit <|im_end|> yet
                        # This is expected for finish_reason="length" (generation cut off)
                        # Verify we don't have more tokens to process before <|im_end|>
                        remaining_tokens = []
                        j = i + 1
                        while j < len(full_tokens) and full_tokens[j] != im_end_id:
                            remaining_tokens.append(self.tokenizer.decode([full_tokens[j]]))
                            j += 1

                        if remaining_tokens:
                            self._dump_debug_info_on_error(
                                error_type="tokens_without_logprobs",
                                assistant_msg_idx=assistant_msg_idx + 1,
                                position=i,
                                consumed_logprobs=len(current_message_logprobs),
                                remaining_tokens_count=len(remaining_tokens),
                                remaining_tokens_sample=remaining_tokens[:5],
                            )
                            raise RuntimeError(
                                f"Consumed all {len(current_message_logprobs)} logprobs for assistant message #{assistant_msg_idx + 1}, "
                                f"but found {len(remaining_tokens)} more tokens before <|im_end|>: {remaining_tokens[:5]}. "
                                f"This indicates a mismatch between SGLang's logprobs and the reconstructed token sequence. "
                                f"Possible cause: chat template is stripping/adding tokens."
                            )
                        break

                    i += 1

                # Verify we consumed all logprobs for this message
                if msg_logprob_idx != len(current_message_logprobs):
                    # Debug: Show what tokens we found vs what we expected
                    # Note: assistant_msg_idx indexes into all_generated_logprobs, not self.messages
                    assistant_messages = [msg for msg in self.messages if msg["role"] == "assistant"]
                    current_msg_content = assistant_messages[assistant_msg_idx]['content'] if assistant_msg_idx < len(assistant_messages) else "N/A"

                    self._dump_debug_info_on_error(
                        error_type="chat_template_logprob_mismatch",
                        assistant_msg_idx=assistant_msg_idx + 1,
                        consumed_tokens=msg_logprob_idx,
                        expected_logprobs=len(current_message_logprobs),
                        difference=len(current_message_logprobs) - msg_logprob_idx,
                        message_content_preview=current_msg_content[:200] if current_msg_content != "N/A" else "N/A",
                    )

                    error_msg = (
                        f"Logprob/token alignment failure for assistant message #{assistant_msg_idx + 1}:\n"
                        f"  Consumed: {msg_logprob_idx} tokens from reconstructed sequence\n"
                        f"  Expected: {len(current_message_logprobs)} logprobs from SGLang\n"
                        f"  Difference: {len(current_message_logprobs) - msg_logprob_idx} logprobs unmatched\n\n"
                        f"LIKELY CAUSE: Chat template is stripping thinking content!\n"
                        f"  The tokenizer's chat template may be removing <think>...</think> content\n"
                        f"  from assistant messages when reconstructing the sequence, but SGLang\n"
                        f"  returns logprobs for ALL generated tokens including thinking.\n\n"
                        f"SOLUTION:\n"
                        f"  1. Run: python modify_chat_template_for_training.py {self.model_path}\n"
                        f"  2. Restart any SGLang/vLLM servers using this model\n"
                        f"  3. See modify_chat_template_for_training.py for details\n\n"
                        f"Message content (first 200 chars): {repr(current_msg_content[:200])}\n"
                    )
                    print(f"\n{'='*80}\n{error_msg}{'='*80}\n")
                    raise RuntimeError(error_msg)

                # Move to next assistant message
                assistant_msg_idx += 1
            else:
                i += 1

        # Verify we processed all assistant messages
        if assistant_msg_idx != num_assistant_messages:
            self._dump_debug_info_on_error(
                error_type="assistant_message_count_mismatch",
                found_count=assistant_msg_idx,
                expected_count=num_assistant_messages,
            )
            raise RuntimeError(
                f"Found {assistant_msg_idx} assistant messages in token sequence "
                f"but expected {num_assistant_messages} based on self.messages."
            )

        # Cache the result along with the token sequence
        self._cached_mask_and_logprobs = (mask, logprob_tensor)
        self._cached_input_sequence = full_tokens

        return mask, logprob_tensor

    def get_assistant_mask(self) -> List[int]:
        """Generate a mask that is 1 for tokens actually generated by the model.

        Excludes template-added tokens (<|im_start|>assistant\n) but includes all
        generated tokens that have logprobs from SGLang, including <|im_end|>.

        For Qwen3-8B: mask includes <think>, thinking content, </think>, response, <|im_end|>

        Returns:
            List of 0s and 1s, where 1 indicates tokens the model generated
        """
        mask, _ = self._build_mask_and_logprobs()
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
        _, logprob_tensor = self._build_mask_and_logprobs()
        return logprob_tensor

    def _get_full_conversation_text(self) -> str:
        """Get the full conversation as text (for debugging).

        Returns:
            The full conversation formatted with the chat template
        """
        self._load_tokenizer()
        return self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _validate_thinking_budget(self, response_text: str, output_tokens: int):
        """Validate that max_thinking_tokens was respected.

        If max_thinking_tokens was set but the thinking block exceeds it significantly,
        this indicates the SGLang patch was not applied. Log loudly but don't crash.
        """
        import logging
        import re

        max_thinking = getattr(self, '_expected_max_thinking_tokens', None)
        if max_thinking is None:
            return  # No limit was set

        # Extract thinking block from response
        # Format: <think>...</think>
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if not think_match:
            return  # No thinking block found

        thinking_content = think_match.group(1)

        # Rough token estimate: ~4 chars per token for English
        # This is imprecise but good enough for detecting violations
        estimated_thinking_tokens = len(thinking_content) // 4

        if estimated_thinking_tokens > max_thinking:
            logger = logging.getLogger(__name__)
            warning_msg = (
                f"[WARNING] max_thinking_tokens={max_thinking} was set but thinking block "
                f"has ~{estimated_thinking_tokens} tokens. "
                f"This suggests the SGLang thinking budget patch is NOT applied! "
                f"Run: python sglang_patch/apply_patch.py"
            )
            logger.warning(warning_msg)
            self.console.print(f"[yellow bold]{warning_msg}[/yellow bold]")

            # Also print to stderr for visibility in logs
            import sys
            print(warning_msg, file=sys.stderr)

    def _dump_max_tokens_exceeded(self, response_text: str, output_tokens: int, max_tokens: int):
        """Dump conversation when a response hits max_tokens limit.

        This indicates degenerate output (e.g., repetition loops) and saves
        the full conversation for debugging.

        Args:
            response_text: The truncated response that hit the limit
            output_tokens: Number of tokens generated
            max_tokens: The max_tokens limit that was hit
        """
        import json
        from datetime import datetime

        # Create dedicated directory for max tokens exceeded dumps
        debug_dir = "max_tokens_exceeded_dumps"
        os.makedirs(debug_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"max_tokens_{self.role}_{timestamp}.json"
        filepath = os.path.join(debug_dir, filename)

        # Collect conversation state
        debug_data = {
            "error_type": "max_tokens_exceeded",
            "role": self.role,
            "model_path": self.model_path,
            "output_tokens": output_tokens,
            "max_tokens": max_tokens,

            # The problematic response
            "truncated_response": response_text,
            "truncated_response_length_chars": len(response_text),

            # Show the end of the response to see repetition pattern
            "response_last_500_chars": response_text[-500:] if len(response_text) > 500 else response_text,

            # Full conversation history
            "messages": self.messages,
            "num_messages": len(self.messages),

            # Config
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }

        try:
            with open(filepath, 'w') as f:
                json.dump(debug_data, f, indent=2, default=str)
            print(f"\n[MAX TOKENS EXCEEDED] Saved debug info to: {filepath}")
        except Exception as e:
            print(f"\n[MAX TOKENS EXCEEDED] Failed to save debug info: {e}")

    def _dump_debug_info_on_error(self, error_type: str, **kwargs):
        """Dump complete state for debugging mask/logprob errors.

        Saves all relevant information to a JSON file for later analysis and replay.

        Args:
            error_type: Type of error (e.g., "logprob_mismatch", "exception")
            **kwargs: Additional error-specific details
        """
        # Recursion guard: prevent infinite loop if error dump itself fails
        if self._in_error_dump:
            print(f"[WARNING] Skipping nested error dump (already dumping) for {error_type}")
            return

        self._in_error_dump = True
        try:
            import json
            from datetime import datetime

            # Create debug dumps directory
            debug_dir = "debug_dumps"
            os.makedirs(debug_dir, exist_ok=True)

            # Unique filename with microsecond precision
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{error_type}_{self.role}_{timestamp}.json"
            filepath = os.path.join(debug_dir, filename)

            # Load tokenizer if needed
            self._load_tokenizer()

            # Safely get full text (may fail if template is broken)
            try:
                full_text = self._get_full_conversation_text()
            except Exception as e:
                full_text = f"<ERROR: Failed to render conversation text: {e}>"

            # Safely get token sequence (recursion guard prevents infinite loop)
            try:
                full_tokens = self.get_input_sequence()
                full_tokens_length = len(full_tokens)
            except Exception as e:
                full_tokens = f"<ERROR: Failed to get input sequence: {e}>"
                full_tokens_length = -1

            # Collect ALL relevant state
            debug_data = {
                "error_type": error_type,
                "error_details": kwargs,
                "role": self.role,
                "model_path": self.model_path,

                # Conversation state
                "messages": self.messages,
                "num_messages": len(self.messages),
                "num_assistant_messages": sum(1 for m in self.messages if m["role"] == "assistant"),

                # Logprob collection
                "all_generated_logprobs": self.all_generated_logprobs,
                "all_generated_tokens": self.all_generated_tokens,  # Token strings from SGLang
                "num_sublists": len(self.all_generated_logprobs),
                "sublist_lengths": [len(s) for s in self.all_generated_logprobs],
                "total_logprobs": sum(len(s) for s in self.all_generated_logprobs),

                # Tokenization (with safe fallbacks)
                "tokenizer_name": self.tokenizer.name_or_path if self.tokenizer else None,
                "full_text": full_text,
                "full_tokens": full_tokens,
                "full_tokens_length": full_tokens_length,

                # Config
                "config": {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_p": self.config.top_p,
                    "server_url": self.config.server_url if hasattr(self.config, 'server_url') else None,
                },
            }

            # Save to file (still inside try block to ensure variables are defined)
            try:
                with open(filepath, 'w') as f:
                    json.dump(debug_data, f, indent=2, default=str)
                print(f"\n[DEBUG DUMP] Saved error context to: {filepath}")
            except Exception as save_err:
                print(f"\n[DEBUG DUMP] Failed to save debug info: {save_err}")

        except Exception as outer_err:
            # If anything goes wrong during debug dump, log it but don't crash
            print(f"\n[DEBUG DUMP] Error while collecting debug data: {outer_err}")
        finally:
            # Always reset recursion guard, even if dump fails
            self._in_error_dump = False

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

        # Add max_thinking_tokens if configured (for Qwen3 thinking budget limitation)
        # NOTE: This requires the SGLang thinking budget patch to be applied.
        max_thinking_tokens = gen_kwargs.get("max_thinking_tokens", getattr(self.config, "max_thinking_tokens", None))
        if max_thinking_tokens is not None:
            sampling_params["max_thinking_tokens"] = max_thinking_tokens
            self._expected_max_thinking_tokens = max_thinking_tokens
        else:
            self._expected_max_thinking_tokens = None

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

            # Validate max_thinking_tokens was respected (detect if patch not applied)
            self._validate_thinking_budget(response_text, output_tokens)

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
