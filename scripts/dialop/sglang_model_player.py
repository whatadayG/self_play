"""
Sglang-backed model player using an OpenAI-compatible HTTP API.
Loads nothing locally; talks to a long-lived sglang server.
"""

import os
import random
from typing import Optional, Dict, Any

import requests
from rich.markup import escape


class SglangModelPlayer:
    """Player that queries an sglang server (OpenAI-compatible API)."""

    def __init__(
        self,
        prompt: str,
        role: str,
        console,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prefix: str = "\nYou:",
        optional: Optional[str] = None,
        sglang_url: str = "http://localhost:30000/v1",
        temperature: float = 0.7,
        timeout_s: float = 120.0,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.prompt = prompt
        self.role = role
        self.console = console
        self.model_path = model_path
        self.prefix = prefix
        self.optional = optional
        self.removed_optional = False
        self.temperature = temperature
        self.timeout_s = timeout_s

        # cost tracking compatibility
        self.api_tracker = None

        # base url must include /v1 for OpenAI-compatible route
        base_url = sglang_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        self.base_url = base_url
        self.completions_url = f"{self.base_url}/chat/completions"

        # optional API key if server requires (often 'EMPTY')
        self.api_key = os.environ.get("SGLANG_API_KEY", "EMPTY")

        self.console.print(f"Using sglang server at {self.base_url} for {role}")

    def observe(self, obs: str, info: bool = False, force: bool = False, ignore_obs: bool = False):
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

    def _remove_optional_context(self):
        if self.optional and not self.removed_optional:
            self.prompt = self.prompt.replace(self.optional, "")
            self.removed_optional = True

    def respond(
        self,
        t: int = 0,
        max_len: int = 1024,
        vary: bool = False,
        propose: bool = False,
        temporal_id: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> str:
        # prepare prompt
        selfprompt = self.prompt
        if strategy:
            selfprompt = selfprompt + "\n" + "Here is your conversational strategy: " + strategy

        if not selfprompt.endswith(self.prefix):
            if propose:
                selfprompt += (self.prefix + "[propose]")
            else:
                selfprompt += self.prefix

        messages = [{"role": "system", "content": selfprompt}]

        # build request
        effective_temperature = 1.8 if vary else self.temperature
        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": effective_temperature,
            "top_p": 0.9,
            "max_tokens": 256,
            "n": 1,
            "stop": ["User:", "Agent:", "You:"],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # send
        resp = requests.post(self.completions_url, json=payload, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible
        response_text = data["choices"][0]["message"]["content"] if data.get("choices") else ""

        # cleanup echoed tags
        stop_strings = ["User:", "Agent:", "You:"]
        response_text = (response_text or "").strip()
        for stop in stop_strings:
            if response_text.startswith(stop):
                response_text = response_text[len(stop):].lstrip()
        for stop in stop_strings:
            idx = response_text.find(stop)
            if idx > 0:
                response_text = response_text[:idx]
                break
        response_text = response_text.strip()

        # log
        self.console.rule(f"{self.role}'s turn")
        self.console.print(f"Response: {escape(response_text)}")

        # token usage (best-effort from API)
        input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
        if input_tokens or output_tokens:
            self.console.print(f"Tokens - Input: {input_tokens}, Output: {output_tokens}")

        # cost tracker (zero cost for local)
        if getattr(self, "api_tracker", None) is not None:
            try:
                self.api_tracker.log_api_call(
                    model=self.model_path,
                    input_text=selfprompt,
                    output_text=response_text,
                    project_id=self.role,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0,
                )
            except Exception as e:
                self.console.print(f"[yellow]Cost tracking error: {e}[/yellow]")

        return response_text 