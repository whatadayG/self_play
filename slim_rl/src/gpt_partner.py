"""Shy GPT-4.1 partner that only shares information when directly asked.

This module implements a "shy" partner behavior for the paper-reviewer
matching game. The partner:
- Only shares specific scores when directly asked
- Gives vague responses to general questions
- Never volunteers information proactively
- Evaluates proposals honestly based on its view
"""
import os
import asyncio
from typing import Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


SHY_PARTNER_SYSTEM_PROMPT = """You are a reviewer coordinator in a paper-reviewer matching game. You can see some affinity scores between reviewers and papers that your partner cannot see.

IMPORTANT BEHAVIORAL RULES - You are SHY:
1. NEVER volunteer information proactively - wait to be asked
2. Only share a specific score when DIRECTLY asked about that exact reviewer-paper pair
   - Good question: "What is the score for Ava Li reviewing BLEU?" -> Share that exact score
   - Vague question: "What do you know about Ava Li?" -> Say "I have some information about Ava Li's scores. Which paper are you interested in?"
3. When asked vague or general questions, give brief non-committal responses
4. Never reveal all your scores at once - make the agent work for each piece of information
5. When the agent proposes an assignment, evaluate it honestly:
   - Calculate the total score based on your known values
   - Accept if it seems good, reject if you think it can be improved
   - You can suggest which assignments might be suboptimal without revealing exact scores

YOUR VIEW OF THE GAME (these are the scores YOU can see):
{partner_view}

Remember: Be helpful but make the agent ask specific questions to get specific information!"""


class ShyGPTPartner:
    """GPT-4.1 partner with shy behavior for paper-reviewer matching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        """Initialize the GPT partner.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4.1)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        if AsyncOpenAI is None:
            raise ImportError("openai package required. Install with: pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def respond(self, dialogue: list[dict], partner_view: str) -> str:
        """Generate a shy response given dialogue history and partner's view.

        Args:
            dialogue: List of {"role": ..., "content": ...} messages
                      from Qwen's perspective (Qwen = assistant, GPT input = user)
            partner_view: String describing what GPT can see (the partner's table)

        Returns:
            GPT's response string
        """
        # Build messages with partner's view injected into system prompt
        system_prompt = SHY_PARTNER_SYSTEM_PROMPT.format(partner_view=partner_view)
        messages = [{"role": "system", "content": system_prompt}]

        # Convert dialogue history
        # From Qwen's perspective: Qwen is "assistant", partner responses are "user"
        # From GPT's perspective: Qwen's messages are "user", GPT's responses are "assistant"
        for msg in dialogue:
            if msg["role"] == "system":
                continue  # Skip Qwen's system prompt

            if msg["role"] == "assistant":
                # This was Qwen's message -> becomes GPT's input
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "user":
                # This was GPT's previous response -> becomes GPT's output
                messages.append({"role": "assistant", "content": msg["content"]})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def respond_sync(self, dialogue: list[dict], partner_view: str) -> str:
        """Synchronous version of respond for testing.

        Args:
            dialogue: Dialogue history
            partner_view: Partner's visible scores

        Returns:
            GPT's response string
        """
        return asyncio.run(self.respond(dialogue, partner_view))


class MockShyPartner:
    """Mock partner for testing without API calls.

    This partner uses simple rules to simulate shy behavior:
    - Responds to specific score queries with made-up scores
    - Gives vague responses to general questions
    - Accepts/rejects proposals randomly
    """

    def __init__(self, seed: int = 42):
        """Initialize mock partner.

        Args:
            seed: Random seed for reproducible responses
        """
        import random
        self.rng = random.Random(seed)
        self.scores_revealed = {}

    async def respond(self, dialogue: list[dict], partner_view: str) -> str:
        """Generate a mock shy response.

        Args:
            dialogue: Dialogue history
            partner_view: Partner's view (parsed to extract scores)

        Returns:
            Mock response string
        """
        if not dialogue:
            return "Hello! I'm ready to help find a good assignment. What would you like to know?"

        # Get the last message from Qwen
        last_msg = None
        for msg in reversed(dialogue):
            if msg["role"] == "assistant":
                last_msg = msg["content"].lower()
                break

        if not last_msg:
            return "I'm here to help. What specific scores would you like to know about?"

        # Check for proposal
        if "[propose]" in last_msg:
            if self.rng.random() > 0.3:
                return "[accept] That looks like a reasonable assignment!"
            else:
                return "[reject] I think we can do better. Some of those assignments don't match well with my information."

        # Check for specific score query
        import re
        # Pattern: "score for X reviewing Y" or "X ... Y ... score"
        if "score" in last_msg and ("for" in last_msg or "between" in last_msg):
            score = self.rng.randint(20, 90)
            return f"The score for that assignment is {score}."

        # Check for question mark (general question)
        if "?" in last_msg:
            responses = [
                "I have some information about that. Could you be more specific about which reviewer-paper pair?",
                "I can help with that. Which specific assignment are you asking about?",
                "Let me check... which reviewer and which paper exactly?",
                "I know some scores there. Ask me about a specific reviewer-paper combination.",
            ]
            return self.rng.choice(responses)

        # Default response
        return "I see. Do you have any specific questions about scores, or would you like to propose an assignment?"

    def respond_sync(self, dialogue: list[dict], partner_view: str) -> str:
        """Synchronous version."""
        return asyncio.run(self.respond(dialogue, partner_view))
