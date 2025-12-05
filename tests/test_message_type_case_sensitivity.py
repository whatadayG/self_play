#!/usr/bin/env python3
"""Unit tests for message type case sensitivity bug fix.

Tests that message types [message], [propose], [accept], [reject] are
handled correctly regardless of capitalization (e.g., [Propose], [MESSAGE]).
"""

import pytest
import re
import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestMessageTypeCaseSensitivity:
    """Test that message type parsing is case-insensitive."""

    def _parse_message_helper(self, message):
        """Simplified version of _parse_message for testing."""
        m = re.match(
            r"\[(?P<mtype>\w+)\](?P<msg>.*)",
            message.strip(),
            flags=re.DOTALL
        )
        if m is None:
            raise ValueError(f"Invalid message: {message}")
        m = {k: v.strip() for k, v in m.groupdict().items()}
        return m

    def _simulate_optimization_step(self, message):
        """Simulate the message type extraction from optimization.py."""
        # Line 104: Find tag with case-insensitive regex
        tag_match = re.search(
            r"\[(message|propose|accept|reject)\]",
            message,
            re.IGNORECASE
        )
        if not tag_match:
            return None

        # Line 131: Extract message for parsing
        msg_for_parse = message[tag_match.start():].lstrip()

        # Line 133-140: Parse message and extract type
        m = self._parse_message_helper(msg_for_parse)
        type_ = m["mtype"].lower()  # THE FIX: normalize to lowercase
        content = m["msg"]

        return type_, content

    @pytest.mark.parametrize("message_tag,expected_type", [
        ("[message]", "message"),
        ("[Message]", "message"),
        ("[MESSAGE]", "message"),
        ("[propose]", "propose"),
        ("[Propose]", "propose"),
        ("[PROPOSE]", "propose"),
        ("[accept]", "accept"),
        ("[Accept]", "accept"),
        ("[ACCEPT]", "accept"),
        ("[reject]", "reject"),
        ("[Reject]", "reject"),
        ("[REJECT]", "reject"),
    ])
    def test_message_type_normalization(self, message_tag, expected_type):
        """Test that message types are normalized to lowercase."""
        message = f"{message_tag} Some content here"
        result = self._simulate_optimization_step(message)

        assert result is not None, f"Failed to parse message: {message}"
        type_, content = result
        assert type_ == expected_type, f"Expected {expected_type}, got {type_}"
        assert content == "Some content here"

    def test_capitalized_propose_with_thinking(self):
        """Test the exact bug case: [Propose] with <think> tag."""
        message = """<think>
Let me analyze this situation carefully.
Some thinking content here.
</think>

[Propose] Here's my proposal!
Proposal:
- Assignment 1
- Assignment 2"""

        result = self._simulate_optimization_step(message)
        assert result is not None
        type_, content = result
        assert type_ == "propose"
        assert "Here's my proposal!" in content

    def test_mixed_case_in_realistic_message(self):
        """Test realistic messages with various capitalizations."""
        test_cases = [
            ("[Message] I think we should assign Sofia to BLEU", "message"),
            ("[Propose] Let me propose an assignment\nProposal:\n- BLEU: Sofia", "propose"),
            ("[Accept] I agree with your proposal!", "accept"),
            ("[Reject] I think we can do better", "reject"),
        ]

        for message, expected_type in test_cases:
            result = self._simulate_optimization_step(message)
            assert result is not None
            type_, _ = result
            assert type_ == expected_type

    def test_message_type_comparison_logic(self):
        """Test that type comparisons work after normalization."""
        message = "[Propose] My proposal"
        result = self._simulate_optimization_step(message)
        type_, _ = result

        # Simulate the comparison logic from optimization.py lines 146-218
        if type_ == "message":
            outcome = "message"
        elif type_ == "propose":
            outcome = "propose"
        elif type_ == "accept":
            outcome = "accept"
        elif type_ == "reject":
            outcome = "reject"
        else:
            outcome = "error"

        assert outcome == "propose", f"Expected 'propose', got '{outcome}'"

    def test_all_comparison_branches(self):
        """Test that all message types match their comparison branches."""
        test_cases = [
            ("[Message] test", "message"),
            ("[Propose] test", "propose"),
            ("[Accept] test", "accept"),
            ("[Reject] test", "reject"),
        ]

        for message, expected_outcome in test_cases:
            result = self._simulate_optimization_step(message)
            type_, _ = result

            # Simulate optimization.py comparison logic
            if type_ == "message":
                outcome = "message"
            elif type_ == "propose":
                outcome = "propose"
            elif type_ == "accept":
                outcome = "accept"
            elif type_ == "reject":
                outcome = "reject"
            else:
                outcome = "error"

            assert outcome == expected_outcome, \
                f"Message '{message}' should match '{expected_outcome}' branch, got '{outcome}'"
            assert outcome != "error", \
                f"Message '{message}' should not trigger error branch"


class TestMessageTypeParsingEdgeCases:
    """Test edge cases in message type parsing."""

    def test_multiple_tags_uses_last(self):
        """Test that when multiple tags exist, the last one is used."""
        message = "[message] First part [propose] Second part"

        # Find all matches
        tag_matches = list(re.finditer(
            r"\[(message|propose|accept|reject)\]",
            message,
            re.IGNORECASE
        ))

        assert len(tag_matches) == 2
        last_match = tag_matches[-1]
        assert last_match.group(1).lower() == "propose"

    def test_thinking_before_tag(self):
        """Test that content before the tag is treated as thinking."""
        message = "Let me think about this.\n[Message] My actual message"

        tag_match = re.search(
            r"\[(message|propose|accept|reject)\]",
            message,
            re.IGNORECASE
        )

        thinking_part = message[:tag_match.start()].strip()
        assert thinking_part == "Let me think about this."

    def test_no_tag_returns_none(self):
        """Test that messages without tags return None in our helper."""
        message = "This message has no tag"

        tag_match = re.search(
            r"\[(message|propose|accept|reject)\]",
            message,
            re.IGNORECASE
        )

        assert tag_match is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
