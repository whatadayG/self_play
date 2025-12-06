#!/usr/bin/env python3
"""
Unit tests for proposal parsing in OptimizationEnv.

Tests that proposals with explanatory text before "Proposal:" are parsed correctly.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dialop.envs.optimization import OptimizationEnv


class TestProposalParser:
    """Test the _parse_proposal method handles various formats correctly."""

    def setup_method(self):
        """Create a fresh OptimizationEnv for each test."""
        self.env = OptimizationEnv()
        self.env.reset(seed=42)

    def test_proposal_with_preamble_text(self):
        """Test that proposals with explanatory text before 'Proposal:' are parsed correctly."""
        # This is the actual format that GPT-5-mini produces
        message = """I'm locking Ava->Electra and Sofia->RoBERTa and returning the favor by giving you Andrei on GLUE (and I also assign Daniel to QuAC since that's another clear top of yours). Please review — I think this balances both our signals.

Proposal:
 - BLEU: Morgan Reed
 - Electra: Ava Li
 - GloVe: Ethan Smith
 - GLUE: Andrei Petrov
 - LLaMA: Joseph Santos
 - RoBERTa: Sofia Patel
 - QuAC: Daniel Nguyen
 - SWAG: Noah Wilson"""

        # Should successfully parse 8 assignments
        result = self.env._parse_proposal(message)
        assert len(result) == 8, f"Expected 8 assignments, got {len(result)}"

    def test_proposal_without_preamble(self):
        """Test that proposals without preamble text still work."""
        message = """Proposal:
 - BLEU: Morgan Reed
 - Electra: Ava Li
 - GloVe: Ethan Smith
 - GLUE: Andrei Petrov
 - LLaMA: Joseph Santos
 - RoBERTa: Sofia Patel
 - QuAC: Daniel Nguyen
 - SWAG: Noah Wilson"""

        result = self.env._parse_proposal(message)
        assert len(result) == 8

    def test_proposal_with_multiline_preamble(self):
        """Test proposals with multiple lines of preamble text."""
        message = """Here's my thinking on this:
I believe we should prioritize the strongest matches.
Let me propose the following allocation.

Proposal:
 - BLEU: Morgan Reed
 - Electra: Ava Li
 - GloVe: Ethan Smith
 - GLUE: Andrei Petrov
 - LLaMA: Joseph Santos
 - RoBERTa: Sofia Patel
 - QuAC: Daniel Nguyen
 - SWAG: Noah Wilson"""

        result = self.env._parse_proposal(message)
        assert len(result) == 8

    def test_proposal_with_too_few_assignments(self):
        """Test that proposals with wrong number of assignments raise error."""
        message = """Proposal:
 - BLEU: Morgan Reed
 - Electra: Ava Li
 - GloVe: Ethan Smith
 - GLUE: Andrei Petrov
 - LLaMA: Joseph Santos
 - RoBERTa: Sofia Patel
 - QuAC: Daniel Nguyen"""

        with pytest.raises(Exception) as exc_info:
            self.env._parse_proposal(message)
        assert "detected 7" in str(exc_info.value).lower()

    def test_proposal_missing_header(self):
        """Test that proposals without 'Proposal:' header raise error."""
        message = """ - BLEU: Morgan Reed
 - Electra: Ava Li
 - GloVe: Ethan Smith
 - GLUE: Andrei Petrov
 - LLaMA: Joseph Santos
 - RoBERTa: Sofia Patel
 - QuAC: Daniel Nguyen
 - SWAG: Noah Wilson"""

        with pytest.raises(Exception) as exc_info:
            self.env._parse_proposal(message)
        assert "proposal" in str(exc_info.value).lower() and "start" in str(exc_info.value).lower()


class TestProposalParserEndToEnd:
    """Test the full step() -> _propose() -> _parse_proposal() flow.

    These tests exercise the actual code path, unlike tests that call
    _parse_proposal() directly. This catches bugs in the preprocessing
    done in step() before _parse_proposal() is called.
    """

    def setup_method(self):
        """Create a fresh OptimizationEnv for each test."""
        self.env = OptimizationEnv()
        self.env.reset(seed=42)

    def test_step_with_propose_newline_after_header(self):
        """Test that [propose] messages with newline after 'Proposal:' are parsed correctly.

        This is the exact format that GPT-5-mini produces:
        [propose] Some preamble text
        Proposal:
         - BLEU: Reviewer1
         - ...

        The bug was that step() did:
            proposal_content = "Proposal:" + content.split("Proposal:")[1].strip()

        The .strip() removed the newline after "Proposal:", merging the header
        with the first assignment line, causing the parser to count 7 instead of 8.
        """
        # This is the exact format from the failing games
        message = """[propose] Here's a proposal I think is strong based on my information.
Proposal:
 - BLEU: Ava Li
 - Electra: Joseph Santos
 - GloVe: Ethan Smith
 - GLUE: Daniel Nguyen
 - LLaMA: Sofia Patel
 - RoBERTa: Noah Wilson
 - QuAC: Andrei Petrov
 - SWAG: Morgan Reed"""

        # Call the actual step() method - this is the end-to-end test
        obs, error = self.env.step(message)

        # Should NOT have an error about wrong assignment count
        assert not error, f"step() returned error: {obs}"

        # Verify the proposal was parsed correctly
        assert self.env.game.proposal is not None, "Proposal was not set"
        assert len(self.env.game.proposal) == 8, f"Expected 8 assignments, got {len(self.env.game.proposal)}"

    def test_step_with_propose_short_paper_names(self):
        """Test proposals using short paper names (without subtitles)."""
        message = """[propose] I propose the following assignments.
Proposal:
 - BLEU: Morgan Reed
 - Electra: Ava Li
 - GloVe: Ethan Smith
 - GLUE: Andrei Petrov
 - LLaMA: Joseph Santos
 - RoBERTa: Sofia Patel
 - QuAC: Daniel Nguyen
 - SWAG: Noah Wilson"""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8

    def test_step_with_propose_full_paper_names(self):
        """Test proposals using full paper names (with subtitles containing colons)."""
        message = """[propose] Here's my proposal.
Proposal:
 - BLEU: a Method for Automatic Evaluation of MT: Noah Wilson
 - Electra: Pre-training Text Encoders as Discriminators: Morgan Reed
 - GloVe: Global Vectors for Word Representation: Daniel Nguyen
 - GLUE: A Multi-Task Benchmark and Analysis Platform for NLU: Andrei Petrov
 - LLaMA: Open and Efficient Foundation Language Models: Sofia Patel
 - RoBERTa: A Robustly Optimized BERT Pretraining Approach: Joseph Santos
 - QuAC: Question Answering in Context: Ethan Smith
 - SWAG: An Adversarial Dataset for Commonsense Inference: Ava Li"""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8

    def test_step_with_propose_no_leading_space(self):
        """Test proposals where assignment lines don't have leading space."""
        message = """[propose] My proposal:
Proposal:
- BLEU: Noah Wilson
- Electra: Ava Li
- GloVe: Sofia Patel
- GLUE: Ethan Smith
- LLaMA: Morgan Reed
- RoBERTa: Andrei Petrov
- QuAC: Daniel Nguyen
- SWAG: Joseph Santos"""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8

    def test_step_with_propose_multiline_preamble(self):
        """Test proposals with extensive preamble text before Proposal:"""
        message = """[propose] I've analyzed the scores carefully.
Based on my view of the table, I see strong matches for several reviewers.
Let me propose an assignment that maximizes the visible high scores.

Proposal:
 - BLEU: Ava Li
 - Electra: Joseph Santos
 - GloVe: Ethan Smith
 - GLUE: Daniel Nguyen
 - LLaMA: Sofia Patel
 - RoBERTa: Noah Wilson
 - QuAC: Andrei Petrov
 - SWAG: Morgan Reed"""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8

    def test_step_with_propose_extra_text_after_assignments(self):
        """Test proposals with extra explanatory text AFTER the 8 assignments.

        This is a common pattern where the model adds helpful text after the
        formal proposal. The parser should ignore this extra text.

        Bug: Previously, the parser would include the extra text as proposal
        lines, causing either 'wrong count' or 'invalid line format' errors.
        """
        message = """[propose] Here's my proposal.
Proposal:
 - BLEU: Morgan Reed
 - Electra: Andrei Petrov
 - GloVe: Sofia Patel
 - GLUE: Noah Wilson
 - LLaMA: Ava Li
 - RoBERTa: Ethan Smith
 - QuAC: Daniel Nguyen
 - SWAG: Joseph Santos

Please accept if this looks good to you, or reject and tell me which single swap you'd prefer (e.g., BLEU/Joseph ↔ BLEU/Morgan, or GLUE↔Noah swapped with someone) and why."""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8

    def test_step_with_propose_extra_text_multiple_paragraphs(self):
        """Test proposals with multiple paragraphs of extra text after assignments."""
        message = """[propose] My proposal based on our discussion.
Proposal:
 - BLEU: Ava Li
 - Electra: Joseph Santos
 - GloVe: Ethan Smith
 - GLUE: Daniel Nguyen
 - LLaMA: Sofia Patel
 - RoBERTa: Noah Wilson
 - QuAC: Andrei Petrov
 - SWAG: Morgan Reed

This assignment gives us strong coverage across all papers.

If you have concerns about any specific assignment, please let me know and we can discuss alternatives. I'm particularly flexible on the GLUE and SWAG assignments."""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8

    def test_step_with_propose_html_br_format(self):
        """Test proposals using HTML <br/> line breaks and &emsp; indentation.

        Some models output proposals in this format, which uses HTML entities
        instead of newlines and spaces. The parser must handle this correctly.
        """
        message = """[propose] My proposal:
Proposal:<br/>&emsp; - BLEU: Joseph Santos<br/>&emsp; - Electra: Andrei Petrov<br/>&emsp; - GloVe: Daniel Nguyen<br/>&emsp; - GLUE: Sofia Patel<br/>&emsp; - LLaMA: Noah Wilson<br/>&emsp; - RoBERTa: Ava Li<br/>&emsp; - QuAC: Ethan Smith<br/>&emsp; - SWAG: Morgan Reed"""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8

    def test_step_with_propose_html_nbsp_format(self):
        """Test proposals using &nbsp; for spacing."""
        message = """[propose] My proposal:
Proposal:<br/>&nbsp;- BLEU: Joseph Santos<br/>&nbsp;- Electra: Andrei Petrov<br/>&nbsp;- GloVe: Daniel Nguyen<br/>&nbsp;- GLUE: Sofia Patel<br/>&nbsp;- LLaMA: Noah Wilson<br/>&nbsp;- RoBERTa: Ava Li<br/>&nbsp;- QuAC: Ethan Smith<br/>&nbsp;- SWAG: Morgan Reed"""

        obs, error = self.env.step(message)
        assert not error, f"step() returned error: {obs}"
        assert len(self.env.game.proposal) == 8


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
