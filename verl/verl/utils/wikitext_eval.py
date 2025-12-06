"""
Utility for evaluating next token prediction loss on wikitext dataset.

This module provides functions to compute language modeling perplexity
on wikitext as a general capability metric during training.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


def load_wikitext_sample(file_path: str) -> str:
    """Load wikitext sample from file.

    Args:
        file_path: Path to wikitext sample text file

    Returns:
        Text content as string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def compute_wikitext_loss(
    model,
    tokenizer: PreTrainedTokenizer,
    wikitext_path: str,
    max_seq_length: int = 2048,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> dict[str, float]:
    """Compute next token prediction loss on wikitext.

    Args:
        model: Language model (should be a HuggingFace model or compatible)
        tokenizer: Tokenizer for the model
        wikitext_path: Path to wikitext sample file
        max_seq_length: Maximum sequence length for evaluation chunks
        batch_size: Number of chunks to process in parallel
        device: Device to run evaluation on (defaults to model device)

    Returns:
        Dictionary containing:
            - 'wikitext/loss': Average cross-entropy loss
            - 'wikitext/perplexity': Perplexity (exp(loss))
            - 'wikitext/num_tokens': Number of tokens evaluated
    """
    # Load text
    text = load_wikitext_sample(wikitext_path)

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Get device
    if device is None:
        device = next(model.parameters()).device

    # Get pad token id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Split into chunks of max_seq_length
    chunks = []
    for i in range(0, len(tokens), max_seq_length):
        chunk = tokens[i : i + max_seq_length]
        if len(chunk) >= 2:  # Need at least 2 tokens (input + target)
            chunks.append(chunk)

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        # Process chunks in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]

            # Find max length in this batch (excluding last token for input)
            max_len = max(len(chunk) - 1 for chunk in batch_chunks)

            # Prepare batched tensors
            input_ids_list = []
            target_ids_list = []
            attention_mask_list = []

            for chunk in batch_chunks:
                # Split into input (all but last) and target (all but first)
                input_tokens = chunk[:-1]
                target_tokens = chunk[1:]

                # Pad to max_len
                pad_len = max_len - len(input_tokens)
                padded_input = input_tokens + [pad_token_id] * pad_len
                padded_target = target_tokens + [-100] * pad_len  # -100 = ignore in loss
                attention_mask = [1] * len(input_tokens) + [0] * pad_len

                input_ids_list.append(padded_input)
                target_ids_list.append(padded_target)
                attention_mask_list.append(attention_mask)

            # Convert to tensors
            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
            target_ids = torch.tensor(target_ids_list, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)

            # Extract logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

            # Compute loss (ignoring padded positions marked with -100)
            # logits: (batch_size, seq_len, vocab_size)
            # target_ids: (batch_size, seq_len)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="sum",
                ignore_index=-100
            )

            total_loss += loss.item()
            total_tokens += (target_ids != -100).sum().item()

    # Compute average loss and perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "wikitext/loss": avg_loss,
        "wikitext/perplexity": perplexity,
        "wikitext/num_tokens": total_tokens,
    }


def compute_wikitext_loss_fsdp(
    worker_group,
    wikitext_path: str,
    max_seq_length: int = 2048,
    batch_size: int = 8,
) -> dict[str, float]:
    """Compute wikitext loss using FSDP worker group.

    This function is designed to work with verl's FSDP worker groups.
    It dispatches the evaluation to the actor workers.

    Args:
        worker_group: FSDP worker group (e.g., actor_rollout_wg)
        wikitext_path: Path to wikitext sample file
        max_seq_length: Maximum sequence length for evaluation chunks
        batch_size: Number of chunks to process in parallel

    Returns:
        Dictionary containing loss, perplexity, and token count
    """
    # Load text on driver
    text = load_wikitext_sample(wikitext_path)

    # Define remote function to run on workers
    def _compute_loss_on_worker(text: str, max_seq_length: int, batch_size: int):
        """Function to run on each worker."""
        import torch
        import torch.nn.functional as F

        # Get model and tokenizer from worker
        model = self.actor  # FSDP model
        tokenizer = self.tokenizer

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Get device
        device = next(model.parameters()).device

        # Get pad token id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Split into chunks of max_seq_length
        chunks = []
        for i in range(0, len(tokens), max_seq_length):
            chunk = tokens[i : i + max_seq_length]
            if len(chunk) >= 2:  # Need at least 2 tokens (input + target)
                chunks.append(chunk)

        total_loss = 0.0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            # Process chunks in batches
            for batch_start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[batch_start : batch_start + batch_size]

                # Find max length in this batch (excluding last token for input)
                max_len = max(len(chunk) - 1 for chunk in batch_chunks)

                # Prepare batched tensors
                input_ids_list = []
                target_ids_list = []
                attention_mask_list = []

                for chunk in batch_chunks:
                    # Split into input (all but last) and target (all but first)
                    input_tokens = chunk[:-1]
                    target_tokens = chunk[1:]

                    # Pad to max_len
                    pad_len = max_len - len(input_tokens)
                    padded_input = input_tokens + [pad_token_id] * pad_len
                    padded_target = target_tokens + [-100] * pad_len  # -100 = ignore in loss
                    attention_mask = [1] * len(input_tokens) + [0] * pad_len

                    input_ids_list.append(padded_input)
                    target_ids_list.append(padded_target)
                    attention_mask_list.append(attention_mask)

                # Convert to tensors
                input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
                target_ids = torch.tensor(target_ids_list, dtype=torch.long, device=device)
                attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0]

                # Compute loss (ignoring padded positions marked with -100)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    reduction="sum",
                    ignore_index=-100
                )

                total_loss += loss.item()
                total_tokens += (target_ids != -100).sum().item()

        return {
            "total_loss": total_loss,
            "total_tokens": total_tokens,
        }

    # Execute on rank 0 worker only (to avoid redundant computation)
    # For FSDP, the model is sharded, so we run on all ranks and average
    try:
        results = worker_group.execute_all_sync(_compute_loss_on_worker, text=text, max_seq_length=max_seq_length, batch_size=batch_size)

        # Average across workers (they should all get the same result)
        # We just take the first result since all workers process the same data
        result = results[0]
        total_loss = result["total_loss"]
        total_tokens = result["total_tokens"]

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "wikitext/loss": avg_loss,
            "wikitext/perplexity": perplexity,
            "wikitext/num_tokens": total_tokens,
        }
    except Exception as e:
        print(f"Warning: Wikitext evaluation failed: {e}")
        return {
            "wikitext/loss": float("nan"),
            "wikitext/perplexity": float("nan"),
            "wikitext/num_tokens": 0,
        }
