#!/usr/bin/env python3
"""
Script to verify gradient masking in multiturn SFT training.
Shows which tokens receive gradients (loss_mask=1) vs which don't (loss_mask=0).
"""

import pandas as pd
from transformers import AutoTokenizer
import json
import sys
import os

# Add verl to path so we can use its utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def process_messages_like_dataset(tokenizer, messages):
    """
    Mimics the logic from MultiturnSFTDataset to show which tokens get gradients.
    """
    # Get full conversation tokens
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
    
    # Track tokens and masks
    all_tokens = []
    all_loss_mask = []
    token_sources = []  # Track which message each token comes from
    
    i = 0
    while i < len(messages):
        message = messages[i]
        
        # Get text up to this point (excluding current message)
        if i > 0:
            prev_text = tokenizer.apply_chat_template(
                messages[:i],
                tokenize=False,
                add_generation_prompt=False,
            )
            if message["role"] == "assistant":
                # For assistant messages, we need the generation prompt
                prev_text_with_prompt = tokenizer.apply_chat_template(
                    messages[:i],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prev_text = ""
            prev_text_with_prompt = ""
        
        # Get text including current message
        cur_text = tokenizer.apply_chat_template(
            messages[:i+1],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Extract tokens for just this message
        if message["role"] == "assistant" and i > 0:
            # Get generation prompt tokens
            generation_prompt = prev_text_with_prompt[len(prev_text):]
            generation_prompt_tokens = tokenizer.encode(generation_prompt, add_special_tokens=False)
            
            # Get assistant message tokens (without the prompt)
            assistant_text = cur_text[len(prev_text_with_prompt):]
            assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
            
            # Combine tokens
            message_tokens = generation_prompt_tokens + assistant_tokens
            
            # Create loss mask: 0 for prompt, 1 for actual assistant response
            loss_mask = [0] * len(generation_prompt_tokens) + [1] * len(assistant_tokens)
            
            # Track sources
            sources = [f"gen_prompt_{i}"] * len(generation_prompt_tokens) + [f"assistant_{i}"] * len(assistant_tokens)
        else:
            # For user/system messages, extract tokens directly
            message_text = cur_text[len(prev_text):]
            message_tokens = tokenizer.encode(message_text, add_special_tokens=False)
            
            # No gradient for user/system messages
            loss_mask = [0] * len(message_tokens)
            sources = [f"{message['role']}_{i}"] * len(message_tokens)
        
        all_tokens.extend(message_tokens)
        all_loss_mask.extend(loss_mask)
        token_sources.extend(sources)
        
        i += 1
    
    return all_tokens, all_loss_mask, token_sources, full_text

def visualize_gradient_masking(tokenizer, tokens, loss_mask, sources, output_file):
    """
    Create a visualization showing which tokens receive gradients.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("GRADIENT MASKING VISUALIZATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("Legend:\n")
        f.write("  🟢 = Token receives gradient (loss_mask=1)\n")
        f.write("  🔴 = Token does NOT receive gradient (loss_mask=0)\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Show token-by-token breakdown
        f.write("TOKEN-BY-TOKEN BREAKDOWN:\n\n")
        f.write(f"{'Token ID':>8} | {'Token':20} | {'Gradient?':10} | {'Source':20}\n")
        f.write("-" * 70 + "\n")
        
        for i, (token_id, mask, source) in enumerate(zip(tokens, loss_mask, sources)):
            token_text = tokenizer.decode([token_id])
            gradient_symbol = "🟢 YES" if mask == 1 else "🔴 NO"
            f.write(f"{token_id:8d} | {token_text:20} | {gradient_symbol:10} | {source:20}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Show the full text with highlighting
        f.write("FULL CONVERSATION WITH GRADIENT HIGHLIGHTING:\n\n")
        
        # Decode tokens one by one and mark gradient regions
        current_pos = 0
        current_mask = loss_mask[0] if loss_mask else 0
        text_parts = []
        
        for i, (token_id, mask) in enumerate(zip(tokens, loss_mask)):
            token_text = tokenizer.decode([token_id])
            
            if mask != current_mask:
                # Mask changed, start new section
                if current_mask == 1:
                    text_parts.append("【GRADIENT ON】")
                else:
                    text_parts.append("【GRADIENT OFF】")
                current_mask = mask
            
            text_parts.append(token_text)
        
        # Add final marker
        if current_mask == 1:
            text_parts.append("【END GRADIENT】")
        
        full_highlighted_text = "".join(text_parts)
        f.write(full_highlighted_text)
        
        f.write("\n\n" + "=" * 80 + "\n\n")
        
        # Summary statistics
        total_tokens = len(tokens)
        gradient_tokens = sum(loss_mask)
        f.write("SUMMARY:\n")
        f.write(f"  Total tokens: {total_tokens}\n")
        f.write(f"  Tokens with gradient: {gradient_tokens} ({gradient_tokens/total_tokens*100:.1f}%)\n")
        f.write(f"  Tokens without gradient: {total_tokens - gradient_tokens} ({(total_tokens-gradient_tokens)/total_tokens*100:.1f}%)\n")

def main():
    # Load the training data
    parquet_file = "/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/sft_qwen3_10k_train.parquet"
    df = pd.read_parquet(parquet_file)
    
    print(f"Loaded {len(df)} training examples")
    
    # Initialize tokenizer (using Qwen3-8B as specified in the script)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    
    # Process first example
    messages = df.iloc[0]['messages']
    
    print(f"\nProcessing example with {len(messages)} messages:")
    for i, msg in enumerate(messages):
        print(f"  {i}: {msg['role']} - {len(msg['content'])} chars")
    
    # Process like the dataset would
    tokens, loss_mask, sources, full_text = process_messages_like_dataset(tokenizer, messages)
    
    # Create visualization
    output_file = "/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/gradient_masking_verification.txt"
    visualize_gradient_masking(tokenizer, tokens, loss_mask, sources, output_file)
    
    print(f"\nVisualization saved to: {output_file}")
    print(f"\nQuick summary:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Tokens with gradient: {sum(loss_mask)} ({sum(loss_mask)/len(tokens)*100:.1f}%)")
    
    # Also save a simpler view showing just the messages and which ones get gradients
    with open("/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/gradient_masking_summary.txt", 'w', encoding='utf-8') as f:
        f.write("MESSAGE-LEVEL GRADIENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for i, msg in enumerate(messages):
            role = msg['role']
            content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            gradient_status = "✅ RECEIVES GRADIENT" if role == "assistant" else "❌ NO GRADIENT"
            
            f.write(f"Message {i}: [{role.upper()}] {gradient_status}\n")
            f.write(f"Content: {content_preview}\n")
            f.write("-" * 40 + "\n")
    
    print(f"Summary saved to: /home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/gradient_masking_summary.txt")

if __name__ == "__main__":
    main()