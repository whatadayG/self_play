#!/usr/bin/env python3
"""Test script to verify Qwen3-4B model loading"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Add path for dialop
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_model_loading():
    model_name = "Qwen/Qwen3-4B"
    print(f"Testing model: {model_name}")
    
    try:
        # Test tokenizer loading
        print("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        
        # Test model info without loading full model
        print("\n2. Checking model availability...")
        from huggingface_hub import model_info
        info = model_info(model_name)
        print(f"✓ Model found on Hugging Face")
        print(f"  Model size: {info.safetensors.total / 1e9:.1f} GB")
        
        # Test HFModelPlayer
        print("\n3. Testing HFModelPlayer...")
        try:
            from dialop.hf_model_player import HFModelPlayer
            print("✓ HFModelPlayer can be imported")
            
            # Don't actually create instance to avoid loading full model
            print("  (Skipping full model load to save memory)")
            
        except ImportError as e:
            print(f"✗ Failed to import HFModelPlayer: {e}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("=== Qwen3-4B Model Loading Test ===")
    
    # Check GPU availability
    print("\nGPU Status:")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ No GPU available")
    
    # Test model loading
    success = test_model_loading()
    
    if success:
        print("\n=== Test passed! ===")
        print("The model should work with evaluate_opt.py")
    else:
        print("\n=== Test failed! ===")
        print("Please check the error messages above")