#!/usr/bin/env python3
"""Check available Qwen models"""

from huggingface_hub import list_models
import torch

print("=== Checking Available Qwen Models ===\n")

# Check GPU memory first
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} - Memory: {props.total_memory / 1e9:.1f} GB")
print()

# Search for Qwen models
print("Searching for Qwen models around 30B size...")
models = list(list_models(search="Qwen", sort="downloads", limit=50))

qwen_models = []
for model in models:
    model_id = model.modelId
    # Look for models with size indicators
    if any(size in model_id.lower() for size in ["30b", "32b", "70b", "72b", "14b"]):
        qwen_models.append(model_id)

print("\nLarge Qwen models found:")
for model_id in sorted(set(qwen_models)):
    if "qwen" in model_id.lower():
        print(f"  - {model_id}")

# Also check some specific known models
print("\nChecking specific Qwen models:")
known_models = [
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct", 
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen2-72B-Instruct",
]

for model_id in known_models:
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        size_gb = info.safetensors.total / 1e9 if hasattr(info, 'safetensors') else 0
        print(f"  ✓ {model_id} - Size: {size_gb:.1f} GB")
    except:
        pass