#!/bin/bash
# Test script to verify Qwen setup before full training

set -e
set -x

# Activate conda environment
echo "=== Activating conda environment ==="
source /home/nickatomlin/sources/conda/etc/profile.d/conda.sh
conda activate py38

# Check Python and key packages
echo ""
echo "=== Checking environment ==="
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import verl; print('VERL import successful')"

# Check GPU availability
echo ""
echo "=== Checking GPUs ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv

# Test model loading
echo ""
echo "=== Testing Qwen model loading ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2.5-7B-Instruct'
print(f'Testing model: {model_name}')

# Just test tokenizer first (lightweight)
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
print('Tokenizer loaded successfully')

# Test a simple tokenization
test_text = 'Hello, world!'
tokens = tokenizer(test_text)
print(f'Test tokenization: {test_text} -> {len(tokens[\"input_ids\"])} tokens')
"

echo ""
echo "=== Setup verification complete ==="
echo "To start training, run: ./run_qwen_training.sh"