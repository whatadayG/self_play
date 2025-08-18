#!/bin/bash
# Debug script to test evaluate_opt.py

set -x

# Activate venv
source .venv/bin/activate

# Show python info
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Change to scripts directory
cd scripts

# Try to run evaluate_opt.py with help
echo "Testing evaluate_opt.py..."
python evaluate_opt.py --help | head -20

# Try a minimal run
echo ""
echo "Running minimal test..."
python evaluate_opt.py \
    --exp-name "debug_test" \
    --game matching \
    --mode selfplay \
    --agent-model-id "gpt-3.5-turbo" \
    --user-model-id "gpt-3.5-turbo" \
    --end 1 \
    --threshold 0.0 \
    --dry-run True