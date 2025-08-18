#!/bin/bash
# Quick test to verify evaluate_opt.py command line interface

cd scripts

# Test with minimal settings
python evaluate_opt.py \
    --exp-name "test_cli" \
    --game matching \
    --mode selfplay \
    --agent-model-id "gpt-3.5-turbo" \
    --user-model-id "gpt-3.5-turbo" \
    --end 1 \
    --dry-run True \
    --threshold 0.0