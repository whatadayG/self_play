#!/bin/bash

# Comprehensive fix for GRPO hanging issue

echo "Fixing potential causes of GRPO hanging..."
echo "=========================================="

# 1. Stop any existing Ray instances
echo "1. Stopping Ray..."
ray stop --force 2>/dev/null || true

# 2. Clear torch compile cache
echo "2. Clearing torch compile cache..."
rm -rf ~/.cache/torch_extensions/* 2>/dev/null || true
rm -rf ~/.cache/torch/kernels/* 2>/dev/null || true

# 3. Clean up old Ray sessions
echo "3. Cleaning old Ray sessions..."
cd /tmp/ray 2>/dev/null && {
    # Keep only the last 3 sessions
    ls -dt session_* 2>/dev/null | tail -n +4 | xargs rm -rf 2>/dev/null || true
}

# 4. Clear shared memory
echo "4. Clearing shared memory..."
rm -rf /dev/shm/plasma* 2>/dev/null || true
rm -rf /dev/shm/nccl* 2>/dev/null || true

# 5. Check for stuck processes
echo "5. Checking for stuck ML processes..."
ps aux | grep -E "sglang|verl" | grep -v grep | awk '{print $2}' | while read pid; do
    echo "  Found process $pid - consider killing if stuck"
done

# 6. Clear any NCCL socket files
echo "6. Clearing NCCL files..."
rm -f /tmp/nccl* 2>/dev/null || true

# 7. Environment variable fixes
echo "7. Setting environment variables..."
echo "export NCCL_P2P_DISABLE=1  # Disable P2P if having issues"
echo "export NCCL_IB_DISABLE=1   # Disable InfiniBand if not using it"
echo "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

echo ""
echo "Fixes applied. Recommendations:"
echo "1. Run this script before training"
echo "2. Use the debug script to test: ./test_grpo_qwen13b_debug.sh"
echo "3. If still hanging, try setting in your script:"
echo "   export NCCL_P2P_DISABLE=1"
echo "   export TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
echo ""
echo "The hanging at 2048MB suggests processes waiting for distributed init."
echo "Common fix: Ensure all GPUs can communicate properly."