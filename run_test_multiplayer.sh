#!/bin/bash
# Script to run multiplayer test with correct environment

set -e

# Activate conda environment
echo "Activating conda environment..."
source /home/nickatomlin/sources/conda/etc/profile.d/conda.sh
conda activate py38

# Run the test
echo "Running multiplayer test..."
cd /home/nickatomlin/georgiazhou/self_play
python test_multiplayer_simple.py