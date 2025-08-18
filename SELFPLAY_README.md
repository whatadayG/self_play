# Self-Play Training Pipeline

This directory contains scripts for running self-play training for the matching game using VERL and GRPO.

## Overview

The self-play pipeline works by:
1. Having a model play against itself (both players use the same model)
2. Converting each game into 2 training instances (one from each player's perspective)
3. Training with GRPO where both perspectives form a natural comparison group
4. Iterating this process to improve the model

## Scripts

### 1. `run_selfplay_generation.sh`
Generates self-play data using the current model.

```bash
# Generate 100 games with default model (Qwen/Qwen2.5-7B-Instruct)
./run_selfplay_generation.sh

# Or specify custom parameters
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct" NUM_GAMES=200 ./run_selfplay_generation.sh
```

Output:
- `data/matching_game_selfplay/train.parquet` - Training data
- `data/matching_game_selfplay/val.parquet` - Validation data

### 2. `run_selfplay_training.sh`
Trains a model using generated self-play data.

```bash
# Train on generated self-play data
./run_selfplay_training.sh

# With custom batch size
TRAIN_BATCH_SIZE=128 MICRO_BATCH_SIZE=16 ./run_selfplay_training.sh
```

**Note**: You must run `run_selfplay_generation.sh` first to generate the data.

### 3. `run_selfplay_iteration.sh`
Runs multiple iterations of self-play → training automatically.

```bash
# Run 3 iterations with default settings
./run_selfplay_iteration.sh

# Custom settings
NUM_ITERATIONS=5 GAMES_PER_ITER=200 EPOCHS_PER_ITER=3 ./run_selfplay_iteration.sh
```

Each iteration:
1. Generates self-play data with current model
2. Trains on that data
3. Uses the new model for the next iteration

## Environment Variables

### Data Generation (`run_selfplay_generation.sh`)
- `MODEL_NAME`: Model to use for self-play (default: "Qwen/Qwen2.5-7B-Instruct")
- `NUM_GAMES`: Number of games to generate (default: 100)
- `MAX_LENGTH`: Maximum conversation length (default: 100)
- `TEMPERATURE`: Sampling temperature (default: 0.7)
- `EXP_NAME`: Experiment name for output files (default: "selfplay_gen")

### Training (`run_selfplay_training.sh`)
- `TRAIN_BATCH_SIZE`: Training batch size (default: 64)
- `MICRO_BATCH_SIZE`: Micro batch size per GPU (default: 8)

### Iterative Training (`run_selfplay_iteration.sh`)
- `INITIAL_MODEL`: Starting model (default: "Qwen/Qwen2.5-7B-Instruct")
- `NUM_ITERATIONS`: Number of self-play iterations (default: 3)
- `GAMES_PER_ITER`: Games to generate per iteration (default: 100)
- `EPOCHS_PER_ITER`: Training epochs per iteration (default: 2)

## GPU Configuration

The scripts are configured to use GPUs 2,3,4,5 by default. Modify the `CUDA_VISIBLE_DEVICES` line in the scripts to change this.

## Prerequisites

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Ensure you have the required models accessible (e.g., through Hugging Face)

3. The scripts expect to be run from the `self_play` directory

## Example Workflow

1. **Single iteration of self-play training**:
   ```bash
   # Generate data
   ./run_selfplay_generation.sh
   
   # Train on the data
   ./run_selfplay_training.sh
   ```

2. **Multiple iterations (recommended)**:
   ```bash
   # Run 5 iterations with 200 games each
   NUM_ITERATIONS=5 GAMES_PER_ITER=200 ./run_selfplay_iteration.sh
   ```

## Output Structure

```
data/
├── matching_game_selfplay/          # Single iteration data
│   ├── selfplay_games.jsonl         # Raw game logs
│   ├── train.parquet                # Training data
│   └── val.parquet                  # Validation data
└── matching_game_selfplay_iterations/  # Multi-iteration data
    ├── iter_1/
    │   ├── selfplay_games.jsonl
    │   ├── train.parquet
    │   ├── val.parquet
    │   └── checkpoint/              # Trained model
    ├── iter_2/
    └── ...
```

## Notes

- The conversion script (`convert_selfplay_to_verl.py`) creates 2 training instances per game
- Both instances from the same game share the same `index` for GRPO grouping
- This naturally creates comparison groups of size 2 for advantage estimation
- The reward is the same for both players (collaborative game)