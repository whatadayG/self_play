#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for VERL FSDP SFT on Qwen2.5-7B-Instruct
# Single-turn:    provide -p PROMPT_KEY and -r RESPONSE_KEY
# Multi-turn:     pass -U to enable, optionally -K messages_key (default: messages), -T tools_key, -H enable_thinking_key
# Required: -t train_file -v val_file
# Optional: -m model_id -P project -E experiment -e epochs -o outdir -g gpus -b micro_bs -B global_bs -L lora_rank

usage() {
  echo "Usage: $0 -t TRAIN.parquet -v VAL.parquet [-U -K messages] [-p PROMPT_KEY -r RESPONSE_KEY] [options]" >&2
  echo "Options:" >&2
  echo "  -m MODEL_ID              HF model id or local path (default: Qwen/Qwen2.5-7B-Instruct)" >&2
  echo "  -P PROJECT_NAME          Project name for logging/checkpoints (default: qwen2_5_7b_sft)" >&2
  echo "  -E EXPERIMENT_NAME       Experiment name (default: run_$(date +%Y%m%d_%H%M%S))" >&2
  echo "  -e EPOCHS                Total epochs (default: 3 for multi-turn; 4 otherwise)" >&2
  echo "  -o OUTDIR                Output checkpoints dir (default: self_play/verl/checkpoints/sft_qwen2_5_7b)" >&2
  echo "  -g GPUS                  Number of GPUs for torchrun (default: auto-detect or 1)" >&2
  echo "  -b MICRO_BS              Micro batch size per GPU (default: 1 for multi-turn)" >&2
  echo "  -B GLOBAL_BS             Global train batch size (default: 64 for multi-turn)" >&2
  echo "  -L LORA_RANK             Enable LoRA with given rank (default: 0 disabled)" >&2
  echo "  -U                      Enable multi-turn dataset mode (uses data.multiturn.*)" >&2
  echo "  -K MESSAGES_KEY         Column/key name for multi-turn messages (default: messages)" >&2
  echo "  -T TOOLS_KEY            Column/key name for tools in multi-turn (default: tools)" >&2
  echo "  -H THINKING_KEY         Column/key name for enable_thinking (default: enable_thinking)" >&2
  echo "  -W NUM_WORKERS          data.num_workers for DataLoader (default: unset; use 0 to avoid mp deadlocks)" >&2
  echo "  -p PROMPT_KEY           Single-turn prompt key (ignored if -U)" >&2
  echo "  -r RESPONSE_KEY         Single-turn response key (ignored if -U)" >&2
  echo "  -h                       Show this help" >&2
}

TRAIN_FILE=""
VAL_FILE=""
PROMPT_KEY=""
RESPONSE_KEY=""
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
PROJECT_NAME="qwen2_5_7b_sft"
EXPERIMENT_NAME="run_$(date +%Y%m%d_%H%M%S)"
# Defaults tuned for ~1k multi-turn dialogues
EPOCHS=3
OUTDIR_DEFAULT="/home/nickatomlin/georgiazhou/self_play/verl/checkpoints/sft_qwen2_5_7b"
OUTDIR="$OUTDIR_DEFAULT"
GPUS=""
MICRO_BS="1"
GLOBAL_BS="64"
LORA_RANK=0
MULTITURN=false
MESSAGES_KEY="messages"
TOOLS_KEY="tools"
THINKING_KEY="enable_thinking"
NUM_WORKERS=""

while getopts ":t:v:p:r:m:P:E:e:o:g:b:B:L:UK:T:H:W:h" opt; do
  case ${opt} in
    t) TRAIN_FILE="$OPTARG" ;;
    v) VAL_FILE="$OPTARG" ;;
    p) PROMPT_KEY="$OPTARG" ;;
    r) RESPONSE_KEY="$OPTARG" ;;
    m) MODEL_ID="$OPTARG" ;;
    P) PROJECT_NAME="$OPTARG" ;;
    E) EXPERIMENT_NAME="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    b) MICRO_BS="$OPTARG" ;;
    B) GLOBAL_BS="$OPTARG" ;;
    L) LORA_RANK="$OPTARG" ;;
    U) MULTITURN=true ;;
    K) MESSAGES_KEY="$OPTARG" ;;
    T) TOOLS_KEY="$OPTARG" ;;
    H) THINKING_KEY="$OPTARG" ;;
    W) NUM_WORKERS="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument" >&2; usage; exit 1 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$TRAIN_FILE" || -z "$VAL_FILE" ]]; then
  echo "Error: -t and -v are required." >&2
  usage
  exit 1
fi

if [[ "$MULTITURN" == false ]]; then
  if [[ -z "$PROMPT_KEY" || -z "$RESPONSE_KEY" ]]; then
    echo "Error: single-turn mode requires -p PROMPT_KEY and -r RESPONSE_KEY." >&2
    usage
    exit 1
  fi
fi

# Resolve absolute OUTDIR
OUTDIR=$(python3 - "$OUTDIR" <<'PY'
import os,sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)

# Detect GPU count if not provided
if [[ -z "$GPUS" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
    [[ "$GPUS" -eq 0 ]] && GPUS=1 || true
  else
    GPUS=1
  fi
fi

VERL_DIR="/home/nickatomlin/georgiazhou/self_play/verl"
# Prepend VERL_DIR to PYTHONPATH safely with set -u
export PYTHONPATH="$VERL_DIR${PYTHONPATH+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

# Base overrides
OVERRIDES=(
  "data.train_files=$TRAIN_FILE"
  "data.val_files=$VAL_FILE"
  "model.partial_pretrain=$MODEL_ID"
  "trainer.project_name=$PROJECT_NAME"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.total_epochs=$EPOCHS"
  "trainer.default_local_dir=$OUTDIR"
  "model.trust_remote_code=true"
  "model.fsdp_config.model_dtype=bf16"
  "data.max_length=2048"
  "data.truncation=left"
  "use_remove_padding=true"
)

# Dataset mode overrides
if [[ "$MULTITURN" == true ]]; then
  OVERRIDES+=(
    "data.multiturn.enable=true"
    "data.multiturn.messages_key=$MESSAGES_KEY"
    "data.multiturn.tools_key=$TOOLS_KEY"
    "data.multiturn.enable_thinking_key=$THINKING_KEY"
    "data.micro_batch_size_per_gpu=$MICRO_BS"
    "data.train_batch_size=$GLOBAL_BS"
  )
else
  OVERRIDES+=(
    "data.prompt_key=$PROMPT_KEY"
    "data.response_key=$RESPONSE_KEY"
  )
  # Set micro/global if user specified
  if [[ -n "$MICRO_BS" ]]; then OVERRIDES+=("data.micro_batch_size_per_gpu=$MICRO_BS"); fi
  if [[ -n "$GLOBAL_BS" ]]; then OVERRIDES+=("data.train_batch_size=$GLOBAL_BS"); fi
fi

# Optional dataloader workers override (use Hydra append so key may not exist)
if [[ -n "$NUM_WORKERS" ]]; then
  OVERRIDES+=("+data.num_workers=$NUM_WORKERS")
fi

# LoRA and LR strategy
if [[ "$LORA_RANK" != "0" ]]; then
  OVERRIDES+=("model.lora_rank=$LORA_RANK")
  OVERRIDES+=("optim.lr=2e-4")
else
  OVERRIDES+=("optim.lr=1e-5")
fi

mkdir -p "$OUTDIR"

echo "Launching SFT with $GPUS GPU(s)"
echo "Checkpoints will be saved to: $OUTDIR"

exec torchrun --nproc_per_node "$GPUS" -m verl.trainer.fsdp_sft_trainer "${OVERRIDES[@]}" 