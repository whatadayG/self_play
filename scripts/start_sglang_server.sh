#!/bin/bash
# Start SGLang server for local model inference with OpenAI-compatible API

set -x

# Optionally activate a specific virtualenv for the SGLang server
if [ -n "$VENV_PATH" ]; then
  echo "Activating venv at $VENV_PATH for SGLang server..."
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

MODEL_PATH=${MODEL_PATH:-"/home/nickatomlin/georgiazhou/self_play/checkpoints/sft_qwen3_8b/global_step_4800_merged/"}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-4}  # Tensor parallel size
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
# Restrict NCCL P2P to NVLink pairs to avoid cross-bridge P2P stalls on GPUs 4-7
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_P2P_LEVEL=NVL

echo "Starting SGLang server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "TP Size: $TP_SIZE"

# Kill any existing server on the port
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true

# Start SGLang server
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --host 127.0.0.1 \
    --tp $TP_SIZE \
    --trust-remote-code \
    --mem-fraction-static $GPU_MEMORY_UTILIZATION \
    --dtype bfloat16 \
#     --enable-torch-compile \

# Store the PID
SERVER_PID=$!
echo "Server PID: $SERVER_PID"


echo ""
echo "SGLang server is running at http://localhost:$PORT"
echo "To use with OpenAI client:"
echo "  export OPENAI_API_BASE=http://localhost:$PORT/v1"
echo "  export OPENAI_API_KEY=dummy"
echo ""
echo "To stop the server: kill $SERVER_PID"
