#!/bin/bash
# Start SGLang server for local model inference with OpenAI-compatible API

set -x

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
PORT=${PORT:-8000}
TP_SIZE=${TP_SIZE:-1}  # Tensor parallel size
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

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
    --host 0.0.0.0 \
    --tp $TP_SIZE \
    --trust-remote-code \
    --mem-fraction-static $GPU_MEMORY_UTILIZATION \
    --dtype bfloat16 \
    --enable-torch-compile \
    &

# Store the PID
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null; then
        echo "Server is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 2
done

# Test the server
echo "Testing server..."
curl -s http://localhost:$PORT/v1/models | jq .

echo ""
echo "SGLang server is running at http://localhost:$PORT"
echo "To use with OpenAI client:"
echo "  export OPENAI_API_BASE=http://localhost:$PORT/v1"
echo "  export OPENAI_API_KEY=dummy"
echo ""
echo "To stop the server: kill $SERVER_PID"