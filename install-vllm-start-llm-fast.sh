#!/bin/bash
# Fixed RunPod Deployment Script for AIDC Lao LLM
# Model: Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it

set -e

echo "=========================================="
echo "AIDC Lao LLM - RunPod Deployment Setup"
echo "=========================================="

# Configuration
MODEL_NAME="Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"
PORT=8000
MAX_MODEL_LEN=24576
GPU_MEMORY_UTIL=0.95

# Disable hf_transfer if not installed (fixes the error)
export HF_HUB_ENABLE_HF_TRANSFER=0

# Check if running on RunPod
if [ ! -z "$RUNPOD_POD_ID" ]; then
    echo "✓ Running on RunPod Pod: $RUNPOD_POD_ID"
else
    echo "⚠ Warning: Not detected as RunPod environment"
fi

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq wget curl git nano htop

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip --quiet

# Install vLLM with all dependencies
echo "Installing vLLM..."
pip install vllm --quiet

# Install additional packages
pip install fastapi uvicorn[standard] --quiet
pip install python-dotenv --quiet
pip install huggingface-hub --quiet

# Optional: Install hf_transfer for faster downloads (if needed)
echo "📥 Installing hf_transfer for faster model downloads..."
pip install hf-transfer --quiet || echo "⚠ hf_transfer install failed, continuing without it"

# Setup Hugging Face token if provided
if [ ! -z "$HF_TOKEN" ]; then
    echo "🔑 Setting up Hugging Face authentication..."
    huggingface-cli login --token $HF_TOKEN
    echo "✓ Hugging Face token configured"
else
    echo "⚠ No HF_TOKEN provided - checking if model is accessible..."
fi

# Check GPU availability
echo "🎮 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"

# Check if model exists on HuggingFace
echo "🔍 Checking if model exists..."
python3 << 'EOF'
from huggingface_hub import model_info
import sys

model_id = "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"

try:
    info = model_info(model_id)
    print(f"✓ Model found: {model_id}")
    print(f"  Model ID: {info.modelId}")
    print(f"  Author: {info.author}")
    if hasattr(info, 'private') and info.private:
        print("  ⚠ This is a PRIVATE model - HF_TOKEN required")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: Cannot access model '{model_id}'")
    print(f"   {str(e)}")
    print("\nPossible solutions:")
    print("1. Make sure the model name is correct")
    print("2. If it's a private model, set HF_TOKEN environment variable")
    print("3. Check https://huggingface.co/Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Model access check failed. Please verify:"
    echo "   1. Model exists at: https://huggingface.co/$MODEL_NAME"
    echo "   2. Model is public or you have provided HF_TOKEN"
    echo ""
    exit 1
fi

# Download model (pre-caching)
echo "📥 Pre-downloading model to cache..."
python3 << EOF
from huggingface_hub import snapshot_download
import os

model_id = "$MODEL_NAME"
print(f"Downloading {model_id}...")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✓ Model downloaded successfully")
except Exception as e:
    print(f"⚠ Download warning: {e}")
    print("Model will be downloaded on first inference")
EOF

# Create startup script
echo "📝 Creating vLLM startup script..."
cat > /workspace/start_vllm.sh << 'VLLM_SCRIPT'
#!/bin/bash

# Disable hf_transfer to avoid errors
export HF_HUB_ENABLE_HF_TRANSFER=0

# Environment variables with defaults
MODEL_NAME="${MODEL_NAME:-Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

echo "Starting vLLM OpenAI-compatible server..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --dtype auto \
    --enable-prefix-caching \
    --enable-log-requests
VLLM_SCRIPT

chmod +x /workspace/start_vllm.sh

# Create test script
echo "📝 Creating test script..."
cat > /workspace/test_model.py << 'TEST_SCRIPT'
#!/usr/bin/env python3
import requests
import json
import sys
import time

def test_chat():
    url = "http://localhost:8000/v1/chat/completions"
    
    # Test in Lao
    messages = [
        {"role": "user", "content": "ສະບາຍດີ, ທ່ານເປັນໃຜ?"}
    ]
    
    payload = {
        "model": "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    print("Testing Lao LLM...")
    print(f"URL: {url}")
    print(f"Request: {json.dumps(messages, ensure_ascii=False)}\n")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result:
            content = result["choices"][0]["message"]["content"]
            print("✓ Response received:")
            print(content)
            return True
        else:
            print("❌ Unexpected response format")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_health():
    url = "http://localhost:8000/health"
    max_retries = 30
    
    print("Checking server health...")
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ Health check passed")
                return True
        except:
            pass
        
        if i < max_retries - 1:
            print(f"Waiting for server... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print("❌ Health check failed - server not responding")
    return False

if __name__ == "__main__":
    print("========================================")
    print("AIDC Lao LLM - Test Suite")
    print("========================================\n")
    
    if test_health():
        print("\n" + "="*40 + "\n")
        test_chat()
    else:
        print("Server not ready. Check logs: tail -f /workspace/vllm.log")
        sys.exit(1)
TEST_SCRIPT

chmod +x /workspace/test_model.py

# Create startup script with better error handling
echo "📝 Creating main startup script..."
cat > /workspace/start_server.sh << 'STARTUP_SCRIPT'
#!/bin/bash

# Disable hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=========================================="
echo "Starting AIDC Lao LLM Server"
echo "=========================================="

# Check if model exists locally or can be accessed
echo "Verifying model access..."
python3 << 'VERIFY'
from huggingface_hub import model_info
import sys

model_id = "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"

try:
    info = model_info(model_id)
    print(f"✓ Model accessible: {model_id}")
except Exception as e:
    print(f"❌ Cannot access model: {e}")
    print("\nPlease check:")
    print("1. Model name is correct")
    print("2. Model is public or HF_TOKEN is set")
    print("3. Internet connection is working")
    sys.exit(1)
VERIFY

if [ $? -ne 0 ]; then
    echo "❌ Model verification failed. Exiting."
    exit 1
fi

# Start vLLM in background
echo "🚀 Starting vLLM server..."
/workspace/start_vllm.sh > /workspace/vllm.log 2>&1 &
VLLM_PID=$!

echo "vLLM PID: $VLLM_PID"
echo "Logs: /workspace/vllm.log"

# Wait for vLLM to be ready
echo "⏳ Waiting for vLLM to be ready (this may take 1-2 minutes)..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM is ready!"
        break
    fi
    
    # Check if process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ vLLM process died. Check logs:"
        tail -50 /workspace/vllm.log
        exit 1
    fi
    
    echo "Waiting... ($i/60)"
    sleep 2
done

# Check if vLLM started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ vLLM failed to start. Last 50 lines of log:"
    tail -50 /workspace/vllm.log
    exit 1
fi

# Display server information
echo ""
echo "=========================================="
echo "✓ Server Started Successfully!"
echo "=========================================="
echo "Model: Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"
echo "vLLM Endpoint: http://localhost:8000"
echo "OpenAI API Compatible: http://localhost:8000/v1"
echo ""
echo "Test with:"
echo 'curl http://localhost:8000/v1/chat/completions \\'
echo '  -H "Content-Type: application/json" \\'
echo '  -d '"'"'{"model": "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it", "messages": [{"role": "user", "content": "ສະບາຍດີ"}]}'"'"
echo ""
echo "Or run: python3 /workspace/test_model.py"
echo ""
echo "Logs: tail -f /workspace/vllm.log"
echo "=========================================="

# Keep container running and show logs
tail -f /workspace/vllm.log
STARTUP_SCRIPT

chmod +x /workspace/start_server.sh

# Create environment file
echo "📝 Creating environment configuration..."
cat > /workspace/.env << 'ENV_FILE'
# AIDC Lao LLM Configuration
MODEL_NAME=Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it
PORT=8000
MAX_MODEL_LEN=24576
GPU_MEMORY_UTIL=0.95
TENSOR_PARALLEL_SIZE=1

# Disable hf_transfer (fixes download errors)
HF_HUB_ENABLE_HF_TRANSFER=0

# Optional: Hugging Face Token (required for private models)
# HF_TOKEN=your_token_here

# Optional: API Key for authentication
# API_KEY=your_secret_key
ENV_FILE

echo ""
echo "=========================================="
echo "✓ RunPod Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start the server: /workspace/start_server.sh"
echo "2. Test the model: python3 /workspace/test_model.py"
echo "3. Access via: http://localhost:8000"
echo ""
echo "If you get errors, check:"
echo "- Model exists: https://huggingface.co/$MODEL_NAME"
echo "- HF_TOKEN is set (if private model)"
echo "- GPU is available: nvidia-smi"
echo ""
echo "=========================================="
