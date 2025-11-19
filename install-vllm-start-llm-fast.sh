#!/bin/bash
# RunPod Deployment Script for AIDC Lao LLM (Gemma 3 4B)
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

# Check if running on RunPod
if [ ! -z "$RUNPOD_POD_ID" ]; then
    echo "✓ Running on RunPod Pod: $RUNPOD_POD_ID"
else
    echo "⚠ Warning: Not detected as RunPod environment"
fi

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq wget curl git nano htop nvtop

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install vllm --quiet
pip install fastapi uvicorn[standard] --quiet
pip install python-dotenv --quiet
pip install huggingface-hub --quiet

# Setup Hugging Face token if provided
if [ ! -z "$HF_TOKEN" ]; then
    echo "🔑 Setting up Hugging Face authentication..."
    huggingface-cli login --token $HF_TOKEN
    echo "✓ Hugging Face token configured"
else
    echo "⚠ No HF_TOKEN provided - using public model access"
fi

# Check GPU availability
echo "🎮 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "✓ Found $GPU_COUNT GPU(s)"

# Download model (optional pre-caching)
echo "📥 Pre-downloading model to cache..."
python3 << EOF
from huggingface_hub import snapshot_download
import os

model_id = "$MODEL_NAME"
print(f"Downloading {model_id}...")
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=f"/root/.cache/huggingface/hub/{model_id.replace('/', '--')}",
        local_dir_use_symlinks=False
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
    --disable-log-requests
VLLM_SCRIPT

chmod +x /workspace/start_vllm.sh

# Create FastAPI wrapper (optional advanced API)
echo "📝 Creating FastAPI wrapper..."
cat > /workspace/api_server.py << 'FASTAPI_SCRIPT'
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import os
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="AIDC Lao LLM API",
    description="Production API for Lao Language Model",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
API_KEY = os.getenv("API_KEY", "")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

def verify_api_key(authorization: Optional[str] = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.get("/")
async def root():
    return {
        "service": "AIDC Lao LLM API",
        "status": "running",
        "model": "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it",
        "endpoints": {
            "health": "/health",
            "chat": "/v1/chat/completions",
            "generate": "/v1/generate"
        }
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VLLM_ENDPOINT}/health")
            return {"status": "healthy", "vllm": response.status_code == 200}
    except:
        return {"status": "unhealthy", "vllm": False}

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    authorized: bool = Header(None, alias="Authorization")
):
    if API_KEY:
        verify_api_key(authorized)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            response = await client.post(
                f"{VLLM_ENDPOINT}/chat/completions",
                json={
                    "model": "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it",
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "stream": request.stream
                }
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/generate")
async def generate(
    request: GenerateRequest,
    authorized: bool = Header(None, alias="Authorization")
):
    if API_KEY:
        verify_api_key(authorized)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{VLLM_ENDPOINT}/completions",
                json={
                    "model": "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it",
                    "prompt": request.prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "stream": request.stream
                }
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
FASTAPI_SCRIPT

# Create Docker-like startup script
echo "📝 Creating main startup script..."
cat > /workspace/start_server.sh << 'STARTUP_SCRIPT'
#!/bin/bash

echo "=========================================="
echo "Starting AIDC Lao LLM Server"
echo "=========================================="

# Start vLLM in background
echo "🚀 Starting vLLM server..."
/workspace/start_vllm.sh > /workspace/vllm.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "⏳ Waiting for vLLM to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ vLLM is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 2
done

# Check if vLLM started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ vLLM failed to start. Check logs at /workspace/vllm.log"
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
echo "Logs: tail -f /workspace/vllm.log"
echo "=========================================="

# Keep container running and show logs
tail -f /workspace/vllm.log
STARTUP_SCRIPT

chmod +x /workspace/start_server.sh

# Create environment file template
echo "📝 Creating environment configuration..."
cat > /workspace/.env << 'ENV_FILE'
# AIDC Lao LLM Configuration
MODEL_NAME=Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it
PORT=8000
MAX_MODEL_LEN=24576
GPU_MEMORY_UTIL=0.95
TENSOR_PARALLEL_SIZE=1

# Optional: Hugging Face Token
# HF_TOKEN=your_token_here

# Optional: API Key for FastAPI wrapper
# API_KEY=your_secret_key
ENV_FILE

# Create test script
echo "📝 Creating test script..."
cat > /workspace/test_model.py << 'TEST_SCRIPT'
#!/usr/bin/env python3
import requests
import json
import sys

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
    try:
        response = requests.get(url, timeout=5)
        print(f"✓ Health check: {response.status_code}")
        return response.status_code == 200
    except:
        print("❌ Health check failed")
        return False

if __name__ == "__main__":
    print("========================================")
    print("AIDC Lao LLM - Test Suite")
    print("========================================\n")
    
    if test_health():
        print("\n" + "="*40 + "\n")
        test_chat()
    else:
        print("Server not ready. Please start the server first.")
        sys.exit(1)
TEST_SCRIPT

chmod +x /workspace/test_model.py

# Create RunPod specific readme
echo "📝 Creating README..."
cat > /workspace/README.md << 'README'
# AIDC Lao LLM - RunPod Deployment

## Model Information
- **Model**: Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it
- **Architecture**: Gemma 3 4B Instruction-Tuned
- **Context Length**: 24,576 tokens
- **Language**: Lao (ພາສາລາວ)

## Quick Start

### 1. Start the Server
```bash
/workspace/start_server.sh
```

### 2. Test the Model
```bash
python3 /workspace/test_model.py
```

### 3. Access via API

**OpenAI Compatible Endpoint:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it",
    "messages": [
      {"role": "user", "content": "ສະບາຍດີ, ເຈົ້າສາມາດຊ່ວຍຂ້ອຍໄດ້ບໍ່?"}
    ],
    "max_tokens": 512
  }'
```

**Python Client:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it",
    messages=[
        {"role": "user", "content": "ສະບາຍດີ"}
    ]
)

print(response.choices[0].message.content)
```

## Configuration

Edit `/workspace/.env` to customize:
- `MODEL_NAME`: Model identifier
- `PORT`: Server port (default: 8000)
- `MAX_MODEL_LEN`: Maximum context length
- `GPU_MEMORY_UTIL`: GPU memory utilization (0.0-1.0)
- `TENSOR_PARALLEL_SIZE`: Number of GPUs for tensor parallelism

## Monitoring

### GPU Usage
```bash
nvidia-smi -l 1
# or
nvtop
```

### Server Logs
```bash
tail -f /workspace/vllm.log
```

### Health Check
```bash
curl http://localhost:8000/health
```

## RunPod Specific

### Expose Port
In RunPod dashboard:
1. Go to your pod settings
2. Add exposed port: 8000
3. Access via: `https://{pod-id}-8000.proxy.runpod.net`

### Environment Variables
Set in RunPod template or pod configuration:
- `HF_TOKEN`: Your Hugging Face token (if needed)
- `MODEL_NAME`: Override default model
- `API_KEY`: Optional API authentication

## Troubleshooting

### Model Download Issues
```bash
# Manually download model
huggingface-cli download Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it
```

### Out of Memory
Reduce GPU memory utilization:
```bash
export GPU_MEMORY_UTIL=0.85
/workspace/start_vllm.sh
```

### Slow Startup
Model is downloading on first run. Check progress:
```bash
watch -n 1 du -sh ~/.cache/huggingface/
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- OpenAPI Spec: http://localhost:8000/openapi.json

## Support

For issues related to:
- Model: Contact AIDC team
- vLLM: https://github.com/vllm-project/vllm
- RunPod: https://docs.runpod.io
README

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
echo "Files created:"
echo "  - /workspace/start_vllm.sh      (vLLM server)"
echo "  - /workspace/start_server.sh     (Main startup)"
echo "  - /workspace/api_server.py       (FastAPI wrapper)"
echo "  - /workspace/test_model.py       (Test script)"
echo "  - /workspace/.env                (Configuration)"
echo "  - /workspace/README.md           (Documentation)"
echo ""
echo "📖 Read README.md for full documentation"
echo "=========================================="
