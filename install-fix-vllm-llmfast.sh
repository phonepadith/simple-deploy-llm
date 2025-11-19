#!/bin/bash
# complete_merge_and_deploy.sh
# Merges LoRA adapter with base model and deploys with vLLM

set -e

echo "============================================================"
echo "  LoRA Adapter Merge & vLLM Deployment"
echo "  Adapter: aidc-llm-laos-24k-gemma-3-4b-it"
echo "  Base: unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
echo "============================================================"
echo ""

# Configuration
ADAPTER_REPO="Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"
ADAPTER_DIR="/workspace/aidc-model"
MERGED_DIR="/workspace/aidc-model-merged"
PORT=8000
MAX_MODEL_LEN=24576
GPU_MEMORY_UTIL=0.95

# Environment setup
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0

# Step 1: Install all required dependencies
echo "[1/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q --upgrade transformers huggingface_hub peft bitsandbytes accelerate safetensors torch
if ! python -c "import vllm" 2>/dev/null; then
    pip install vllm
fi
echo "✓ Dependencies installed"
echo ""

# Step 2: Download adapter if needed
echo "[2/5] Checking adapter model..."
if [ ! -d "$ADAPTER_DIR" ]; then
    echo "Downloading adapter..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$ADAPTER_REPO', local_dir='$ADAPTER_DIR', local_dir_use_symlinks=False)
"
fi
echo "✓ Adapter ready at $ADAPTER_DIR"
echo ""

# Step 3: Merge adapter with base model
echo "[3/5] Merging LoRA adapter with base model..."
echo "This process will:"
echo "  1. Download base model: unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
echo "  2. Load LoRA adapter weights"
echo "  3. Merge them into a full model"
echo "  4. Save to: $MERGED_DIR"
echo ""

if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ]; then
    echo "✓ Merged model already exists"
    read -p "Re-merge? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing merged model"
    else
        rm -rf "$MERGED_DIR"
    fi
fi

if [ ! -d "$MERGED_DIR" ]; then
    python << 'MERGE_SCRIPT'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

print("="*60)
print("Starting merge process...")
print("="*60)

ADAPTER_DIR = "/workspace/aidc-model"
MERGED_DIR = "/workspace/aidc-model-merged"

try:
    # Read adapter config to get base model
    import json
    with open(f"{ADAPTER_DIR}/adapter_config.json", 'r') as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config['base_model_name_or_path']
    print(f"\n[1/4] Loading base model: {base_model_name}")
    print("This will download ~8GB and may take 5-10 minutes...")
    
    # Load base model in 16-bit (we'll convert to bf16 after merge)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"\n[2/4] Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        torch_dtype=torch.float16
    )
    
    print("\n[3/4] Merging adapter weights with base model...")
    model = model.merge_and_unload()
    
    # Convert to bfloat16 for vLLM
    print("Converting to bfloat16...")
    model = model.to(torch.bfloat16)
    
    print(f"\n[4/4] Saving merged model to: {MERGED_DIR}")
    os.makedirs(MERGED_DIR, exist_ok=True)
    
    model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    
    print("\n" + "="*60)
    print("✓ Merge completed successfully!")
    print("="*60)
    print(f"Merged model saved to: {MERGED_DIR}")
    
except Exception as e:
    print(f"\n✗ Merge failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
MERGE_SCRIPT

    if [ $? -ne 0 ]; then
        echo "✗ Merge failed"
        exit 1
    fi
fi

echo ""

# Step 4: Fix config for vLLM compatibility
echo "[4/5] Preparing config for vLLM..."
cd "$MERGED_DIR"

python << 'FIX_CONFIG'
import json

with open('config.json', 'r') as f:
    config = json.load(f)

# Backup original
with open('config.json.original', 'w') as f:
    json.dump(config, f, indent=2)

# Ensure vLLM-compatible config
if 'text_config' in config:
    text_config = config['text_config']
else:
    text_config = config

# Build clean config for vLLM
new_config = {
    "architectures": ["GemmaForCausalLM"],
    "model_type": "gemma2",
    "torch_dtype": "bfloat16",
    "hidden_size": text_config.get("hidden_size", 2560),
    "intermediate_size": text_config.get("intermediate_size", 10240),
    "num_hidden_layers": text_config.get("num_hidden_layers", 34),
    "num_attention_heads": text_config.get("num_attention_heads", 8),
    "num_key_value_heads": text_config.get("num_key_value_heads", 4),
    "head_dim": text_config.get("head_dim", 256),
    "max_position_embeddings": text_config.get("max_position_embeddings", 131072),
    "hidden_act": text_config.get("hidden_activation") or text_config.get("hidden_act", "gelu_pytorch_tanh"),
    "rms_norm_eps": text_config.get("rms_norm_eps", 1e-06),
    "rope_theta": text_config.get("rope_theta", 1000000.0),
    "attention_bias": text_config.get("attention_bias", False),
    "attention_dropout": text_config.get("attention_dropout", 0.0),
    "query_pre_attn_scalar": text_config.get("query_pre_attn_scalar", 256),
    "sliding_window": text_config.get("sliding_window", 4096),
    "bos_token_id": config.get("bos_token_id", 2),
    "eos_token_id": config.get("eos_token_id", 1),
    "pad_token_id": config.get("pad_token_id", 0),
    "vocab_size": text_config.get("vocab_size", 256000),
    "initializer_range": text_config.get("initializer_range", 0.02),
    "use_cache": True,
}

# Add optional params
if "rope_scaling" in text_config:
    new_config["rope_scaling"] = text_config["rope_scaling"]

# Remove None values
new_config = {k: v for k, v in new_config.items() if v is not None}

with open('config.json', 'w') as f:
    json.dump(new_config, f, indent=2)

print("✓ Config prepared for vLLM")
print(f"  Architecture: {new_config['architectures'][0]}")
print(f"  Model type: {new_config['model_type']}")
print(f"  Layers: {new_config['num_hidden_layers']}")
print(f"  Vocab size: {new_config['vocab_size']}")
FIX_CONFIG

cd /workspace
echo ""

# Step 5: Start vLLM server
echo "[5/5] Starting vLLM server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  vLLM Server Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MERGED_DIR"
echo "  Host: 0.0.0.0"
echo "  Port: $PORT"
echo "  Max Length: $MAX_MODEL_LEN tokens"
echo "  GPU Memory: ${GPU_MEMORY_UTIL}"
echo ""
echo "  API Endpoint: http://localhost:$PORT"
echo "  Models: http://localhost:$PORT/v1/models"
echo "  Completions: http://localhost:$PORT/v1/completions"
echo "  Chat: http://localhost:$PORT/v1/chat/completions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Starting in 3 seconds..."
sleep 3

python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_DIR" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    --enable-prefix-caching \
    --max-num-seqs 256

echo ""
echo "============================================================"
echo "  Server stopped"
echo "============================================================"
