#!/bin/bash
# complete_merge_and_deploy_fixed.sh
# Fixed version that handles meta tensors correctly

set -e

echo "============================================================"
echo "  LoRA Adapter Merge & vLLM Deployment"
echo "  Adapter: aidc-llm-laos-24k-gemma-3-4b-it"
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

# Step 1: Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q transformers accelerate peft bitsandbytes safetensors torch
if ! python -c "import vllm" 2>/dev/null; then
    pip install vllm
fi
echo "✓ Dependencies installed"
echo ""

# Step 2: Check adapter
echo "[2/5] Checking adapter model..."
if [ ! -d "$ADAPTER_DIR" ]; then
    echo "Downloading adapter..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$ADAPTER_REPO', local_dir='$ADAPTER_DIR', local_dir_use_symlinks=False)
"
fi
echo "✓ Adapter ready"
echo ""

# Step 3: Merge with proper handling
echo "[3/5] Merging LoRA adapter with base model..."

if [ -d "$MERGED_DIR" ]; then
    echo "Merged model exists. Remove it? (y/N)"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$MERGED_DIR"
    else
        echo "Using existing merged model"
        MODEL_DIR="$MERGED_DIR"
    fi
fi

if [ ! -d "$MERGED_DIR" ]; then
    python << 'MERGE_SCRIPT'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import gc

print("="*60)
print("Merging LoRA adapter with base model")
print("="*60)

ADAPTER_DIR = "/workspace/aidc-model"
MERGED_DIR = "/workspace/aidc-model-merged"

try:
    # Get base model info
    print("\n[1/5] Reading adapter configuration...")
    peft_config = PeftConfig.from_pretrained(ADAPTER_DIR)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_name}")
    
    # Load base model WITHOUT device_map to avoid meta tensor issues
    print("\n[2/5] Loading base model...")
    print("This may take 5-10 minutes and download ~8GB...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # Don't use device_map="auto" to avoid meta tensors
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        print("Moving base model to GPU...")
        base_model = base_model.cuda()
    
    print("\n[3/5] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        torch_dtype=torch.bfloat16
    )
    
    print("\n[4/5] Merging adapter with base model...")
    model = model.merge_and_unload()
    
    # Move back to CPU for saving
    print("Moving to CPU for saving...")
    model = model.cpu()
    
    print("\n[5/5] Saving merged model...")
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
    
    # Cleanup
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
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
        echo ""
        echo "Merge failed. Trying alternative method..."
        
        # Try alternative merge method using unsloth
        python << 'ALT_MERGE'
print("Trying alternative merge with unsloth...")

try:
    from unsloth import FastLanguageModel
    import torch
    
    ADAPTER_DIR = "/workspace/aidc-model"
    MERGED_DIR = "/workspace/aidc-model-merged"
    
    print("Loading model with unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=24576,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    
    print("Saving merged model...")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )
    
    print("✓ Alternative merge successful!")
    
except ImportError:
    print("unsloth not available, installing...")
    import subprocess
    subprocess.run(["pip", "install", "unsloth"], check=True)
    print("Please run the script again")
    exit(1)
except Exception as e:
    print(f"✗ Alternative merge also failed: {e}")
    exit(1)
ALT_MERGE

        if [ $? -ne 0 ]; then
            echo "✗ All merge methods failed"
            exit 1
        fi
    fi
fi

MODEL_DIR="$MERGED_DIR"
echo ""

# Step 4: Fix config
echo "[4/5] Preparing config for vLLM..."
cd "$MODEL_DIR"

python << 'FIX_CONFIG'
import json

with open('config.json', 'r') as f:
    config = json.load(f)

# Check structure
text_config = config.get('text_config', config)

# Build vLLM-compatible config
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
    "initializer_range": 0.02,
    "use_cache": True,
}

# Save original
with open('config.json.backup', 'w') as f:
    json.dump(config, f, indent=2)

# Remove None values
new_config = {k: v for k, v in new_config.items() if v is not None}

with open('config.json', 'w') as f:
    json.dump(new_config, f, indent=2)

print("✓ Config ready")
print(f"  Model type: {new_config['model_type']}")
print(f"  Architecture: {new_config['architectures'][0]}")
FIX_CONFIG

cd /workspace
echo ""

# Step 5: Start vLLM
echo "[5/5] Starting vLLM server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Server Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL_DIR"
echo "  Port: $PORT"
echo "  Max Length: $MAX_MODEL_LEN"
echo "  GPU Memory: ${GPU_MEMORY_UTIL}"
echo ""
echo "  Endpoints:"
echo "    Models:      http://localhost:$PORT/v1/models"
echo "    Completions: http://localhost:$PORT/v1/completions"
echo "    Chat:        http://localhost:$PORT/v1/chat/completions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    --enable-prefix-caching \
    --max-num-seqs 256
