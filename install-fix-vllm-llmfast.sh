#!/bin/bash
# install-vllm-start-llm-fast-fixed.sh
# Fixed deployment script that handles PEFT/LoRA adapter models

set -e

echo "============================================================"
echo "  vLLM Deployment for Lao Language Model on RunPod"
echo "  Model: aidc-llm-laos-24k-gemma-3-4b-it"
echo "============================================================"
echo ""

# Configuration
MODEL_REPO="Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"
MODEL_DIR="/workspace/aidc-model"
PORT=8000
MAX_MODEL_LEN=24576
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=256

# Step 1: Environment setup
echo "[1/8] Preparing system environment..."
cd /workspace
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0
echo "✓ Environment ready"
echo ""

# Step 2: Install dependencies
echo "[2/8] Installing dependencies..."
pip install -q --upgrade pip transformers huggingface_hub safetensors torch 2>/dev/null || pip install --upgrade transformers huggingface_hub safetensors torch
if ! python -c "import vllm" 2>/dev/null; then
    pip install vllm -q 2>/dev/null || pip install vllm
fi
echo "✓ Dependencies installed"
echo ""

# Step 3: Download model
echo "[3/8] Checking model..."
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "✓ Model exists"
else
    echo "Downloading model (this may take several minutes)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', local_dir='$MODEL_DIR', local_dir_use_symlinks=False, resume_download=True)
print('✓ Download complete')
"
fi
echo ""

# Step 4: Check if model has PEFT/LoRA adapter structure
echo "[4/8] Checking model structure..."
cd "$MODEL_DIR"

python << 'EOF'
import os
import json

# Check for adapter config (PEFT/LoRA model)
if os.path.exists('adapter_config.json'):
    print("⚠ WARNING: This is a PEFT/LoRA adapter model")
    print("vLLM cannot load adapter models directly")
    print("You need to merge the adapter with the base model first")
    
    with open('adapter_config.json', 'r') as f:
        adapter_config = json.load(f)
    base_model = adapter_config.get('base_model_name_or_path', 'Unknown')
    print(f"Base model: {base_model}")
    print("\nTo fix this, you need to merge the adapter:")
    print("1. Load the base model")
    print("2. Load the PEFT adapter")
    print("3. Merge them")
    print("4. Save the merged model")
    exit(1)
else:
    print("✓ This is a full model (not an adapter)")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "  SOLUTION: Merge the LoRA adapter with base model"
    echo "============================================================"
    echo ""
    echo "Creating merge script..."
    
    cat > /workspace/merge_adapter.py << 'MERGE_EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

MODEL_DIR = "/workspace/aidc-model"
OUTPUT_DIR = "/workspace/aidc-model-merged"

print("Loading base model and adapter...")
print("This may take several minutes...")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load adapter
print("Loading PEFT adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)

# Merge
print("Merging adapter with base model...")
model = model.merge_and_unload()

# Save merged model
print(f"Saving merged model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✓ Merge complete!")
print(f"Merged model saved to: {OUTPUT_DIR}")
MERGE_EOF

    echo ""
    echo "Installing PEFT library..."
    pip install -q peft
    
    echo ""
    echo "Merging adapter with base model..."
    python /workspace/merge_adapter.py
    
    if [ $? -eq 0 ]; then
        MODEL_DIR="/workspace/aidc-model-merged"
        echo "✓ Using merged model at $MODEL_DIR"
    else
        echo "✗ Merge failed"
        exit 1
    fi
fi

cd /workspace
echo ""

# Step 5: Backup and fix config
echo "[5/8] Fixing configuration..."
cd "$MODEL_DIR"

if [ ! -f "config.json.original" ]; then
    cp config.json config.json.original
fi

python << 'EOF'
import json
import sys

try:
    with open('config.json', 'r') as f:
        old_config = json.load(f)
    
    text_config = old_config.get('text_config', {})
    source_config = text_config if text_config else old_config
    
    new_config = {
        "architectures": ["GemmaForCausalLM"],
        "model_type": "gemma2",
        "torch_dtype": "bfloat16",
        "hidden_size": source_config.get("hidden_size", 2560),
        "intermediate_size": source_config.get("intermediate_size", 10240),
        "num_hidden_layers": source_config.get("num_hidden_layers", 34),
        "num_attention_heads": source_config.get("num_attention_heads", 8),
        "num_key_value_heads": source_config.get("num_key_value_heads", 4),
        "head_dim": source_config.get("head_dim", 256),
        "max_position_embeddings": source_config.get("max_position_embeddings", 131072),
        "hidden_act": source_config.get("hidden_activation") or source_config.get("hidden_act", "gelu_pytorch_tanh"),
        "rms_norm_eps": source_config.get("rms_norm_eps", 1e-06),
        "rope_theta": source_config.get("rope_theta", 1000000.0),
        "attention_bias": source_config.get("attention_bias", False),
        "attention_dropout": source_config.get("attention_dropout", 0.0),
        "query_pre_attn_scalar": source_config.get("query_pre_attn_scalar", 256),
        "sliding_window": source_config.get("sliding_window", 4096),
        "bos_token_id": old_config.get("bos_token_id", 2),
        "eos_token_id": old_config.get("eos_token_id", 1),
        "pad_token_id": old_config.get("pad_token_id", 0),
        "vocab_size": source_config.get("vocab_size", 256000),
        "initializer_range": source_config.get("initializer_range", 0.02),
        "use_cache": True,
    }
    
    rope_scaling = source_config.get("rope_scaling")
    if rope_scaling:
        new_config["rope_scaling"] = rope_scaling
    
    new_config = {k: v for k, v in new_config.items() if v is not None}
    
    with open('config.json', 'w') as f:
        json.dump(new_config, f, indent=2)
    
    print("✓ Config fixed")
    print(f"  Model type: {new_config['model_type']}")
    print(f"  Architecture: {new_config['architectures'][0]}")
    
except Exception as e:
    print(f"✗ Config fix failed: {e}")
    sys.exit(1)
EOF

cd /workspace
echo ""

# Step 6: Verify config
echo "[6/8] Verifying configuration..."
python -c "
import json
with open('$MODEL_DIR/config.json', 'r') as f:
    config = json.load(f)
assert config['architectures'][0] == 'GemmaForCausalLM'
print('✓ Configuration valid')
"
echo ""

# Step 7: Check model weights structure
echo "[7/8] Checking model weights..."
python << 'EOF'
import os
from safetensors import safe_open

model_dir = os.environ.get('MODEL_DIR', '/workspace/aidc-model')
safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]

if not safetensors_files:
    print("✗ No safetensors files found")
    exit(1)

print(f"Found {len(safetensors_files)} safetensors file(s)")

# Check first file for key structure
first_file = os.path.join(model_dir, safetensors_files[0])
with safe_open(first_file, framework="pt") as f:
    keys = list(f.keys())[:5]  # Check first 5 keys
    print(f"Sample weight keys: {keys}")
    
    # Check if keys have unwanted prefixes
    has_base_model_prefix = any(k.startswith('base_model.') for k in keys)
    if has_base_model_prefix:
        print("✗ ERROR: Weights have 'base_model.' prefix")
        print("This model needs to be properly merged")
        exit(1)
    
print("✓ Weight structure looks good")
EOF

if [ $? -ne 0 ]; then
    echo "✗ Weight structure check failed"
    echo "This model cannot be loaded by vLLM in its current state"
    exit 1
fi
echo ""

# Step 8: Start vLLM
echo "[8/8] Starting vLLM server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MODEL_DIR"
echo "  Port: $PORT"
echo "  Max Length: $MAX_MODEL_LEN tokens"
echo "  GPU Memory: ${GPU_MEMORY_UTIL}"
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
    --max-num-seqs $MAX_NUM_SEQS
