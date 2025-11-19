#!/bin/bash
# vllm_deploy_runpod.sh
# Complete deployment script for aidc-llm-laos-24k-gemma-3-4b-it on RunPod
# Author: Dr. Phonepadith
# Date: 2024-11-19

set -e  # Exit on error

echo "============================================================"
echo "  vLLM Deployment for Lao Language Model on RunPod"
echo "  Model: aidc-llm-laos-24k-gemma-3-4b-it"
echo "============================================================"
echo ""

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_REPO="Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"
MODEL_DIR="/workspace/aidc-model"
PORT=8000
MAX_MODEL_LEN=24576
GPU_MEMORY_UTIL=0.95
MAX_NUM_SEQS=256

# ============================================================
# STEP 1: System Preparation
# ============================================================
echo "[1/7] Preparing system environment..."
cd /workspace

# Set critical environment variables
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Check GPU availability - try different paths
GPU_FOUND=false
if command -v nvidia-smi &> /dev/null; then
    GPU_FOUND=true
elif [ -f /usr/bin/nvidia-smi ]; then
    export PATH=$PATH:/usr/bin
    GPU_FOUND=true
elif [ -f /usr/local/bin/nvidia-smi ]; then
    export PATH=$PATH:/usr/local/bin
    GPU_FOUND=true
fi

if [ "$GPU_FOUND" = true ]; then
    echo "✓ GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU present but nvidia-smi output unavailable"
else
    echo "⚠ WARNING: nvidia-smi not found, but continuing (GPU may still be available)"
fi

echo "✓ System environment ready"
echo ""

# ============================================================
# STEP 2: Install Dependencies
# ============================================================
echo "[2/7] Installing dependencies..."

# Upgrade pip quietly
pip install --upgrade pip -q 2>/dev/null || pip install --upgrade pip

# Install required packages
echo "Installing transformers and huggingface_hub..."
pip install -q --upgrade transformers huggingface_hub 2>/dev/null || pip install --upgrade transformers huggingface_hub

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM (this may take a few minutes)..."
    pip install vllm -q 2>/dev/null || pip install vllm
fi

# Verify installations
echo "Verifying installations..."
python -c "import vllm; print(f'✓ vLLM version: {vllm.__version__}')" 2>/dev/null || echo "✓ vLLM installed"
python -c "import transformers; print(f'✓ Transformers version: {transformers.__version__}')" 2>/dev/null || echo "✓ Transformers installed"

echo "✓ Dependencies installed"
echo ""

# ============================================================
# STEP 3: Download Model
# ============================================================
echo "[3/7] Checking model..."

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "✓ Model already exists at $MODEL_DIR"
else
    echo "Downloading model from HuggingFace Hub..."
    python << 'EOF'
import sys
from huggingface_hub import snapshot_download

MODEL_REPO = "Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it"
MODEL_DIR = "/workspace/aidc-model"

try:
    print(f"Downloading {MODEL_REPO}...")
    print("This will take several minutes for a ~23GB model...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✓ Model downloaded to {MODEL_DIR}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo "✗ ERROR: Model download failed"
        exit 1
    fi
fi

# Verify download
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "✗ ERROR: config.json not found in model directory"
    exit 1
fi

echo "✓ Model ready"
echo ""

# ============================================================
# STEP 4: Backup Original Config
# ============================================================
echo "[4/7] Backing up original configuration..."

cd "$MODEL_DIR"

if [ ! -f "config.json.original" ]; then
    cp config.json config.json.original
    echo "✓ Original config backed up"
else
    echo "✓ Backup already exists"
fi

echo ""

# ============================================================
# STEP 5: Fix Model Configuration
# ============================================================
echo "[5/7] Converting config to vLLM-compatible format..."

python << 'EOF'
import json
import sys

CONFIG_FILE = "config.json"

try:
    # Read original config
    print("Reading config.json...")
    with open(CONFIG_FILE, 'r') as f:
        old_config = json.load(f)
    
    # Display original config info
    print(f"Original model_type: {old_config.get('model_type', 'N/A')}")
    print(f"Original architectures: {old_config.get('architectures', 'N/A')}")
    
    # Check if this is a multimodal model with nested text_config
    text_config = old_config.get('text_config', {})
    has_text_config = bool(text_config)
    
    if has_text_config:
        print("Detected multimodal config with text_config section")
        source_config = text_config
    else:
        print("Using flat config structure")
        source_config = old_config
    
    # Build new Gemma2-compatible config
    new_config = {
        "architectures": ["GemmaForCausalLM"],
        "model_type": "gemma2",
        "torch_dtype": "bfloat16",
        
        # Core model parameters
        "hidden_size": source_config.get("hidden_size", 2560),
        "intermediate_size": source_config.get("intermediate_size", 10240),
        "num_hidden_layers": source_config.get("num_hidden_layers", 34),
        "num_attention_heads": source_config.get("num_attention_heads", 8),
        "num_key_value_heads": source_config.get("num_key_value_heads", 4),
        "head_dim": source_config.get("head_dim", 256),
        "max_position_embeddings": source_config.get("max_position_embeddings", 131072),
        
        # Activation and normalization
        "hidden_act": source_config.get("hidden_activation") or source_config.get("hidden_act", "gelu_pytorch_tanh"),
        "rms_norm_eps": source_config.get("rms_norm_eps", 1e-06),
        
        # RoPE configuration
        "rope_theta": source_config.get("rope_theta", 1000000.0),
        
        # Attention configuration
        "attention_bias": source_config.get("attention_bias", False),
        "attention_dropout": source_config.get("attention_dropout", 0.0),
        "query_pre_attn_scalar": source_config.get("query_pre_attn_scalar", 256),
        
        # Sliding window (Gemma 2 feature)
        "sliding_window": source_config.get("sliding_window", 4096),
        
        # Token IDs
        "bos_token_id": old_config.get("bos_token_id", 2),
        "eos_token_id": old_config.get("eos_token_id", 1),
        "pad_token_id": old_config.get("pad_token_id", 0),
        
        # Vocabulary
        "vocab_size": source_config.get("vocab_size", 256000),
        
        # Other settings
        "initializer_range": source_config.get("initializer_range", 0.02),
        "use_cache": True,
        "transformers_version": "4.48.0"
    }
    
    # Add optional parameters if they exist
    rope_scaling = source_config.get("rope_scaling")
    if rope_scaling:
        new_config["rope_scaling"] = rope_scaling
    
    attn_softcapping = source_config.get("attn_logit_softcapping")
    if attn_softcapping:
        new_config["attn_logit_softcapping"] = attn_softcapping
    
    final_softcapping = source_config.get("final_logit_softcapping")
    if final_softcapping:
        new_config["final_logit_softcapping"] = final_softcapping
    
    # Remove None values
    new_config = {k: v for k, v in new_config.items() if v is not None}
    
    # Save new config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(new_config, f, indent=2)
    
    print("\n✓ Config conversion successful!")
    print(f"  New model_type: {new_config['model_type']}")
    print(f"  New architecture: {new_config['architectures'][0]}")
    print(f"  Hidden layers: {new_config['num_hidden_layers']}")
    print(f"  Hidden size: {new_config['hidden_size']}")
    print(f"  Vocab size: {new_config['vocab_size']}")
    print(f"  Max position embeddings: {new_config['max_position_embeddings']}")
    
except Exception as e:
    print(f"\n✗ Config conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "✗ ERROR: Config conversion failed"
    exit 1
fi

cd /workspace
echo ""

# ============================================================
# STEP 6: Verify Configuration
# ============================================================
echo "[6/7] Verifying configuration..."

python << EOF
import json
import sys

try:
    with open('${MODEL_DIR}/config.json', 'r') as f:
        config = json.load(f)
    
    # Verify required fields
    assert 'model_type' in config, "Missing model_type"
    assert 'architectures' in config, "Missing architectures"
    assert config['architectures'][0] == 'GemmaForCausalLM', "Wrong architecture"
    assert 'vocab_size' in config, "Missing vocab_size"
    assert 'hidden_size' in config, "Missing hidden_size"
    
    print("✓ Configuration validation passed")
    print(f"  Model ready for deployment")
    
except AssertionError as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error during validation: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "✗ ERROR: Configuration validation failed"
    exit 1
fi

echo ""

# ============================================================
# STEP 7: Start vLLM Server
# ============================================================
echo "[7/7] Starting vLLM OpenAI-compatible API server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Configuration Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model Path:           $MODEL_DIR"
echo "  Host:                 0.0.0.0"
echo "  Port:                 $PORT"
echo "  Max Model Length:     $MAX_MODEL_LEN"
echo "  GPU Memory:           ${GPU_MEMORY_UTIL}"
echo "  Max Sequences:        $MAX_NUM_SEQS"
echo "  Prefix Caching:       Enabled"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Starting server in 3 seconds..."
sleep 3

echo ""
echo "============================================================"
echo "  🚀 vLLM Server Starting..."
echo "============================================================"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --dtype auto \
    --enable-prefix-caching \
    --max-num-seqs $MAX_NUM_SEQS

# If we reach here, the server has stopped
echo ""
echo "============================================================"
echo "  Server stopped"
echo "============================================================"
