#!/bin/bash
# merge_with_unsloth.sh
# Use unsloth to merge the LoRA adapter (since it was trained with unsloth)

set -e

echo "============================================================"
echo "  Merging LoRA Adapter with Unsloth"
echo "  Model: aidc-llm-laos-24k-gemma-3-4b-it"
echo "============================================================"
echo ""

ADAPTER_DIR="/workspace/aidc-model"
MERGED_DIR="/workspace/aidc-model-merged"
PORT=8000

# Setup environment
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0

echo "[1/4] Installing unsloth..."
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --upgrade transformers
echo "✓ Unsloth installed"
echo ""

echo "[2/4] Checking adapter..."
if [ ! -d "$ADAPTER_DIR" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it', local_dir='$ADAPTER_DIR', local_dir_use_symlinks=False)
"
fi
echo "✓ Adapter ready"
echo ""

echo "[3/4] Merging with unsloth..."
echo "This will take 5-15 minutes..."
echo ""

python << 'UNSLOTH_MERGE'
from unsloth import FastLanguageModel
import torch
import os

ADAPTER_DIR = "/workspace/aidc-model"
MERGED_DIR = "/workspace/aidc-model-merged"

print("="*60)
print("Loading model with unsloth...")
print("="*60)

try:
    # Load the adapter with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,
        max_seq_length=24576,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # Load in full precision for merging
    )
    
    print("\nSaving merged model in 16-bit format...")
    print(f"Output directory: {MERGED_DIR}")
    
    # Save the merged model
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",  # Save as full 16-bit model
    )
    
    print("\n" + "="*60)
    print("✓ Merge successful!")
    print("="*60)
    print(f"Merged model saved to: {MERGED_DIR}")
    
except Exception as e:
    print(f"\n✗ Merge failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
UNSLOTH_MERGE

if [ $? -ne 0 ]; then
    echo "✗ Merge failed"
    exit 1
fi

echo ""

echo "[4/4] Fixing config for vLLM..."
cd "$MERGED_DIR"

python << 'FIX_CONFIG'
import json
import os

if not os.path.exists('config.json'):
    print("✗ config.json not found!")
    exit(1)

with open('config.json', 'r') as f:
    config = json.load(f)

# Save backup
with open('config.json.backup', 'w') as f:
    json.dump(config, f, indent=2)

# Get config source
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

new_config = {k: v for k, v in new_config.items() if v is not None}

with open('config.json', 'w') as f:
    json.dump(new_config, f, indent=2)

print("✓ Config fixed for vLLM")
print(f"  Architecture: {new_config['architectures'][0]}")
print(f"  Model type: {new_config['model_type']}")
FIX_CONFIG

cd /workspace

echo ""
echo "============================================================"
echo "✓ Model ready for vLLM deployment!"
echo "============================================================"
echo ""
echo "Now starting vLLM server..."
echo ""

# Install vLLM if needed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm
fi

# Start vLLM server
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  vLLM Server Starting"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $MERGED_DIR"
echo "  Port: $PORT"
echo ""
echo "  API Endpoints:"
echo "    http://localhost:$PORT/v1/models"
echo "    http://localhost:$PORT/v1/completions"
echo "    http://localhost:$PORT/v1/chat/completions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_DIR" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --enable-prefix-caching \
    --max-num-seqs 256
