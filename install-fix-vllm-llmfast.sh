#!/bin/bash
# fix_and_deploy.sh
# Fix unsloth weight keys and deploy with vLLM

set -e

echo "============================================================"
echo "  Fixing Unsloth Model Weights for vLLM"
echo "============================================================"
echo ""

MERGED_DIR="/workspace/aidc-model-merged"
FIXED_DIR="/workspace/aidc-model-vllm-ready"
PORT=8000

if [ ! -d "$MERGED_DIR" ]; then
    echo "✗ Merged model not found at $MERGED_DIR"
    echo "Please run the merge script first"
    exit 1
fi

echo "[1/3] Fixing weight keys..."

if [ -d "$FIXED_DIR" ]; then
    echo "Fixed model already exists. Remove it? (y/N)"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$FIXED_DIR"
    else
        echo "Using existing fixed model"
    fi
fi

if [ ! -d "$FIXED_DIR" ]; then
python << 'FIX_WEIGHTS'
import torch
from safetensors.torch import load_file, save_file
import os
import json
from pathlib import Path

MERGED_DIR = "/workspace/aidc-model-merged"
FIXED_DIR = "/workspace/aidc-model-vllm-ready"

print("="*60)
print("Fixing weight key names for vLLM compatibility")
print("="*60)

os.makedirs(FIXED_DIR, exist_ok=True)

# Get all safetensors files
safetensor_files = list(Path(MERGED_DIR).glob("*.safetensors"))
print(f"\nFound {len(safetensor_files)} safetensors file(s)")

for idx, file_path in enumerate(safetensor_files, 1):
    print(f"\n[{idx}/{len(safetensor_files)}] Processing {file_path.name}...")
    
    # Load weights
    weights = load_file(str(file_path))
    
    # Check and fix key names
    fixed_weights = {}
    changes_made = False
    
    for key, tensor in weights.items():
        # Remove unwanted prefixes
        new_key = key
        
        # Remove 'language_model.' prefix if present
        if new_key.startswith('language_model.'):
            new_key = new_key.replace('language_model.', '', 1)
            changes_made = True
        
        # Remove 'base_model.model.' prefix if present
        if new_key.startswith('base_model.model.'):
            new_key = new_key.replace('base_model.model.', '', 1)
            changes_made = True
        
        # Remove 'model.' prefix if it's the first component
        if new_key.startswith('model.') and not new_key.startswith('model.layers'):
            # Keep 'model.layers' but remove standalone 'model.'
            parts = new_key.split('.', 1)
            if len(parts) > 1 and not parts[1].startswith('layers'):
                new_key = parts[1]
                changes_made = True
        
        fixed_weights[new_key] = tensor
        
        if new_key != key:
            print(f"  Renamed: {key[:60]}... -> {new_key[:60]}...")
    
    if not changes_made:
        print(f"  No changes needed for {file_path.name}")
    
    # Save fixed weights
    output_path = Path(FIXED_DIR) / file_path.name
    save_file(fixed_weights, str(output_path))
    print(f"  Saved to: {output_path.name}")

# Copy config and tokenizer files
print("\nCopying config and tokenizer files...")
for file_name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 
                  'special_tokens_map.json', 'tokenizer.model', 'generation_config.json']:
    src = Path(MERGED_DIR) / file_name
    if src.exists():
        dst = Path(FIXED_DIR) / file_name
        import shutil
        shutil.copy2(src, dst)
        print(f"  Copied: {file_name}")

print("\n" + "="*60)
print("✓ Weight fixing complete!")
print("="*60)
print(f"Fixed model saved to: {FIXED_DIR}")
FIX_WEIGHTS

    if [ $? -ne 0 ]; then
        echo "✗ Weight fixing failed"
        exit 1
    fi
fi

echo ""

echo "[2/3] Verifying fixed model..."
python << 'VERIFY'
from safetensors.torch import load_file
from pathlib import Path

FIXED_DIR = "/workspace/aidc-model-vllm-ready"

# Check first safetensors file
safetensor_files = list(Path(FIXED_DIR).glob("*.safetensors"))
if not safetensor_files:
    print("✗ No safetensors files found!")
    exit(1)

weights = load_file(str(safetensor_files[0]))
sample_keys = list(weights.keys())[:5]

print("Sample weight keys:")
for key in sample_keys:
    print(f"  {key}")

# Check for bad prefixes
bad_prefixes = ['language_model.', 'base_model.']
has_bad_prefix = any(key.startswith(prefix) for prefix in bad_prefixes for key in sample_keys)

if has_bad_prefix:
    print("\n✗ ERROR: Model still has incompatible prefixes!")
    exit(1)

print("\n✓ Model structure looks good for vLLM")
VERIFY

echo ""

echo "[3/3] Starting vLLM server..."

if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  vLLM Server Starting"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $FIXED_DIR"
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
    --model "$FIXED_DIR" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --enable-prefix-caching \
    --max-num-seqs 256
