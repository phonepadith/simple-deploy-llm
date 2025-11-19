#!/bin/bash
# final_fix_and_deploy.sh
# Add proper 'model.' prefix to weight keys

set -e

echo "============================================================"
echo "  Final Weight Key Fix for vLLM Compatibility"
echo "============================================================"
echo ""

MERGED_DIR="/workspace/aidc-model-vllm-ready"
FINAL_DIR="/workspace/aidc-model-final"
PORT=8000

if [ ! -d "$MERGED_DIR" ]; then
    echo "✗ Model not found at $MERGED_DIR"
    exit 1
fi

echo "[1/2] Adding 'model.' prefix to weight keys..."

if [ -d "$FINAL_DIR" ]; then
    rm -rf "$FINAL_DIR"
fi

python << 'FINAL_FIX'
import torch
from safetensors.torch import load_file, save_file
import os
import json
from pathlib import Path
import shutil

MERGED_DIR = "/workspace/aidc-model-vllm-ready"
FINAL_DIR = "/workspace/aidc-model-final"

print("="*60)
print("Adding proper 'model.' prefix to all weights")
print("="*60)

os.makedirs(FINAL_DIR, exist_ok=True)

# Get all safetensors files
safetensor_files = list(Path(MERGED_DIR).glob("*.safetensors"))
print(f"\nFound {len(safetensor_files)} safetensors file(s)")

for idx, file_path in enumerate(safetensor_files, 1):
    print(f"\n[{idx}/{len(safetensor_files)}] Processing {file_path.name}...")
    
    # Load weights
    weights = load_file(str(file_path))
    
    # Fix key names - add 'model.' prefix where needed
    fixed_weights = {}
    
    for key, tensor in weights.items():
        new_key = key
        
        # Keys that should have 'model.' prefix
        if not key.startswith('model.') and not key.startswith('lm_head.'):
            # Add 'model.' prefix to everything except lm_head
            if key == 'embed_tokens.weight':
                new_key = 'model.embed_tokens.weight'
            elif key.startswith('layers.') or key.startswith('norm.'):
                new_key = f'model.{key}'
            else:
                new_key = f'model.{key}'
        
        fixed_weights[new_key] = tensor
        
        if new_key != key:
            print(f"  {key} -> {new_key}")
    
    # Save fixed weights
    output_path = Path(FINAL_DIR) / file_path.name
    save_file(fixed_weights, str(output_path))
    print(f"  ✓ Saved: {output_path.name}")

# Copy other files
print("\nCopying config and tokenizer files...")
for file_name in os.listdir(MERGED_DIR):
    if not file_name.endswith('.safetensors'):
        src = Path(MERGED_DIR) / file_name
        dst = Path(FINAL_DIR) / file_name
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"  ✓ {file_name}")

# Verify
print("\nVerifying final weight structure...")
first_file = list(Path(FINAL_DIR).glob("*.safetensors"))[0]
weights = load_file(str(first_file))
sample_keys = sorted(weights.keys())[:10]

print("\nSample weight keys:")
for key in sample_keys:
    print(f"  {key}")

# Check structure
has_model_prefix = any(k.startswith('model.') for k in sample_keys)
has_embed = 'model.embed_tokens.weight' in weights
has_norm = any('model.norm' in k for k in weights.keys())

print(f"\n✓ Has 'model.' prefix: {has_model_prefix}")
print(f"✓ Has 'model.embed_tokens.weight': {has_embed}")
print(f"✓ Has norm layer: {has_norm}")

if not (has_model_prefix and has_embed):
    print("\n✗ ERROR: Weight structure still incorrect!")
    exit(1)

print("\n" + "="*60)
print("✓ Weight keys fixed successfully!")
print("="*60)
print(f"Final model saved to: {FINAL_DIR}")
FINAL_FIX

if [ $? -ne 0 ]; then
    echo "✗ Final fix failed"
    exit 1
fi

echo ""

echo "[2/2] Starting vLLM server..."

if ! python -c "import vllm" 2>/dev/null; then
    pip install vllm
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 vLLM Server Starting - Final Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: $FINAL_DIR"
echo "  Port: $PORT"
echo ""
echo "  Test with:"
echo "    curl http://localhost:$PORT/v1/models"
echo ""
echo "    curl http://localhost:$PORT/v1/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"aidc-model-final\","
echo "           \"prompt\": \"ສະບາຍດີ\","
echo "           \"max_tokens\": 50}'"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$FINAL_DIR" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --enable-prefix-caching \
    --max-num-seqs 256
