#!/bin/bash
set -e

# Update packages
apt update -y
apt install -y lshw wget

# Install Ollama (x86_64)
curl -s https://ollama.com/install.sh | bash

# Stop any existing ollama service to apply new configs
systemctl stop ollama || true
pkill ollama || true

# --- CONFIGURATION FOR EXPOSING PORT ---
# 0.0.0.0 allows connections from any IP
export OLLAMA_HOST=0.0.0.0:11434
# "*" allows Open WebUI to connect from a different browser/domain
export OLLAMA_ORIGINS="*"

# Run Ollama server in background with new configs
nohup ollama serve > ollama.log 2>&1 &
echo "Ollama server started in background on 0.0.0.0:11434"

# Wait for Ollama to be fully ready (Check loop)
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
    echo "Retrying..."
done

# Pull BGE-M3 embedding model
echo "Pulling BGE-M3 and creating embedding model..."
ollama pull bge-m3 

# Create directory
cd /home/workspace || exit
mkdir -p AIDC-LLM
cd AIDC-LLM

# Download model file
echo "Downloading model AIDC-12B-IT..."
wget -O aidc-llm-laos-24k-gemma-3-12b-it-q8.gguf \
"https://huggingface.co/Phonepadith/aidc-llm-laos-24k-gemma-3-12b-it/resolve/main/aidc-llm-laos-24k-gemma-3-12b-it-q8.gguf"

# Download Speed Model of AIDC
echo "Downloading model AIDC-4B-IT..."
wget -O aidc-llm-laos-24k-gemma-3-4b-it-q8.gguf \
"https://huggingface.co/Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it/resolve/main/aidc-llm-laos-24k-gemma-3-4b-it-Q8.gguf"


# Create RAG-optimized Modelfile (Standard)
cat <<'EOF' > Modelfile
FROM ./aidc-llm-laos-24k-gemma-3-12b-it-q8.gguf

# RAG-optimized template for Gemma-3
TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ if .Context }}ຂໍ້ມູນອ້າງອີງ (Context):
{{ .Context }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

SYSTEM """
ເຈົ້າເປັນ AI Assistant ທີ່ສະຫຼາດ ແລະ ມີຄວາມຮັບຜິດຊອບ.

ເມື່ອມີຂໍ້ມູນອ້າງອີງ (Context) ໃຫ້:
- ໃຊ້ຂໍ້ມູນອ້າງອີງເພື່ອຕອບຄຳຖາມຢ່າງແມ່ນຍຳ
- ອ້າງອີງຂໍ້ມູນຈາກ Context ໂດຍກົງ
- ຖ້າຂໍ້ມູນໃນ Context ບໍ່ພຽງພໍ ໃຫ້ບອກຢ່າງຊັດເຈນ

ເມື່ອບໍ່ມີຂໍ້ມູນອ້າງອີງ:
- ຕອບຕາມຄວາມຮູ້ທົ່ວໄປຂອງເຈົ້າ
- ໃຫ້ຄຳແນະນຳທີ່ເປັນປະໂຫຍດ

ຕອບເປັນພາສາລາວທີ່ຊັດເຈນ ແລະ ເຂົ້າໃຈງ່າຍ.
"""

# RAG-optimized parameters for 12B model
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER num_predict 15000
PARAMETER repeat_penalty 1.1

# Gemma stop tokens
PARAMETER stop <start_of_turn>
PARAMETER stop <end_of_turn>
PARAMETER stop <|im_end|>
EOF

# Create RAG-optimized Modelfile (Fast)
cat <<'EOF' > Modelfile-SP
FROM ./aidc-llm-laos-24k-gemma-3-4b-it-q8.gguf

# RAG-optimized template for Gemma-3
TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ if .Context }}ຂໍ້ມູນອ້າງອີງ (Context):
{{ .Context }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

SYSTEM """
ເຈົ້າເປັນ AI Assistant ທີ່ສະຫຼາດ ແລະ ມີຄວາມຮັບຜິດຊອບ.

ເມື່ອມີຂໍ້ມູນອ້າງອີງ (Context) ໃຫ້:
- ໃຊ້ຂໍ້ມູນອ້າງອີງເພື່ອຕອບຄຳຖາມຢ່າງແມ່ນຍຳ
- ອ້າງອີງຂໍ້ມູນຈາກ Context ໂດຍກົງ
- ຖ້າຂໍ້ມູນໃນ Context ບໍ່ພຽງພໍ ໃຫ້ບອກຢ່າງຊັດເຈນ

ເມື່ອບໍ່ມີຂໍ້ມູນອ້າງອີງ:
- ຕອບຕາມຄວາມຮູ້ທົ່ວໄປຂອງເຈົ້າ
- ໃຫ້ຄຳແນະນຳທີ່ເປັນປະໂຫຍດ

ຕອບເປັນພາສາລາວທີ່ຊັດເຈນ ແລະ ເຂົ້າໃຈງ່າຍ.
"""

# RAG-optimized parameters for 4B model
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER num_predict 15000
PARAMETER repeat_penalty 1.1

# Gemma stop tokens
PARAMETER stop <start_of_turn>
PARAMETER stop <end_of_turn>
PARAMETER stop <|im_end|>
EOF


# Create Ollama models
echo "Creating Ollama models..."
ollama create AIDC-STANDARD-LLM -f Modelfile
ollama create AIDC-FAST-LLM -f Modelfile-SP


echo "============================================="
echo "Models created successfully!"
echo "High Quality Model: AIDC-STANDARD-LLM"
echo "High Speed Model:   AIDC-FAST-LLM"
echo "Embedding Model:    bge-m3"
echo "============================================="
echo "Test the model:"
echo "ollama run AIDC-STANDARD-LLM"
echo ""
echo "Connect to Open WebUI / External Apps:"
echo "- Ollama API URL: http://<YOUR-SERVER-IP>:11434"
echo "- Model name: AIDC-STANDARD-LLM"
echo "- Embedding model: bge-m3"
