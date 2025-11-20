#!/bin/bash
set -e

# Update packages
apt update -y
apt install -y lshw wget

# Install Ollama (x86_64)
curl -s https://ollama.com/install.sh | bash

# Run Ollama server in background
nohup ollama serve > ollama.log 2>&1 &
echo "Ollama server started in background. Logs: /root/AIDC-LLM/ollama.log"

# Wait for Ollama to be ready
sleep 5

# Pull BGE-M3 embedding model
echo "Pulling BGE-M3 embedding model..."
ollama pull bge-m3

# Create directory
cd /root || exit
mkdir -p AIDC-LLM
cd AIDC-LLM

# Download model file
echo "Downloading model AIDC-12B-IT..."
wget -O aidc-llm-laos-24k-gemma-3-12b-it-q8.gguf \
"https://huggingface.co/Phonepadith/aidc-llm-laos-24k-gemma-3-12b-it/resolve/main/aidc-llm-laos-24k-gemma-3-12b-it-q8.gguf"

#Download Speed Model of AIDC
echo "Downloading model AIDC-4B-IT..."
wget -O aidc-llm-laos-24k-gemma-3-4b-it-q8.gguf \
"https://huggingface.co/Phonepadith/aidc-llm-laos-24k-gemma-3-4b-it/resolve/main/aidc-llm-laos-24k-gemma-3-4b-it-Q8.gguf"


# Create RAG-optimized Modelfile
cat <<'EOF' > Modelfile
FROM ./aidc-llm-laos-24k-gemma-3-12b-it-q8.gguf

# RAG-optimized template for Gemma 3
TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ if .Context }}ຂໍ້ມູນອ້າງອີງ (Context):
{{ .Context }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

# System prompt optimized for RAG
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

# RAG-optimized parameters
# Temperature
PARAMETER temperature 0.1

# Max response length (tokens)
PARAMETER num_predict 15000

# Sampling settings
PARAMETER top_k 40
PARAMETER repeat_penalty 1.0       # OFF in UI = neutral (1.0)
PARAMETER min_p 0.04
PARAMETER top_p 1.0

# Gemma 3 stop tokens
PARAMETER stop <start_of_turn>
PARAMETER stop <end_of_turn>
PARAMETER stop <|im_end|>
EOF

# Create RAG-optimized Modelfile
cat <<'EOF' > Modelfile
FROM ./aidc-llm-laos-24k-gemma-3-4b-it-q8.gguf

# ---- RAG Template (strict: use context, but natural output) ----
TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ if .Context }}ຂໍ້ມູນອ້າງອີງ (Context):
{{ .Context }}

ກະລຸນາຕອບຈາກ Context ເທົ່ານັ້ນ. ຢ່າຄາດເດົາ. 
{{ end }}
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

# ---- System Prompt Optimized for Strong RAG ----
SYSTEM """
ເຈົ້າແມ່ນ AI Assistant ທີ່ຕອບຕາມຂໍ້ມູນອ້າງອີງ (Context) ເທົ່ານັ້ນ.

1) ຖ້າ Context ມີຂໍ້ມູນ -> ຕອບຈາກ Context ໂດຍກົງ  
2) ຖ້າ Context ບໍ່ມີຫຼືບໍ່ພໍ -> ບອກວ່າ “ບໍ່ພົບຂໍ້ມູນໃນ Context”  
3) ບໍ່ຄວນສ້າງຂໍ້ມູນໃໝ່ ຫຼື ຄາດເດົາ  
4) ຕອບໃຫ້ລາຍລະອຽດ ແຕ່ອີງຈາກ Context ເທົ່ານັ້ນ

ຕອບເປັນພາສາລາວທີ່ຊັດເຈນ.
"""

# ---- Parameters for Strong RAG Control ----
# Temperature
PARAMETER temperature 0.1

# Max response length (tokens)
PARAMETER num_predict 15000

# Sampling settings
PARAMETER top_k 40
PARAMETER repeat_penalty 1.0       # OFF in UI = neutral (1.0)
PARAMETER min_p 0.04
PARAMETER top_p 1.0


# Stop tokens for Gemma 3
PARAMETER stop <start_of_turn>
PARAMETER stop <end_of_turn>
PARAMETER stop <|im_end|>
EOF


# Create Ollama model
echo "Creating Ollama model..."
ollama create BOL-CH-2 -f Modelfile
ollama create AIDC-STANDARD-LLM -f Modelfile
ollama create AIDC-FAST-LLM -f Modelfile-SP
ollama cp BOL-CH-2 BOL-CH-3
ollama cp BOL-CH-2 BOL-CH-4
ollama cp BOL-CH-2 ROBOT-AIDC-LLM

echo "Model created successfully!"
echo "Model name: aidc-llm-laos"
echo "Embedding model: bge-m3"
echo ""
echo "Test the model:"
echo "ollama run aidc-llm-laos"
echo ""
echo "Connect to Open WebUI:"
echo "- Ollama API URL: http://localhost:11434"
echo "- Model name: aidc-llm-laos"
echo "- Embedding model: bge-m3"
