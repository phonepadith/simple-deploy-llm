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
echo "Pulling BGE-M3 and creating embedding model..."
# Create Modelfile
cat << 'EOF' > Modelfile-EMD
FROM hf.co/amiya/bge-m3.gguf
TEMPLATE """{{ .Prompt }}"""
EOF

# Build Ollama model
echo "Building bge-m3 embedding model..."
ollama create bge-m3 -f Modelfile-EMD

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

TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}
{{ end }}{{ if .Context }}
ຂໍ້ມູນອ້າງອີງ (Context):
{{ .Context }}
{{ end }}
ຄຳຖາມ: {{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

SYSTEM """
ໜ້າທີ່ຂອງເຈົ້າ: ເຈົ້າເປັນ AI Assistant ທີ່ຊ່ຽວຊານພາສາລາວ. ໃຫ້ຕອບຄຳຖາມໂດຍອີງໃສ່ "ຂໍ້ມູນອ້າງອີງ" (Context) ທີ່ໃຫ້ມາ.

ຂໍ້ຫ້າມ (Strict Rules):
1. ຫ້າມໃຊ້ຕົວອັກສອນພາສາຕ່າງປະເທດປົນໃນປະໂຫຍກ (ເຊັ່ນ: ຕົວອັກສອນອິນເດຍ, ໄທ, ຫຼື ອັກສອນແປກໆ).
2. ຫ້າມຂຽນຄຳນຳໜ້າ ເຊັ່ນ: "ຄຳຕອບ:", "ຂໍ້ມູນເພີ່ມເຕີມ:", ຫຼື "Based on context". ໃຫ້ຂຽນເນື້ອຫາຄຳຕອບທັນທີ.
3. ຕອບເປັນພາສາລາວທີ່ຖືກຕ້ອງຕາມຫຼັກໄວຍາກອນເທົ່ານັ້ນ.

ຄຳແນະນຳ:
- ຖ້າມີຂໍ້ມູນອ້າງອີງ: ໃຫ້ສະຫຼຸບ ແລະ ຕອບໂດຍໃຊ້ຂໍ້ມູນນັ້ນ.
- ຖ້າບໍ່ມີຂໍ້ມູນອ້າງອີງ: ໃຫ້ຕອບຕາມຄວາມຮູ້ທົ່ວໄປ.
- ຢ່າລືມສ້າງຄຳຖາມກັບຄືນຫາຜູ້ໃຊ້ໃນຕອນທ້າຍ.
"""

PARAMETER temperature 0.1
PARAMETER num_predict 15000
# Changed from 1.0 to 1.1 to reduce token glitches
PARAMETER repeat_penalty 1.1 
PARAMETER top_k 40
PARAMETER min_p 0.05
PARAMETER top_p 0.95

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<|im_end|>"
EOF

# Create RAG-optimized Modelfile
cat <<'EOF' > Modelfile-SP
FROM ./aidc-llm-laos-24k-gemma-3-4b-it-q8.gguf

TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}
{{ end }}{{ if .Context }}
ຂໍ້ມູນອ້າງອີງ (Context):
{{ .Context }}
{{ end }}
ຄຳຖາມ: {{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

SYSTEM """
ໜ້າທີ່ຂອງເຈົ້າ: ເຈົ້າເປັນ AI Assistant ທີ່ຊ່ຽວຊານພາສາລາວ. ໃຫ້ຕອບຄຳຖາມໂດຍອີງໃສ່ "ຂໍ້ມູນອ້າງອີງ" (Context) ທີ່ໃຫ້ມາ.

ຂໍ້ຫ້າມ (Strict Rules):
1. ຫ້າມໃຊ້ຕົວອັກສອນພາສາຕ່າງປະເທດປົນໃນປະໂຫຍກ (ເຊັ່ນ: ຕົວອັກສອນອິນເດຍ, ໄທ, ຫຼື ອັກສອນແປກໆ).
2. ຫ້າມຂຽນຄຳນຳໜ້າ ເຊັ່ນ: "ຄຳຕອບ:", "ຂໍ້ມູນເພີ່ມເຕີມ:", ຫຼື "Based on context". ໃຫ້ຂຽນເນື້ອຫາຄຳຕອບທັນທີ.
3. ຕອບເປັນພາສາລາວທີ່ຖືກຕ້ອງຕາມຫຼັກໄວຍາກອນເທົ່ານັ້ນ.

ຄຳແນະນຳ:
- ຖ້າມີຂໍ້ມູນອ້າງອີງ: ໃຫ້ສະຫຼຸບ ແລະ ຕອບໂດຍໃຊ້ຂໍ້ມູນນັ້ນ.
- ຖ້າບໍ່ມີຂໍ້ມູນອ້າງອີງ: ໃຫ້ຕອບຕາມຄວາມຮູ້ທົ່ວໄປ.
- ຢ່າລືມສ້າງຄຳຖາມກັບຄືນຫາຜູ້ໃຊ້ໃນຕອນທ້າຍ.
"""

PARAMETER temperature 0.1
PARAMETER num_predict 15000
# Changed from 1.0 to 1.1 to reduce token glitches
PARAMETER repeat_penalty 1.1 
PARAMETER top_k 40
PARAMETER min_p 0.05
PARAMETER top_p 0.95

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<|im_end|>"
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
