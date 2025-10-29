#!/bin/bash
cd ~/AIDC-LLM
mkdir MOIC-CH-2-1
# Download Gemma-SEA-LION model file
echo "Downloading Gemma-SEA-LION-v4-27B model..."
wget -O Gemma-SEA-LION-v4-27B-IT-Q8_0.gguf \
"https://huggingface.co/aisingapore/Gemma-SEA-LION-v4-27B-IT-GGUF/resolve/main/Gemma-SEA-LION-v4-27B-IT-Q8_0.gguf"

# Create RAG-optimized Modelfile for SEA-LION
cat <<'EOF' > Modelfile
FROM ./Gemma-SEA-LION-v4-27B-IT-Q8_0.gguf

# RAG-optimized template for Gemma-SEA-LION
TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ if .Context }}ຂໍ້ມູນອ້າງອີງ (Context):
{{ .Context }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

# System prompt optimized for RAG (Lao language)
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

# RAG-optimized parameters for 27B model
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
PARAMETER repeat_penalty 1.1

# Gemma stop tokens
PARAMETER stop <start_of_turn>
PARAMETER stop <end_of_turn>
PARAMETER stop <|im_end|>
EOF

# Create Ollama model
echo "Creating Ollama model..."
ollama create MOIC-CH-2-1 -f Modelfile

echo "Model created successfully!"
echo "Model name: sealion-laos-rag"
echo "Embedding model: bge-m3"
echo ""
echo "Test the model:"
echo "ollama run sealion-laos-rag"
echo ""
echo "Connect to Open WebUI:"
echo "- Ollama API URL: http://localhost:11434"
echo "- Model name: sealion-laos-rag"
echo "- Embedding model: bge-m3"