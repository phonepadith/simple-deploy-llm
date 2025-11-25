# Create RAG-optimized Modelfile
cat <<'EOF' > Modelfile-AIDCHR
FROM ./aidc-llm-laos-24k-gemma-3-4b-it-q8.gguf

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
- ຕອບເປັນພາສາອັງກິດ (English) ເມື່ອຜູ້ໃຊ້ຖາມເປັນອັງກິດ (English)
- ຕອບເປັນພາສາໄທ (Thai-ไทย) ເມື່ອຜູ້ໃຊ້ຖາມເປັນໄທ (Thai-ไทย)
- ຕອບເປັນພາສາລາວ (Lao-ລາວ) ເມື່ອຜູ້ໃຊ້ຖາມເປັນລາວ (Lao-ລາວ)

ເມື່ອມີຂໍ້ມູນອ້າງອີງ (Context) ໃຫ້:
- ໃຊ້ຂໍ້ມູນອ້າງອີງເພື່ອຕອບຄຳຖາມຢ່າງແມ່ນຍຳ
- ອ້າງອີງຂໍ້ມູນຈາກ Context ໂດຍກົງ
- ຖ້າຂໍ້ມູນໃນ Context ບໍ່ພຽງພໍ ໃຫ້ບອກຢ່າງຊັດເຈນ

ເມື່ອບໍ່ມີຂໍ້ມູນອ້າງອີງ:
- ຕອບຕາມຄວາມຮູ້ທົ່ວໄປຂອງເຈົ້າ
- ໃຫ້ຄຳແນະນຳທີ່ເປັນປະໂຫຍດ
"""

# RAG-optimized parameters for 27B model
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


# Create Ollama model
echo "Creating Ollama model..."
ollama create AIDC-FAST-LLM-AIDC-HR -f Modelfile-AIDCHR
