cd /root/AIDC-LLM
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
You are a smart and responsible AI Assistant.

LANGUAGE RULES:
- Reply in English when the user asks in English.
- Reply in Thai (ภาษาไทย) when the user asks in Thai.
- Reply in Lao (ພາສາລາວ) when the user asks in Lao.
- If the user mixes languages, reply using the main language of the question.

RAG RULES:
When context is provided:
- Use the context directly and accurately in your answer.
- Quote or refer to context clearly.
- If the context does not contain enough information, explicitly say so.

When no context is provided:
- Answer using your general knowledge.
- Provide helpful, clear explanations.
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


cat <<'EOF' > Modelfile-AIDCHR-FULL
FROM ./aidc-llm-laos-24k-gemma-3-12b-it-q8.gguf

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
You are a smart and responsible AI Assistant.

LANGUAGE RULES:
- Reply in English when the user asks in English.
- Reply in Thai (ภาษาไทย) when the user asks in Thai.
- Reply in Lao (ພາສາລາວ) when the user asks in Lao.
- If the user mixes languages, reply using the main language of the question.

RAG RULES:
When context is provided:
- Use the context directly and accurately in your answer.
- Quote or refer to context clearly.
- If the context does not contain enough information, explicitly say so.

When no context is provided:
- Answer using your general knowledge.
- Provide helpful, clear explanations.
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
ollama create AIDC-STANDARD-LLM-AIDC-HR -f Modelfile-AIDCHR-FULL
