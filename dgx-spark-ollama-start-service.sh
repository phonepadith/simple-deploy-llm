#!/bin/bash
sudo mkdir -p /etc/systemd/system/ollama.service.d

echo '[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"' | sudo tee /etc/systemd/system/ollama.service.d/override.conf

sudo systemctl daemon-reload
sudo systemctl restart ollama
echo "- Ollama Start Service for Public"
