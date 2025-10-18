#!/usr/bin/env bash
set -e

# Clone SupportAgent if not present
if [ ! -d "SupportAgent" ]; then
  git clone https://github.com/Dealtors/SupportAgent.git
fi

# Unpack provided archives
if [ -f "ml_orchestrator.zip" ]; then
  mkdir -p app/external/ml_orchestrator
  unzip -o ml_orchestrator.zip -d app/external/ml_orchestrator >/dev/null
fi

if [ -f "reglament_ops_bundle.zip" ]; then
  mkdir -p app/data/kb
  unzip -o reglament_ops_bundle.zip -d app/data/kb >/dev/null
fi

# Pull Ollama models if CLI is available (in container ollama is separate)
if command -v ollama >/dev/null 2>&1; then
  ollama pull ${OLLAMA_CHAT_MODEL:-llama3.1} || true
  ollama pull ${OLLAMA_EMBED_MODEL:-nomic-embed-text} || true
fi

echo "Setup complete."
