#!/usr/bin/env bash
set -euo pipefail
echo "[setup] Pulling models..."
ollama pull qwen2:7b-q4_K_M || true
ollama pull nomic-embed-text || true

echo "[setup] Creating venv and installing deps..."
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt

echo "[setup] Ingest sample KB..."
python app/ingest_kb.py

echo "[setup] Done. Run: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload"
