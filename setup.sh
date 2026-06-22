#!/usr/bin/env bash
set -euo pipefail

echo "=== local-fugu setup ==="

# 1. Python deps
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt

# 2. Check Ollama
echo "[2/3] Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Ollama found: $(ollama --version)"
    echo "  Pulling default models (edit config.yaml to change)..."
    ollama pull qwen3:8b   || echo "  Warning: qwen3:8b pull failed — pull manually"
    ollama pull qwen3:32b  || echo "  Warning: qwen3:32b pull failed — pull manually"
    ollama pull qwen3:14b  || echo "  Warning: qwen3:14b pull failed — pull manually"
else
    echo "  Ollama not found. Install from https://ollama.com or switch backend to vllm in config.yaml"
fi

# 3. Smoke test
echo "[3/3] Running smoke test..."
python - <<'EOF'
from agents.base import load_config
cfg = load_config()
print(f"  Backend : {cfg['backend']}")
print(f"  Conductor model: {cfg['models']['conductor']}")
print(f"  Coder model    : {cfg['models']['coder']}")
print(f"  Reviewer model : {cfg['models']['reviewer']}")
print("  Config loaded OK")
EOF

echo ""
echo "=== Setup complete ==="
echo "Run: python pipeline.py \"Write a Python function that reverses a string\""
