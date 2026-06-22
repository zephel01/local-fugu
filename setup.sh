#!/usr/bin/env bash
# local-fugu setup
# Usage:
#   bash setup.sh                 # check deps + verify models in config.yaml
#   bash setup.sh --pull-missing  # also pull any config models not yet in Ollama
#   bash setup.sh --merged        # download HuggingFace base models for mergekit
#   bash setup.sh --skip-models   # skip all model checks (CI / re-setup)
set -euo pipefail

MODE="default"
for arg in "$@"; do
    case "$arg" in
        --pull-missing) MODE="pull-missing" ;;
        --merged)       MODE="merged" ;;
        --skip-models)  MODE="skip" ;;
    esac
done

echo "=== local-fugu setup (mode: $MODE) ==="

# ── 1. Python deps ─────────────────────────────────────────────────────────
echo ""
echo "[1/3] Installing Python dependencies..."

# Create venv if not already inside one
VENV_DIR=".venv"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "  Creating virtualenv at $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "  Activated: $VENV_DIR"
fi

pip install -q openai pyyaml
pip install -q datasets || echo "  Warning: datasets install failed (needed for SWE-Bench eval)"
echo "  Done"
echo ""
echo "  ★ To activate the venv in future sessions:"
echo "    source $VENV_DIR/bin/activate"

# ── 2. Models ──────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Model check..."

if [[ "$MODE" == "skip" ]]; then
    echo "  Skipping (--skip-models)"

elif [[ "$MODE" == "merged" ]]; then
    echo "  Downloading HuggingFace base models for mergekit..."
    pip install huggingface_hub --break-system-packages -q
    hf download Qwen/Qwen3-32B               --local-dir models/Qwen3-32B
    hf download Qwen/Qwen2.5-Coder-32B-Instruct \
        --local-dir models/Qwen2.5-Coder-32B-Instruct
    hf download Qwen/Qwen3-14B               --local-dir models/Qwen3-14B
    hf download Qwen/Qwen2.5-Coder-14B-Instruct \
        --local-dir models/Qwen2.5-Coder-14B-Instruct
    echo "  Next: bash merges/run_merge.sh merges/coder_32b_dare_ties.yaml"

elif command -v ollama &>/dev/null; then
    echo "  Ollama $(ollama --version)"
    echo ""

    # Read model names from config.yaml via Python
    MODELS=$(python3 - <<'PYEOF'
import yaml, sys
try:
    cfg = yaml.safe_load(open("config.yaml"))
    agents = cfg.get("agents", {})
    seen = set()
    for role, model in agents.items():
        if model not in seen:
            print(f"{role}:{model}")
            seen.add(model)
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
PYEOF
    )

    # Get list of already-downloaded models
    AVAILABLE=$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')

    echo "  Models in config.yaml:"
    MISSING=()
    while IFS= read -r line; do
        role="${line%%:*}"
        model="${line#*:}"
        if echo "$AVAILABLE" | grep -qF "$model"; then
            echo "    ✓ $role: $model"
        else
            echo "    ✗ $role: $model  ← NOT FOUND locally"
            MISSING+=("$model")
        fi
    done <<< "$MODELS"

    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo ""
        if [[ "$MODE" == "pull-missing" ]]; then
            echo "  Pulling missing models..."
            for m in "${MISSING[@]}"; do
                echo "  → ollama pull $m"
                ollama pull "$m" || echo "  Warning: pull failed for $m"
            done
        else
            echo "  Missing models above — run with --pull-missing to auto-pull"
        fi
    else
        echo ""
        echo "  All models present"
    fi

    echo ""
    echo "  Available coding/planning models on this machine:"
    echo "$AVAILABLE" | grep -iE "coder|qwen3|devstral|gemma4" | sort | sed 's/^/    /'
    echo ""
    echo "  Recommended additional download (not yet in config, but worth evaluating):"
    if echo "$AVAILABLE" | grep -qF "devstral-small-2"; then
        echo "    ✓ devstral-small-2 already present"
    else
        echo "    → ollama pull devstral-small-2:24b-instruct-2512-q4_K_M  (~14 GB)"
        echo "      Purpose-built coding agent (Mistral × All Hands AI)"
        echo "      Swap into reviewer slot in config.yaml to compare"
    fi

else
    echo "  Ollama not found — install from https://ollama.com"
    echo "  Or use vLLM: set backend: vllm in config.yaml"
fi

# ── 3. Smoke test ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Smoke test..."
python3 - <<'EOF'
import sys
sys.path.insert(0, ".")
from agents.base import load_config
from agents import build_pool
cfg = load_config()
pool = build_pool(cfg)
print(f"  Backend   : {cfg['backend']}")
print(f"  Conductor : {cfg['agents']['conductor']}")
for role, agent in pool.items():
    print(f"  {role:10s}: {agent.model}")
print("  Config OK")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Quick start:"
echo "  python pipeline.py \"Write a Python function that reverses a string\""
echo ""
echo "SWE-Bench smoke test (10 instances):"
echo "  python eval/run_swebench.py --limit 10 --output eval/preds_smoke.jsonl"
