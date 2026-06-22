#!/usr/bin/env bash
# Run mergekit to produce merged models.
# Prerequisites: pip install mergekit  (and enough disk space — ~60GB per 32B model)
set -euo pipefail

RECIPE="${1:-merges/coder_32b_dare_ties.yaml}"
OUT_DIR="${2:-./models/$(basename "${RECIPE%.yaml}")}"

echo "=== mergekit merge ==="
echo "Recipe  : $RECIPE"
echo "Output  : $OUT_DIR"
echo ""

# Install mergekit if needed
if ! python3 -c "import mergekit" 2>/dev/null; then
    echo "[1/2] Installing mergekit…"
    pip install mergekit --break-system-packages -q
fi

# Run merge (CPU offload enabled for large models)
echo "[2/2] Merging…"
mergekit-merge "$RECIPE" \
    --out-path "$OUT_DIR" \
    --copy-tokenizer \
    --allow-crimes \
    --lazy-unpickle

echo ""
echo "=== Done ==="
echo "Merged model saved to: $OUT_DIR"
echo ""
echo "Next steps:"
echo "  # Convert to GGUF for Ollama:"
echo "  python llama.cpp/convert_hf_to_gguf.py $OUT_DIR --outfile models/coder-32b.gguf --outtype q4_k_m"
echo "  ollama create local-coder-32b -f models/coder-32b.Modelfile"
echo ""
echo "  # Or load directly with vLLM:"
echo "  vllm serve $OUT_DIR --port 8000"
echo ""
echo "  # Then update config.yaml:"
echo "  #   agents:"
echo "  #     coder: \"local-coder-32b\"   # Ollama model name"
