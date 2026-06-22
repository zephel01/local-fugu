# Merge Recipes

Mergekit configurations for producing optimized local models.

## Model Targets

| Model | Recipe | VRAM (Q4_K_M) | Role |
|-------|--------|--------------|------|
| `coder-32b-merged` | `coder_32b_dare_ties.yaml` | ~20GB | Primary coder agent |
| `coder-32b-slerp` | `coder_32b_slerp.yaml` | ~20GB | Alternative coder (tune `t`) |
| `reviewer-14b-merged` | `reviewer_14b.yaml` | ~10GB | Reviewer agent |

With RTX 5090 32GB: run coder-32b in full bfloat16 OR reviewer-14b + coder-32b both in Q4.

## Quick Start

```bash
# Default: DARE-TIES coder merge
bash merges/run_merge.sh

# Specific recipe + output path
bash merges/run_merge.sh merges/reviewer_14b.yaml ./models/reviewer-14b-merged

# Convert to GGUF and add to Ollama
python llama.cpp/convert_hf_to_gguf.py ./models/coder-32b-merged \
    --outfile ./models/coder-32b.gguf --outtype q4_k_m
ollama create local-coder-32b -f - <<EOF
FROM ./models/coder-32b.gguf
EOF

# Update config.yaml
# agents:
#   coder: "local-coder-32b"
```

## Merge Strategy Guide

### DARE-TIES (`coder_32b_dare_ties.yaml`) — Recommended
Best for combining models that share the same base (Qwen3) but different fine-tune targets.
- `density`: what fraction of the task vector to keep. Start at 0.7; lower if outputs are noisy.
- `weight`: relative contribution. 0.5/0.5 is balanced; shift toward 0.6/0.4 to favor one model.

### SLERP (`coder_32b_slerp.yaml`) — Simpler baseline
Smooth interpolation. Easier to tune — just adjust `t`:
- `t=0.4`: 40% Qwen2.5-Coder flavor, 60% Qwen3
- `t=0.6`: 60% Qwen2.5-Coder → stronger coding execution
- `t=0.7+`: often too specialized; general reasoning degrades

## Tuning Loop

1. Merge with initial config
2. Run smoke eval: `python eval/run_swebench.py --limit 50 --output eval/preds_smoke.jsonl`
3. Check stats: `python eval/run_swebench.py --stats-only --output eval/preds_smoke.jsonl`
4. Adjust `density` / `weight` / `t` → repeat
