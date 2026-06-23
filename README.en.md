<h1 align="center">local-fugu</h1>

<p align="center">
  <strong>Run multi-agent coding on local LLMs.<br>Ships a SWE-bench harness that evaluates "without breaking files, without overflowing context, with verification."</strong>
</p>

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python"></a>
  <a href=""><img src="https://img.shields.io/badge/backend-ollama%20%7C%20vllm-orange" alt="backend"></a>
  <a href=""><img src="https://img.shields.io/badge/eval-SWE--bench%20Verified-purple" alt="swebench"></a>
  <a href=""><img src="https://img.shields.io/badge/license-MIT-yellow" alt="license"></a>
</p>

<p align="center">
  <a href="./README.md">日本語</a> · <strong>English</strong> · <a href="./ROADMAP.md">Roadmap</a>
</p>

---

## What it does — in 30 seconds

```
                    your task / a SWE-bench issue
                              │
                              ▼
        ┌────────────── local-fugu ──────────────┐
        │  Conductor  (dynamic workflow)          │
        │     ├─ planner   locate the bug         │
        │     ├─ coder     fix via SEARCH/REPLACE │
        │     └─ reviewer  verify                 │
        └────────────────┬────────────────────────┘
                         ▼
        validated patch (difflib + git apply --check + ast)
                         ▼
            SWE-bench scoring (Docker) / execution-guided repair
```

A research/experiment project that runs Fugu-style (Sakana AI) multi-agent
orchestration entirely on **your local ollama / vLLM**, plus a SWE-bench harness
that small local models can run **safely**.

---

## Highlights

- **Multi-agent pipeline** — a Conductor builds a dynamic `planner → coder → reviewer` workflow; agent isolation curbs cascade bias.
- **Patches that don't break files** — the model emits SEARCH/REPLACE; the unified diff is produced by `difflib` and only kept if it passes a strict `git apply --check` + `ast.parse`. **Zero file corruption.**
- **Context compression** — large sources are AST-sliced to just the relevant function, verbatim (measured 216KB → ~12KB), so the small local window never overflows.
- **Symbol-based localization** — symbols from the issue / failing tests are traced to their defining file via `git grep`.
- **Execution-guided repair (opt-in)** — run the failing tests on a candidate and feed the traceback back to the coder.

---

## Install

```bash
git clone https://github.com/zephel01/local-fugu.git
cd local-fugu
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash setup.sh
```

## Usage

```bash
# one-off coding
python pipeline.py "Write a thread-safe LRU cache in Python"

# SWE-bench: generate predictions, then score with Docker
python -m eval.run_swebench --limit 3
python eval/run_swebench.py --score-only --output eval/predictions.jsonl

# with execution-guided repair (slow; needs SWE-bench Docker)
python -m eval.run_swebench --limit 3 --exec-repair
```

> **Mind `num_ctx`**: ollama's default context window is small and silently
> truncates overflow. Run the coder at 16k+ (`PARAMETER num_ctx 16384`).

## Status (honest)

The harness is solid (no corruption, no timeouts, verified patches; 5 offline
golden suites pass). **Resolve rate is bound by the local ~30B model** and many
hard instances stay unresolved — a 50–100 instance run is needed to know the
real baseline (see ROADMAP). This is a research project, not a benchmark claim.

## License
MIT
