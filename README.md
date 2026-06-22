# local-fugu

Fugu-Ultra inspired local multi-agent coding pipeline.

The Conductor generates a **dynamic workflow** of `(subtask, agent, access_list)` steps.  
Steps with no shared dependencies run **in parallel**. Each agent sees only what's in its `access_list` — **agent isolation** prevents cascade bias (Fugu Technical Report §3.2).

## Architecture

```
User Query
    │
    ▼
[Conductor]  qwen3:8b
    │
    │  Outputs a workflow, e.g.:
    │
    │  Step 1 [planner]  access=[]         ─┐ parallel
    │  Step 2 [coder]    access=[]         ─┘
    │  Step 3 [reviewer] access=[1,2]      ← sees both outputs
    │  Step 4 [coder]    access=[1,2,3]    ← refines based on review
    ▼
[WorkflowExecutor]
    └─ asyncio parallel execution
    └─ access_list → context injection (agent isolation enforced)
```

**Supported topologies** (chosen by Conductor per task):
- Simple: `coder → reviewer`
- Sequential: `planner → coder → reviewer`
- Parallel: two coders → reviewer → final coder

## Quick Start

```bash
# 1. Install deps + pull models
bash setup.sh

# 2. Run
python pipeline.py "Write a Python class for a thread-safe LRU cache"

# 3. Save output
python pipeline.py --output result.py "Implement merge sort with tests"

# 4. JSON output (for scripting)
python pipeline.py --json "Write a REST API client for GitHub"
```

## Configuration

`config.yaml`:

```yaml
backend: ollama   # or "vllm"

agents:
  conductor: "qwen3:8b"    # workflow generator (not in pool)
  coder:     "qwen3:32b"   # put your merged model here
  reviewer:  "qwen3:14b"
  planner:   "qwen3:8b"

pipeline:
  max_parallel_steps: 4
```

Add agents to the pool by adding entries under `agents:` — the Conductor prompt automatically picks them up.

### Using vLLM

```yaml
backend: vllm
vllm:
  base_url: "http://localhost:8000/v1"
```

```bash
vllm serve Qwen/Qwen3-32B --port 8000
```

## Customizing Prompts

All agent instructions are in `prompts/`:

| File | Role |
|------|------|
| `conductor.md` | Workflow generation instructions + topology guidelines |
| `coder.md` | Implementation instructions |
| `reviewer.md` | Review criteria (PASS/FAIL JSON) |
| `planner.md` | Decomposition instructions |

## Key Design Decisions (from Fugu Technical Report)

**Agent isolation**: Each agent's context is built exclusively from its `access_list`. This prevents the first agent's approach from biasing all subsequent agents ("orchestration collapse").

**Dynamic topology**: The Conductor chooses the graph shape per query — simple tasks get `coder→reviewer`, complex tasks get parallel exploration + synthesis.

**Parallel execution**: Independent steps (same access_list depth) run concurrently via `asyncio` + thread pool, bounded by `max_parallel_steps`.

## Roadmap

- [ ] Phase 2: SWE-Bench evaluation harness
- [ ] Phase 2: mergekit configs for Coder/Reviewer merged models
- [ ] Phase 3: Multi-turn conversation with persistent shared memory
- [ ] Phase 3: Tool use (code execution, file editing) inside agent steps
