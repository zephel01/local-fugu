"""
local-fugu pipeline
====================
Fugu-Ultra style orchestration:
  Conductor generates a dynamic workflow of (subtask, agent, access_list) steps.
  Steps with no inter-dependencies execute in parallel.
  Each agent sees ONLY the outputs explicitly listed in its access_list (agent isolation).

Usage:
    python pipeline.py "Write a Python class for a thread-safe LRU cache"
    python pipeline.py --config config.yaml --output result.py "..."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.base import load_config
from agents import ConductorAgent, build_pool


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    id: int
    agent: str
    subtask: str
    output: str
    duration_s: float


@dataclass
class WorkflowResult:
    goal: str
    step_results: list[StepResult] = field(default_factory=list)


# ── Workflow executor ─────────────────────────────────────────────────────────

class WorkflowExecutor:
    """
    Executes a Conductor-generated workflow with:
    - Parallel execution of independent steps (asyncio)
    - Access-list driven context injection
    - Agent isolation (each agent sees only its access_list outputs)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.pool = build_pool(config)
        self.semaphore = asyncio.Semaphore(
            config["pipeline"].get("max_parallel_steps", 4)
        )

    async def execute(self, workflow: list[dict[str, Any]]) -> list[StepResult]:
        """Execute all steps, respecting access_list dependencies."""
        completed: dict[int, StepResult] = {}
        pending = {step["id"]: step for step in workflow}

        while pending:
            # Find all steps whose access_list is fully satisfied
            ready = [
                step for step in pending.values()
                if all(dep in completed for dep in step.get("access_list", []))
            ]

            if not ready:
                raise RuntimeError(
                    f"Workflow deadlock — pending steps {list(pending.keys())} "
                    f"but none are ready. Check access_list for cycles."
                )

            # Run all ready steps in parallel
            tasks = [self._run_step(step, completed) for step in ready]
            results = await asyncio.gather(*tasks)

            for result in results:
                completed[result.id] = result
                del pending[result.id]

        return [completed[step["id"]] for step in workflow]

    async def _run_step(
        self,
        step: dict[str, Any],
        completed: dict[int, StepResult],
    ) -> StepResult:
        async with self.semaphore:
            step_id = step["id"]
            agent_role = step["agent"]
            subtask = step["subtask"]
            access_list = step.get("access_list", [])

            if agent_role not in self.pool:
                raise ValueError(
                    f"Step {step_id} references unknown agent '{agent_role}'. "
                    f"Available: {list(self.pool.keys())}"
                )

            # Build context from access_list — agent isolation enforced here
            context = [
                {
                    "id": completed[dep_id].id,
                    "agent": completed[dep_id].agent,
                    "output": completed[dep_id].output,
                }
                for dep_id in access_list
                if dep_id in completed
            ]

            print(f"  [Step {step_id}] {agent_role} ← {subtask[:60]}…")
            t0 = time.perf_counter()

            # Run with retry — model-load failures (500) can happen when a previous
            # model is still being unloaded from RAM. Wait and retry up to 3 times.
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    output = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.pool[agent_role].run,
                        subtask,
                        context or None,
                    )
                    break
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    wait = 20 * attempt  # 20s, 40s — give Ollama time to unload
                    print(f"  [Step {step_id}] attempt {attempt} failed ({e.__class__.__name__}), "
                          f"retrying in {wait}s…")
                    await asyncio.sleep(wait)

            duration = time.perf_counter() - t0
            print(f"  [Step {step_id}] done in {duration:.1f}s")

            return StepResult(
                id=step_id,
                agent=agent_role,
                subtask=subtask,
                output=output,
                duration_s=duration,
            )


# ── Main pipeline ─────────────────────────────────────────────────────────────

class FuguPipeline:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = load_config(config_path)
        self.conductor = ConductorAgent(self.config)

    def run(self, user_query: str) -> WorkflowResult:
        # Step 1: Conductor generates workflow
        print("[Conductor] Generating workflow…")
        t0 = time.perf_counter()
        plan = self.conductor.plan(user_query)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

        goal: str = plan["goal"]
        workflow: list[dict] = plan["workflow"]
        notes: str = plan.get("notes", "")

        print(f"\n  Goal    : {goal}")
        if notes:
            print(f"  Notes   : {notes}")
        print(f"  Steps   : {len(workflow)}")
        self._print_workflow(workflow)

        # Step 2: Execute workflow
        print("\n[Executing workflow]")
        executor = WorkflowExecutor(self.config)
        step_results = asyncio.run(executor.execute(workflow))

        return WorkflowResult(goal=goal, step_results=step_results)

    @staticmethod
    def _print_workflow(workflow: list[dict]) -> None:
        for step in workflow:
        	deps = step.get("access_list", [])
        	dep_str = f" ← [{', '.join(str(d) for d in deps)}]" if deps else ""
        	print(f"  Step {step['id']:2d} [{step['agent']:10s}]{dep_str}: {step['subtask'][:55]}…")


# ── Result logger ─────────────────────────────────────────────────────────────

def _slug(text: str, max_len: int = 40) -> str:
    """Convert query text to a safe filename slug."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len]


def save_result(
    query: str,
    result: WorkflowResult,
    elapsed: float,
    config: dict[str, Any],
    log_dir: str = "results",
) -> Path:
    """
    Save a full run log to results/<YYYYMMDD_HHMMSS>_<slug>/

    Directory layout:
        results/
          20260623_093000_implement_a_thread/
            run.json        ← full structured log (machine-readable)
            summary.md      ← human-readable summary
            step_01_coder.txt
            step_02_reviewer.txt
            ...
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"{ts}_{_slug(query)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── run.json ──
    run_data = {
        "timestamp": ts,
        "query": query,
        "goal": result.goal,
        "elapsed_s": round(elapsed, 2),
        "config": {
            "backend": config.get("backend"),
            "agents": config.get("agents", {}),
        },
        "steps": [
            {
                "id": s.id,
                "agent": s.agent,
                "subtask": s.subtask,
                "duration_s": round(s.duration_s, 2),
                "output": s.output,
            }
            for s in result.step_results
        ],
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_data, ensure_ascii=False, indent=2)
    )

    # ── per-step txt files ──
    for s in result.step_results:
        fname = f"step_{s.id:02d}_{s.agent}.txt"
        (run_dir / fname).write_text(
            f"# Step {s.id} [{s.agent}] ({s.duration_s:.1f}s)\n"
            f"# Subtask: {s.subtask}\n\n"
            f"{s.output}\n"
        )

    # ── summary.md ──
    verdicts = []
    for s in result.step_results:
        if s.agent == "reviewer":
            m = re.search(r'"verdict"\s*:\s*"(\w+)"', s.output)
            verdicts.append(m.group(1) if m else "?")

    lines = [
        f"# Run: {ts}",
        f"",
        f"**Query**: {query}",
        f"**Goal**: {result.goal}",
        f"**Total time**: {elapsed:.1f}s",
        f"**Steps**: {len(result.step_results)}",
        f"**Reviewer verdicts**: {', '.join(verdicts) if verdicts else '—'}",
        f"",
        f"## Steps",
        f"",
    ]
    for s in result.step_results:
        lines.append(f"### Step {s.id} [{s.agent}] ({s.duration_s:.1f}s)")
        lines.append(f"*{s.subtask[:100]}*")
        lines.append(f"")
        # First 20 lines of output
        preview = "\n".join(s.output.splitlines()[:20])
        lines.append(f"```\n{preview}\n```")
        lines.append(f"")

    (run_dir / "summary.md").write_text("\n".join(lines))

    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="local-fugu: Fugu-Ultra style local coding pipeline")
    parser.add_argument("query", nargs="+", help="Programming task description")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", help="Save all step outputs to this file")
    parser.add_argument("--log-dir", default="results", help="Directory for run logs (default: results/)")
    parser.add_argument("--no-log", action="store_true", help="Disable automatic result logging")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    args = parser.parse_args()

    query = " ".join(args.query)
    pipeline = FuguPipeline(config_path=args.config)

    t_total = time.perf_counter()
    result = pipeline.run(query)
    elapsed = time.perf_counter() - t_total

    print(f"\n{'=' * 60}")
    print(f"COMPLETE  ({elapsed:.1f}s total)")
    print(f"{'=' * 60}")

    if args.json:
        out = {
            "goal": result.goal,
            "steps": [
                {
                    "id": s.id,
                    "agent": s.agent,
                    "subtask": s.subtask,
                    "output": s.output,
                    "duration_s": round(s.duration_s, 2),
                }
                for s in result.step_results
            ],
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for s in result.step_results:
            print(f"\n── Step {s.id} [{s.agent}] ({s.duration_s:.1f}s)")
            print(s.output)

    if args.output:
        with open(args.output, "w") as f:
            for s in result.step_results:
                f.write(f"# Step {s.id} [{s.agent}]: {s.subtask}\n\n")
                f.write(s.output)
                f.write("\n\n")
        print(f"\nSaved to {args.output}")

    if not args.no_log:
        run_dir = save_result(query, result, elapsed, pipeline.config, args.log_dir)
        print(f"Log saved → {run_dir}/")


if __name__ == "__main__":
    main()
