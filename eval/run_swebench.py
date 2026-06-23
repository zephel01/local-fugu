"""
SWE-Bench Verified evaluation runner for local-fugu.

Flow:
  1. Load SWE-Bench Verified from HuggingFace (princeton-nlp/SWE-bench_Verified)
  2. For each instance: format prompt → run FuguPipeline → extract patch
  3. Save predictions JSONL (swebench-compatible format)
  4. (Optional) Run official swebench harness to score

Usage:
    # Run all 500 instances
    python eval/run_swebench.py --output eval/predictions.jsonl

    # Run first 10 instances (smoke test)
    python eval/run_swebench.py --limit 10 --output eval/predictions_smoke.jsonl

    # Resume interrupted run (skips already-saved instance_ids)
    python eval/run_swebench.py --resume --output eval/predictions.jsonl

    # Score after generating predictions
    python eval/run_swebench.py --score-only --output eval/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add repo root to path so we can import pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import FuguPipeline, WorkflowExecutor, WorkflowResult
from agents.base import load_config
from agents import build_pool
from eval.swe_prompt import format_query, format_conductor_hint
from eval.extract_patch import extract, is_valid_patch


# ── Fixed SWE-bench workflow (bypasses Conductor) ─────────────────────────────

def run_fixed_workflow(instance: dict, config: dict) -> WorkflowResult:
    """
    Run a hardcoded planner→coder→reviewer workflow without calling the Conductor.
    This is more reliable for SWE-bench because:
    - The workflow is always the same (no need for dynamic planning)
    - Avoids Conductor failures on long problem statements
    - Saves ~1-5min per instance on conductor retries
    """
    import asyncio

    query = format_query(instance)
    repo = instance.get("repo", "unknown")

    fixed_workflow = [
        {
            "id": 1,
            "agent": "planner",
            "subtask": (
                f"You are working on the {repo} repository.\n\n"
                "Your job: carefully read the issue below and identify:\n"
                "1. The exact file path(s) in the repository that need to be changed\n"
                "2. The root cause of the bug\n"
                "3. The minimal change needed\n\n"
                "Be specific about file paths — they will be used to generate a git patch.\n\n"
                f"{query}"
            ),
            "access_list": [],
        },
        {
            "id": 2,
            "agent": "coder",
            "subtask": (
                "Using the file paths and analysis from the planner, implement the fix "
                "as a valid unified diff (git patch).\n\n"
                "STRICT requirements:\n"
                "- Use REAL file paths from the planner (never placeholders like 'path/to/file')\n"
                "- Format: `--- a/actual/path.py` and `+++ b/actual/path.py`\n"
                "- Include correct @@ line numbers and context lines\n"
                "- Output ONLY the patch in a ```diff ... ``` block\n\n"
                f"Original issue for reference:\n{query}"
            ),
            "access_list": [1],
        },
        {
            "id": 3,
            "agent": "reviewer",
            "subtask": (
                "Review the patch from the coder. Check:\n"
                "1. File paths in `--- a/...` and `+++ b/...` are real paths (not placeholders)\n"
                "2. The patch addresses the root cause described in the issue\n"
                "3. The diff format is valid (correct @@ headers, context lines)\n\n"
                "If the patch looks valid, output it unchanged in a ```diff ... ``` block.\n"
                "If it has placeholder paths, correct them using the planner's file paths."
            ),
            "access_list": [2],
        },
    ]

    executor = WorkflowExecutor(config)
    step_results = asyncio.run(executor.execute(fixed_workflow))
    return WorkflowResult(goal=f"Fix: {instance.get('instance_id', '')}", step_results=step_results)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(split: str = "test") -> list[dict]:
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        print("ERROR: Install datasets: pip install datasets")
        sys.exit(1)
    ds = hf_load("princeton-nlp/SWE-bench_Verified", split=split)
    return list(ds)


def load_existing_ids(path: str) -> set[str]:
    """Return set of instance_ids already in the predictions file."""
    ids: set[str] = set()
    p = Path(path)
    if not p.exists():
        return ids
    with open(p) as f:
        for line in f:
            try:
                ids.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def run_instance(
    instance: dict,
    log_dir: str = "results/swebench",
    pipeline: FuguPipeline | None = None,
    config: dict | None = None,
    bypass_conductor: bool = True,
) -> dict:
    """Run the pipeline on one SWE-Bench instance. Returns a prediction dict."""
    iid = instance["instance_id"]

    t0 = time.time()
    try:
        if bypass_conductor and config is not None:
            result = run_fixed_workflow(instance, config)
        else:
            assert pipeline is not None
            query = format_query(instance)
            hint = format_conductor_hint(instance)
            result = pipeline.run(f"{hint}\n\n{query}")

        elapsed = time.time() - t0
        all_output = "\n\n".join(s.output for s in result.step_results)
        patch = extract(all_output)

        # Save per-instance log
        _save_instance_log(
            log_dir=log_dir,
            instance=instance,
            result=result,
            patch=patch,
            elapsed=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {iid}: {e}")
        patch = ""

    return {
        "instance_id": iid,
        "model_patch": patch,
        "model_name_or_path": "local-fugu",
        "patch_valid": is_valid_patch(patch),
    }


def _save_instance_log(
    log_dir: str,
    instance: dict,
    result: Any,
    patch: str,
    elapsed: float,
) -> None:
    """Save a structured log for one SWE-bench instance."""
    import json as _json
    from datetime import datetime, timezone

    iid = instance["instance_id"]
    out_dir = Path(log_dir) / iid
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "instance_id": iid,
        "repo": instance.get("repo", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed, 2),
        "patch_valid": is_valid_patch(patch),
        "patch": patch,
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
    (out_dir / "log.json").write_text(_json.dumps(log, ensure_ascii=False, indent=2))
    if patch:
        (out_dir / "patch.diff").write_text(patch)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_predictions(predictions_path: str) -> None:
    """
    Run the official swebench harness to evaluate predictions.
    Requires: pip install swebench  and  Docker running.
    """
    try:
        from swebench.harness.run_evaluation import main as swe_eval
    except ImportError:
        print("swebench not installed. Run: pip install swebench")
        return

    print("\n[Scoring] Running official swebench harness (Docker required)…")
    swe_eval(
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        instance_ids=[],
        predictions_path=predictions_path,
        max_workers=4,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id="local-fugu",
        timeout=1800,
        namespace="swebench",
        rewrite_reports=False,
        modal=False,
        report_dir="results/swebench_reports",
    )


def print_quick_stats(predictions_path: str) -> None:
    """Print patch extraction stats without running Docker."""
    preds = []
    with open(predictions_path) as f:
        for line in f:
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    total = len(preds)
    valid = sum(1 for p in preds if p.get("patch_valid", False))
    empty = sum(1 for p in preds if not p.get("model_patch", "").strip())

    print(f"\n── Prediction stats ({predictions_path})")
    print(f"  Total instances : {total}")
    print(f"  Valid patches   : {valid} ({valid/total*100:.1f}%)")
    print(f"  Empty patches   : {empty} ({empty/total*100:.1f}%)")
    print(f"\nTo score with Docker: python eval/run_swebench.py --score-only --output {predictions_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval/predictions.jsonl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Max instances to run")
    parser.add_argument("--resume", action="store_true", help="Skip already-saved instances")
    parser.add_argument("--score-only", action="store_true", help="Only score existing predictions")
    parser.add_argument("--stats-only", action="store_true", help="Print extraction stats, no Docker")
    parser.add_argument(
        "--use-conductor", action="store_true",
        help="Use Conductor for dynamic workflow (default: bypass with fixed planner→coder→reviewer)"
    )
    args = parser.parse_args()

    if args.stats_only:
        print_quick_stats(args.output)
        return

    if args.score_only:
        score_predictions(args.output)
        return

    # Load dataset
    print("[Loading] SWE-Bench Verified from HuggingFace…")
    instances = load_dataset("test")
    print(f"  {len(instances)} instances loaded")

    # Resume support
    done_ids: set[str] = set()
    if args.resume:
        done_ids = load_existing_ids(args.output)
        print(f"  Resuming: {len(done_ids)} already done")

    # Apply limit
    to_run = [i for i in instances if i["instance_id"] not in done_ids]
    if args.limit:
        to_run = to_run[: args.limit]
    print(f"  Running: {len(to_run)} instances\n")

    bypass_conductor = not args.use_conductor
    if bypass_conductor:
        print("[Mode] Fixed workflow (planner→coder→reviewer) — Conductor bypassed\n")
        config = load_config(args.config)
        pipeline = None
    else:
        print("[Mode] Dynamic workflow via Conductor\n")
        config = None
        pipeline = FuguPipeline(config_path=args.config)

    # Run
    log_dir = f"results/swebench/{Path(args.output).stem}"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    with open(args.output, "a") as out_f:
        for idx, instance in enumerate(to_run, 1):
            iid = instance["instance_id"]
            print(f"[{idx}/{len(to_run)}] {iid}")
            t0 = time.perf_counter()

            pred = run_instance(
                instance,
                log_dir=log_dir,
                pipeline=pipeline,
                config=config,
                bypass_conductor=bypass_conductor,
            )
            out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            out_f.flush()

            elapsed = time.perf_counter() - t0
            status = "✓ patch" if pred["patch_valid"] else "✗ no patch"
            print(f"  {status}  ({elapsed:.1f}s)\n")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nDone. {len(to_run)} instances in {total_elapsed/60:.1f}min")
    print_quick_stats(args.output)


if __name__ == "__main__":
    main()
