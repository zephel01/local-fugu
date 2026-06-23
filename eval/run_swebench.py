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

from pipeline import FuguPipeline, WorkflowResult, StepResult
from agents.base import load_config
from agents import build_pool
from eval.swe_prompt import format_query, format_conductor_hint
from eval.extract_patch import extract, is_valid_patch


# ── Fixed SWE-bench workflow (bypasses Conductor) ─────────────────────────────

def run_fixed_workflow(
    instance: dict,
    config: dict,
    use_repo_context: bool = True,
) -> WorkflowResult:
    """
    Run planner→coder→reviewer without the Conductor.

    With use_repo_context=True (default):
      - Clones the repo at base_commit (cached in /tmp/local_fugu_*)
      - Passes file tree to planner; actual file content to coder
      - Dramatically improves diff correctness (real line numbers & context)

    With use_repo_context=False:
      - Falls back to asking the model to recall file paths from training data
    """
    import asyncio

    query = format_query(instance)
    repo = instance.get("repo", "unknown")
    base_commit = instance.get("base_commit", "")
    pkg_name = repo.split("/")[-1].replace("-", "_")
    iid = instance.get("instance_id", "unknown")

    pool = build_pool(config)

    async def _run() -> list[StepResult]:
        steps: list[StepResult] = []

        # ── Repo context (optional) ───────────────────────────────────────────
        repo_dir = None
        file_tree = ""
        if use_repo_context and base_commit:
            try:
                from eval.repo_context import (
                    get_repo_at_commit, get_file_tree,
                    extract_file_paths, read_files,
                )
                repo_dir = get_repo_at_commit(repo, base_commit)
                file_tree = get_file_tree(repo_dir, pkg_name)
            except Exception as e:
                print(f"  [repo] Clone/checkout failed: {e} — falling back to no-context", flush=True)
                repo_dir = None
                file_tree = ""

        # ── Step 1: Planner ───────────────────────────────────────────────────
        if file_tree:
            planner_subtask = (
                f"You are working on the {repo} repository.\n\n"
                f"## Available Python files\n{file_tree}\n\n"
                "## Your task\n"
                "Read the issue below and identify:\n"
                "1. The EXACT file path(s) from the list above that need to be changed\n"
                "2. The root cause of the bug\n"
                "3. The minimal change needed\n\n"
                "List the file path(s) explicitly — they will be used to read the actual code.\n\n"
                f"{query}"
            )
        else:
            planner_subtask = (
                f"You are working on the {repo} repository.\n\n"
                "Read the issue below and identify:\n"
                "1. The exact file path(s) that need to be changed\n"
                "2. The root cause of the bug\n"
                "3. The minimal change needed\n\n"
                f"{query}"
            )

        print(f"  [Step 1] planner ← {planner_subtask[:60]}…")
        t0 = time.perf_counter()
        planner_output = await asyncio.get_event_loop().run_in_executor(
            None, pool["planner"].run, planner_subtask, None
        )
        dur1 = time.perf_counter() - t0
        print(f"  [Step 1] done in {dur1:.1f}s")
        steps.append(StepResult(id=1, agent="planner", subtask=planner_subtask, output=planner_output, duration_s=dur1))

        # ── Read actual file content ──────────────────────────────────────────
        file_content_section = ""
        if repo_dir is not None:
            try:
                from eval.repo_context import extract_file_paths, read_files
                file_paths = extract_file_paths(planner_output, repo_dir, pkg_name)
                if file_paths:
                    print(f"  [repo] Reading: {file_paths}")
                    file_content_section = (
                        "\n\n## Actual file content from repository\n"
                        + read_files(repo_dir, file_paths)
                    )
                else:
                    print("  [repo] No matching files found in planner output")
            except Exception as e:
                print(f"  [repo] File read failed: {e}")

        # ── Step 2: Coder ─────────────────────────────────────────────────────
        coder_subtask = (
            "Using the planner's analysis, implement the fix as a valid unified diff.\n\n"
            "STRICT requirements:\n"
            "- Use the EXACT file paths identified by the planner\n"
            "- `--- a/real/path.py` and `+++ b/real/path.py` (no placeholders)\n"
            "- Context lines MUST match the actual file content exactly\n"
            "- @@ line numbers must be correct\n"
            "- Output ONLY a ```diff ... ``` block\n"
            + file_content_section
        )

        context1 = [{"id": 1, "agent": "planner", "output": planner_output}]
        print(f"  [Step 2] coder ← {coder_subtask[:60]}…")
        t0 = time.perf_counter()
        coder_output = await asyncio.get_event_loop().run_in_executor(
            None, pool["coder"].run, coder_subtask, context1
        )
        dur2 = time.perf_counter() - t0
        print(f"  [Step 2] done in {dur2:.1f}s")
        steps.append(StepResult(id=2, agent="coder", subtask=coder_subtask, output=coder_output, duration_s=dur2))

        # ── Step 3: Reviewer ──────────────────────────────────────────────────
        reviewer_subtask = (
            "Review the patch from the coder:\n"
            "1. File paths in `--- a/...` headers must be real (not placeholders)\n"
            "2. The fix must address the root cause\n"
            "3. Diff format must be valid (correct @@ headers and context lines)\n\n"
            "Output the verified patch in ```diff ... ``` block. "
            "If paths look wrong, correct them using the planner's identified paths."
        )
        context2 = [{"id": 2, "agent": "coder", "output": coder_output}]
        print(f"  [Step 3] reviewer ← {reviewer_subtask[:60]}…")
        t0 = time.perf_counter()
        reviewer_output = await asyncio.get_event_loop().run_in_executor(
            None, pool["reviewer"].run, reviewer_subtask, context2
        )
        dur3 = time.perf_counter() - t0
        print(f"  [Step 3] done in {dur3:.1f}s")
        steps.append(StepResult(id=3, agent="reviewer", subtask=reviewer_subtask, output=reviewer_output, duration_s=dur3))

        return steps

    step_results = asyncio.run(_run())
    return WorkflowResult(goal=f"Fix: {iid}", step_results=step_results)


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
    use_repo_context: bool = True,
) -> dict:
    """Run the pipeline on one SWE-Bench instance. Returns a prediction dict."""
    iid = instance["instance_id"]

    t0 = time.time()
    try:
        if bypass_conductor and config is not None:
            result = run_fixed_workflow(instance, config, use_repo_context=use_repo_context)
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
    parser.add_argument(
        "--no-clone", action="store_true",
        help="Skip repo cloning — model infers file paths from training data (faster but less accurate)"
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
    use_repo_context = not args.no_clone
    if bypass_conductor:
        mode = "Fixed workflow (planner→coder→reviewer)"
        mode += " + repo clone" if use_repo_context else " [no clone]"
        print(f"[Mode] {mode}\n")
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
                use_repo_context=use_repo_context,
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
