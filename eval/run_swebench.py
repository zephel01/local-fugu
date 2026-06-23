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

from pipeline import FuguPipeline
from eval.swe_prompt import format_query, format_conductor_hint
from eval.extract_patch import extract, is_valid_patch


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


def run_instance(pipeline: FuguPipeline, instance: dict, log_dir: str = "results/swebench") -> dict:
    """Run the pipeline on one SWE-Bench instance. Returns a prediction dict."""
    iid = instance["instance_id"]
    query = format_query(instance)
    hint = format_conductor_hint(instance)
    full_query = f"{hint}\n\n{query}"

    t0 = time.time()
    try:
        result = pipeline.run(full_query)
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

    # Init pipeline
    pipeline = FuguPipeline(config_path=args.config)

    # Run
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    with open(args.output, "a") as out_f:
        for idx, instance in enumerate(to_run, 1):
            iid = instance["instance_id"]
            print(f"[{idx}/{len(to_run)}] {iid}")
            t0 = time.perf_counter()

            pred = run_instance(pipeline, instance, log_dir=f"results/swebench/{Path(args.output).stem}")
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
