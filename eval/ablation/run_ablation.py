"""
local-fugu  P1 ablation runner  (2026-06-25)
============================================
Measures the *net value of agent decomposition* on SWE-bench Verified by running
the SAME instances through parametrized arms, holding the harness constant.

ARMS (which reasoning agents run):
  A  coder            symbol+test localize -> coder SEARCH/REPLACE + retry
  B  planner+coder    + planner reasoning (and, in `real` mode, planner localize hints)
  C  full             + WIRED reviewer (its SEARCH/REPLACE is re-validated and
                      ADOPTED if it produces a safe non-empty patch; else the
                      coder patch is kept).  NB: this differs from the repo's
                      run_fixed_workflow, where the reviewer never touches the
                      submitted patch.

MODES:
  pure   localize = symbol+test ONLY for every arm; temperature = 0 everywhere.
         Isolates the decomposition variable.
  real   planner hints feed localize (B/C); temperature = config default.
         A_real == A_pure except temperature (A has no planner).

Design notes live in PLAN.md.  This module imports the existing harness pieces
and NEVER edits the repository in place.

Usage:
  python run_ablation.py --arm A --mode pure --limit 20 --output preds_A_pure.jsonl
  python run_ablation.py --score-only --output preds_A_pure.jsonl
  python run_ablation.py --stats-only --output preds_A_pure.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

# ── make the repo root importable (this file lives in _OUTPUTS/local-fugu-ablation/) ──
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline import WorkflowResult, StepResult          # noqa: E402
from agents.base import load_config                       # noqa: E402
from agents import build_pool                             # noqa: E402
from eval.swe_prompt import (                             # noqa: E402
    format_query, coder_instructions, coder_retry_instructions, verifier_instructions,
)
from eval.extract_patch import is_valid_patch             # noqa: E402
from eval.run_swebench import (                           # noqa: E402
    load_dataset, load_existing_ids, _save_instance_log,
    score_predictions, print_quick_stats,
)

ARMS = ("A", "B", "C")
MODES = ("pure", "real")


def _apply_mode_temperature(config: dict, mode: str) -> dict:
    """Return a deep-copied config with temperature pinned to 0 in `pure` mode."""
    cfg = copy.deepcopy(config)
    if mode == "pure":
        cfg.setdefault("pipeline", {})["temperature"] = {
            "default": 0.0, "planner": 0.0, "coder": 0.0,
            "reviewer": 0.0, "conductor": 0.0,
        }
    return cfg


# ── parametrized workflow ─────────────────────────────────────────────────────

def run_arm_workflow(
    instance: dict,
    config: dict,
    *,
    arm: str,
    mode: str,
    use_repo_context: bool = True,
) -> WorkflowResult:
    """planner?/coder/reviewer? with the variable isolated to the agent set."""
    assert arm in ARMS and mode in MODES

    use_planner = arm in ("B", "C")
    use_reviewer = arm == "C"
    # planner output may inform localization ONLY in real mode (pure keeps localize
    # identical across arms so the only variable is "who reasons").
    planner_feeds_localize = use_planner and mode == "real"

    cfg = _apply_mode_temperature(config, mode)
    pool = build_pool(cfg)

    query = format_query(instance)
    repo = instance.get("repo", "unknown")
    base_commit = instance.get("base_commit", "")
    pkg_name = repo.split("/")[-1].replace("-", "_")
    iid = instance.get("instance_id", "unknown")

    async def _run() -> list[StepResult]:
        steps: list[StepResult] = []
        repo_dir = None
        file_tree = ""

        if use_repo_context and base_commit:
            try:
                from eval.repo_context import get_repo_at_commit, get_file_tree
                repo_dir = get_repo_at_commit(repo, base_commit)
                file_tree = get_file_tree(repo_dir, pkg_name)
            except Exception as e:
                print(f"  [repo] clone/checkout failed: {e} — no-context fallback", flush=True)
                repo_dir, file_tree = None, ""

        # ── Step 1: planner (arms B/C only) ──────────────────────────────────
        planner_output = ""
        if use_planner:
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
            steps.append(StepResult(id=1, agent="planner", subtask=planner_subtask,
                                    output=planner_output, duration_s=dur1))

        # ── Localization (harness; constant except planner hints in real mode) ─
        file_content_section = ""
        file_paths: list[str] = []
        src_for_edit: list[str] = []
        if repo_dir is not None:
            try:
                from eval.repo_context import paths_from_tests, extract_file_paths, read_files
                from eval.repo_focus import (
                    symbols_for_localization, locate_source_by_symbols,
                    parse_fail_to_pass, scope_dirs_from_tests,
                    focus_source_files, identifiers_from_text, failing_tests_section,
                )
                from eval.patch_utils import source_files_only, test_to_source_candidates

                test_paths = paths_from_tests(instance, repo_dir)
                planner_paths = (
                    extract_file_paths(planner_output, repo_dir, pkg_name)
                    if planner_feeds_localize and planner_output else []
                )

                _ft = parse_fail_to_pass(instance.get("FAIL_TO_PASS", []))
                _testnames = set().union(*_ft.values()) if _ft else set()
                # planner text only contributes to the symbol set in real mode
                _sym_texts = [instance.get("problem_statement", "")]
                if planner_feeds_localize:
                    _sym_texts.append(planner_output)
                _sym_texts.append(" ".join(_testnames))
                _syms = symbols_for_localization(*_sym_texts)
                _scope = scope_dirs_from_tests(instance.get("FAIL_TO_PASS", []))
                located_paths = locate_source_by_symbols(
                    repo_dir, pkg_name, _syms, scope_dirs=_scope, require_basename_match=True)

                seen: set[str] = set()
                for p in located_paths + test_paths + planner_paths:
                    if p not in seen:
                        seen.add(p)
                        file_paths.append(p)
                file_paths = file_paths[:6]

                if file_paths:
                    source_files = [p for p in file_paths
                                    if "/tests/" not in p and not p.endswith("_test.py")]
                    test_files = [p for p in file_paths
                                  if "/tests/" in p or p.endswith("_test.py")]
                    _hints = set(_syms)
                    if planner_feeds_localize:
                        _hints |= identifiers_from_text(planner_output)
                    _src = (source_files if source_files else file_paths)[:2]
                    _focus, _stats = focus_source_files(repo_dir, _src, _hints)
                    print(f"  [focus] {_stats['orig_chars']}->{_stats['out_chars']} chars "
                          f"(~{_stats['est_tokens']} tok)")
                    file_content_section = (
                        "\n\n## Source files to fix (SEARCH/REPLACE these; excerpts are verbatim)\n"
                        + _focus
                    )
                    _ftsec = failing_tests_section(repo_dir, instance.get("FAIL_TO_PASS", []))
                    if _ftsec:
                        file_content_section += _ftsec
                    elif test_files and source_files:
                        file_content_section += (
                            "\n\n## Test files (READ-ONLY — do NOT modify these)\n"
                            + read_files(repo_dir, test_files, max_chars=3000)
                        )

                # source files for editing (never tests)
                src_for_edit = source_files_only(file_paths)[:2]
                if not src_for_edit:
                    for tp in file_paths:
                        for cand in test_to_source_candidates(tp):
                            if (repo_dir / cand).exists():
                                src_for_edit.append(cand)
                                break
            except Exception as e:
                print(f"  [repo] localize failed: {e}")

        # ── Step 2: coder (always) — SEARCH/REPLACE + verify-and-retry ────────
        from eval.patch_utils import generate_patch_with_retry, model_output_to_diff
        from eval.repo_focus import clip as _clip

        coder_output = ""
        coder_patch = ""
        _planner_brief = _clip(planner_output, 1500) if planner_output else ""
        context_coder = (
            [{"id": 1, "agent": "planner", "output": _planner_brief}]
            if _planner_brief else None
        )

        if repo_dir is not None and file_content_section and src_for_edit:
            head = (("Planner analysis:\n" + _planner_brief + "\n\n") if _planner_brief else "")
            base_prompt = head + coder_instructions() + file_content_section
            _last = {"out": ""}

            def _coder_fn(prompt: str) -> str:
                out = pool["coder"].run(prompt, context_coder)
                _last["out"] = out
                return out

            print(f"  [Step 2] coder (retry loop) on {src_for_edit}")
            t0 = time.perf_counter()
            coder_patch = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: generate_patch_with_retry(
                    _coder_fn, repo_dir, src_for_edit,
                    base_prompt=base_prompt,
                    retry_feedback=coder_retry_instructions,
                    max_retries=2,
                ).patch,
            )
            dur2 = time.perf_counter() - t0
            coder_output = _last["out"]
            print(f"  [Step 2] done in {dur2:.1f}s — "
                  f"{'patch' if coder_patch else 'empty'}")
            steps.append(StepResult(id=2, agent="coder", subtask="(retry loop)",
                                    output=coder_output, duration_s=dur2))
        else:
            coder_subtask = ("Using the available analysis, implement the fix.\n\n"
                             + coder_instructions())
            print(f"  [Step 2] coder (no-context)")
            t0 = time.perf_counter()
            coder_output = await asyncio.get_event_loop().run_in_executor(
                None, pool["coder"].run, coder_subtask, context_coder
            )
            dur2 = time.perf_counter() - t0
            steps.append(StepResult(id=2, agent="coder", subtask=coder_subtask,
                                    output=coder_output, duration_s=dur2))

        final_patch = coder_patch

        # ── Step 3: reviewer (arm C only) — WIRED: revision adopted if safe ───
        if use_reviewer and repo_dir is not None and src_for_edit:
            reviewer_subtask = verifier_instructions() + (file_content_section or "")
            context_rev = [{"id": 2, "agent": "coder", "output": coder_output}]
            print(f"  [Step 3] reviewer (wired)")
            t0 = time.perf_counter()
            reviewer_output = await asyncio.get_event_loop().run_in_executor(
                None, pool["reviewer"].run, reviewer_subtask, context_rev
            )
            dur3 = time.perf_counter() - t0
            steps.append(StepResult(id=3, agent="reviewer", subtask=reviewer_subtask,
                                    output=reviewer_output, duration_s=dur3))
            # Re-validate the reviewer's SEARCH/REPLACE against the real source.
            rev_patch = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model_output_to_diff(reviewer_output, repo_dir, src_for_edit)
            )
            if rev_patch:
                final_patch = rev_patch
                print(f"  [Step 3] reviewer revision ADOPTED")
            else:
                print(f"  [Step 3] reviewer produced no safe patch — kept coder patch")

        # Final submitted patch (mirrors run_swebench's __patch__ convention).
        steps.append(StepResult(id=99, agent="__patch__",
                                subtask=f"arm={arm} mode={mode}",
                                output=final_patch, duration_s=0.0))
        return steps

    step_results = asyncio.run(_run())
    return WorkflowResult(goal=f"Fix: {iid} [{arm}/{mode}]", step_results=step_results)


# ── prediction wrapper ────────────────────────────────────────────────────────

def run_instance_arm(
    instance: dict, config: dict, *, arm: str, mode: str,
    log_dir: str, use_repo_context: bool = True,
) -> dict:
    iid = instance["instance_id"]
    t0 = time.time()
    try:
        result = run_arm_workflow(instance, config, arm=arm, mode=mode,
                                  use_repo_context=use_repo_context)
        elapsed = time.time() - t0
        patch = ""
        for s in result.step_results:
            if getattr(s, "agent", "") == "__patch__":
                patch = s.output or ""
                break
        _save_instance_log(log_dir=log_dir, instance=instance, result=result,
                           patch=patch, elapsed=elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {iid}: {e}")
        patch = ""
    return {
        "instance_id": iid,
        "model_patch": patch,
        "model_name_or_path": f"local-fugu-{arm}-{mode}",
        "patch_valid": is_valid_patch(patch),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="local-fugu agent-decomposition ablation")
    ap.add_argument("--arm", choices=ARMS, help="A=coder, B=planner+coder, C=full(wired reviewer)")
    ap.add_argument("--mode", choices=MODES, default="pure")
    ap.add_argument("--config", default=str(_REPO_ROOT / "config.yaml"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--instances", default=None, help="comma-separated instance_ids")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-clone", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()

    if args.stats_only:
        print_quick_stats(args.output)
        return
    if args.score_only:
        score_predictions(args.output)
        return
    if not args.arm:
        ap.error("--arm is required unless --score-only/--stats-only")

    config = load_config(args.config)
    use_repo_context = not args.no_clone

    print("[Loading] SWE-Bench Verified…")
    instances = load_dataset("test")
    print(f"  {len(instances)} instances")

    done_ids = load_existing_ids(args.output) if args.resume else set()
    to_run = [i for i in instances if i["instance_id"] not in done_ids]
    if args.instances:
        want = {x.strip() for x in args.instances.split(",") if x.strip()}
        to_run = [i for i in to_run if i["instance_id"] in want]
    elif args.limit:
        to_run = to_run[: args.limit]

    print(f"[Mode] arm={args.arm} mode={args.mode} "
          f"{'+clone' if use_repo_context else '[no clone]'} — {len(to_run)} instances\n")

    log_dir = str(Path(args.output).with_suffix("")) + "_logs"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    file_mode = "a" if args.resume else "w"
    with open(args.output, file_mode) as out_f:
        for idx, inst in enumerate(to_run, 1):
            iid = inst["instance_id"]
            print(f"[{idx}/{len(to_run)}] {iid}")
            t0 = time.perf_counter()
            pred = run_instance_arm(inst, config, arm=args.arm, mode=args.mode,
                                    log_dir=log_dir, use_repo_context=use_repo_context)
            out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            out_f.flush()
            status = "✓ patch" if pred["patch_valid"] else "✗ no patch"
            print(f"  {status}  ({time.perf_counter() - t0:.1f}s)\n")

    print(f"\nDone. {len(to_run)} instances in {(time.perf_counter() - t_start)/60:.1f}min")
    print_quick_stats(args.output)


if __name__ == "__main__":
    main()
