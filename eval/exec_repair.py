"""
Execution-guided repair for local-fugu SWE-bench (eval/exec_repair.py).

The harness is now reliable, but the model produces "right area, wrong detail"
patches (resolved=0). The decisive lever is to ACTUALLY RUN the failing tests
against a candidate patch and feed the real failure back to the coder:

    candidate -> run FAIL_TO_PASS in the instance env -> compress the failure
    -> feed it back -> regenerate -> repeat (bounded).

This module is RUNNER-AGNOSTIC: you inject a `TestRunner`. That keeps the loop
fully unit-testable offline (fake runner) while the real runner executes tests
in the SWE-bench Docker image on the Mac (see SWEBenchSingleInstanceRunner).

Context safety (per the "compress before trim" principle): test output is
summarized to the FAILED / assertion / traceback lines + tail, capped, so the
repair feedback can never blow the local model's context window.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


# ── test result + runner protocol ─────────────────────────────────────────────

@dataclass
class TestOutcome:
    passed: bool                       # True iff all FAIL_TO_PASS now pass and no PASS_TO_PASS broke
    failed_tests: list[str] = field(default_factory=list)
    summary: str = ""                  # compressed failure text for feedback
    error: str = ""                    # runner-level error (env/docker), not a test failure


class TestRunner(Protocol):
    def run(self, patch: str) -> TestOutcome: ...


# ── compress test output for feedback ─────────────────────────────────────────

_KEEP = re.compile(
    r"(FAILED|ERROR\b|^E\s|Traceback|assert|AssertionError|Error:|Exception|"
    r"\bpassed\b|\bfailed\b)",
    re.IGNORECASE | re.MULTILINE,
)


def summarize_test_output(text: str, *, max_chars: int = 2000, tail_lines: int = 25) -> str:
    """Keep only failure-signal lines + the tail; drop the noise. Capped."""
    if not text:
        return ""
    lines = text.splitlines()
    keep: list[str] = [ln for ln in lines if _KEEP.search(ln)]
    tail = lines[-tail_lines:]
    # merge keep + tail preserving order, dedup adjacent
    seen: set[int] = set()
    out: list[str] = []
    for ln in keep + tail:
        h = hash(ln)
        if h in seen:
            continue
        seen.add(h)
        out.append(ln)
    s = "\n".join(out)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n… [truncated]"
    return s


# ── the repair loop (runner-agnostic, offline-testable) ───────────────────────

@dataclass
class RepairResult:
    patch: str = ""
    status: str = "empty"        # "resolved" | "unresolved" | "empty"
    rounds: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "resolved"


def build_repair_feedback(outcome: TestOutcome) -> str:
    failed = ", ".join(outcome.failed_tests[:8]) if outcome.failed_tests else "(see output)"
    return (
        "Your patch APPLIED but the tests still FAIL.\n"
        f"Still failing: {failed}\n\n"
        "Test output (failures only):\n"
        f"{outcome.summary}\n\n"
        "Revise the SOURCE so these tests pass WITHOUT breaking others. "
        "Pay attention to the exact assertion / exception above. "
        "Output corrected SEARCH/REPLACE blocks only."
    )


def repair_patch(
    candidate_fn: Callable[[str | None], str],
    runner: TestRunner,
    *,
    max_rounds: int = 2,
    verbose: bool = True,
) -> RepairResult:
    """
    candidate_fn(feedback) -> a VALIDATED patch string ("" if none). `feedback`
    is None on the first round, else the failure text from the previous run.

    Returns the resolved patch if found; otherwise the last non-empty candidate
    (best-effort submission — still scores 0 but is our best attempt), or "".
    """
    res = RepairResult()
    feedback: str | None = None
    best = ""

    for rnd in range(1, max_rounds + 2):  # 1 initial + max_rounds repairs
        res.rounds = rnd
        patch = candidate_fn(feedback)
        if not patch:
            res.log.append(f"round {rnd}: no candidate patch")
            break
        best = patch

        outcome = runner.run(patch)
        if outcome.error:
            res.log.append(f"round {rnd}: runner error: {outcome.error[:120]}")
            # cannot get test signal — keep the best candidate, stop looping
            break
        if outcome.passed:
            res.patch = patch
            res.status = "resolved"
            res.log.append(f"round {rnd}: PASSED")
            return res

        res.log.append(f"round {rnd}: failing {outcome.failed_tests[:6]}")
        feedback = build_repair_feedback(outcome)

    res.patch = best
    res.status = "unresolved" if best else "empty"
    res.log.append(f"done: {res.status} after {res.rounds} round(s)")
    return res


# ── reference runner: SWE-bench Docker, single instance (runs on the Mac) ──────

class SWEBenchSingleInstanceRunner:
    """
    Run ONE instance's tests against a candidate patch using the SWE-bench
    harness, then parse the report.

    Calls `swebench.harness.run_evaluation.main` as a PYTHON FUNCTION with the
    SAME arguments the project's working `score_predictions()` uses — this is
    far more robust than shelling out to the CLI (the CLI flags / report paths
    vary by version and produced "no report.json" for some instances).

    The harness writes the per-instance report to:
        logs/run_evaluation/<run_id>/<model_name>/<instance_id>/report.json
    (relative to CWD), with test_output.txt alongside.
    """

    MODEL_NAME = "local-fugu-repair"

    def __init__(self, instance: dict, *, workdir: str | None = None, timeout: int = 1800):
        self.instance = instance
        self.iid = instance["instance_id"]
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="fugu_repair_"))
        self.timeout = timeout
        self._round = 0

    def _find_report(self, run_id: str) -> Path | None:
        # primary documented location
        base = Path("logs") / "run_evaluation" / run_id / self.MODEL_NAME / self.iid
        cand = base / "report.json"
        if cand.exists():
            return cand
        # fallback: any report.json under this run_id
        root = Path("logs") / "run_evaluation" / run_id
        if root.exists():
            return next(root.rglob("report.json"), None)
        return None

    def run(self, patch: str) -> TestOutcome:
        self._round += 1
        run_id = f"repair_{self.iid.replace('/', '_')}_{self._round}"
        preds = self.workdir / f"pred_{self._round}.jsonl"
        preds.write_text(json.dumps({
            "instance_id": self.iid,
            "model_patch": patch,
            "model_name_or_path": self.MODEL_NAME,
        }) + "\n", encoding="utf-8")

        try:
            from swebench.harness.run_evaluation import main as swe_eval
        except Exception as e:
            return TestOutcome(passed=False, error=f"swebench import failed: {e}")

        try:
            swe_eval(
                dataset_name="princeton-nlp/SWE-bench_Verified",
                split="test",
                instance_ids=[self.iid],
                predictions_path=str(preds),
                max_workers=1,
                force_rebuild=False,
                cache_level="env",
                clean=False,
                open_file_limit=4096,
                run_id=run_id,
                timeout=self.timeout,
                namespace="swebench",
                rewrite_reports=False,
                modal=False,
                report_dir=str(self.workdir),
            )
        except TypeError as e:
            # signature drift across swebench versions — surface it clearly
            return TestOutcome(passed=False, error=f"swe_eval signature mismatch: {e}")
        except Exception as e:
            return TestOutcome(passed=False, error=f"swe_eval raised: {e}")

        report = self._find_report(run_id)
        if report is None:
            return TestOutcome(passed=False, error=f"no report.json under logs/run_evaluation/{run_id}")

        try:
            data = json.loads(report.read_text())
            rec = data[self.iid] if self.iid in data else next(iter(data.values()))
            resolved = bool(rec.get("resolved", False))
            ts = rec.get("tests_status", {})
            failed = (ts.get("FAIL_TO_PASS", {}).get("failure", [])
                      + ts.get("PASS_TO_PASS", {}).get("failure", []))
        except Exception as e:
            return TestOutcome(passed=False, error=f"report parse failed: {e}")

        out_txt = ""
        tout = report.parent / "test_output.txt"
        if tout.exists():
            try:
                out_txt = tout.read_text(errors="replace")
            except OSError:
                pass
        return TestOutcome(
            passed=resolved,
            failed_tests=failed,
            summary=summarize_test_output(out_txt),
        )
