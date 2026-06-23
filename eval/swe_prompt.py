"""
Format a SWE-Bench instance into local-fugu pipeline prompts.

v2 (2026-06-23) REBUILD — Phase 1: single patch contract
--------------------------------------------------------
The previous version told the model to "output ONLY a unified diff with exact
`@@` line numbers". Small local models cannot count line numbers, so they
emitted broken diffs that the harness then applied and corrupted files with.

v2 removes ALL unified-diff instructions from the model-facing text. The issue
description is now used purely to (a) explain the bug and (b) help LOCALIZE the
source file. The actual edit format (SEARCH/REPLACE) is specified in the coder
step (see run_swebench.run_fixed_workflow), and the unified diff is generated
deterministically by difflib in patch_utils. One contract, no contradictions.

SWE-Bench instance fields used:
  instance_id, repo, problem_statement, hints_text, base_commit, FAIL_TO_PASS
"""
from __future__ import annotations

import json as _json


def _as_list(val) -> list[str]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = _json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def format_query(instance: dict) -> str:
    """
    Natural-language description of the bug, for the planner / localization.

    NOTE: deliberately contains NO output-format instructions. It never asks
    for a diff. The FAIL_TO_PASS test paths are provided only as *hints for
    locating the source file* — with an explicit warning not to edit tests.
    """
    repo = instance.get("repo", "")
    problem = instance.get("problem_statement", "").strip()
    hints = instance.get("hints_text", "").strip()
    repo_pkg = repo.split("/")[-1].replace("-", "_") if repo else ""

    fail_to_pass = _as_list(instance.get("FAIL_TO_PASS", []))

    lines = [
        f"Repository: {repo}",
        f"Package root: {repo_pkg}/",
        "",
        "## Issue",
        problem,
    ]
    if hints:
        lines += ["", "## Hints from issue thread", hints]

    if fail_to_pass:
        lines += [
            "",
            "## Failing tests (for LOCATING the buggy source file only)",
            "These tests currently fail and must pass after the fix. Use their "
            "paths to locate the SOURCE module that contains the bug. "
            "Do NOT edit these test files — the bug is in the library code they exercise.",
        ]
        for t in fail_to_pass[:8]:
            lines.append(f"  - {t}")

    lines += [
        "",
        "## Goal",
        "Find the root cause in the library SOURCE code and fix it. "
        "The fix must make the failing tests pass without breaking other tests.",
    ]
    return "\n".join(lines)


def format_conductor_hint(instance: dict) -> str:
    """Steer the Conductor toward a SWE-appropriate topology (no diff talk)."""
    repo = instance.get("repo", "")
    repo_pkg = repo.split("/")[-1].replace("-", "_") if repo else ""
    return (
        "This is a software bug-fix task on a real repository. "
        "Recommended workflow: planner (locate the exact SOURCE file inside "
        f"{repo_pkg}/ that contains the bug — never a test file) → "
        "coder (edit that source file using SEARCH/REPLACE blocks copied "
        "verbatim from the provided file content) → "
        "verifier (confirm the change targets the root cause and edits only "
        "source files). Never modify files under a /tests/ directory."
    )


# ── Coder / Verifier instruction blocks (single source of truth) ──────────────
# These are imported by run_swebench so the *exact same* contract lives in one
# place instead of being duplicated (and contradicted) across files.

def coder_instructions() -> str:
    return (
        "Implement the fix by editing the SOURCE file(s) shown below.\n\n"
        "Output your change as one or more SEARCH/REPLACE blocks in EXACTLY this format:\n\n"
        "<<<<<<< SEARCH\n"
        "<lines copied verbatim from the file — exact whitespace>\n"
        "=======\n"
        "<the replacement lines>\n"
        ">>>>>>> REPLACE\n\n"
        "Rules:\n"
        "- Copy the SEARCH lines EXACTLY as they appear (indentation, spacing). No paraphrasing.\n"
        "- Make the SEARCH block large enough to be UNIQUE in the file (include surrounding context).\n"
        "- Use multiple SEARCH/REPLACE blocks if several spots change.\n"
        "- Edit ONLY source files. NEVER edit files under a /tests/ directory.\n"
        "- Do NOT output a unified diff, code fences, or commentary — SEARCH/REPLACE blocks only.\n"
    )


def coder_retry_instructions(failures: list[str]) -> str:
    """Feedback appended on a retry when the previous attempt failed validation."""
    bullet = "\n".join(f"  - {f}" for f in failures)
    return (
        "Your previous attempt could not be applied. Problems:\n"
        f"{bullet}\n\n"
        "Fix these and output corrected SEARCH/REPLACE blocks only. "
        "Remember: the SEARCH text must match the file content character-for-character.\n"
    )


def verifier_instructions() -> str:
    return (
        "Review the coder's proposed fix.\n"
        "1. Does it address the root cause described in the issue?\n"
        "2. Could it break other (passing) tests?\n"
        "3. Does it edit ONLY source files (no /tests/ files)?\n\n"
        "If correct, output the SEARCH/REPLACE blocks UNCHANGED.\n"
        "If wrong, output corrected SEARCH/REPLACE blocks. Keep the SEARCH/REPLACE "
        "format — never switch to a unified diff.\n"
    )
