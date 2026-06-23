"""
Format a SWE-Bench Verified instance into a local-fugu pipeline query.

SWE-Bench instance fields used:
  instance_id      — e.g. "django__django-12345"
  repo             — e.g. "django/django"
  problem_statement — the GitHub issue text
  hints_text       — optional hints from the issue thread
  base_commit      — the commit to apply the patch on top of
"""
from __future__ import annotations


def format_query(instance: dict) -> str:
    """Return the natural-language query to feed into FuguPipeline.run()."""
    repo = instance.get("repo", "")
    problem = instance.get("problem_statement", "").strip()
    hints = instance.get("hints_text", "").strip()

    # FAIL_TO_PASS contains test IDs like "astropy/units/tests/test_q.py::Class::method"
    # → directly reveals the module/file paths involved
    fail_to_pass = instance.get("FAIL_TO_PASS", [])
    if isinstance(fail_to_pass, str):
        import json as _json
        try:
            fail_to_pass = _json.loads(fail_to_pass)
        except Exception:
            fail_to_pass = []

    # Derive the top-level package name from repo (e.g. "astropy/astropy" → "astropy")
    repo_pkg = repo.split("/")[-1].replace("-", "_") if repo else ""

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
        lines += ["", "## Tests that must pass after fix (use these paths to locate files)"]
        for t in fail_to_pass[:8]:  # cap at 8 to avoid context bloat
            lines.append(f"  - {t}")

    lines += [
        "",
        "## Your Task",
        "Produce a minimal unified diff (git patch) that fixes the issue described above.",
        "",
        "CRITICAL diff format requirements:",
        f"1. File paths MUST use the real repository layout — e.g. `{repo_pkg}/module/file.py`",
        "2. Header lines: `--- a/REAL_PATH` and `+++ b/REAL_PATH` (never placeholders like 'path/to/file')",
        "3. Hunk headers must be exact: `@@ -LINE,COUNT +LINE,COUNT @@`",
        "4. Context lines (unchanged) must match the actual source exactly",
        "5. The patch must apply cleanly with `git apply`",
        "6. Fix the root cause — not just suppress the symptom",
        "7. Do not break existing tests",
        "",
        "Output ONLY the patch inside a ```diff ... ``` block. No explanation before or after.",
    ]

    return "\n".join(lines)


def format_conductor_hint(instance: dict) -> str:
    """
    Extra system-level hint injected into the Conductor's user message
    to steer it toward a SWE-appropriate workflow topology.
    """
    repo = instance.get("repo", "")
    repo_pkg = repo.split("/")[-1].replace("-", "_") if repo else ""
    return (
        "This is a software engineering bug-fix task on a real repository. "
        "Recommended workflow: planner (understand issue + identify the exact file path "
        f"inside {repo_pkg}/ that needs changing) → "
        "coder (implement fix as a valid unified diff using the exact path from planner) → "
        "reviewer (verify the diff header paths are real, not placeholders). "
        "The final output MUST be a valid unified diff with real file paths."
    )
