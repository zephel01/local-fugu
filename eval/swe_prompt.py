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

    lines = [
        f"Repository: {repo}",
        "",
        "## Issue",
        problem,
    ]
    if hints:
        lines += ["", "## Hints from issue thread", hints]

    lines += [
        "",
        "## Your Task",
        "Produce a minimal unified diff (git patch) that fixes the issue described above.",
        "The patch must:",
        "1. Apply cleanly on top of the repository's current state",
        "2. Fix the root cause — not just suppress the symptom",
        "3. Not break existing tests",
        "4. Follow the repository's existing code style",
        "",
        "Output the patch inside a ```diff ... ``` block.",
    ]

    return "\n".join(lines)


def format_conductor_hint(instance: dict) -> str:
    """
    Extra system-level hint injected into the Conductor's user message
    to steer it toward a SWE-appropriate workflow topology.
    """
    return (
        "This is a software engineering bug-fix task on a real repository. "
        "Recommended workflow: planner (understand issue + locate files) → "
        "coder (implement fix as unified diff) → reviewer (verify patch correctness). "
        "The final output MUST be a valid unified diff."
    )
