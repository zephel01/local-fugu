"""
Extract a unified diff patch from the pipeline's output text.

Strategies (in order):
1. Fenced ```diff ... ``` block
2. Any block starting with "diff --git" or "--- a/"
3. Fallback: return empty string (instance will score 0)
"""
from __future__ import annotations

import re


_FENCE_RE = re.compile(r"```diff\s*([\s\S]+?)\s*```", re.IGNORECASE)
_DIFF_START_RE = re.compile(r"(diff --git .+|--- a/.+)", re.MULTILINE)


def _only_tests(patch: str) -> bool:
    """True if every modified file in the patch is a test file."""
    files = re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)
    if not files:
        return False
    return all("/tests/" in f or f.endswith("_test.py") or "/test_" in f for f in files)


def extract(text: str) -> str:
    """Return the patch string, or '' if none found."""
    # Strategy 1: fenced block
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if not _only_tests(candidate):
            return candidate

    # Strategy 2: bare diff starting with known headers
    m2 = _DIFF_START_RE.search(text)
    if m2:
        candidate = text[m2.start():].strip()
        if not _only_tests(candidate):
            return candidate

    return ""


def is_valid_patch(patch: str) -> bool:
    """Minimal sanity check — does it look like a real diff?"""
    if not patch:
        return False
    has_diff_header = "diff --git" in patch or "--- a/" in patch
    has_hunk = "@@" in patch
    return has_diff_header and has_hunk
