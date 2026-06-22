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


def extract(text: str) -> str:
    """Return the patch string, or '' if none found."""
    # Strategy 1: fenced block
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()

    # Strategy 2: bare diff starting with known headers
    m2 = _DIFF_START_RE.search(text)
    if m2:
        return text[m2.start():].strip()

    return ""


def is_valid_patch(patch: str) -> bool:
    """Minimal sanity check — does it look like a real diff?"""
    if not patch:
        return False
    has_diff_header = "diff --git" in patch or "--- a/" in patch
    has_hunk = "@@" in patch
    return has_diff_header and has_hunk
