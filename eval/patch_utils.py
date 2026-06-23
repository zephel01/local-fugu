"""
Patch generation utilities for local-fugu SWE-bench evaluation.

v2 (2026-06-23) REBUILD — Phases 1-3
------------------------------------
Single, safe patch contract:

  model emits SEARCH/REPLACE  →  apply to the REAL source file  →
  difflib makes the unified diff  →  is_patch_safe() gates it  →
  on failure, feed the error back and RETRY (bounded).

Key changes vs v1:
  * apply_search_replace() now requires a UNIQUE match (ambiguous matches are
    rejected with a reason instead of silently editing the first occurrence).
  * source_files_only() / test_to_source_candidates(): localization helpers
    that keep edits on SOURCE files and never on /tests/ files.
  * generate_patch_with_retry(): the Phase-3 verify-and-retry loop. It is model
    agnostic — you inject a `coder_fn(prompt)->str`, so it is fully testable
    offline with a scripted fake model (see the golden test).
  * is_patch_safe() (from v1) unchanged: strict `git apply --check` (no fuzz) in
    a throwaway worktree + ast.parse of every touched .py file.

This file supersedes both eval/patch_utils.py and eval/extract_patch.py.
"""
from __future__ import annotations

import ast
import difflib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── basics ────────────────────────────────────────────────────────────────────

def _is_valid_python(content: str) -> bool:
    try:
        ast.parse(content)
        return True
    except SyntaxError:
        return False


def is_test_path(path: str) -> bool:
    return "/tests/" in path or "/test/" in path or path.endswith("_test.py") \
        or os.path.basename(path).startswith("test_")


def source_files_only(paths: list[str]) -> list[str]:
    """Drop test files; keep source files, order preserved, deduped."""
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if p in seen or is_test_path(p):
            continue
        seen.add(p)
        out.append(p)
    return out


def test_to_source_candidates(test_path: str) -> list[str]:
    """
    Heuristic: map a test path to candidate SOURCE module paths.

      astropy/timeseries/tests/test_sampled.py -> astropy/timeseries/sampled.py
      astropy/table/tests/test_mixin.py        -> astropy/table/mixin.py
      pkg/tests/test_foo.py                    -> pkg/foo.py

    Candidates are returned best-first; the caller should keep only those that
    actually exist in the repo.
    """
    p = Path(test_path)
    name = p.name
    if name.startswith("test_"):
        mod = name[len("test_"):]
    elif name.endswith("_test.py"):
        mod = name[: -len("_test.py")] + ".py"
    else:
        mod = name
    # parent without the trailing tests/ segment
    parts = [seg for seg in p.parent.parts if seg not in ("tests", "test")]
    base = Path(*parts) if parts else Path(".")
    cands = [str(base / mod)]
    # also try one level up (some repos nest tests under the package)
    if len(parts) >= 1:
        cands.append(str(Path(*parts) / mod))
    # dedupe preserve order
    out, seen = [], set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ── SEARCH/REPLACE extraction ─────────────────────────────────────────────────

SEARCH_REPLACE_PATTERN = re.compile(
    r"<<<+\s*SEARCH\s*\n(.*?)\n=====*\n(.*?)\n>>>+\s*REPLACE",
    re.DOTALL | re.IGNORECASE,
)
SEARCH_PATTERN2 = re.compile(
    r"SEARCH:\s*```(?:python)?\n(.*?)```\s*\nREPLACE:\s*```(?:python)?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_search_replace(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in SEARCH_REPLACE_PATTERN.finditer(text):
        pairs.append((m.group(1).rstrip("\n"), m.group(2).rstrip("\n")))
    if not pairs:
        for m in SEARCH_PATTERN2.finditer(text):
            pairs.append((m.group(1).rstrip("\n"), m.group(2).rstrip("\n")))
    return pairs


# ── apply one SEARCH/REPLACE (v2: unique-match required) ──────────────────────

def apply_search_replace(original: str, search: str, replace: str) -> tuple[str | None, str]:
    """
    Apply a search/replace. Returns (modified_or_None, reason).

    Requires a UNIQUE match:
      - exact unique match            -> applied
      - exact but multiple matches    -> rejected ("ambiguous")
      - no exact, unique normalized   -> applied (whitespace-tolerant)
      - otherwise                     -> rejected ("not found")
    """
    if not search.strip():
        return None, "empty SEARCH block"

    # 1) exact, unique
    n_exact = original.count(search)
    if n_exact == 1:
        return original.replace(search, replace, 1), "exact"
    if n_exact > 1:
        return None, f"ambiguous: SEARCH matches {n_exact} places (add more context)"

    # 2) whitespace-normalized (rstrip per line), unique
    def norm(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.splitlines())

    no, ns = norm(original), norm(search)
    n_norm = no.count(ns)
    if n_norm == 1:
        idx = no.index(ns)
        before = no[:idx].count("\n")
        nlines = ns.count("\n") + 1
        olines = original.splitlines(keepends=True)
        tail = olines[before + nlines:]
        repl = replace if replace.endswith("\n") else replace + "\n"
        return "".join(olines[:before] + [repl] + tail), "normalized"
    if n_norm > 1:
        return None, f"ambiguous: SEARCH matches {n_norm} places after normalization"

    return None, "not found (SEARCH text does not match the file)"


# ── unified diff via difflib ──────────────────────────────────────────────────

def make_unified_diff(original: str, modified: str, filepath: str, context_lines: int = 3) -> str:
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
        n=context_lines,
    )
    return "".join(diff)


# ── pre-apply safety (unchanged from v1) ──────────────────────────────────────

def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def patched_files(patch: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(.+?)\s*$", patch, re.MULTILINE)


def _run_git_apply_check(patch: str, cwd: Path) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as tf:
        tf.write(patch)
        pf = tf.name
    try:
        return _git(["apply", "--check", "--whitespace=nowarn", pf], cwd).returncode == 0
    finally:
        try:
            os.unlink(pf)
        except OSError:
            pass


def is_patch_safe(patch: str, repo_dir: Path) -> bool:
    """True iff patch applies cleanly (strict, no fuzz) AND every touched .py
    file stays syntactically valid. Verified in a throwaway worktree; the
    caller's checkout is never mutated. Fails CLOSED."""
    if not patch or not patch.strip():
        return False
    if "@@" not in patch or ("--- a/" not in patch and "diff --git" not in patch):
        return False
    files = patched_files(patch)
    if not files:
        return False
    repo_dir = Path(repo_dir)
    head = _git(["rev-parse", "HEAD"], repo_dir)
    if head.returncode != 0:
        return False
    commit = head.stdout.strip()
    with tempfile.TemporaryDirectory(prefix="fugu_patchcheck_") as tmp:
        wt = Path(tmp) / "wt"
        if _git(["worktree", "add", "--detach", str(wt), commit], repo_dir).returncode != 0:
            return _run_git_apply_check(patch, repo_dir)
        try:
            if not _run_git_apply_check(patch, wt):
                return False
            pf = Path(tmp) / "candidate.patch"
            pf.write_text(patch, encoding="utf-8")
            if _git(["apply", "--whitespace=nowarn", str(pf)], wt).returncode != 0:
                return False
            for f in files:
                if f.endswith(".py"):
                    try:
                        if not _is_valid_python((wt / f).read_text(encoding="utf-8", errors="replace")):
                            return False
                    except OSError:
                        return False
            return True
        finally:
            _git(["worktree", "remove", "--force", str(wt)], repo_dir)


# ── build a unified diff from SEARCH/REPLACE pairs (one shot) ──────────────────

def pairs_to_diff(
    pairs: list[tuple[str, str]],
    repo_dir: Path,
    source_files: list[str],
) -> tuple[str, list[str]]:
    """
    Apply pairs to the given SOURCE files and return (combined_diff, failures).
    Each pair must apply to exactly one file. `failures` explains every pair
    that could not be applied (used as retry feedback).
    """
    repo_dir = Path(repo_dir)
    src = source_files_only(source_files)
    if not src:
        return "", ["no source files to edit (only test files were located)"]

    # read originals
    originals: dict[str, str] = {}
    for f in src:
        fp = repo_dir / f
        if fp.exists():
            try:
                originals[f] = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    if not originals:
        return "", [f"none of the source files exist in the repo: {src}"]

    modified = dict(originals)
    failures: list[str] = []
    applied = 0
    for i, (search, replace) in enumerate(pairs, 1):
        # try each file; a pair applies to the unique file that contains it
        hits = []
        for f in modified:
            res, reason = apply_search_replace(modified[f], search, replace)
            if res is not None:
                hits.append((f, res))
        if len(hits) == 1:
            f, res = hits[0]
            modified[f] = res
            applied += 1
        elif len(hits) == 0:
            snippet = search.strip().splitlines()[0][:80] if search.strip() else ""
            failures.append(f"SEARCH block #{i} did not match any source file "
                            f"(first line: {snippet!r}). Copy the exact text from the file.")
        else:
            failures.append(f"SEARCH block #{i} matched multiple files "
                            f"{[h[0] for h in hits]}; make it more specific.")

    if applied == 0:
        return "", failures or ["no SEARCH/REPLACE block applied"]

    # generate + syntax-check + assemble
    diffs = []
    for f, content in modified.items():
        if content == originals[f]:
            continue
        if f.endswith(".py") and not _is_valid_python(content):
            failures.append(f"applying blocks to {f} produced invalid Python — "
                            f"check indentation in your REPLACE text")
            continue
        d = make_unified_diff(originals[f], content, f)
        if d:
            diffs.append(d)

    return ("\n".join(diffs), failures)


# ── Phase 3: verify-and-retry loop ────────────────────────────────────────────

@dataclass
class PatchResult:
    patch: str = ""
    status: str = "empty"          # "ok" | "empty"
    attempts: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.patch) and self.status == "ok"


def generate_patch_with_retry(
    coder_fn: Callable[[str], str],
    repo_dir: Path,
    source_files: list[str],
    *,
    base_prompt: str,
    retry_feedback: Callable[[list[str]], str],
    max_retries: int = 2,
    verbose: bool = True,
) -> PatchResult:
    """
    Drive the coder to produce a SAFE patch, retrying with targeted feedback.

      coder_fn(prompt) -> model output text   (inject the real LLM or a fake)
      base_prompt        the coder instructions + source file content
      retry_feedback(fs) build the feedback string from a list of failures

    Returns a PatchResult. status="ok" only if the diff passed is_patch_safe().
    Otherwise status="empty" (an empty patch scores 0 but never corrupts files).
    """
    repo_dir = Path(repo_dir)
    res = PatchResult()
    prompt = base_prompt
    last_failures: list[str] = []

    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
        res.attempts = attempt
        if attempt > 1:
            prompt = base_prompt + "\n\n" + retry_feedback(last_failures)

        output = coder_fn(prompt)
        pairs = extract_search_replace(output)
        if not pairs:
            last_failures = ["No SEARCH/REPLACE block found in your output. "
                             "Output blocks in the required format."]
            if verbose:
                res.log.append(f"attempt {attempt}: no SEARCH/REPLACE blocks")
            continue

        diff, failures = pairs_to_diff(pairs, repo_dir, source_files)
        if not diff:
            last_failures = failures
            if verbose:
                res.log.append(f"attempt {attempt}: no applicable diff — {failures}")
            continue

        if not is_patch_safe(diff, repo_dir):
            last_failures = (failures or []) + [
                "the assembled patch failed validation (it does not apply "
                "cleanly or breaks Python syntax)."]
            if verbose:
                res.log.append(f"attempt {attempt}: diff failed is_patch_safe")
            continue

        # success
        res.patch = diff
        res.status = "ok"
        if verbose:
            res.log.append(f"attempt {attempt}: OK ({len(patched_files(diff))} file(s))")
        return res

    if verbose:
        res.log.append(f"exhausted {res.attempts} attempts — empty patch")
    return res


# ── legacy one-shot entry (kept for compatibility, now fully validated) ────────

def model_output_to_diff(model_output: str, repo_dir: Path, file_paths: list[str]) -> str:
    """
    One-shot conversion (no retry). Returns a validated diff or "".
    Prefer generate_patch_with_retry() for the live pipeline.
    """
    repo_dir = Path(repo_dir)
    src = source_files_only(file_paths)
    pairs = extract_search_replace(model_output)
    if pairs and src:
        diff, _ = pairs_to_diff(pairs, repo_dir, src)
        if diff and is_patch_safe(diff, repo_dir):
            return diff
    return ""
