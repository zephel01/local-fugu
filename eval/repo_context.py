"""
Repository context utilities for SWE-bench evaluation.

Provides actual file content to the coder agent so it can generate
correct unified diffs (correct line numbers, context lines, file paths).

Architecture:
  - One base clone per repo (blobless, metadata only)
  - git worktree per (repo, base_commit) — avoids checkout conflicts
  - File tree passed to planner → planner identifies files
  - File content (up to max_chars) passed to coder

Cache layout:
  /tmp/local_fugu_repos/
    astropy__astropy/           ← base clone (blobless)
    django__django/
    ...
  /tmp/local_fugu_worktrees/
    astropy__astropy__abc12345/ ← worktree at specific commit
    ...
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


BASE_REPOS_DIR = Path("/tmp/local_fugu_repos")
WORKTREES_DIR = Path("/tmp/local_fugu_worktrees")

# Limit file content sent to coder (chars, not tokens)
DEFAULT_MAX_CHARS = 20_000


# ── Clone / worktree management ───────────────────────────────────────────────

def _run(cmd: list[str], cwd: str | Path | None = None, timeout: int = 600) -> None:
    """Run a subprocess, raise on failure."""
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")


def _has_commit(base_repo: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=str(base_repo),
        capture_output=True,
    )
    return result.returncode == 0


def get_repo_at_commit(repo: str, base_commit: str) -> Path:
    """
    Return a Path to the repository checked out at base_commit.

    First call per repo clones it (blobless).
    Each (repo, commit) pair gets its own git worktree for safe concurrent use.
    Subsequent calls return the cached worktree instantly.
    """
    repo_key = repo.replace("/", "__")
    commit_short = base_commit[:16]
    worktree_dir = WORKTREES_DIR / f"{repo_key}__{commit_short}"

    # Fast path: worktree already exists
    if worktree_dir.exists() and (worktree_dir / ".git").exists():
        return worktree_dir

    base_repo = BASE_REPOS_DIR / repo_key
    url = f"https://github.com/{repo}.git"

    # Clone if not present (blobless = only metadata, blobs fetched on checkout)
    if not base_repo.exists():
        print(f"  [repo] Cloning {repo} (blobless)…", flush=True)
        BASE_REPOS_DIR.mkdir(parents=True, exist_ok=True)
        _run(
            ["git", "clone", "--filter=blob:none", url, str(base_repo)],
            timeout=600,
        )

    # Fetch the specific commit if not present
    if not _has_commit(base_repo, base_commit):
        print(f"  [repo] Fetching commit {base_commit[:8]}…", flush=True)
        _run(
            ["git", "fetch", "--filter=blob:none", "origin", base_commit],
            cwd=base_repo,
            timeout=300,
        )

    # Create worktree at this commit
    print(f"  [repo] Creating worktree at {base_commit[:8]}…", flush=True)
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "worktree", "add", "--detach", str(worktree_dir), base_commit],
        cwd=base_repo,
        timeout=120,
    )

    return worktree_dir


# ── File tree ─────────────────────────────────────────────────────────────────

def get_file_tree(repo_dir: Path, pkg_name: str, max_files: int = 300) -> str:
    """
    Return a newline-separated list of Python file paths relative to repo root.
    Focuses on the main package directory (pkg_name/) if it exists.
    """
    search_dir = repo_dir / pkg_name
    if not search_dir.is_dir():
        search_dir = repo_dir  # fallback: search whole repo

    files: list[str] = []
    for p in sorted(search_dir.rglob("*.py")):
        if any(part.startswith(".") or part in ("__pycache__", "node_modules")
               for part in p.parts):
            continue
        try:
            rel = str(p.relative_to(repo_dir))
            files.append(rel)
        except ValueError:
            pass
        if len(files) >= max_files:
            break

    return "\n".join(files)


# ── FAIL_TO_PASS → source file derivation ────────────────────────────────────

def paths_from_tests(
    instance: dict,
    repo_dir: Path,
    max_files: int = 4,
) -> list[str]:
    """
    Derive source file paths from FAIL_TO_PASS test IDs.

    Example:
      "astropy/modeling/tests/test_separable.py::TestClass::test_fn"
      → "astropy/modeling/separable.py"

    This is more reliable than parsing planner output because test paths
    are explicitly provided in the SWE-bench dataset.
    """
    import json as _json

    fail_to_pass = instance.get("FAIL_TO_PASS", [])
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = _json.loads(fail_to_pass)
        except Exception:
            fail_to_pass = []

    candidates: list[str] = []
    seen: set[str] = set()

    for test_id in fail_to_pass:
        # Extract the file path part (before ::)
        test_file = test_id.split("::")[0]  # e.g. "astropy/modeling/tests/test_separable.py"

        # Derive source file: remove /tests/test_ prefix pattern
        # astropy/modeling/tests/test_separable.py → astropy/modeling/separable.py
        source = re.sub(r'/tests/test_', '/', test_file)
        # Also try: astropy/modeling/tests/separable_test.py → astropy/modeling/separable.py
        source2 = re.sub(r'/tests/', '/', test_file)
        source2 = re.sub(r'_test\.py$', '.py', source2)

        for candidate in [source, source2]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (repo_dir / candidate).exists():
                candidates.append(candidate)
                break
        # Do NOT fall back to test_file itself — never modify test files

        if len(candidates) >= max_files:
            break

    return candidates


# ── File path extraction ──────────────────────────────────────────────────────

def extract_file_paths(
    planner_output: str,
    repo_dir: Path,
    pkg_name: str,
    max_files: int = 4,
    debug: bool = True,
) -> list[str]:
    """
    Parse planner output and return paths that actually exist in the repo.

    Tries several heuristics:
      1. Direct match:    "astropy/modeling/separable.py"
      2. With pkg prefix: "modeling/separable.py" → "astropy/modeling/separable.py"
      3. Module notation: "astropy.modeling.separable" → "astropy/modeling/separable.py"
      4. Filename only:   "separable.py" → search repo for matching file
    """
    # Strategy 1 & 2: slash-separated paths
    slash_candidates: list[str] = re.findall(r'[\w][\w/]*\.py', planner_output)

    # Strategy 3: Python module notation (e.g. astropy.modeling.separable)
    # Convert to path candidates
    module_candidates: list[str] = []
    for m in re.findall(r'\b[\w]+(?:\.[\w]+){1,6}\b', planner_output):
        # Skip if it looks like a version string or URL fragment
        if any(c in m for c in ('-', ':', '//')):
            continue
        as_path = m.replace('.', '/') + '.py'
        module_candidates.append(as_path)

    # Strategy 4: bare filenames
    filename_candidates: list[str] = re.findall(r'\b([\w]+\.py)\b', planner_output)

    all_candidates = slash_candidates + module_candidates

    if debug:
        print(f"  [repo] Slash candidates: {slash_candidates[:6]}")
        print(f"  [repo] Module candidates: {module_candidates[:6]}")

    valid: list[str] = []
    seen: set[str] = set()

    def _try_add(path: str) -> bool:
        if path in seen:
            return False
        seen.add(path)
        if (repo_dir / path).exists():
            valid.append(path)
            return True
        # Try with pkg_name prefix
        with_pkg = f"{pkg_name}/{path}"
        if with_pkg not in seen and (repo_dir / with_pkg).exists():
            seen.add(with_pkg)
            valid.append(with_pkg)
            return True
        return False

    for c in all_candidates:
        _try_add(c)
        if len(valid) >= max_files:
            break

    # Strategy 4 fallback: search by filename across the repo
    if not valid and filename_candidates:
        for fname in dict.fromkeys(filename_candidates):  # deduplicate, preserve order
            # rglob the repo for this filename
            matches = list(repo_dir.rglob(fname))
            for m in matches:
                try:
                    rel = str(m.relative_to(repo_dir))
                    if rel not in seen:
                        seen.add(rel)
                        valid.append(rel)
                except ValueError:
                    pass
            if len(valid) >= max_files:
                break

    if debug and not valid:
        print(f"  [repo] All candidates tried: {all_candidates[:10]}")

    return valid


# ── File content reader ───────────────────────────────────────────────────────

def read_files(
    repo_dir: Path,
    file_paths: list[str],
    max_chars: int = DEFAULT_MAX_CHARS,
    line_numbers: bool = True,
) -> str:
    """
    Read file contents from the repo.
    Returns a formatted string with each file in a fenced code block.
    Line numbers are included by default so the coder can generate correct
    @@ hunk headers and context lines.
    Truncates if total exceeds max_chars.
    """
    sections: list[str] = []
    total = 0

    for path in file_paths:
        full_path = repo_dir / path
        if not full_path.exists():
            # Try stripping leading a/ or b/ (diff artifact)
            stripped = re.sub(r'^[ab]/', '', path)
            full_path = repo_dir / stripped
            if not full_path.exists():
                continue

        try:
            raw = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        remaining = max_chars - total
        if remaining <= 0:
            break

        if line_numbers:
            lines = raw.splitlines()
            # Add 1-based line numbers: "  42  code here"
            numbered = "\n".join(f"{i+1:5d}  {line}" for i, line in enumerate(lines))
            content = numbered
        else:
            content = raw

        if len(content) > remaining:
            content = content[:remaining] + "\n... (truncated)"

        sections.append(f"### {path}\n```\n{content}\n```")
        total += len(content)

    if not sections:
        return "(No matching files found in repository)"

    return "\n\n".join(sections)
