"""
Context compression for the coder step (local-LLM context windows are small).

Design follows the "compress before trim, fail-safe to original" principle
(zephel01, CodeRouter compress plugin):

  1. COMPRESS: AST-extract only the function(s)/class(es) the planner pointed at,
     verbatim (so SEARCH/REPLACE still matches the real file).
  2. TRIM: if the excerpt is still over budget, truncate with a marker.
  3. FAIL-SAFE: if AST parse fails or nothing matches, fall back to the capped
     whole file — never return empty content (an empty section silently breaks
     the coder step, the way an overflowed window does).
  4. MEASURABLE: return stats (orig/out chars, per-file mode) for logging.

Why verbatim matters: pairs_to_diff() applies SEARCH/REPLACE against the REAL
full file, so a verbatim excerpt guarantees the model's copied SEARCH text still
matches. The per-slice `# --- lines A-B ---` headers are NOT code and the model
is told to copy only the code lines.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

CHARS_PER_TOKEN = 4  # rough heuristic, same as the article's char/4


def est_tokens(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


# common English/Python words that must never be treated as a symbol hint
_STOP = {
    "self", "cls", "the", "and", "for", "not", "with", "return", "class", "def",
    "if", "else", "elif", "true", "false", "none", "import", "from", "in", "is",
    "test", "tests", "value", "values", "data", "file", "files", "fix", "bug",
    "issue", "code", "line", "lines", "this", "that", "should", "when", "which",
}


def identifiers_from_text(text: str) -> set[str]:
    """Pull candidate symbol names (functions/classes/methods) from planner text."""
    out: set[str] = set()
    if not text:
        return out
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        if len(tok) >= 3 and tok.lower() not in _STOP:
            out.add(tok)
    # dotted refs like Column.__init__ -> both parts
    for a, b in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)", text):
        for p in (a, b):
            if len(p) >= 3 and p.lower() not in _STOP:
                out.add(p)
    return out


def extract_relevant_source(
    content: str,
    hints: set[str],
    *,
    context_lines: int = 1,
    max_def_lines: int = 200,
) -> str:
    """Return a verbatim excerpt of the defs whose names are in `hints`, or ""."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""
    lines = content.splitlines()
    n = len(lines)
    if n == 0:
        return ""

    ranges: list[tuple[int, int, str]] = []  # 1-based inclusive

    def _start(node) -> int:
        s = node.lineno
        if getattr(node, "decorator_list", None):
            s = min(s, min(d.lineno for d in node.decorator_list))
        return s

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in hints:
                ranges.append((_start(node), node.end_lineno, node.name))
        elif isinstance(node, ast.ClassDef):
            method_hits = [s for s in node.body
                           if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and s.name in hints]
            cls_hit = node.name in hints
            if cls_hit and not method_hits:
                end = node.end_lineno
                if end - node.lineno + 1 > max_def_lines:
                    end = node.lineno + max_def_lines - 1
                ranges.append((_start(node), end, node.name))
            elif method_hits:
                # class header (decorators..signature line) for context
                hend = node.body[0].lineno - 1 if node.body else node.lineno
                ranges.append((_start(node), max(_start(node), hend), f"class {node.name}"))
                for s in method_hits:
                    ranges.append((_start(s), s.end_lineno, f"{node.name}.{s.name}"))

    if not ranges:
        return ""

    # expand by context, clamp to file, merge overlaps/adjacent
    exp = [(max(1, s - context_lines), min(n, e + context_lines), lbl) for s, e, lbl in ranges]
    exp.sort()
    merged: list[tuple[int, int, str]] = []
    for s, e, lbl in exp:
        if merged and s <= merged[-1][1] + 1:
            ps, pe, plbl = merged[-1]
            merged[-1] = (ps, max(pe, e), f"{plbl}; {lbl}")
        else:
            merged.append((s, e, lbl))

    parts = []
    for s, e, lbl in merged:
        body = "\n".join(lines[s - 1:e])
        parts.append(f"# --- lines {s}-{e}: {lbl} ---\n{body}")
    return "\n\n".join(parts)


def focus_source_files(
    repo_dir,
    source_files: list[str],
    hints: set[str],
    *,
    budget_chars: int = 12000,
    cap_per_file: int = 8000,
) -> tuple[str, dict]:
    """
    Build the coder's source section under a hard char budget.

    compress (AST) -> trim (per-file cap then global budget) -> fail-safe (full
    file capped). Never returns empty if any source file exists.
    """
    repo_dir = Path(repo_dir)
    sections: list[str] = []
    stats = {"files": [], "orig_chars": 0, "out_chars": 0, "est_tokens": 0}
    remaining = budget_chars

    for f in source_files:
        fp = repo_dir / f
        if not fp.exists():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        stats["orig_chars"] += len(content)

        excerpt = extract_relevant_source(content, hints)
        if excerpt and len(excerpt) < len(content):
            body, mode = excerpt, "focused"
        else:
            body, mode = content, "full"

        if len(body) > cap_per_file:
            body = body[:cap_per_file] + "\n# … [truncated to per-file cap] …"
            mode += "+cap"
        if len(body) > remaining:
            body = body[:max(0, remaining)] + "\n# … [truncated: context budget exhausted] …"
            mode += "+budget"
        remaining -= len(body)

        stats["files"].append({"file": f, "mode": mode, "chars": len(body)})
        sections.append(f"### FILE: {f}  [{mode}]\n{body}")
        if remaining <= 0:
            break

    section = "\n\n".join(sections)
    stats["out_chars"] = len(section)
    stats["est_tokens"] = est_tokens(section)
    return section, stats


def clip(text: str, max_chars: int) -> str:
    """Hard clip helper for planner analysis / feedback re-sent each retry."""
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …[clipped]"


# ── improvement 1: show the coder the EXACT failing tests ──────────────────────

def _as_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        import json
        try:
            v = json.loads(val)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def parse_fail_to_pass(fail_to_pass) -> dict[str, set[str]]:
    """
    Map FAIL_TO_PASS entries to {test_file_path: {test_function_names}}.

      "astropy/table/tests/test_mixin.py::test_ndarray_mixin[False]"
        -> {"astropy/table/tests/test_mixin.py": {"test_ndarray_mixin"}}
      "pkg/tests/test_x.py::TestClass::test_method"
        -> {"pkg/tests/test_x.py": {"TestClass", "test_method"}}
    """
    out: dict[str, set[str]] = {}
    for entry in _as_list(fail_to_pass):
        if "::" not in entry:
            continue
        parts = entry.split("::")
        path = parts[0]
        names = set()
        for seg in parts[1:]:
            seg = seg.split("[")[0].strip()  # drop pytest params
            if seg:
                names.add(seg)
        if names:
            out.setdefault(path, set()).update(names)
    return out


def failing_tests_section(repo_dir, fail_to_pass, *, budget_chars: int = 6000) -> str:
    """
    Build a section containing the VERBATIM source of the failing test
    functions, so the coder knows exactly which assertions must pass.
    """
    repo_dir = Path(repo_dir)
    mapping = parse_fail_to_pass(fail_to_pass)
    if not mapping:
        return ""
    chunks: list[str] = []
    remaining = budget_chars
    for tpath, names in mapping.items():
        fp = repo_dir / tpath
        if not fp.exists():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        excerpt = extract_relevant_source(content, names, context_lines=0)
        if not excerpt:
            continue
        if len(excerpt) > remaining:
            excerpt = excerpt[:max(0, remaining)] + "\n# … [truncated] …"
        remaining -= len(excerpt)
        chunks.append(f"### TEST FILE: {tpath}\n{excerpt}")
        if remaining <= 0:
            break
    if not chunks:
        return ""
    return ("\n\n## Failing tests — your fix MUST make these pass "
            "(do NOT edit them)\n" + "\n\n".join(chunks))


# ── improvement 2: locate source files by symbol (git grep) ───────────────────

_CAMEL = re.compile(r"\b[A-Z][a-zA-Z0-9]*[a-z][a-zA-Z0-9]*\b")
_SNAKE = re.compile(r"\b_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_BACKTICK = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)`")


def symbols_for_localization(*texts: str, limit: int = 15) -> list[str]:
    """Class/function-like identifiers from issue text + test names, best-first."""
    found: list[str] = []
    seen: set[str] = set()

    def add(s: str):
        s = s.split(".")[-1]
        if s and s.isidentifier() and len(s) >= 3 and s.lower() not in _STOP and s not in seen:
            seen.add(s)
            found.append(s)

    for t in texts:
        if not t:
            continue
        for m in _BACKTICK.findall(t):
            add(m)
    for t in texts:
        if not t:
            continue
        for m in _CAMEL.findall(t):  # class names first (most precise)
            add(m)
    for t in texts:
        if not t:
            continue
        for m in _SNAKE.findall(t):
            add(m)
    return found[:limit]


def scope_dirs_from_tests(fail_to_pass) -> list[str]:
    """
    Sub-package dirs to restrict symbol search to, derived from failing test
    paths: "astropy/table/tests/test_mixin.py" -> "astropy/table".
    Keeps localization precise (avoids matching same-named files in unrelated
    sub-packages, e.g. astropy/io/fits/column.py vs astropy/table/column.py).
    """
    dirs: list[str] = []
    seen: set[str] = set()
    for tpath in parse_fail_to_pass(fail_to_pass).keys():
        parts = [p for p in Path(tpath).parent.parts if p not in ("tests", "test")]
        if parts:
            d = str(Path(*parts))
            if d not in seen:
                seen.add(d)
                dirs.append(d)
    return dirs


def locate_source_by_symbols(
    repo_dir,
    pkg_name: str,
    symbols: list[str],
    *,
    scope_dirs: list[str] | None = None,
    require_basename_match: bool = True,
    max_files: int = 3,
) -> list[str]:
    """
    Use `git grep` to find SOURCE files DEFINING any of `symbols` (class/def).

    Precision-first (v5):
      * search is restricted to `scope_dirs` (the failing test's sub-package)
        when provided, else the whole package — avoids cross-package noise.
      * with require_basename_match=True, only files whose module name equals a
        symbol are kept (e.g. `Column` -> column.py), which filters out the
        long tail of incidental `class/def` matches that caused a regression.
    Test files are always excluded.
    """
    import subprocess
    repo_dir = Path(repo_dir)
    syms = [s for s in symbols if s.isidentifier()][:15]
    if not syms:
        return []
    alt = "|".join(re.escape(s) for s in syms)
    pattern = r"^[[:space:]]*(class|def)[[:space:]]+(" + alt + r")"
    scope = scope_dirs if scope_dirs else ([f"{pkg_name}/"] if pkg_name else ["."])
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), "grep", "-lE", pattern, "--", *scope],
            capture_output=True, text=True,
        )
    except Exception:
        return []

    sym_low = {s.lower() for s in syms}
    out: list[str] = []
    seen: set[str] = set()
    for f in res.stdout.splitlines():
        f = f.strip()
        if not f.endswith(".py") or f in seen:
            continue
        if "/tests/" in f or "/test/" in f or os.path.basename(f).startswith("test_"):
            continue
        stem = os.path.splitext(os.path.basename(f))[0].lower()
        if require_basename_match and stem not in sym_low:
            continue
        seen.add(f)
        out.append(f)
    # basename-matching files first (they already passed the filter; keep stable)
    out.sort(key=lambda p: 0 if os.path.splitext(os.path.basename(p))[0].lower() in sym_low else 1)
    return out[:max_files]
