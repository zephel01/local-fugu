"""
Compare ablation arms — paired resolve-rate analysis.  (2026-06-25)

Reads, per condition, a predictions JSONL (instance_ids + patch_valid) and the
official swebench report JSON (resolved_ids).  Emits a markdown report with:
  * per-condition: N, valid-patch %, resolved, resolve % + Wilson 95% CI
  * pairwise McNemar (exact binomial) on the COMMON instance set
    — the right test for paired binary outcomes with small, low-rate samples.

Usage:
  # explicit conditions
  python compare.py \
      --cond A_pure:preds_A_pure.jsonl:reports/A_pure.json \
      --cond B_pure:preds_B_pure.jsonl:reports/B_pure.json \
      --cond C_pure:preds_C_pure.jsonl:reports/C_pure.json \
      --pair B_pure:A_pure --pair C_pure:B_pure --pair C_pure:A_pure \
      --out report.md

  # or a manifest JSON: {"conditions":[{label,predictions,report}], "pairs":[["B","A"]]}
  python compare.py --manifest manifest.json --out report.md

  # verify the statistics implementation (no data needed)
  python compare.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ── statistics ────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi) in [0,1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact (binomial) McNemar p-value for discordant counts b and c.
      b = pairs where arm1 solved & arm2 did NOT
      c = pairs where arm2 solved & arm1 did NOT
    Under H0 each discordant pair is 50/50; p = P(X as extreme as observed),
    X ~ Binom(b+c, 0.5).  Concordant pairs are (correctly) ignored.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: sum tails <= k and >= n-k
    tail = sum(_binom_pmf(i, n, 0.5) for i in range(0, k + 1))
    p = 2 * tail
    return min(1.0, p)


# ── data loading ──────────────────────────────────────────────────────────────

def load_predictions(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d["instance_id"]] = d
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def load_resolved_ids(report_path: str) -> set[str]:
    """Parse resolved_ids from a swebench report JSON (or a plain id list / .txt)."""
    p = Path(report_path)
    if not p.exists():
        raise FileNotFoundError(f"report not found: {report_path}")
    text = p.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # plain text: one instance_id per line
        return {ln.strip() for ln in text.splitlines() if ln.strip()}
    if isinstance(data, list):
        return set(data)
    for key in ("resolved_ids", "resolved", "resolved_instances_ids"):
        if isinstance(data.get(key), list):
            return set(data[key])
    raise ValueError(f"no resolved_ids list found in {report_path} "
                     f"(keys: {list(data.keys())[:10]})")


# ── condition assembly ────────────────────────────────────────────────────────

class Condition:
    def __init__(self, label: str, predictions: str, report: str):
        self.label = label
        self.preds = load_predictions(predictions)
        self.resolved = load_resolved_ids(report)
        # an instance counts as solved only if it appears in this run's predictions
        self.solved = {iid for iid in self.preds if iid in self.resolved}

    @property
    def n(self) -> int:
        return len(self.preds)

    @property
    def n_resolved(self) -> int:
        return len(self.solved)

    @property
    def n_valid(self) -> int:
        return sum(1 for d in self.preds.values() if d.get("patch_valid"))

    def solved_vector(self, ids: list[str]) -> list[int]:
        return [1 if i in self.solved else 0 for i in ids]


def _fmt_pct(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {k/n*100:.1f}%  [{lo*100:.1f}, {hi*100:.1f}]"


def build_report(conds: list[Condition], pairs: list[tuple[str, str]]) -> str:
    by_label = {c.label: c for c in conds}
    L: list[str] = []
    L.append("# Ablation report — agent decomposition (P1)\n")
    L.append("## Per-condition\n")
    L.append("| condition | N | valid patch | resolved (Wilson 95% CI) |")
    L.append("|---|---|---|---|")
    for c in conds:
        L.append(f"| {c.label} | {c.n} | {_fmt_pct(c.n_valid, c.n)} | {_fmt_pct(c.n_resolved, c.n)} |")
    L.append("")

    L.append("## Pairwise McNemar (paired, common instances)\n")
    L.append("`b` = arm1 solved & arm2 not · `c` = arm2 solved & arm1 not · "
             "p = two-sided exact binomial.\n")
    L.append("| arm1 vs arm2 | common N | arm1 res | arm2 res | b | c | p |")
    L.append("|---|---|---|---|---|---|---|")
    for a1, a2 in pairs:
        if a1 not in by_label or a2 not in by_label:
            L.append(f"| {a1} vs {a2} | missing condition | | | | | |")
            continue
        c1, c2 = by_label[a1], by_label[a2]
        common = sorted(set(c1.preds) & set(c2.preds))
        v1, v2 = c1.solved_vector(common), c2.solved_vector(common)
        b = sum(1 for x, y in zip(v1, v2) if x == 1 and y == 0)
        c = sum(1 for x, y in zip(v1, v2) if x == 0 and y == 1)
        p = mcnemar_exact(b, c)
        L.append(f"| {a1} vs {a2} | {len(common)} | {sum(v1)} | {sum(v2)} | {b} | {c} | {p:.4f} |")
    L.append("")
    L.append("## Reading guide\n")
    L.append("- `b - c` is the net instances the first arm wins by; a small p means the "
             "difference is unlikely under chance.\n")
    L.append("- With low resolve rates and N≈100, expect wide CIs — treat single-digit "
             "deltas as suggestive, not conclusive.\n")
    return "\n".join(L)


# ── self-test (verification of the stats, no data needed) ─────────────────────

def selftest() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    # Wilson: known value 8/10 -> ~[0.490, 0.943]
    lo, hi = wilson_ci(8, 10)
    check("wilson 8/10 lo≈0.490", abs(lo - 0.490) < 0.01)
    check("wilson 8/10 hi≈0.943", abs(hi - 0.943) < 0.01)
    check("wilson 0/0 -> (0,0)", wilson_ci(0, 0) == (0.0, 0.0))

    # McNemar exact: b=c -> p=1.0 ; classic b=10,c=0 -> p=2*0.5^10
    check("mcnemar b=c symmetric p=1", abs(mcnemar_exact(5, 5) - 1.0) < 1e-9)
    check("mcnemar (10,0) p=2/1024", abs(mcnemar_exact(10, 0) - 2 * 0.5 ** 10) < 1e-9)
    check("mcnemar (0,0) p=1", mcnemar_exact(0, 0) == 1.0)
    # known textbook: b=1,c=9 two-sided exact p≈0.0215
    check("mcnemar (1,9) p≈0.0215", abs(mcnemar_exact(1, 9) - 0.021484375) < 1e-6)

    print("OK" if ok else "FAILURES")
    return 0 if ok else 1


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", action="append", default=[],
                    help="label:predictions.jsonl:report.json (repeatable)")
    ap.add_argument("--pair", action="append", default=[],
                    help="arm1:arm2 (repeatable)")
    ap.add_argument("--manifest", help="JSON manifest with conditions+pairs")
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    conds: list[Condition] = []
    pairs: list[tuple[str, str]] = []

    if args.manifest:
        m = json.loads(Path(args.manifest).read_text())
        for c in m.get("conditions", []):
            conds.append(Condition(c["label"], c["predictions"], c["report"]))
        pairs = [tuple(p) for p in m.get("pairs", [])]
    for spec in args.cond:
        label, preds, report = spec.split(":", 2)
        conds.append(Condition(label, preds, report))
    for spec in args.pair:
        a1, a2 = spec.split(":", 1)
        pairs.append((a1, a2))

    if not conds:
        ap.error("no conditions given (use --cond or --manifest)")

    report = build_report(conds, pairs)
    Path(args.out).write_text(report)
    print(report)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
