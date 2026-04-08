"""
Report: read a per_case.jsonl from a candidate run, compute composite,
per-category breakdown, bootstrap 95% CI, hard-disqualification gates.
Print to console + save summary.json.

Usage:
    python report.py results/qwen3-0.6b/<timestamp>
    python report.py results/qwen3-0.6b/<timestamp> --compare results/other/...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import scorers

CORPUS_PATH = ROOT / "corpus" / "english_gold.jsonl"


def load_corpus_lookup() -> Dict[str, Dict]:
    out = {}
    with CORPUS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def load_results(run_dir: Path) -> List[Dict]:
    results = []
    with (run_dir / "per_case.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def bootstrap_ci(values: List[float], n_resamples: int = 1000, alpha: float = 0.05) -> tuple:
    """Bootstrap 95% CI for the mean."""
    if not values:
        return (0.0, 0.0)
    means = []
    n = len(values)
    rng = random.Random(42)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_resamples * alpha / 2)]
    hi = means[int(n_resamples * (1 - alpha / 2))]
    return (lo, hi)


def report(run_dir: Path) -> Dict:
    corpus = load_corpus_lookup()
    results = load_results(run_dir)
    if not results:
        print(f"No results in {run_dir}")
        return {}

    composites = [r["scores"]["composite"] for r in results]
    cat_success_all = [r["scores"]["category_success"] for r in results]
    avg_comp = sum(composites) / len(composites)
    ci = bootstrap_ci(composites)

    # Per-category breakdown
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_rows = []
    for cat, rows in sorted(by_cat.items()):
        cat_comp = sum(r["scores"]["composite"] for r in rows) / len(rows)
        cat_succ = sum(r["scores"]["category_success"] for r in rows) / len(rows)
        cat_rows.append((cat, len(rows), cat_succ, cat_comp))

    # Disqualification gates
    cases = [corpus[r["id"]] for r in results]
    per_case_scores = [r["scores"] for r in results]
    dq = scorers.disqualify(per_case_scores, cases)

    # Latency
    ttlts = [r.get("ttlt_ms", 0) for r in results if r.get("ttlt_ms")]
    if ttlts:
        ttlts_sorted = sorted(ttlts)
        p50 = ttlts_sorted[len(ttlts_sorted) // 2]
        p95 = ttlts_sorted[int(len(ttlts_sorted) * 0.95)]
        p99 = ttlts_sorted[int(len(ttlts_sorted) * 0.99)]
    else:
        p50 = p95 = p99 = 0

    # Console output
    name = run_dir.parent.name
    ts = run_dir.name
    print(f"\n{'='*64}")
    print(f"  {name}  ({ts})")
    print(f"{'='*64}")
    print(f"  cases:           {len(results)}")
    print(f"  composite:       {avg_comp:.3f}  (95% CI {ci[0]:.3f}–{ci[1]:.3f})")
    print(f"  cat_success avg: {sum(cat_success_all)/len(cat_success_all):.3f}")
    print(f"  TTLT p50/p95/p99: {p50:.0f} / {p95:.0f} / {p99:.0f} ms")
    print()
    print(f"  per-category breakdown:")
    print(f"  {'category':<35} {'n':>4} {'cat_succ':>10} {'composite':>11}")
    print(f"  {'-'*35} {'-'*4} {'-'*10} {'-'*11}")
    for cat, n, succ, comp in cat_rows:
        print(f"  {cat:<35} {n:>4} {succ*100:>9.1f}% {comp:>11.3f}")
    print()

    if dq:
        print(f"  [!] DISQUALIFIED:")
        for r in dq:
            print(f"    - {r}")
    else:
        print(f"  [ok] all hard gates pass (no hallucination, no paraphrase, "
              f"self-correction OK)")

    # Find worst cases
    sorted_results = sorted(results, key=lambda r: r["scores"]["composite"])
    print(f"\n  worst 5 cases:")
    for r in sorted_results[:5]:
        print(f"    [{r['id']}] {r['scores']['composite']:.2f}  {r['category']}")
        print(f"        IN:  {r['input']}")
        print(f"        REF: {r['reference']}")
        print(f"        OUT: {r['output']}")

    summary = {
        "candidate": name,
        "timestamp": ts,
        "n_cases": len(results),
        "composite_mean": avg_comp,
        "composite_ci_low": ci[0],
        "composite_ci_high": ci[1],
        "category_success_mean": sum(cat_success_all) / len(cat_success_all),
        "ttlt_p50_ms": p50,
        "ttlt_p95_ms": p95,
        "ttlt_p99_ms": p99,
        "by_category": [
            {"category": cat, "n": n, "category_success": succ, "composite": comp}
            for cat, n, succ, comp in cat_rows
        ],
        "disqualifications": dq,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    report(args.run_dir)


if __name__ == "__main__":
    main()
