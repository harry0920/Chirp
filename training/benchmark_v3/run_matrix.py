"""
Run the full Phase C matrix: every candidate in candidates.yaml against
the full corpus, sequentially. Saves per-candidate runs and an aggregated
matrix_summary.json + console table.

Usage:
    python run_matrix.py            # all candidates
    python run_matrix.py --skip qwen3-0.6b   # skip baseline (already run)
    python run_matrix.py --only gemma-4-e2b-it,gemma-4-e4b-it
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import runner
import report

CANDIDATES_PATH = ROOT / "candidates.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", default="", help="comma-separated candidates to skip")
    ap.add_argument("--only", default="", help="comma-separated candidates to run")
    ap.add_argument("--strategy", default="prod-v13", help="prompt strategy")
    ap.add_argument("--limit", type=int, default=None, help="cases per candidate")
    args = ap.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    only = set(s.strip() for s in args.only.split(",") if s.strip())

    cands = yaml.safe_load(CANDIDATES_PATH.read_text())["candidates"]
    names = [n for n in cands if n not in skip and (not only or n in only)]
    print(f"Running {len(names)} candidate(s): {names}", flush=True)

    summaries = []
    for i, name in enumerate(names, 1):
        print(f"\n[{i}/{len(names)}] === {name} ===", flush=True)
        cand = cands[name]
        # Sanity-check files exist BEFORE spawning
        if not Path(cand["model"]).exists():
            print(f"  [SKIP] model not found: {cand['model']}", flush=True)
            summaries.append({"candidate": name, "status": "model_missing"})
            continue
        try:
            t0 = time.time()
            run_dir = runner.run_candidate(name, args.limit, greedy=False, strategy_name=args.strategy)
            elapsed = time.time() - t0
            summary = report.report(run_dir)
            summary["wall_seconds"] = elapsed
            summary["status"] = "ok"
            summaries.append(summary)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}", flush=True)
            summaries.append({"candidate": name, "status": "error", "error": str(e)})

    # Final ranking table
    print("\n" + "=" * 78)
    print("  Phase C matrix — final ranking")
    print("=" * 78)
    print(f"  {'candidate':<28} {'composite':>10} {'cat_succ':>10} {'p95(ms)':>9}  status")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*9}  {'-'*15}")
    ranked = sorted(
        [s for s in summaries if s.get("status") == "ok"],
        key=lambda s: -s.get("composite_mean", 0),
    )
    for s in ranked:
        comp = s.get("composite_mean", 0)
        succ = s.get("category_success_mean", 0) * 100
        p95 = s.get("ttlt_p95_ms", 0)
        dq = "[!] DQ" if s.get("disqualifications") else "[ok]"
        print(f"  {s['candidate']:<28} {comp:>10.3f} {succ:>9.1f}% {p95:>9.0f}  {dq}")
    for s in summaries:
        if s.get("status") != "ok":
            print(f"  {s['candidate']:<28} {'--':>10} {'--':>10} {'--':>9}  {s.get('status')}")

    # Save matrix summary
    out = ROOT / "results" / "matrix_summary.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nMatrix summary: {out}")


if __name__ == "__main__":
    main()
