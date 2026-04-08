"""
Run multiple prompt strategies against ONE candidate, rank by composite.

Usage:
    python sweep_prompts.py --candidate qwen3-1.7b
    python sweep_prompts.py --candidate qwen3-1.7b \
        --strategies prod-v13,v125-json,v125-improved
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import runner
import report
import prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument(
        "--strategies",
        default=",".join(prompts.STRATEGIES.keys()),
        help="comma-separated strategy names (default = all)",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    print(f"Sweeping {len(names)} strategies on {args.candidate}: {names}", flush=True)

    summaries = []
    for i, strat in enumerate(names, 1):
        print(f"\n[{i}/{len(names)}] === {strat} ===", flush=True)
        try:
            t0 = time.time()
            run_dir = runner.run_candidate(args.candidate, args.limit, greedy=False, strategy_name=strat)
            elapsed = time.time() - t0
            summary = report.report(run_dir)
            summary["strategy"] = strat
            summary["wall_seconds"] = elapsed
            summaries.append(summary)
        except Exception as e:
            print(f"  [FAIL] {strat}: {e}", flush=True)
            import traceback; traceback.print_exc()
            summaries.append({"strategy": strat, "status": "error", "error": str(e)})

    # Final ranking table
    print("\n" + "=" * 78)
    print(f"  Prompt sweep — {args.candidate} — final ranking")
    print("=" * 78)
    print(f"  {'strategy':<22} {'composite':>10} {'cat_succ':>10} {'p95(ms)':>9}  status")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*9}  {'-'*15}")
    ranked = sorted(
        [s for s in summaries if "composite_mean" in s],
        key=lambda s: -s.get("composite_mean", 0),
    )
    for s in ranked:
        comp = s.get("composite_mean", 0)
        succ = s.get("category_success_mean", 0) * 100
        p95 = s.get("ttlt_p95_ms", 0)
        dq = "[!] DQ" if s.get("disqualifications") else "[ok]"
        print(f"  {s['strategy']:<22} {comp:>10.3f} {succ:>9.1f}% {p95:>9.0f}  {dq}")

    out = ROOT / "results" / f"sweep_{args.candidate}.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\nSweep summary: {out}")


if __name__ == "__main__":
    main()
