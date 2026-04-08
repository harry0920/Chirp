"""
Re-score existing per_case.jsonl files in-place using current scorers.

Useful when fixing a metric bug — avoids re-running the LLMs. Walks
results/ for any per_case.jsonl, recomputes scores from output strings,
overwrites the file, and regenerates summary.json via report.report().

Usage:
    python rescore.py                    # everything under results/
    python rescore.py --candidate qwen2.5-3b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import scorers
import report

CORPUS_PATH = ROOT / "corpus" / "english_gold.jsonl"
RESULTS_DIR = ROOT / "results"


def load_corpus_lookup():
    out = {}
    with CORPUS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def rescore_run(run_dir: Path, corpus: dict) -> None:
    pc_path = run_dir / "per_case.jsonl"
    if not pc_path.exists():
        return
    new_lines = []
    with pc_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            case = corpus.get(rec["id"])
            if case is None:
                new_lines.append(line)
                continue
            rec["scores"] = scorers.score_case(case, rec.get("output", ""))
            new_lines.append(json.dumps(rec, ensure_ascii=False))
    with pc_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    # Regenerate summary.json without printing the verbose report
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        report.report(run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None)
    args = ap.parse_args()

    corpus = load_corpus_lookup()
    pattern = "**/per_case.jsonl"
    base = RESULTS_DIR / args.candidate if args.candidate else RESULTS_DIR
    runs = sorted(base.glob(pattern))
    print(f"Re-scoring {len(runs)} run(s)")
    for pc in runs:
        run_dir = pc.parent
        print(f"  {run_dir.relative_to(RESULTS_DIR)}")
        rescore_run(run_dir, corpus)
    print("Done.")


if __name__ == "__main__":
    main()
