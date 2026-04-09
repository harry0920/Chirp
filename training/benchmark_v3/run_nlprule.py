"""
Benchmark nlprule (Rust port of LanguageTool's rule engine) on the v3 corpus.

Uses the nlprule-shim binary built from /tmp/nlprule-shim. The shim takes
file input via --input and writes JSON to stdout containing the corrected
text and the list of suggestions.

Usage:
    py run_nlprule.py
    py run_nlprule.py --self-correction-only
    py run_nlprule.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import scorers
import report

CORPUS_PATH = ROOT / "corpus" / "english_gold.jsonl"
RESULTS_DIR = ROOT / "results" / "nlprule"

# Built artifacts. MSYS /tmp = C:\Users\dutch\AppData\Local\Temp on Windows;
# Python runs in Windows context, so we use the Windows-form path.
import os
_TMP = os.environ.get("TEMP", r"C:\Users\dutch\AppData\Local\Temp")
SHIM_BIN = str(Path(_TMP) / "nlprule-target" / "release" / "nlprule-shim.exe")
TOKENIZER_BIN = str(Path(_TMP) / "nlprule-models" / "en_tokenizer.bin")
RULES_BIN = str(Path(_TMP) / "nlprule-models" / "en_rules.bin")

SELF_CORRECTION_CATEGORIES = {
    "explicit_self_correction",
    "implicit_self_correction",
    "cross_sentence_self_correction",
}


def load_corpus():
    cases = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def nlprule_correct(text: str) -> tuple[str, list[dict]]:
    """Run the shim against text. Returns (corrected_text, suggestion_list)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(text)
        tmp_path = tf.name
    try:
        proc = subprocess.run(
            [SHIM_BIN, "--tokenizer", TOKENIZER_BIN, "--rules", RULES_BIN, "--input", tmp_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"shim failed (rc={proc.returncode}): {proc.stderr}")
    if not proc.stdout.strip():
        return text, []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"shim output not JSON: {proc.stdout[:200]}") from e
    return data.get("corrected", text), data.get("suggestions", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--self-correction-only",
        action="store_true",
        help="Run only the 50 self-correction cases",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Optional tag for the results directory name",
    )
    args = ap.parse_args()

    # Sanity check
    if not Path(SHIM_BIN).exists():
        print(f"ERROR: shim binary not found at {SHIM_BIN}")
        sys.exit(1)
    if not Path(TOKENIZER_BIN).exists() or not Path(RULES_BIN).exists():
        print(f"ERROR: model files missing at {TOKENIZER_BIN} / {RULES_BIN}")
        sys.exit(1)

    cases = load_corpus()
    if args.self_correction_only:
        cases = [c for c in cases if c["category"] in SELF_CORRECTION_CATEGORIES]
        print(f"Filtered to {len(cases)} self-correction cases", flush=True)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Running {len(cases)} cases through nlprule-shim...", flush=True)

    results = []
    t_start = time.time()
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        try:
            out, suggs = nlprule_correct(case["input"])
            err = None
        except Exception as e:
            out = case["input"]  # fall back to passthrough on shim failure
            suggs = []
            err = str(e)
        ttlt_ms = (time.perf_counter() - t0) * 1000

        score = scorers.score_case(case, out)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "reference": case["reference"],
            "output": out,
            "raw": out,
            "suggestion_count": len(suggs),
            "ttlt_ms": ttlt_ms,
            "scores": score,
            "error": err,
        })
        if (i + 1) % 25 == 0 or i + 1 == len(cases):
            avg = sum(r["scores"]["composite"] for r in results) / len(results)
            print(f"  [{i+1}/{len(cases)}] composite={avg:.3f} ({ttlt_ms:.0f}ms)", flush=True)

    elapsed = time.time() - t_start

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base = "self-correction" if args.self_correction_only else "full"
    suffix = f"{base}-{args.tag}" if args.tag else base
    out_dir = RESULTS_DIR / f"{suffix}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "candidate": "nlprule",
        "strategy": suffix,
        "shim_binary": SHIM_BIN,
        "tokenizer": TOKENIZER_BIN,
        "rules": RULES_BIN,
        "n_cases": len(results),
        "elapsed_seconds": elapsed,
        "timestamp": ts,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults: {out_dir}\n", flush=True)
    report.report(out_dir)

    # Show cases where nlprule actually made an edit
    print("\n========== CASES WHERE NLPRULE EDITED ==========")
    edits = [r for r in results if r["output"] != r["input"]][:15]
    if not edits:
        print("(no edits made on any case)")
    for r in edits:
        marker = "OK " if r["scores"]["composite"] >= 0.85 else "BAD"
        print(f"\n[{marker}] {r['id']} ({r['category']}) sugg={r['suggestion_count']} comp={r['scores']['composite']:.2f}")
        print(f"  IN:   {r['input']}")
        print(f"  OUT:  {r['output']}")


if __name__ == "__main__":
    main()
