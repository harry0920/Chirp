"""
Benchmark Harper (rule-based grammar checker) on the v3 corpus.

Pipes each input through `harper-cli lint --format json` and applies all
suggestions to produce a corrected output, then scores against the v3 corpus.

By default runs the full 250 cases and reports the per-category breakdown.
Use --self-correction-only to focus on the 50 self-correction cases (the
specific question that motivated this run: can rule-based grammar handle
self-corrections at all?).

Usage:
    py run_harper.py
    py run_harper.py --self-correction-only
    py run_harper.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
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
RESULTS_DIR = ROOT / "results" / "harper"
HARPER_BIN = "harper-cli"  # must be on PATH after `cargo install`

# Rules that damage technical/proper-noun content. Identified empirically from
# the first Harper run by inspecting the JSON output of failing cases:
#   - SpellCheck: "SHA256" -> "SHA's", "KubeCon" -> "Rubicon", "EAGAIN" -> "Again"
#   - SplitWords: "errno" -> "err no", "syscall" -> "sys call", "oldco" -> "old co"
DAMAGING_RULES = ["SpellCheck", "SplitWords"]

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


# harper-cli formats suggestions as: 'Replace with: "actual text"' (smart quotes)
# We need the actual text. Smart quotes are U+201C / U+201D.
SUGGEST_RE = re.compile(r'^(?:Replace with|Insert):\s*[\u201c"](.*?)[\u201d"]\s*$')


def parse_suggestion(s: str) -> str | None:
    """Extract the literal replacement text from a harper-cli suggestion string."""
    m = SUGGEST_RE.match(s.strip())
    if m:
        return m.group(1)
    # Some suggestion kinds may be 'Remove the word' etc — treat as deletion
    if s.strip().lower().startswith("remove"):
        return ""
    return None


def harper_lint(text: str, ignore_rules: list[str] | None = None) -> list[dict]:
    """Run harper-cli on text via a temp file (stdin path is buggy in this version)."""
    # Use a per-call temp file. harper-cli's stdin parsing currently misses lints
    # that the file path catches — verified empirically on the same input.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(text)
        tmp_path = tf.name
    cmd = [HARPER_BIN, "lint", "--format", "json", "--no-color"]
    if ignore_rules:
        cmd += ["--ignore", ",".join(ignore_rules)]
    cmd += [tmp_path]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if proc.returncode not in (0, 1):
        raise RuntimeError(f"harper-cli failed (rc={proc.returncode}): {proc.stderr}")
    if not proc.stdout.strip():
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"harper-cli output not JSON: {proc.stdout[:200]}") from e

    # Output shape per harper-cli/src/lint.rs:
    #   { "files": [ { "file": "...", "lints": [ { rule, span: {char_start, char_end},
    #                                              suggestions: [str], ... } ] } ] }
    # We pipe stdin so there's exactly one "file" entry.
    lints = []
    if isinstance(data, list):
        # could be a flat list if there's only one file
        for entry in data:
            lints.extend(entry.get("lints", []))
    elif isinstance(data, dict):
        if "files" in data:
            for entry in data["files"]:
                lints.extend(entry.get("lints", []))
        elif "lints" in data:
            lints.extend(data["lints"])
    return lints


def apply_suggestions(text: str, lints: list[dict]) -> str:
    """Apply harper-cli suggestions to text. Reverse-order to keep char offsets stable."""
    if not lints:
        return text
    # Filter to lints with at least one suggestion
    actionable = []
    for lint in lints:
        suggestions = lint.get("suggestions") or []
        if not suggestions:
            continue
        span = lint.get("span") or {}
        cs = span.get("char_start")
        ce = span.get("char_end")
        if cs is None or ce is None:
            continue
        # Take the first suggestion. Parse out the literal replacement text.
        repl = parse_suggestion(suggestions[0])
        if repl is None:
            continue
        actionable.append((cs, ce, repl))
    # Sort descending by start so later edits don't shift earlier offsets
    actionable.sort(key=lambda x: x[0], reverse=True)
    out = text
    for cs, ce, repl in actionable:
        out = out[:cs] + repl + out[ce:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--self-correction-only",
        action="store_true",
        help="Run only the 50 self-correction cases",
    )
    ap.add_argument(
        "--ignore-damaging",
        action="store_true",
        help="Disable rules empirically known to damage technical/proper-noun content (SpellCheck, SplitWords)",
    )
    ap.add_argument(
        "--ignore",
        default=None,
        help="Custom comma-separated rule names to disable (overrides --ignore-damaging)",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Optional tag for the results directory name",
    )
    args = ap.parse_args()

    if args.ignore:
        ignore_rules = [r.strip() for r in args.ignore.split(",") if r.strip()]
    elif args.ignore_damaging:
        ignore_rules = list(DAMAGING_RULES)
    else:
        ignore_rules = None
    if ignore_rules:
        print(f"Ignoring Harper rules: {ignore_rules}", flush=True)

    # Sanity check that harper-cli is on PATH
    try:
        v = subprocess.run(
            [HARPER_BIN, "--version"], capture_output=True, text=True, check=True
        )
        print(f"Using {HARPER_BIN}: {v.stdout.strip()}", flush=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"ERROR: harper-cli not found on PATH. Install with:")
        print(f"  cargo install --git https://github.com/Automattic/harper.git harper-cli")
        sys.exit(1)

    cases = load_corpus()
    if args.self_correction_only:
        cases = [c for c in cases if c["category"] in SELF_CORRECTION_CATEGORIES]
        print(f"Filtered to {len(cases)} self-correction cases", flush=True)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Running {len(cases)} cases through harper-cli...", flush=True)

    results = []
    t_start = time.time()
    for i, case in enumerate(cases):
        t0 = time.perf_counter()
        try:
            lints = harper_lint(case["input"], ignore_rules=ignore_rules)
            out = apply_suggestions(case["input"], lints)
            err = None
        except Exception as e:
            out = case["input"]  # fall back to passthrough on harper failure
            lints = []
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
            "lint_count": len(lints),
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
        "candidate": "harper-cli",
        "strategy": suffix,
        "binary": HARPER_BIN,
        "n_cases": len(results),
        "elapsed_seconds": elapsed,
        "timestamp": ts,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults: {out_dir}\n", flush=True)
    report.report(out_dir)

    # Show 10 sample I/O pairs from self-correction cases regardless of mode,
    # because that's the diagnostic question
    print("\n========== SELF-CORRECTION CASE SAMPLES ==========")
    sc = [r for r in results if r["category"] in SELF_CORRECTION_CATEGORIES][:10]
    for r in sc:
        marker = "OK" if r["scores"]["composite"] >= 0.85 else "FAIL"
        print(f"\n[{marker}] {r['id']} ({r['category']}) lints={r['lint_count']}")
        print(f"  IN:   {r['input']}")
        print(f"  REF:  {r['reference']}")
        print(f"  OUT:  {r['output']}")


if __name__ == "__main__":
    main()
