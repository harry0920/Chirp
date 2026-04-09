"""
Multi-stage pipeline viability spike.

Tests whether running a deletion-only stage 1 (regex) BEFORE the rewriting
stage 3 (CoEdIT-large) collapses CoEdIT's hallucination failure mode.

Measures four configurations on the v3 corpus with real per-stage timings:
  A. passthrough              — sanity baseline
  B. regex stage 1 only       — how far does aggressive regex get us?
  C. CoEdIT stage 3 only      — yesterday's 0.499 baseline (rerun for parity)
  D. regex -> CoEdIT pipeline — the actual spike question

Stage 2 (self-correction resolver) is intentionally NOT in this spike. It's
the unsolved hard step and we're testing the *architecture*, not stage 2.
Self-correction failures will show up as DQ on configs B and D — that's
expected and tells us how much of the gap stage 2 has to close.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import scorers
import report

CORPUS_PATH = ROOT / "corpus" / "english_gold.jsonl"
RESULTS_DIR = ROOT / "results" / "pipeline-spike"
COEDIT_MODEL_ID = "grammarly/coedit-large"
COEDIT_INSTRUCTION = "Fix grammatical errors in this sentence:"

# ---------------------------------------------------------------------------
# Stage 1: structural disfluency removal (regex only)
# ---------------------------------------------------------------------------
# Expanded vs production cleanup.rs: full filler list (not comma-gated),
# explicit stutter and false-start handling. This is intentionally more
# aggressive than what ships today, because the spike question is whether
# pre-cleaning helps stage 3 — not whether production regex is sufficient.

FILLER_PATTERNS = [
    r"\bum+\b",
    r"\buh+\b",
    r"\buh huh\b",
    r"\bmm+[ -]?hmm+\b",
    r"\bhmm+\b",
    r"\byou know\b",
    r"\bI mean\b",
    r"\bI guess\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bkinda\b",
    r"\bsorta\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bhonestly\b",
    r"\bliterally\b",
    r"\bfrankly\b",
    r"\bobviously\b",
    r"\banyway\b",
    # "like" only when used as filler (before pronoun/article/adv)
    r"\blike\b(?=\s+(?:the|a|an|i|we|they|he|she|it|my|our|this|that|really|just)\b)",
    # "well" / "so" only at sentence start
    r"(?:^|(?<=[.!?]\s))well,?\s+",
    r"(?:^|(?<=[.!?]\s))so,\s+",
    # "right?" tag question
    r",?\s*right\?",
]
FILLER_RE = re.compile("|".join(FILLER_PATTERNS), flags=re.IGNORECASE)

# Stutter: a word repeated immediately (case-insensitive). Apply iteratively.
STUTTER_RE = re.compile(r"\b(\w+)(\s+\1\b)+", flags=re.IGNORECASE)

# Word-level false start: a word followed by a hyphen and a space
# ("I tried- I attempted" -> "I attempted")
FALSE_START_RE = re.compile(r"\b\w+-\s+")

# Cleanup helpers
DOUBLE_SPACE_RE = re.compile(r"\s{2,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")
LEADING_COMMA_RE = re.compile(r"^[\s,]+")
DOUBLE_COMMA_RE = re.compile(r",\s*,")


def stage1_regex(text: str) -> str:
    """Aggressive deletion-only stage. Removes false starts, stutters, fillers."""
    if not text:
        return text
    out = text
    # Order matters: false starts first (they contain hyphens that confuse stutter regex)
    out = FALSE_START_RE.sub("", out)
    # Stutter removal — single pass handles arbitrary-length runs via the + quantifier
    out = STUTTER_RE.sub(r"\1", out)
    # Filler removal
    out = FILLER_RE.sub("", out)
    # Whitespace + punctuation cleanup
    out = DOUBLE_COMMA_RE.sub(",", out)
    out = SPACE_BEFORE_PUNCT_RE.sub(r"\1", out)
    out = DOUBLE_SPACE_RE.sub(" ", out)
    out = LEADING_COMMA_RE.sub("", out)
    out = out.strip()
    # Re-capitalize first letter (filler removal may have stripped a leading "Um,")
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


# ---------------------------------------------------------------------------
# Stage 3: CoEdIT-large grammar polish (loaded once, reused across configs)
# ---------------------------------------------------------------------------

class CoEditStage:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {COEDIT_MODEL_ID} on {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(COEDIT_MODEL_ID)
        self.model = T5ForConditionalGeneration.from_pretrained(COEDIT_MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def run(self, text: str) -> str:
        if not text:
            return text
        prompt = f"{COEDIT_INSTRUCTION} {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.2,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_corpus():
    cases = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_config(name: str, cases, transform, timing_buckets):
    """Run a config across all cases. transform(input_text) -> (output, per_stage_ms_dict)."""
    print(f"\n=== {name} ===", flush=True)
    results = []
    t0 = time.time()
    for i, case in enumerate(cases):
        out, stage_ms = transform(case["input"])
        score = scorers.score_case(case, out)
        # Total ttlt is sum of stage timings
        ttlt_ms = sum(stage_ms.values())
        for k, v in stage_ms.items():
            timing_buckets.setdefault(k, []).append(v)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "reference": case["reference"],
            "output": out,
            "raw": out,
            "stage_ms": stage_ms,
            "ttlt_ms": ttlt_ms,
            "scores": score,
            "error": None,
        })
        if (i + 1) % 50 == 0 or i + 1 == len(cases):
            avg = sum(r["scores"]["composite"] for r in results) / len(results)
            print(f"  [{i+1}/{len(cases)}] composite={avg:.3f}", flush=True)

    elapsed = time.time() - t0

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / f"{name}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "config": name,
        "n_cases": len(results),
        "elapsed_seconds": elapsed,
        "timestamp": ts,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Results: {out_dir}")
    report.report(out_dir)
    return out_dir


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((len(s) - 1) * p))
    return s[k]


def main():
    cases = load_corpus()
    print(f"Loaded {len(cases)} cases")

    coedit = CoEditStage()

    timings: dict = {}

    # Config A: passthrough
    def passthrough(text):
        t0 = time.perf_counter()
        out = text
        return out, {"passthrough": (time.perf_counter() - t0) * 1000}

    # Config B: regex only
    def regex_only(text):
        t0 = time.perf_counter()
        out = stage1_regex(text)
        return out, {"stage1_regex": (time.perf_counter() - t0) * 1000}

    # Config C: CoEdIT only
    def coedit_only(text):
        t0 = time.perf_counter()
        out = coedit.run(text)
        return out, {"stage3_coedit": (time.perf_counter() - t0) * 1000}

    # Config D: regex -> CoEdIT pipeline
    def pipeline(text):
        t1 = time.perf_counter()
        cleaned = stage1_regex(text)
        t2 = time.perf_counter()
        out = coedit.run(cleaned)
        t3 = time.perf_counter()
        return out, {
            "stage1_regex": (t2 - t1) * 1000,
            "stage3_coedit": (t3 - t2) * 1000,
        }

    run_config("A-passthrough", cases, passthrough, timings)
    run_config("B-regex-only", cases, regex_only, timings)
    run_config("C-coedit-only", cases, coedit_only, timings)
    run_config("D-regex-coedit", cases, pipeline, timings)

    # Latency summary across all configs
    print("\n\n========== STAGE LATENCY SUMMARY (real measurements) ==========")
    print(f"{'stage':<20} {'n':>6} {'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10}")
    print("-" * 60)
    for stage_name, vals in timings.items():
        print(
            f"{stage_name:<20} {len(vals):>6} "
            f"{percentile(vals, 0.50):>10.2f} "
            f"{percentile(vals, 0.95):>10.2f} "
            f"{percentile(vals, 0.99):>10.2f}"
        )

    print("\n========== SPIKE COMPLETE ==========")


if __name__ == "__main__":
    main()
