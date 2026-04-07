"""
Quick proof-of-concept: Decider/Editor pipeline with PRETRAINED FLAN-T5 (no fine-tuning).

Tests the architecture concept:
  1. Decider (flan-t5-small): classifies what needs fixing
  2. Editor (flan-t5-base): applies only the flagged fixes

Run:
    pip install transformers torch
    python benchmark_decider_editor.py
    python benchmark_decider_editor.py --device cuda   # if you have GPU
"""

import argparse
import time
from difflib import SequenceMatcher

from transformers import T5ForConditionalGeneration, AutoTokenizer

from benchmark_enc_dec import BENCHMARK, score_result

# ── Categories the decider can flag ──────────────────────────────────

CATEGORIES = [
    "fillers",          # um, uh, like, basically, so, you know
    "stutter",          # repeated words (we we, the the)
    "self-correction",  # "no wait", "actually", "scratch that"
    "merging",          # choppy "And ... And ... And" run-ons
    "grammar",          # gonna→going to, missing articles, agreement
    "punctuation",      # missing periods, commas, question marks, capitalization
    "proper-nouns",     # lowercase proper nouns (google → Google)
    "numbers",          # spoken numbers → digits (twenty three → 23)
]

# ─��� Decider prompt ──────────────���────────────────────────────────────

DECIDER_PROMPT = """Classify which issues are present in this speech transcription. Choose from: fillers, stutter, self-correction, merging, grammar, punctuation, proper-nouns, numbers. If the text is already clean, say "none".

Text: {text}

Issues:"""

# ���─ Editor prompt ────────────────��──────────────────────────────���────

EDITOR_PROMPT = """Fix the following issues in this speech transcription: {flags}. Only fix those specific issues — do not change anything else. Output only the corrected text.

Text: {text}

Corrected:"""

# ── Single-model baseline for comparison ─────────────────────────────

SINGLE_PROMPT = """Rewrite as typed text: {text}"""


def generate(model, tokenizer, prompt, max_new_tokens=256, device="cpu"):
    """Generate text from a T5 model."""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.strip(), elapsed


def run_decider(model, tokenizer, text, device="cpu"):
    """Run the decider to classify issues."""
    prompt = DECIDER_PROMPT.format(text=text)
    result, elapsed = generate(model, tokenizer, prompt, max_new_tokens=32, device=device)
    return result.lower().strip(), elapsed


def run_editor(model, tokenizer, text, flags, device="cpu"):
    """Run the editor to fix flagged issues."""
    prompt = EDITOR_PROMPT.format(text=text, flags=flags)
    result, elapsed = generate(model, tokenizer, prompt, max_new_tokens=256, device=device)
    return result, elapsed


def run_single(model, tokenizer, text, device="cpu"):
    """Run single-model baseline (just 'Rewrite as typed text')."""
    prompt = SINGLE_PROMPT.format(text=text)
    result, elapsed = generate(model, tokenizer, prompt, max_new_tokens=256, device=device)
    return result, elapsed


def print_score_table(results, label):
    """Print a score summary table."""
    by_cat = {}
    total_exact = total_close = total_ok = total_halluc = total_fail = 0
    total_sim = 0
    total_time = 0

    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"exact": 0, "close": 0, "ok": 0, "halluc": 0, "fail": 0, "count": 0}
        sc = r["score"]
        by_cat[cat]["count"] += 1
        total_time += r.get("time_ms", 0)

        sim = sc["similarity"]
        total_sim += sim

        if sc["exact"]:
            by_cat[cat]["exact"] += 1
            total_exact += 1
        elif sim >= 0.90:
            by_cat[cat]["close"] += 1
            total_close += 1
        elif sim >= 0.70:
            by_cat[cat]["ok"] += 1
            total_ok += 1
        elif sc.get("hallucination", False) or sim < 0.50:
            by_cat[cat]["halluc"] += 1
            total_halluc += 1
        else:
            by_cat[cat]["fail"] += 1
            total_fail += 1

    n = len(results)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  EXACT: {total_exact}  CLOSE: {total_close}  OK: {total_ok}  HALLUC: {total_halluc}  FAIL: {total_fail}")
    print(f"  Avg similarity: {total_sim/n:.3f}   Avg time: {total_time/n:.0f}ms")
    print()
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"  {cat:20s}  E:{c['exact']} C:{c['close']} O:{c['ok']} H:{c['halluc']} F:{c['fail']}  (n={c['count']})")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--skip-single", action="store_true", help="Skip single-model baseline")
    args = parser.parse_args()

    device = args.device

    # Load models
    print("Loading decider (flan-t5-small)...")
    dec_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    dec_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device).eval()

    print("Loading editor (flan-t5-base)...")
    ed_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    ed_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device).eval()

    # ── Run decider/editor pipeline ──────────────────────────────────
    print("\n" + "="*70)
    print("  DECIDER/EDITOR PIPELINE (zero-shot, no fine-tuning)")
    print("="*70)

    pipeline_results = []

    for i, case in enumerate(BENCHMARK):
        text = case["input"]
        ideal = case["ideal"]

        # Step 1: Decider
        flags, dec_time = run_decider(dec_model, dec_tok, text, device)

        # Step 2: Editor (or passthrough)
        if flags == "none" or not flags:
            output = text
            ed_time = 0
        else:
            output, ed_time = run_editor(ed_model, ed_tok, text, flags, device)

        total_ms = (dec_time + ed_time) * 1000
        sc = score_result(output, ideal)

        pipeline_results.append({
            "category": case["category"],
            "input": text,
            "ideal": ideal,
            "flags": flags,
            "output": output,
            "score": sc,
            "time_ms": total_ms,
        })

        # Show details
        sim = sc["similarity"]
        tag = "EXACT" if sc["exact"] else f"sim={sim:.2f}"
        print(f"\n[{i+1:2d}/{len(BENCHMARK)}] {case['category']}")
        print(f"  Input:  {text[:80]}{'...' if len(text)>80 else ''}")
        print(f"  Flags:  {flags}")
        print(f"  Output: {output[:80]}{'...' if len(output)>80 else ''}")
        print(f"  Ideal:  {ideal[:80]}{'...' if len(ideal)>80 else ''}")
        print(f"  Score:  {tag}  Time: {total_ms:.0f}ms")

    print_score_table(pipeline_results, "DECIDER/EDITOR PIPELINE (zero-shot)")

    # ─��� Run single-model baseline ────────────────────────────────────
    if not args.skip_single:
        print("\n" + "="*70)
        print("  SINGLE MODEL BASELINE (flan-t5-base, 'Rewrite as typed text')")
        print("="*70)

        single_results = []
        for i, case in enumerate(BENCHMARK):
            text = case["input"]
            ideal = case["ideal"]
            output, elapsed = run_single(ed_model, ed_tok, text, device)
            sc = score_result(output, ideal)

            single_results.append({
                "category": case["category"],
                "input": text,
                "ideal": ideal,
                "output": output,
                "score": sc,
                "time_ms": elapsed * 1000,
            })

            sim = sc["similarity"]
            tag = "EXACT" if sc["exact"] else f"sim={sim:.2f}"
            print(f"  [{i+1:2d}] {case['category']:20s} {tag:12s} {elapsed*1000:.0f}ms")

        print_score_table(single_results, "SINGLE MODEL BASELINE (flan-t5-base)")

    # ── Summary comparison ───────────────────────────────────────────
    print("\n" + "="*70)
    print("  COMPARISON")
    print("="*70)

    pipe_exact = sum(1 for r in pipeline_results if r["score"]["exact"])
    pipe_sim = sum(r["score"]["similarity"] for r in pipeline_results) / len(pipeline_results)
    pipe_time = sum(r["time_ms"] for r in pipeline_results) / len(pipeline_results)
    pipe_passthrough = sum(1 for r in pipeline_results if r["flags"] == "none" or not r["flags"])

    print(f"  Pipeline:  {pipe_exact}/50 exact, {pipe_sim:.3f} avg sim, {pipe_time:.0f}ms avg, {pipe_passthrough} passthrough")

    if not args.skip_single:
        sing_exact = sum(1 for r in single_results if r["score"]["exact"])
        sing_sim = sum(r["score"]["similarity"] for r in single_results) / len(single_results)
        sing_time = sum(r["time_ms"] for r in single_results) / len(single_results)
        print(f"  Single:    {sing_exact}/50 exact, {sing_sim:.3f} avg sim, {sing_time:.0f}ms avg")


if __name__ == "__main__":
    main()
