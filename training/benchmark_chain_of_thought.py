"""
Chain-of-thought T5 pipeline: analyze first, then fix.

Pass 1 (T5-base): "What's wrong with this transcription?" → natural language analysis
Pass 2 (T5-base): "Fix these issues: [analysis]. Text: [original]" → cleaned text

Also tests single-model baseline for comparison.

Run:
    python benchmark_chain_of_thought.py --device cpu
    python benchmark_chain_of_thought.py --device cuda
"""

import argparse
import time
from difflib import SequenceMatcher

from transformers import T5ForConditionalGeneration, AutoTokenizer
from benchmark_enc_dec import BENCHMARK, score_result

# ── Prompts ──────────────────────────────────────────────────────────

ANALYZE_PROMPT = """What problems exist in this speech transcription? List each issue briefly (filler words, repeated words, self-corrections, run-on sentences, missing punctuation, lowercase proper nouns, spoken numbers). If the text is already clean, say "no issues".

Text: {text}

Problems:"""

FIX_PROMPT = """Fix the following issues in this speech-to-text transcription: {analysis}

Only fix those issues. Do not add or remove meaning. Output only the corrected text.

Text: {text}

Corrected text:"""

SINGLE_PROMPT = """Rewrite as typed text: {text}"""

# ── Also test a few alternative single-model prompts ─────────────────

ALT_PROMPTS = {
    "rewrite": "Rewrite as typed text: {text}",
    "cleanup": "Clean up this dictated speech so it reads like typed text: {text}",
    "fix-grammar": "Fix the grammar: {text}",
    "polish": "Polish this speech transcription into clean written text. Only fix errors, do not change meaning: {text}",
}


def generate(model, tokenizer, prompt, max_new_tokens=256, device="cpu"):
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


def score_and_bucket(sc):
    """Return bucket label from score dict."""
    if sc["exact"]:
        return "EXACT"
    sim = sc["similarity"]
    if sim >= 0.90:
        return "CLOSE"
    if sim >= 0.70:
        return "OK"
    if sim < 0.50:
        return "HALLUC"
    return "FAIL"


def print_summary(results, label):
    by_cat = {}
    totals = {"exact": 0, "close": 0, "ok": 0, "halluc": 0, "fail": 0}
    total_sim = 0
    total_time = 0

    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"exact": 0, "close": 0, "ok": 0, "halluc": 0, "fail": 0, "count": 0}
        by_cat[cat]["count"] += 1
        total_time += r["time_ms"]
        sim = r["score"]["similarity"]
        total_sim += sim

        bucket = score_and_bucket(r["score"])
        key = bucket.lower()
        totals[key] = totals.get(key, 0) + 1
        by_cat[cat][key] = by_cat[cat].get(key, 0) + 1

    n = len(results)
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  EXACT: {totals['exact']}  CLOSE: {totals['close']}  OK: {totals['ok']}  HALLUC: {totals['halluc']}  FAIL: {totals['fail']}")
    print(f"  Avg similarity: {total_sim/n:.3f}   Avg time: {total_time/n:.0f}ms")
    print()
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"  {cat:20s}  E:{c['exact']} C:{c['close']} O:{c['ok']} H:{c.get('halluc',0)} F:{c.get('fail',0)}  (n={c['count']})")
    print()
    return totals, total_sim / n, total_time / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-alts", action="store_true", help="Skip alternative prompt tests")
    args = parser.parse_args()
    device = args.device

    print("Loading T5-base...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device).eval()

    # ── Chain-of-thought pipeline ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CHAIN-OF-THOUGHT (T5-base -> T5-base)")
    print(f"{'='*70}")

    cot_results = []
    for i, case in enumerate(BENCHMARK):
        text = case["input"]
        ideal = case["ideal"]

        # Pass 1: Analyze
        analysis, t1 = generate(model, tokenizer, ANALYZE_PROMPT.format(text=text),
                                max_new_tokens=64, device=device)

        # Pass 2: Fix (or passthrough)
        if "no issue" in analysis.lower() or "no problem" in analysis.lower() or "already clean" in analysis.lower():
            output = text
            t2 = 0
            passthrough = True
        else:
            output, t2 = generate(model, tokenizer, FIX_PROMPT.format(analysis=analysis, text=text),
                                  max_new_tokens=256, device=device)
            passthrough = False

        total_ms = (t1 + t2) * 1000
        sc = score_result(output, ideal)

        cot_results.append({
            "category": case["category"],
            "score": sc,
            "time_ms": total_ms,
            "analysis": analysis,
            "output": output,
            "passthrough": passthrough,
        })

        bucket = score_and_bucket(sc)
        pt = " [PASS]" if passthrough else ""
        print(f"\n[{i+1:2d}/{len(BENCHMARK)}] {case['category']}")
        print(f"  Input:    {text[:80]}{'...' if len(text)>80 else ''}")
        print(f"  Analysis: {analysis[:80]}{'...' if len(analysis)>80 else ''}")
        print(f"  Output:   {output[:80]}{'...' if len(output)>80 else ''}")
        print(f"  Ideal:    {ideal[:80]}{'...' if len(ideal)>80 else ''}")
        print(f"  Score:    {bucket} (sim={sc['similarity']:.2f})  Time: {total_ms:.0f}ms{pt}")

    cot_totals, cot_sim, cot_time = print_summary(cot_results, "CHAIN-OF-THOUGHT (analyze -> fix)")
    cot_passthrough = sum(1 for r in cot_results if r["passthrough"])
    print(f"  Passthrough: {cot_passthrough}/50")

    # ── Single model baseline ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SINGLE MODEL BASELINE ('Rewrite as typed text')")
    print(f"{'='*70}")

    single_results = []
    for i, case in enumerate(BENCHMARK):
        text = case["input"]
        ideal = case["ideal"]
        output, elapsed = generate(model, tokenizer, SINGLE_PROMPT.format(text=text),
                                   max_new_tokens=256, device=device)
        sc = score_result(output, ideal)
        single_results.append({
            "category": case["category"],
            "score": sc,
            "time_ms": elapsed * 1000,
        })
        bucket = score_and_bucket(sc)
        print(f"  [{i+1:2d}] {case['category']:20s} {bucket:6s} sim={sc['similarity']:.2f}  {elapsed*1000:.0f}ms")

    single_totals, single_sim, single_time = print_summary(single_results, "SINGLE MODEL ('Rewrite as typed text')")

    # ── Alternative prompts ──────────────────────────────────────────
    if not args.skip_alts:
        alt_summaries = {}
        for name, template in ALT_PROMPTS.items():
            if name == "rewrite":
                continue  # already tested above
            print(f"\n  Testing prompt: '{name}'...")
            alt_results = []
            for case in BENCHMARK:
                text = case["input"]
                ideal = case["ideal"]
                output, elapsed = generate(model, tokenizer, template.format(text=text),
                                           max_new_tokens=256, device=device)
                sc = score_result(output, ideal)
                alt_results.append({"category": case["category"], "score": sc, "time_ms": elapsed * 1000})

            totals, avg_sim, avg_time = print_summary(alt_results, f"PROMPT: '{name}'")
            alt_summaries[name] = (totals, avg_sim, avg_time)

    # ── Final comparison ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<35s} {'Exact':>5s} {'Close':>5s} {'OK':>5s} {'Hall':>5s} {'Fail':>5s} {'Sim':>6s} {'ms':>6s}")
    print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")

    print(f"  {'Chain-of-thought (analyze->fix)':<35s} {cot_totals['exact']:>5d} {cot_totals['close']:>5d} {cot_totals['ok']:>5d} {cot_totals['halluc']:>5d} {cot_totals['fail']:>5d} {cot_sim:>6.3f} {cot_time:>6.0f}")
    print(f"  {'Single: rewrite as typed text':<35s} {single_totals['exact']:>5d} {single_totals['close']:>5d} {single_totals['ok']:>5d} {single_totals['halluc']:>5d} {single_totals['fail']:>5d} {single_sim:>6.3f} {single_time:>6.0f}")

    if not args.skip_alts:
        for name, (totals, avg_sim, avg_time) in alt_summaries.items():
            print(f"  {'Single: ' + name:<35s} {totals['exact']:>5d} {totals['close']:>5d} {totals['ok']:>5d} {totals['halluc']:>5d} {totals['fail']:>5d} {avg_sim:>6.3f} {avg_time:>6.0f}")


if __name__ == "__main__":
    main()
