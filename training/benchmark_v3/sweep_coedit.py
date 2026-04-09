"""
Sweep multiple instruction prefixes against grammarly/coedit-large on the v3
benchmark corpus. Loads the model once and reuses it across all strategies
(model is ~3GB, reloading per run is wasteful).

CoEdIT was trained on a specific instruction set covering grammar correction,
clarity/conciseness, paraphrasing, simplification, and neutralization. We test
instructions from each family to find the closest match for spoken-text cleanup.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import scorers
import report

CORPUS_PATH = ROOT / "corpus" / "english_gold.jsonl"
RESULTS_DIR = ROOT / "results"
MODEL_ID = "grammarly/coedit-large"

# Instruction prefixes to test. Each is (strategy_name, instruction_text).
# Picked to cover CoEdIT's known training task families:
#   - grammar-fix (already benched: "Fix grammatical errors in this sentence:")
#   - conciseness / redundancy
#   - paraphrasing
#   - neutralization / formal rewriting
#   - off-distribution disfluency-removal asks
STRATEGIES = [
    ("coedit-remove-redundant", "Remove redundant words from this sentence:"),
    ("coedit-make-concise", "Make this sentence more concise:"),
    ("coedit-paraphrase", "Paraphrase this sentence:"),
    ("coedit-rewrite", "Rewrite this sentence:"),
    ("coedit-fix-disfluencies", "Remove filler words and disfluencies from this sentence:"),
    ("coedit-clean-spoken", "Clean up this spoken text:"),
]


def load_corpus():
    cases = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_strategy(model, tokenizer, device, cases, instruction, strategy_name):
    import torch

    print(f"\n=== {strategy_name} ===")
    print(f"Instruction: {instruction!r}")

    results = []
    t_start = time.time()
    for i, case in enumerate(cases):
        prompt = f"{instruction} {case['input']}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    do_sample=False,
                    repetition_penalty=1.2,
                )
            output = tokenizer.decode(out[0], skip_special_tokens=True)
            err = None
        except Exception as e:
            output = ""
            err = str(e)
        ttlt_ms = (time.time() - t0) * 1000

        score = scorers.score_case(case, output)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "reference": case["reference"],
            "output": output,
            "raw": output,
            "ttlt_ms": ttlt_ms,
            "scores": score,
            "error": err,
        })
        if (i + 1) % 50 == 0 or i + 1 == len(cases):
            avg = sum(r["scores"]["composite"] for r in results) / len(results)
            print(f"  [{i+1}/{len(cases)}] composite={avg:.3f}", flush=True)

    elapsed = time.time() - t_start

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / "coedit-large" / f"{strategy_name}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "candidate": "coedit-large",
        "strategy": strategy_name,
        "model_id": MODEL_ID,
        "instruction": instruction,
        "device": device,
        "num_beams": 4,
        "max_new_tokens": 256,
        "n_cases": len(results),
        "elapsed_seconds": elapsed,
        "timestamp": ts,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Results: {out_dir}")
    report.report(out_dir)
    return out_dir


def main():
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    cases = load_corpus()
    print(f"Loaded {len(cases)} cases", flush=True)

    for strategy_name, instruction in STRATEGIES:
        run_strategy(model, tokenizer, device, cases, instruction, strategy_name)

    print("\n\n========== SWEEP COMPLETE ==========")


if __name__ == "__main__":
    main()
