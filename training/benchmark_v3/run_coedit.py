"""
Run grammarly/coedit-large against the v3 benchmark corpus.

CoEdIT is a T5-large fine-tune from Grammarly trained on instruction-style
text editing tasks. Unlike our internal FLAN-T5 fine-tune, it expects natural
language instructions like "Fix grammatical errors in this sentence: <text>".

This runner uses HuggingFace transformers directly (no CTranslate2 conversion).
Model is auto-downloaded to the HF cache on first run (~3GB for T5-large).

Usage:
    py run_coedit.py
    py run_coedit.py --limit 10
    py run_coedit.py --device cuda
    py run_coedit.py --instruction "Fix grammar and remove disfluencies:"
"""

from __future__ import annotations

import argparse
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
DEFAULT_INSTRUCTION = "Fix grammatical errors in this sentence:"


def load_corpus():
    cases = []
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument(
        "--strategy-name",
        default="coedit-fix-grammar",
        help="Label for the results subdirectory",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading {MODEL_ID} on {device} (beams={args.num_beams})", flush=True)
    print(f"Instruction prefix: {args.instruction!r}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    cases = load_corpus()
    if args.limit:
        cases = cases[: args.limit]
    print(f"Running {len(cases)} cases...", flush=True)

    results = []
    t_start = time.time()
    for i, case in enumerate(cases):
        prompt = f"{args.instruction} {case['input']}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
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
        if (i + 1) % 25 == 0 or i + 1 == len(cases):
            avg = sum(r["scores"]["composite"] for r in results) / len(results)
            print(f"  [{i+1}/{len(cases)}] composite={avg:.3f} ({ttlt_ms:.0f}ms)", flush=True)

    elapsed = time.time() - t_start

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / "coedit-large" / f"{args.strategy_name}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "candidate": "coedit-large",
        "strategy": args.strategy_name,
        "model_id": MODEL_ID,
        "instruction": args.instruction,
        "device": device,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "n_cases": len(results),
        "elapsed_seconds": elapsed,
        "timestamp": ts,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults: {out_dir}\n", flush=True)
    report.report(out_dir)


if __name__ == "__main__":
    main()
