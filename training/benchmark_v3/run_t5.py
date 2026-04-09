"""
Run the existing FLAN-T5-small fine-tune (from training/data/ct2_models/) against
the v3 benchmark corpus. This is the model from project_t5_finetuning.md —
77M params, CTranslate2 int8 runtime, prompt prefix "Rewrite as typed text: ".

The T5 fine-tune does not use the v2-fewshot-hard prompt — it has its own
trained prefix. So this is a fair "did the existing T5 model already solve
this?" check, not a prompt comparison.

Usage:
    python run_t5.py
    python run_t5.py --limit 10
    python run_t5.py --device cuda
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

# Model + tokenizer locations baked from training/ct2_server.py
MODEL_DIR = ROOT.parent / "data" / "ct2_models" / "flan-t5-small-finetuned-int8"
TOKENIZER_NAME = "sitelift/chirp-cleanup"
PREFIX = "Rewrite as typed text: "


def load_corpus():
    cases = []
    with CORPUS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--beam-size", type=int, default=4)
    args = ap.parse_args()

    import ctranslate2
    from transformers import AutoTokenizer

    device = args.device
    if device == "auto":
        try:
            import torch  # noqa: F401
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    print(f"Loading T5 from {MODEL_DIR} on {device} (beam={args.beam_size})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    translator = ctranslate2.Translator(
        str(MODEL_DIR),
        device=device,
        inter_threads=1,
        intra_threads=__import__("os").cpu_count() if device == "cpu" else 1,
    )

    cases = load_corpus()
    if args.limit:
        cases = cases[: args.limit]
    print(f"Running {len(cases)} cases...", flush=True)

    results = []
    t_start = time.time()
    for i, case in enumerate(cases):
        prompt = f"{PREFIX}{case['input']}"
        tokens = tokenizer(prompt, return_tensors="np")
        token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

        t0 = time.time()
        try:
            res = translator.translate_batch(
                [token_list],
                beam_size=args.beam_size,
                max_decoding_length=256,
                repetition_penalty=1.2,
            )
            output_tokens = res[0].hypotheses[0]
            output = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(output_tokens),
                skip_special_tokens=True,
            )
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

    # Save like other runs
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / "chirp-flan-t5-small" / f"trained-prompt-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "candidate": "chirp-flan-t5-small",
        "strategy": "trained-prompt",
        "model_path": str(MODEL_DIR),
        "tokenizer": TOKENIZER_NAME,
        "prefix": PREFIX,
        "device": device,
        "beam_size": args.beam_size,
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
