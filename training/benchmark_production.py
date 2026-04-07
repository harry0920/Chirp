"""
Production-representative benchmark: All models on CPU.

This benchmarks what users will actually experience on laptops with integrated GPUs.
Every model runs through a production-grade runtime, not Python transformers.

Runtimes:
  - CTranslate2 (INT8 quantized) for T5 models — C++ inference, no Python loop
  - llama-server on CPU for Qwen — same binary Chirp ships, but without GPU offload

Usage:
    python benchmark_production.py
    python benchmark_production.py --skip-qwen
    python benchmark_production.py --only flan-t5-small flan-t5-base
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import ctranslate2
import requests
from transformers import AutoTokenizer

from benchmark_enc_dec import BENCHMARK, score_result, datamark, QWEN_SYSTEM_PROMPT

LLAMA_SERVER = os.path.join(os.environ.get("APPDATA", ""), "com.chirp.app", "llm", "llama-server.exe")
QWEN_MODEL = os.path.join(os.environ.get("APPDATA", ""), "com.chirp.app", "llm", "qwen2.5-3b-instruct-q4_k_m.gguf")
PORT = 9998

CT2_DIR = Path("data/ct2_models")

T5_MODELS = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
}

PROMPT_MAP = {
    "flan-t5-small": "Fix the grammar: {text}",
    "flan-t5-base": "Clean up this dictated speech so it reads like typed text: {text}",
    "flan-t5-large": "Clean up this dictated speech so it reads like typed text: {text}",
}


# ── CTranslate2 model conversion ─────────────────────────────────────

def convert_to_ct2(hf_id: str, output_dir: Path, quantization: str = "int8"):
    """Convert HuggingFace T5 model to CTranslate2 format with quantization."""
    if output_dir.exists() and (output_dir / "model.bin").exists():
        print(f"    Already converted: {output_dir}")
        return

    print(f"    Converting {hf_id} -> CT2 ({quantization})...")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    from transformers import T5ForConditionalGeneration
    # Monkey-patch to fix transformers/ctranslate2 version incompatibility
    _orig = T5ForConditionalGeneration.from_pretrained.__func__
    def _patched(cls, *args, **kwargs):
        kwargs.pop('dtype', None)
        return _orig(cls, *args, **kwargs)
    T5ForConditionalGeneration.from_pretrained = classmethod(_patched)

    converter = ctranslate2.converters.TransformersConverter(hf_id)
    converter.convert(str(output_dir), quantization=quantization, force=True)
    print(f"    Done. Size: {sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1e6:.0f} MB")


# ── CTranslate2 T5 inference ─────────────────────────────────────────

def run_t5_ct2(model_name: str, hf_id: str, ct2_path: Path, prompt_template: str):
    """Run benchmark with CTranslate2 on CPU."""
    print(f"\n  {model_name} (CTranslate2 INT8, CPU)")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    translator = ctranslate2.Translator(str(ct2_path), device="cpu", inter_threads=1, intra_threads=os.cpu_count())

    # Warmup
    warmup_tokens = tokenizer("Fix the grammar: hello world", return_tensors="np")
    warmup_input = tokenizer.convert_ids_to_tokens(warmup_tokens["input_ids"][0])
    translator.translate_batch([warmup_input], max_decoding_length=32)

    results = []
    for bench in BENCHMARK:
        prompt = prompt_template.format(text=bench["input"])
        input_tokens = tokenizer(prompt, return_tensors="np")
        token_list = tokenizer.convert_ids_to_tokens(input_tokens["input_ids"][0])

        start = time.perf_counter()
        output = translator.translate_batch(
            [token_list],
            max_decoding_length=256,
            beam_size=1,  # greedy for speed
            repetition_penalty=1.2,
        )
        elapsed = time.perf_counter() - start

        output_tokens = output[0].hypotheses[0]
        result_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )
        score = score_result(result_text, bench["ideal"])
        results.append({
            "output": result_text,
            "time_ms": elapsed * 1000,
            "category": bench["category"],
            **score,
        })

    del translator
    return results


def run_t5_ct2_beam(model_name: str, hf_id: str, ct2_path: Path, prompt_template: str):
    """Run benchmark with CTranslate2 on CPU with beam_size=4."""
    print(f"\n  {model_name} (CTranslate2 INT8, CPU, beam=4)")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    translator = ctranslate2.Translator(str(ct2_path), device="cpu", inter_threads=1, intra_threads=os.cpu_count())

    # Warmup
    warmup_tokens = tokenizer("Fix the grammar: hello world", return_tensors="np")
    warmup_input = tokenizer.convert_ids_to_tokens(warmup_tokens["input_ids"][0])
    translator.translate_batch([warmup_input], max_decoding_length=32, beam_size=4)

    results = []
    for bench in BENCHMARK:
        prompt = prompt_template.format(text=bench["input"])
        input_tokens = tokenizer(prompt, return_tensors="np")
        token_list = tokenizer.convert_ids_to_tokens(input_tokens["input_ids"][0])

        start = time.perf_counter()
        output = translator.translate_batch(
            [token_list],
            max_decoding_length=256,
            beam_size=4,
            repetition_penalty=1.2,
        )
        elapsed = time.perf_counter() - start

        output_tokens = output[0].hypotheses[0]
        result_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )
        score = score_result(result_text, bench["ideal"])
        results.append({
            "output": result_text,
            "time_ms": elapsed * 1000,
            "category": bench["category"],
            **score,
        })

    del translator
    return results


# ── Qwen via llama-server on CPU ─────────────────────────────────────

def run_qwen_cpu():
    """Run Qwen 3B via llama-server with 0 GPU layers (pure CPU)."""
    print(f"\n  qwen-3b (llama-server Q4_K_M, CPU only)")

    # Warmup
    try:
        requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "Hello"}],
                  "max_tokens": 5, "stream": False}, timeout=30)
    except Exception:
        pass

    results = []
    for bench in BENCHMARK:
        marked = datamark(bench["input"])
        input_words = len(bench["input"].split())
        max_tokens = min(int(input_words * 2) + 30, 1024)

        payload = {
            "model": "qwen",
            "messages": [
                {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "Clean up the following speech-to-text transcription. "
                    "The text uses ^ as word separators. Remove the ^ markers, fix grammar, "
                    "and output only the cleaned text.\n\n"
                    f"<transcription>\n{marked}\n</transcription>"
                )},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"cleaned_text": {"type": "string"}},
                    "required": ["cleaned_text"],
                },
            },
        }

        start = time.perf_counter()
        try:
            resp = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=60)
            elapsed = time.perf_counter() - start
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            try:
                result_text = json.loads(raw)["cleaned_text"]
            except (json.JSONDecodeError, KeyError):
                result_text = raw
            result_text = result_text.replace("^", " ")
            result_text = " ".join(result_text.split())
        except Exception as e:
            elapsed = time.perf_counter() - start
            result_text = f"[ERROR: {e}]"

        score = score_result(result_text, bench["ideal"])
        results.append({
            "output": result_text,
            "time_ms": elapsed * 1000,
            "category": bench["category"],
            **score,
        })

    return results


# ── Reporting ─────────────────────────────────────────────────────────

def summarize(name, results, disk_mb=None):
    times = sorted(r["time_ms"] for r in results)
    grades = [r["grade"] for r in results]
    exact = sum(1 for g in grades if g == "EXACT")
    close = sum(1 for g in grades if g == "CLOSE")
    ok = sum(1 for g in grades if g == "OK")
    halluc = sum(1 for g in grades if g == "HALLUC")
    fail = sum(1 for g in grades if g == "FAIL")
    avg_sim = sum(r["similarity"] for r in results) / len(results)

    # Per-category
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = []
        cats[c].append(r["grade"])

    return {
        "name": name,
        "exact": exact, "close": close, "ok": ok, "halluc": halluc, "fail": fail,
        "good": exact + close,
        "avg_sim": avg_sim,
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(len(times) * 0.95)],
        "min_ms": times[0],
        "max_ms": times[-1],
        "disk_mb": disk_mb,
        "categories": cats,
    }


def print_table(summaries):
    print(f"\n{'=' * 110}")
    print(f"  PRODUCTION BENCHMARK RESULTS (all CPU)")
    print(f"{'=' * 110}")
    print(f"  {'Model':<45} {'EXACT':>5} {'GOOD':>5} {'HALL':>5} {'FAIL':>5} {'Sim':>5} {'p50':>7} {'p95':>7} {'Disk':>7}")
    print(f"  {'-' * 105}")
    for s in summaries:
        disk = f"{s['disk_mb']:.0f}MB" if s['disk_mb'] else "?"
        print(f"  {s['name']:<45} {s['exact']:>5} {s['good']:>5} {s['halluc']:>5} {s['fail']:>5}"
              f" {s['avg_sim']:>5.3f} {s['p50_ms']:>6.0f}ms {s['p95_ms']:>6.0f}ms {disk:>7}")

    # Per-category breakdown for top models
    print(f"\n  Per-category grades:")
    print(f"  {'Model':<30} {'self_corr':<12} {'merge':<10} {'stutter':<10} {'question':<10} {'proper':<10} {'passthru':<10} {'mixed':<12}")
    print(f"  {'-' * 105}")
    for s in summaries:
        cats = s["categories"]
        def cat_summary(cat):
            if cat not in cats:
                return "-"
            grades = cats[cat]
            e = sum(1 for g in grades if g == "EXACT")
            c = sum(1 for g in grades if g == "CLOSE")
            return f"{e}E {c}C/{len(grades)}"

        print(f"  {s['name']:<30} {cat_summary('self_correction'):<12} {cat_summary('sentence_merging'):<10}"
              f" {cat_summary('stutter'):<10} {cat_summary('question'):<10} {cat_summary('proper_nouns'):<10}"
              f" {cat_summary('passthrough'):<10} {cat_summary('mixed'):<12}")
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    models = args.only or list(T5_MODELS.keys())
    summaries = []

    print("=" * 110)
    print("  PRODUCTION BENCHMARK")
    print("  Target: laptop with integrated GPU (all inference on CPU)")
    print(f"  T5 runtime: CTranslate2 with INT8 quantization")
    print(f"  Qwen runtime: llama-server with --gpu-layers 0 (CPU only)")
    print(f"  Models: {', '.join(models)}")
    print("=" * 110)

    # ── Convert T5 models to CT2 ──
    print("\n--- Model Conversion ---")
    for model_key in models:
        hf_id = T5_MODELS[model_key]
        ct2_path = CT2_DIR / f"{model_key}-int8"
        convert_to_ct2(hf_id, ct2_path, "int8")

    # ── Qwen on CPU ──
    if not args.skip_qwen and os.path.exists(LLAMA_SERVER) and os.path.exists(QWEN_MODEL):
        print("\n--- Qwen 3B (CPU) ---")
        print("  Starting llama-server with --gpu-layers 0...")
        proc = subprocess.Popen(
            [LLAMA_SERVER, "--model", QWEN_MODEL, "--port", str(PORT),
             "--ctx-size", "1024", "--n-predict", "200", "--gpu-layers", "0",
             "--flash-attn", "on", "--batch-size", "512", "--parallel", "1",
             "--log-disable"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x08000000,
        )
        for _ in range(120):
            time.sleep(0.5)
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
                if r.json().get("status") == "ok":
                    print("  llama-server ready (CPU mode)")
                    break
            except Exception:
                pass
        else:
            print("  WARNING: llama-server may not be ready")

        results = run_qwen_cpu()
        model_size = os.path.getsize(QWEN_MODEL) / 1e6
        summaries.append(summarize("qwen-3b (llama-server CPU)", results, model_size))
        proc.kill(); proc.wait(); time.sleep(1)

    # ── T5 models via CTranslate2 ──
    for model_key in models:
        hf_id = T5_MODELS[model_key]
        ct2_path = CT2_DIR / f"{model_key}-int8"
        prompt = PROMPT_MAP.get(model_key, PROMPT_MAP["flan-t5-base"])

        disk_mb = sum(f.stat().st_size for f in ct2_path.rglob("*") if f.is_file()) / 1e6

        print(f"\n--- {model_key} (CT2 INT8) ---")

        # Greedy
        results = run_t5_ct2(model_key, hf_id, ct2_path, prompt)
        summaries.append(summarize(f"{model_key} (CT2 INT8 greedy)", results, disk_mb))

        # Beam=4
        results = run_t5_ct2_beam(model_key, hf_id, ct2_path, prompt)
        summaries.append(summarize(f"{model_key} (CT2 INT8 beam=4)", results, disk_mb))

    # ── Results ──
    print_table(summaries)

    # Save
    output_path = Path("data/benchmark_production_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([{k: v for k, v in s.items()} for s in summaries], f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
