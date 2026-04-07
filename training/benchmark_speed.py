"""
Speed-focused benchmark: T5 models with greedy decoding + ONNX Runtime vs Qwen.

Tests inference latency under realistic production conditions.
"""

import json
import os
import time
import subprocess
import requests
from pathlib import Path
from difflib import SequenceMatcher

# Reuse benchmark data from main benchmark
from benchmark_enc_dec import BENCHMARK, score_result, datamark, QWEN_SYSTEM_PROMPT

LLAMA_SERVER = os.path.join(os.environ.get("APPDATA", ""), "com.chirp.app", "llm", "llama-server.exe")
QWEN_MODEL = os.path.join(os.environ.get("APPDATA", ""), "com.chirp.app", "llm", "qwen2.5-3b-instruct-q4_k_m.gguf")
PORT = 9998

T5_MODELS = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
}

# Best prompt per model size from previous benchmark
PROMPT = "Clean up this dictated speech so it reads like typed text: {text}"
PROMPT_SMALL = "Fix the grammar: {text}"


def run_t5_greedy(model_name, hf_id, prompt_template):
    """T5 with greedy decoding (num_beams=1) on CUDA FP16."""
    import torch
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    use_cuda = torch.cuda.is_available()
    device_label = "FP16 CUDA" if use_cuda else "FP32 CPU"
    print(f"\n  {model_name} (transformers, greedy, {device_label})")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = T5ForConditionalGeneration.from_pretrained(hf_id)
    if use_cuda:
        model = model.cuda().half()
    model.eval()

    # Warmup
    inputs = tokenizer("Fix the grammar: hello world", return_tensors="pt", max_length=512, truncation=True)
    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32)

    results = []
    for bench in BENCHMARK:
        prompt = prompt_template.format(text=bench["input"])
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        if use_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        elapsed = time.perf_counter() - start

        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score = score_result(result_text, bench["ideal"])
        results.append({"output": result_text, "time_ms": elapsed * 1000, **score})

    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_t5_onnx(model_name, hf_id, prompt_template):
    """T5 via ONNX Runtime with CUDA, fallback to CPU."""
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from transformers import AutoTokenizer

    provider = "CUDAExecutionProvider"
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = ORTModelForSeq2SeqLM.from_pretrained(hf_id, export=True, provider=provider)
    except Exception:
        provider = "CPUExecutionProvider"
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = ORTModelForSeq2SeqLM.from_pretrained(hf_id, export=True, provider=provider)
    print(f"\n  {model_name} (ONNX Runtime, {provider})")

    # Warmup
    inputs = tokenizer("Fix the grammar: hello world", return_tensors="pt", max_length=512, truncation=True)
    model.generate(**inputs, max_new_tokens=32)

    results = []
    for bench in BENCHMARK:
        prompt = prompt_template.format(text=bench["input"])
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        start = time.perf_counter()
        outputs = model.generate(**inputs, max_new_tokens=256)
        elapsed = time.perf_counter() - start

        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score = score_result(result_text, bench["ideal"])
        results.append({"output": result_text, "time_ms": elapsed * 1000, **score})

    del model
    return results


def run_qwen():
    """Qwen 3B via llama-server."""
    print(f"\n  qwen-3b (llama-server, Q4_K_M, Vulkan)")

    # Warmup
    try:
        requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "Hello"}],
                  "max_tokens": 5, "stream": False}, timeout=10)
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
            resp = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=30)
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
        results.append({"output": result_text, "time_ms": elapsed * 1000, **score})

    return results


def summarize(name, results):
    times = sorted(r["time_ms"] for r in results)
    grades = [r["grade"] for r in results]
    exact = sum(1 for g in grades if g == "EXACT")
    close = sum(1 for g in grades if g == "CLOSE")
    ok = sum(1 for g in grades if g == "OK")
    halluc = sum(1 for g in grades if g == "HALLUC")
    fail = sum(1 for g in grades if g == "FAIL")
    avg_sim = sum(r["similarity"] for r in results) / len(results)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return {
        "name": name,
        "exact": exact, "close": close, "ok": ok, "halluc": halluc, "fail": fail,
        "avg_sim": avg_sim, "p50_ms": p50, "p95_ms": p95,
        "good": exact + close,
    }


def main():
    print("=" * 95)
    print("  SPEED BENCHMARK: Greedy Decoding + ONNX Runtime vs Qwen llama-server")
    print(f"  {len(BENCHMARK)} transcripts")
    print("=" * 95)

    all_summaries = []

    # Qwen baseline
    if os.path.exists(LLAMA_SERVER) and os.path.exists(QWEN_MODEL):
        print("\nStarting llama-server...")
        proc = subprocess.Popen(
            [LLAMA_SERVER, "--model", QWEN_MODEL, "--port", str(PORT),
             "--ctx-size", "1024", "--n-predict", "200", "--gpu-layers", "99",
             "--flash-attn", "on", "--batch-size", "512", "--parallel", "1",
             "--mlock", "--no-mmap", "--log-disable"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x08000000,
        )
        for _ in range(60):
            time.sleep(0.5)
            try:
                r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
                if r.json().get("status") == "ok":
                    print("  llama-server ready")
                    break
            except Exception:
                pass

        results = run_qwen()
        all_summaries.append(summarize("qwen-3b (llama-server Q4_K_M)", results))
        proc.kill(); proc.wait(); time.sleep(1)

    # T5 models: greedy + ONNX
    for model_key, hf_id in T5_MODELS.items():
        prompt = PROMPT_SMALL if model_key == "flan-t5-small" else PROMPT

        # Greedy (transformers)
        try:
            results = run_t5_greedy(model_key, hf_id, prompt)
            all_summaries.append(summarize(f"{model_key} (torch greedy FP16)", results))
        except Exception as e:
            print(f"  ERROR: {e}")

        # ONNX Runtime
        try:
            results = run_t5_onnx(model_key, hf_id, prompt)
            all_summaries.append(summarize(f"{model_key} (ONNX CUDA)", results))
        except Exception as e:
            print(f"  ONNX ERROR: {e}")

    # Final table
    print(f"\n{'=' * 95}")
    print(f"  RESULTS")
    print(f"{'=' * 95}")
    print(f"  {'Model':<42} {'EXACT':>5} {'GOOD':>5} {'HALL':>5} {'FAIL':>5} {'AvgSim':>7} {'p50ms':>7} {'p95ms':>7}")
    print(f"  {'-' * 90}")
    for s in all_summaries:
        print(f"  {s['name']:<42} {s['exact']:>5} {s['good']:>5} {s['halluc']:>5} {s['fail']:>5} {s['avg_sim']:>7.3f} {s['p50_ms']:>6.0f}ms {s['p95_ms']:>6.0f}ms")
    print()


if __name__ == "__main__":
    main()
