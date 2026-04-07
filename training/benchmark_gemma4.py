"""
Benchmark Gemma 4 E2B vs Qwen 2.5 3B on the same 50 ASR cleanup cases.

Both run via llama-server on GPU. Tests multiple prompt strategies for Gemma.

Usage:
    python benchmark_gemma4.py --model gemma
    python benchmark_gemma4.py --model qwen
    python benchmark_gemma4.py --model both
"""

import argparse
import json
import os
import subprocess
import time
import requests
from benchmark_enc_dec import BENCHMARK, score_result, datamark, QWEN_SYSTEM_PROMPT

LLAMA_SERVER = os.path.join(os.environ.get("APPDATA", ""), "com.chirp.app", "llm", "llama-server.exe")
QWEN_MODEL = os.path.join(os.environ.get("APPDATA", ""), "com.chirp.app", "llm", "qwen2.5-3b-instruct-q4_k_m.gguf")
GEMMA_MODEL = os.path.join(os.environ.get("APPDATA", ""), "com.chirp.app", "llm", "gemma-4-e2b-it-q4_k_m.gguf")

PORT = 9998

# ── Prompt strategies for Gemma 4 ───────────────────────────────────

GEMMA_PROMPTS = {
    "simple": {
        "system": "You are a speech-to-text cleanup tool. Rewrite dictated speech so it reads like typed text. Only fix errors - do not add or remove meaning. Output only the cleaned text, nothing else.",
        "user": "{text}",
    },
    "detailed": {
        "system": (
            "You clean up speech-to-text transcriptions. Rules:\n"
            "1. Resolve self-corrections: when the speaker says 'wait', 'no', 'I mean', 'actually', 'scratch that', discard the wrong part, keep ONLY the correction.\n"
            "2. Merge choppy 'And... And... And...' sentences into flowing prose.\n"
            "3. Remove stutters and repeated words.\n"
            "4. Convert spoken numbers to digits.\n"
            "5. Preserve the speaker's vocabulary. Do not add information they didn't say.\n"
            "Output ONLY the cleaned text. No JSON, no markdown, no commentary."
        ),
        "user": "{text}",
    },
    "json": {
        "system": QWEN_SYSTEM_PROMPT,
        "user": (
            "Clean up the following speech-to-text transcription. "
            "The text uses ^ as word separators. Remove the ^ markers, fix grammar, "
            "and output only the cleaned text.\n\n"
            "<transcription>\n{marked}\n</transcription>"
        ),
        "use_datamark": True,
        "use_json_schema": True,
    },
}


def start_server(model_path, port=PORT):
    """Start llama-server and wait for ready."""
    cmd = [
        LLAMA_SERVER,
        "--model", model_path,
        "--port", str(port),
        "--ctx-size", "2048",
        "--n-predict", "1024",
        "--gpu-layers", "99",
        "--flash-attn", "on",
        "--batch-size", "512",
        "--parallel", "1",
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=0x08000000)  # CREATE_NO_WINDOW

    for i in range(60):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.json().get("status") == "ok":
                print(f"  Server ready after {i+1}s (PID {proc.pid})")
                return proc
        except:
            pass

    proc.kill()
    raise RuntimeError("Server failed to start")


def stop_server(proc):
    proc.kill()
    proc.wait()


def run_benchmark(model_name, prompt_config, port=PORT):
    """Run 50-case benchmark with a given prompt configuration."""
    use_dm = prompt_config.get("use_datamark", False)
    use_json = prompt_config.get("use_json_schema", False)

    results = []
    for i, case in enumerate(BENCHMARK):
        text = case["input"]
        ideal = case["ideal"]

        if use_dm:
            marked = datamark(text)
            user_msg = prompt_config["user"].format(text=text, marked=marked)
        else:
            user_msg = prompt_config["user"].format(text=text)

        word_count = len(text.split())
        max_tokens = min(max(int(word_count * 2.5), 64), 1024)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": prompt_config["system"]},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if use_json:
            payload["response_format"] = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {"cleaned_text": {"type": "string"}},
                    "required": ["cleaned_text"],
                },
            }

        start = time.perf_counter()
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                json=payload, timeout=30)
            elapsed = time.perf_counter() - start

            raw = resp.json()["choices"][0]["message"]["content"].strip()

            if use_json:
                try:
                    parsed = json.loads(raw)
                    output = parsed.get("cleaned_text", raw).replace("^", " ").strip()
                except json.JSONDecodeError:
                    output = raw.replace("^", " ").strip()
            else:
                output = raw.strip()
        except Exception as e:
            elapsed = time.perf_counter() - start
            output = text
            print(f"  ERROR on case {i+1}: {e}")

        sc = score_result(output, ideal)
        ms = elapsed * 1000

        results.append({
            "category": case["category"],
            "score": sc,
            "time_ms": ms,
            "output": output,
            "input": text,
            "ideal": ideal,
        })

        sim = sc["similarity"]
        tag = "EXACT" if sc["exact"] else f"sim={sim:.2f}"
        print(f"  [{i+1:2d}] {case['category']:20s} {tag:12s} {ms:.0f}ms  {output[:60]}")

    return results


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

        if r["score"]["exact"]:
            key = "exact"
        elif sim >= 0.90:
            key = "close"
        elif sim >= 0.70:
            key = "ok"
        elif sim < 0.50:
            key = "halluc"
        else:
            key = "fail"

        totals[key] += 1
        by_cat[cat][key] += 1

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
    parser.add_argument("--model", default="both", choices=["gemma", "qwen", "both"])
    parser.add_argument("--gemma-prompt", default="all", choices=["simple", "detailed", "json", "all"])
    args = parser.parse_args()

    all_summaries = []

    # ── Gemma 4 E2B ──────────────────────────────────────────────────
    if args.model in ("gemma", "both"):
        if not os.path.exists(GEMMA_MODEL):
            print(f"ERROR: Gemma model not found at {GEMMA_MODEL}")
            return

        print(f"\nStarting Gemma 4 E2B server...")
        proc = start_server(GEMMA_MODEL)

        prompts_to_test = GEMMA_PROMPTS if args.gemma_prompt == "all" else {args.gemma_prompt: GEMMA_PROMPTS[args.gemma_prompt]}

        for pname, pconfig in prompts_to_test.items():
            print(f"\n{'='*70}")
            print(f"  GEMMA 4 E2B - prompt: '{pname}'")
            print(f"{'='*70}")
            results = run_benchmark("gemma", pconfig)
            totals, avg_sim, avg_time = print_summary(results, f"GEMMA 4 E2B ({pname})")
            all_summaries.append((f"Gemma 4 E2B ({pname})", totals, avg_sim, avg_time))

        stop_server(proc)
        time.sleep(2)

    # ── Qwen 2.5 3B ─────────────────────────────────────────────────
    if args.model in ("qwen", "both"):
        print(f"\nStarting Qwen 2.5 3B server...")
        proc = start_server(QWEN_MODEL)

        print(f"\n{'='*70}")
        print(f"  QWEN 2.5 3B (production prompt)")
        print(f"{'='*70}")
        results = run_benchmark("qwen", GEMMA_PROMPTS["json"])
        totals, avg_sim, avg_time = print_summary(results, "QWEN 2.5 3B (JSON+datamark)")
        all_summaries.append(("Qwen 2.5 3B", totals, avg_sim, avg_time))

        stop_server(proc)

    # ── Final comparison ─────────────────────────────────────────────
    if all_summaries:
        print(f"\n{'='*70}")
        print(f"  FINAL COMPARISON (all on GPU)")
        print(f"{'='*70}")
        print(f"  {'Model':<35s} {'Exact':>5s} {'Close':>5s} {'OK':>5s} {'Hall':>5s} {'Fail':>5s} {'Sim':>6s} {'ms':>6s}")
        print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")
        for name, totals, avg_sim, avg_time in all_summaries:
            print(f"  {name:<35s} {totals['exact']:>5d} {totals['close']:>5d} {totals['ok']:>5d} {totals['halluc']:>5d} {totals['fail']:>5d} {avg_sim:>6.3f} {avg_time:>6.0f}")


if __name__ == "__main__":
    main()
