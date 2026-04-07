"""
Benchmark sitelift/chirp-cleanup-v2 (Qwen3 0.6B fine-tune) vs Gemma 4 E2B (current production).

Both models run via llama-server on the same GPU, on the same 50 BENCHMARK cases
from benchmark_enc_dec.py, scored with the same score_result function.

For chirp-cleanup-v2 we test BOTH known training-data prompt styles since the
HF README is empty:
  - "simple": training_qwen.jsonl style (short system prompt, plain text in/out)
  - "production": Gemma's actual production system prompt (BASE_SYSTEM_PROMPT from llm.rs)

Gemma is benchmarked with its real production prompt + sampling settings.

Usage:
    python benchmark_chirp_v2.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from benchmark_enc_dec import BENCHMARK, score_result

LLM_DIR = Path(os.environ["APPDATA"]) / "com.chirp.app" / "llm"
LLAMA_SERVER = LLM_DIR / "llama-server.exe"
GEMMA_MODEL = LLM_DIR / "gemma-4-e2b-it-q4_k_m.gguf"
CHIRP_V2_MODEL = LLM_DIR / "chirp-cleanup-0.6b-q4_k_m.gguf"

PORT = 9998

# ─── Prompts ────────────────────────────────────────────────────────────────

# Mirrors src-tauri/src/llm.rs BASE_SYSTEM_PROMPT exactly (production Gemma).
GEMMA_PRODUCTION_PROMPT = """You clean up speech-to-text output. Make it read like the person typed it themselves. Preserve their voice and tone.

Remove:
- Filler sounds: um, uh, uh huh, hmm, mmhmm
- Filler "like" (but keep meaningful "like" as in "I like pizza" or "something like that")
- Stutters and repeated words

Keep exactly as spoken:
- Words like alright, yeah, so, also, I mean, basically, honestly
- Profanity
- The beginning of sentences -- never drop opening words
- The speaker's exact phrasing -- do not rephrase or restructure

Self-corrections: when someone says "X actually no Y" or "X wait Y" or "X no I mean Y", they are correcting X to Y. Remove X entirely and keep Y.

Format:
- Ordinal lists (first/second/third, number one/two/three) -> numbered list
- "New paragraph" or "new line" -> actual line break

Examples:
Input: So I think we should launch on Tuesday actually no Wednesday because um the design team needs one more day to finish the icons
Output: So I think we should launch on Wednesday because the design team needs one more day to finish the icons.

Input: Send it to John no I mean send it to Mike
Output: Send it to Mike.

Input: For the release we need to number one finish the migration number two update the docs and number three notify the customers
Output: For the release we need to:
1. Finish the migration
2. Update the docs
3. Notify the customers

Output only the cleaned text."""

# Mirrors training_qwen.jsonl system content — likely (or close to) what
# chirp-cleanup-v2 was trained with.
CHIRP_SIMPLE_PROMPT = (
    "Clean up dictated speech. Remove fillers, fix stutters, "
    "resolve self-corrections (keep only the final version). "
    "Output only the cleaned text."
)

# ─── Server lifecycle ───────────────────────────────────────────────────────

def start_server(model_path: Path, port: int = PORT) -> subprocess.Popen:
    cmd = [
        str(LLAMA_SERVER),
        "--model", str(model_path),
        "--port", str(port),
        "--ctx-size", "2048",
        "--n-predict", "1024",
        "--gpu-layers", "99",
        "--flash-attn", "on",
        "--batch-size", "512",
        "--parallel", "1",
        "--reasoning-budget", "0",   # disable thinking mode for both Gemma + Qwen3
        "--log-disable",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )
    for i in range(60):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.json().get("status") == "ok":
                print(f"  server ready after {i+1}s (pid {proc.pid})")
                return proc
        except Exception:
            pass
    proc.kill()
    raise RuntimeError(f"llama-server failed to start for {model_path.name}")


def stop_server(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait()
    time.sleep(1)


# ─── Single benchmark run ───────────────────────────────────────────────────

def run_benchmark(label: str, system_prompt: str, temperature: float = 0.3,
                  port: int = PORT) -> list[dict]:
    print(f"\n{'='*78}")
    print(f"  {label}")
    print(f"{'='*78}")

    results = []
    for i, case in enumerate(BENCHMARK):
        text = case["input"]
        ideal = case["ideal"]

        word_count = len(text.split())
        max_tokens = min(max(int(word_count * 2.5), 64), 1024)

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": max_tokens,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            resp = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload,
                timeout=60,
            )
            elapsed = time.perf_counter() - start
            output = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            elapsed = time.perf_counter() - start
            output = text
            print(f"  ERR case {i+1}: {e}")

        sc = score_result(output, ideal)
        ms = elapsed * 1000
        results.append({
            "category": case["category"],
            "input": text,
            "ideal": ideal,
            "output": output,
            "time_ms": ms,
            "exact": sc["exact"],
            "similarity": sc["similarity"],
            "grade": sc["grade"],
        })
        tag = "EXACT" if sc["exact"] else f"sim={sc['similarity']:.2f}"
        print(f"  [{i+1:2d}] {case['category']:18s} {sc['grade']:6s} {tag:10s} {ms:5.0f}ms  {output[:55]}")

    return results


# ─── Summary ────────────────────────────────────────────────────────────────

def summarise(label: str, results: list[dict]) -> dict:
    grades = {"EXACT": 0, "CLOSE": 0, "OK": 0, "FAIL": 0, "HALLUC": 0}
    total_sim, total_time = 0.0, 0.0
    for r in results:
        grades[r["grade"]] = grades.get(r["grade"], 0) + 1
        total_sim += r["similarity"]
        total_time += r["time_ms"]
    n = len(results)
    summary = {
        "label": label,
        "n": n,
        "exact": grades["EXACT"],
        "close": grades["CLOSE"],
        "ok": grades["OK"],
        "fail": grades["FAIL"],
        "halluc": grades["HALLUC"],
        "avg_similarity": total_sim / n,
        "avg_time_ms": total_time / n,
    }
    print(
        f"\n  {label}: "
        f"EXACT {summary['exact']}  CLOSE {summary['close']}  OK {summary['ok']}  "
        f"FAIL {summary['fail']}  HALLUC {summary['halluc']}  "
        f"sim {summary['avg_similarity']:.3f}  avg {summary['avg_time_ms']:.0f}ms"
    )
    return summary


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if not LLAMA_SERVER.exists():
        sys.exit(f"missing: {LLAMA_SERVER}")
    if not GEMMA_MODEL.exists():
        sys.exit(f"missing: {GEMMA_MODEL}")
    if not CHIRP_V2_MODEL.exists():
        sys.exit(f"missing: {CHIRP_V2_MODEL}")

    all_results = {}
    summaries = []

    # ── chirp-cleanup-v2 with simple training-style prompt ──
    print("\nstarting chirp-cleanup-v2 (Qwen3 0.6B)...")
    proc = start_server(CHIRP_V2_MODEL)
    try:
        results = run_benchmark(
            "chirp-cleanup-v2 (simple prompt)",
            CHIRP_SIMPLE_PROMPT,
            temperature=0.3,
        )
        all_results["chirp_v2_simple"] = results
        summaries.append(summarise("chirp-cleanup-v2 (simple)", results))

        results = run_benchmark(
            "chirp-cleanup-v2 (production prompt)",
            GEMMA_PRODUCTION_PROMPT,
            temperature=0.3,
        )
        all_results["chirp_v2_production"] = results
        summaries.append(summarise("chirp-cleanup-v2 (production)", results))
    finally:
        stop_server(proc)

    # ── Gemma 4 E2B with production prompt ──
    print("\nstarting gemma-4-e2b...")
    proc = start_server(GEMMA_MODEL)
    try:
        results = run_benchmark(
            "gemma-4-E2B-it (production prompt)",
            GEMMA_PRODUCTION_PROMPT,
            temperature=0.3,
        )
        all_results["gemma_production"] = results
        summaries.append(summarise("gemma-4-E2B (production)", results))
    finally:
        stop_server(proc)

    # ── Final comparison ──
    print(f"\n{'='*78}")
    print("  FINAL COMPARISON (RTX 4080, GPU, all 50 cases)")
    print(f"{'='*78}")
    print(f"  {'Model':<38s} {'EX':>4s} {'CL':>4s} {'OK':>4s} {'F':>4s} {'H':>4s} {'sim':>6s} {'ms':>7s}")
    print(f"  {'-'*38} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*7}")
    for s in summaries:
        print(
            f"  {s['label']:<38s} "
            f"{s['exact']:>4d} {s['close']:>4d} {s['ok']:>4d} "
            f"{s['fail']:>4d} {s['halluc']:>4d} "
            f"{s['avg_similarity']:>6.3f} {s['avg_time_ms']:>7.0f}"
        )

    out_path = Path(__file__).parent / "data" / "benchmark_chirp_v2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "results": all_results}, f, indent=2)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
