"""Quick Qwen 3B benchmark on the same 50 cases (GPU via llama-server)."""

import json
import time
import requests
from benchmark_enc_dec import BENCHMARK, score_result, datamark, QWEN_SYSTEM_PROMPT

PORT = 9998

results = []

for i, case in enumerate(BENCHMARK):
    text = case["input"]
    ideal = case["ideal"]
    marked = datamark(text)
    word_count = len(text.split())
    input_tokens = int(word_count * 1.3)
    max_tokens = min(max(input_tokens * 2, 64), 1024)

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
        parsed = json.loads(raw)
        output = parsed.get("cleaned_text", raw).replace("^", " ").strip()
    except Exception as e:
        elapsed = time.perf_counter() - start
        output = text
        print(f"  ERROR: {e}")

    sc = score_result(output, ideal)
    ms = elapsed * 1000

    results.append({
        "category": case["category"],
        "score": sc,
        "time_ms": ms,
        "output": output,
    })

    sim = sc["similarity"]
    tag = "EXACT" if sc["exact"] else f"sim={sim:.2f}"
    print(f"[{i+1:2d}] {case['category']:20s} {tag:12s} {ms:.0f}ms")

# Summary
print(f"\n{'='*70}")
print(f"  QWEN 2.5 3B (GPU, llama-server)")
print(f"{'='*70}")

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
    total_time += r["time_ms"]
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
    elif sim < 0.50:
        by_cat[cat]["halluc"] += 1
        total_halluc += 1
    else:
        by_cat[cat]["fail"] += 1
        total_fail += 1

n = len(results)
print(f"  EXACT: {total_exact}  CLOSE: {total_close}  OK: {total_ok}  HALLUC: {total_halluc}  FAIL: {total_fail}")
print(f"  Avg similarity: {total_sim/n:.3f}   Avg time: {total_time/n:.0f}ms")
print()
for cat in sorted(by_cat):
    c = by_cat[cat]
    print(f"  {cat:20s}  E:{c['exact']} C:{c['close']} O:{c['ok']} H:{c['halluc']} F:{c['fail']}  (n={c['count']})")
