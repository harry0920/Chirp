"""Quick test of additional Gemma 4 E2B prompts. Server must be running on port 9998."""
import json, time, sys, requests
from benchmark_enc_dec import BENCHMARK, score_result

PORT = 9998

PROMPTS = {
    "v1_explicit_corrections": {
        "system": (
            "You clean up speech-to-text transcriptions so they read like typed text.\n\n"
            "CRITICAL RULES:\n"
            "1. SELF-CORRECTIONS: When someone says 'wait', 'no', 'I mean', 'actually', "
            "'scratch that', 'never mind', 'or rather', 'sorry' to correct themselves, "
            "DELETE everything before the correction and keep ONLY the corrected version.\n"
            "2. Merge choppy sentences into flowing prose.\n"
            "3. Remove stutters and repeated words.\n"
            "4. Preserve the speaker's exact meaning. Do not add or remove information.\n\n"
            "Output ONLY the cleaned text. Nothing else."
        ),
        "user": "{text}",
    },
    "v2_examples": {
        "system": (
            "You clean up speech-to-text transcriptions. Examples:\n\n"
            "Input: The meeting is at 2. No wait, 3 PM.\n"
            "Output: The meeting is at 3 PM.\n\n"
            "Input: Send it to John. Actually, send it to Mike.\n"
            "Output: Send it to Mike.\n\n"
            "Input: I went to the store. And I got food. And I came home.\n"
            "Output: I went to the store, got food, and came home.\n\n"
            "Input: We we need to finish this.\n"
            "Output: We need to finish this.\n\n"
            "Input: The deployment went smoothly.\n"
            "Output: The deployment went smoothly.\n\n"
            "Rules: Resolve self-corrections (keep only the correction). "
            "Merge choppy sentences. Remove stutters. Preserve meaning exactly. "
            "Output ONLY the cleaned text."
        ),
        "user": "{text}",
    },
    "v3_minimal": {
        "system": "Rewrite this speech transcription as clean typed text. Resolve any self-corrections by keeping only the final version. Output only the result.",
        "user": "{text}",
    },
}

def run(prompt_name, config):
    results = []
    for i, case in enumerate(BENCHMARK):
        text = case["input"]
        ideal = case["ideal"]
        payload = {
            "model": "gemma",
            "messages": [
                {"role": "system", "content": config["system"]},
                {"role": "user", "content": config["user"].format(text=text)},
            ],
            "temperature": 0.0,
            "max_tokens": min(max(len(text.split()) * 3, 64), 1024),
            "stream": False,
        }
        start = time.perf_counter()
        try:
            resp = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=30)
            elapsed = time.perf_counter() - start
            output = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            elapsed = time.perf_counter() - start
            output = text
        sc = score_result(output, ideal)
        results.append({"category": case["category"], "score": sc, "time_ms": elapsed*1000, "output": output})

    # Summary
    totals = {"exact":0,"close":0,"ok":0,"halluc":0,"fail":0}
    total_sim = 0
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"exact":0,"close":0,"ok":0,"halluc":0,"fail":0,"count":0}
        by_cat[cat]["count"] += 1
        sim = r["score"]["similarity"]
        total_sim += sim
        if r["score"]["exact"]: key="exact"
        elif sim >= 0.90: key="close"
        elif sim >= 0.70: key="ok"
        elif sim < 0.50: key="halluc"
        else: key="fail"
        totals[key] += 1
        by_cat[cat][key] += 1

    n = len(results)
    avg_time = sum(r["time_ms"] for r in results) / n
    print(f"\n  {prompt_name}: E:{totals['exact']} C:{totals['close']} O:{totals['ok']} H:{totals['halluc']} F:{totals['fail']}  sim={total_sim/n:.3f}  {avg_time:.0f}ms")
    for cat in sorted(by_cat):
        c = by_cat[cat]
        print(f"    {cat:20s}  E:{c['exact']} C:{c['close']} O:{c['ok']} H:{c.get('halluc',0)} F:{c.get('fail',0)}  (n={c['count']})")
    return totals, total_sim/n

# Check server is up
try:
    r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
    assert r.json().get("status") == "ok"
except:
    print("ERROR: llama-server not running on port 9998. Start it first.")
    sys.exit(1)

for name, config in PROMPTS.items():
    run(name, config)
