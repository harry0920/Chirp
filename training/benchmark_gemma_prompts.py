"""
Benchmark different system prompts and sampling parameters for Gemma 4 E2B cleanup.

Usage:
    # Start llama-server first (or let the script start it):
    python training/benchmark_gemma_prompts.py

    # If llama-server is already running on a specific port:
    python training/benchmark_gemma_prompts.py --port 8080

    # Start llama-server automatically:
    python training/benchmark_gemma_prompts.py --start-server
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

# ── Test cases ──────────────────────────────────────────────────────────
# Each test case has raw ASR input and an ideal output for comparison.

TEST_CASES = [
    {
        "name": "technical_with_misheard_names",
        "input": "We just swapped out clen 2.53b for Gemma for E2B. And immediately we were getting better results with lower inference time. Something that we were exploring is integrating a new feature for Cherp to automatically screenshot and process which type of application you are working. in that way it can write out the words that you are saying with the proper formatting so for example if you are in an email client then it will detect that and ensure that the formatted output is made for an email client because Gemma has multimodal capacity where it can see images. We were exploring if Gemma could be used for that application. And at what point in the pipeline would it make sense for that step to go of the detection of which program you're working in.",
        "ideal": "We just swapped out Qwen 2.5 3B for Gemma 4 E2B. Immediately we were getting better results with lower inference time. We're exploring integrating a new feature for Chirp to automatically screenshot and detect which application the user is working in, so the formatted output matches the context — for example, if you're in an email client, it formats the output accordingly. Since Gemma has multimodal capabilities including image understanding, we're evaluating whether it could handle this app detection step and where in the pipeline it should go.",
        "dictionary": ["Chirp", "Qwen", "Gemma", "Parakeet"],
    },
    {
        "name": "self_correction_and_fillers",
        "input": "So I was thinking we should um we should probably move the the database to um actually no not the database the the cache layer to Redis because it's it's much faster and you know it handles concurrent connections better than what we have right now.",
        "ideal": "I was thinking we should move the cache layer to Redis because it's much faster and handles concurrent connections better than what we have right now.",
        "dictionary": [],
    },
    {
        "name": "garbled_ending",
        "input": "The main thing I want to focus on is making sure that the API endpoints are properly authenticated and that we have rate limiting in place because right now anyone can just hit the endpoint and there's no there's no protection against that so we need to add some kind of middleware that checks for valid tokens and also tracks how many requests each user is making per minute or per hour or whatever we decide on.",
        "ideal": "The main thing I want to focus on is making sure the API endpoints are properly authenticated and have rate limiting in place. Right now anyone can hit the endpoint with no protection, so we need to add middleware that checks for valid tokens and tracks how many requests each user is making per minute or hour.",
        "dictionary": [],
    },
    {
        "name": "repeated_ideas",
        "input": "We need to improve the onboarding flow. The onboarding experience right now is not great. Users are dropping off during onboarding. So basically the whole onboarding thing needs to be reworked. I think we should start with the first screen, the welcome screen, and make it simpler.",
        "ideal": "We need to rework the onboarding flow. Users are dropping off, so we should start by simplifying the welcome screen.",
        "dictionary": [],
    },
    {
        "name": "email_style",
        "input": "Hey Sarah I wanted to follow up on the um the meeting we had yesterday about the Q3 roadmap I think we should prioritize the mobile app redesign over the the analytics dashboard because our mobile users are growing much faster and the current app is really showing its age. Let me know what you think. Thanks John",
        "ideal": "Hey Sarah,\n\nI wanted to follow up on the meeting we had yesterday about the Q3 roadmap. I think we should prioritize the mobile app redesign over the analytics dashboard because our mobile users are growing much faster and the current app is really showing its age.\n\nLet me know what you think.\n\nThanks,\nJohn",
        "dictionary": [],
        "tone": "email",
    },
    {
        "name": "short_simple",
        "input": "Can you send me the report by Friday.",
        "ideal": "Can you send me the report by Friday?",
        "dictionary": [],
    },
    {
        "name": "numbers_and_stats",
        "input": "So our conversion rate went from like 2.3 percent to 4.7 percent after we made those changes which is basically a hundred percent improvement and the um the average order value also went up by about fifteen dollars.",
        "ideal": "Our conversion rate went from 2.3% to 4.7% after we made those changes, basically a 100% improvement. The average order value also went up by about $15.",
        "dictionary": [],
    },
    {
        "name": "complex_technical",
        "input": "The the problem with the current architecture is that we're using a monolithic approach where everything is in one big service and when one part fails it takes down the whole thing. So what I'm proposing is we break it into microservices, specifically we'd have a separate service for authentication, one for user management, one for the API gateway, and then the core business logic would be its own service. Each one would communicate through message queues, probably RabbitMQ or maybe Kafka depending on on the throughput requirements.",
        "ideal": "The problem with the current architecture is that we're using a monolithic approach where everything is in one service, and when one part fails it takes down the whole thing. I'm proposing we break it into microservices: separate services for authentication, user management, the API gateway, and core business logic. Each would communicate through message queues — probably RabbitMQ or Kafka depending on throughput requirements.",
        "dictionary": [],
    },
]


# ── System prompt variants ──────────────────────────────────────────────

PROMPT_VARIANTS = {
    "v1_original": (
        "You are a speech-to-text cleanup tool. Rewrite dictated speech so it reads like typed text. "
        "Remove filler words (um, uh, like, you know, basically, so). Remove stutters and repeated words. "
        "Resolve self-corrections by keeping only the corrected version. Merge choppy sentences into flowing prose. "
        "Preserve the speaker's meaning exactly. Output only the cleaned text, nothing else."
    ),
    "v2_transformation": (
        "You are a speech-to-text post-processor. The input is raw ASR output that may contain "
        "recognition errors, stutters, filler words, self-corrections, and spoken-style phrasing. "
        "Rewrite it as if the person had typed it directly — natural written prose, not speech.\n\n"
        "Rules:\n"
        "- Fix obvious ASR misrecognitions using surrounding context\n"
        "- Remove any remaining filler words, stutters, and false starts\n"
        "- When the speaker corrects themselves, keep only the correction\n"
        "- Restructure fragmented or garbled sentences into clear written prose\n"
        "- Merge redundant repetitions of the same idea\n"
        "- Preserve the speaker's meaning exactly — do not add, editorialize, or summarize\n"
        "- Output only the cleaned text, nothing else\n\n"
        "Example:\n"
        "Input: We were um looking at the the new model and it's basically it's really fast actually no "
        "it's not that fast but it's faster than what we had before\n"
        "Output: We were looking at the new model. It's faster than what we had before, though not extremely fast."
    ),
    "v3_concise": (
        "Rewrite this speech-to-text output as clean typed text. Fix recognition errors from context, "
        "remove fillers/stutters, resolve self-corrections (keep only the final version), merge repeated ideas, "
        "and restructure into clear written prose. Preserve meaning exactly. Output only the cleaned text."
    ),
    "v4_strict_fidelity": (
        "You clean up speech-to-text output. Your job is to make it read like the person typed it, "
        "while preserving every piece of information they communicated.\n\n"
        "Do:\n"
        "- Fix ASR errors (misheard words) using context clues\n"
        "- Remove filler words (um, uh, like, you know, so, basically)\n"
        "- Remove stutters and word repetitions\n"
        "- Keep only the corrected version when someone corrects themselves\n"
        "- Combine fragmented sentences into clear prose\n"
        "- Deduplicate when the same idea is stated multiple times\n\n"
        "Do not:\n"
        "- Add information the speaker didn't say\n"
        "- Summarize or shorten the content beyond removing noise\n"
        "- Change the speaker's word choices (except fixing ASR errors)\n"
        "- Add formatting, headers, or bullet points\n\n"
        "Output only the cleaned text."
    ),
    "v5_few_shot_heavy": (
        "You are a speech-to-text post-processor. Rewrite raw ASR output as if the person typed it.\n\n"
        "Example 1:\n"
        "Input: We were um looking at the the new model and it's basically it's really fast actually no "
        "it's not that fast but it's faster than what we had before\n"
        "Output: We were looking at the new model. It's faster than what we had before, though not extremely fast.\n\n"
        "Example 2:\n"
        "Input: So I think we should um we should probably move the the database to actually no not the "
        "database the cache to Redis because it's faster\n"
        "Output: I think we should move the cache to Redis because it's faster.\n\n"
        "Example 3:\n"
        "Input: The problem is that the the users are complaining about the users are saying that the "
        "load time is too slow especially on mobile\n"
        "Output: The problem is that users are saying the load time is too slow, especially on mobile.\n\n"
        "Rules: Fix ASR errors using context. Remove fillers, stutters, false starts. Keep only corrections. "
        "Deduplicate repeated ideas. Preserve meaning exactly. Output only cleaned text."
    ),
}

# Email-specific variants
EMAIL_PROMPT_VARIANTS = {
    "v2_email": (
        "You are a speech-to-text post-processor that formats output for email. "
        "The input is raw ASR output. Rewrite it as if the person had typed the email directly.\n\n"
        "Rules:\n"
        "- Fix obvious ASR misrecognitions using surrounding context\n"
        "- Remove any remaining filler words, stutters, and false starts\n"
        "- When the speaker corrects themselves, keep only the correction\n"
        "- Restructure fragmented or garbled sentences into clear written prose\n"
        "- Preserve the speaker's meaning exactly — do not add content they didn't say\n"
        "- Output only the cleaned text, nothing else\n\n"
        "Email formatting:\n"
        "- If the speech starts with a greeting (Hey/Hi/Hello/Dear + name), format as a full email: "
        "greeting on its own line, blank line, body paragraphs, blank line, sign-off\n"
        "- If the speech ends with a sign-off but no greeting, add a blank line before the sign-off\n"
        "- If there is no greeting or sign-off, just clean up the text normally"
    ),
}

# ── Sampling parameter variants ─────────────────────────────────────────

SAMPLING_VARIANTS = {
    "greedy": {"temperature": 0.0},
    "google_recommended": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "low_temp": {"temperature": 0.3, "top_p": 0.95, "top_k": 64},
    "mid_temp": {"temperature": 0.7, "top_p": 0.95, "top_k": 64},
}


# ── Scoring ─────────────────────────────────────────────────────────────

def word_overlap_score(output: str, ideal: str) -> float:
    """Simple word-level overlap score (Jaccard-ish)."""
    out_words = set(output.lower().split())
    ideal_words = set(ideal.lower().split())
    if not ideal_words:
        return 1.0 if not out_words else 0.0
    intersection = out_words & ideal_words
    union = out_words | ideal_words
    return len(intersection) / len(union) if union else 0.0


def length_ratio(output: str, ideal: str) -> float:
    """How close is output length to ideal? 1.0 = perfect match."""
    out_len = len(output.split())
    ideal_len = len(ideal.split())
    if ideal_len == 0:
        return 1.0 if out_len == 0 else 0.0
    ratio = out_len / ideal_len
    # Score: 1.0 at ratio=1.0, dropping off as it diverges
    return max(0.0, 1.0 - abs(1.0 - ratio))


def check_issues(output: str, test_case: dict) -> list[str]:
    """Check for specific quality issues."""
    issues = []
    # Fillers that should be removed
    fillers = ["um ", " um ", " uh ", "uh ", "you know", "basically", "like "]
    for f in fillers:
        if f in output.lower() and f not in test_case["ideal"].lower():
            issues.append(f"filler_retained: '{f.strip()}'")

    # Stutters (repeated words)
    words = output.lower().split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and words[i] not in {"the", "a", "had", "that"}:
            issues.append(f"stutter: '{words[i]} {words[i+1]}'")

    # Length sanity
    out_len = len(output.split())
    ideal_len = len(test_case["ideal"].split())
    if out_len > ideal_len * 1.5:
        issues.append(f"too_long: {out_len} words vs {ideal_len} ideal")
    if out_len < ideal_len * 0.5:
        issues.append(f"too_short: {out_len} words vs {ideal_len} ideal")

    return issues


# ── LLM call ────────────────────────────────────────────────────────────

def call_llm(
    port: int,
    system_prompt: str,
    user_text: str,
    sampling: dict,
    dictionary: list[str] | None = None,
) -> tuple[str, float]:
    """Send a cleanup request. Returns (output_text, latency_seconds)."""
    prompt = system_prompt
    if dictionary:
        terms = ", ".join(dictionary)
        prompt += (
            f"\n\nThe speaker frequently uses these terms: {terms}. "
            "When ASR output sounds similar to one of these, use the correct spelling."
        )

    word_count = len(user_text.split())
    max_tokens = max(128, min(1024, int(word_count * 2.0)))

    payload = {
        "model": "gemma",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        "stream": False,
        **sampling,
    }

    start = time.perf_counter()
    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    elapsed = time.perf_counter() - start

    resp.raise_for_status()
    body = resp.json()
    result = body["choices"][0]["message"]["content"].strip()
    return result, elapsed


# ── Main benchmark ──────────────────────────────────────────────────────

def run_benchmark(port: int, save_path: str | None = None):
    # Verify server is up
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        if r.json().get("status") != "ok":
            print(f"Server not healthy: {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot reach llama-server on port {port}: {e}")
        print("Start it first or use --start-server")
        sys.exit(1)

    print(f"Connected to llama-server on port {port}")
    print(f"Running {len(TEST_CASES)} test cases x {len(PROMPT_VARIANTS)} prompts x {len(SAMPLING_VARIANTS)} sampling configs")
    print("=" * 80)

    results = []

    for test in TEST_CASES:
        tone = test.get("tone", "message")
        dictionary = test.get("dictionary", [])

        # Select prompt variants based on tone
        prompts = EMAIL_PROMPT_VARIANTS if tone == "email" else PROMPT_VARIANTS

        for prompt_name, prompt_text in prompts.items():
            for sampling_name, sampling_params in SAMPLING_VARIANTS.items():
                label = f"{test['name']} | {prompt_name} | {sampling_name}"
                print(f"\n{'─' * 80}")
                print(f"Test: {label}")

                try:
                    output, latency = call_llm(
                        port, prompt_text, test["input"], sampling_params, dictionary
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        "test": test["name"],
                        "prompt": prompt_name,
                        "sampling": sampling_name,
                        "error": str(e),
                    })
                    continue

                overlap = word_overlap_score(output, test["ideal"])
                length = length_ratio(output, test["ideal"])
                issues = check_issues(output, test)

                result = {
                    "test": test["name"],
                    "prompt": prompt_name,
                    "sampling": sampling_name,
                    "tone": tone,
                    "latency_s": round(latency, 2),
                    "word_overlap": round(overlap, 3),
                    "length_ratio": round(length, 3),
                    "issues": issues,
                    "issue_count": len(issues),
                    "output": output,
                    "ideal": test["ideal"],
                    "input": test["input"],
                }
                results.append(result)

                print(f"  Latency: {latency:.2f}s")
                print(f"  Overlap: {overlap:.3f}  Length ratio: {length:.3f}  Issues: {len(issues)}")
                if issues:
                    for iss in issues:
                        print(f"    - {iss}")
                print(f"  Output: {output[:120]}...")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by prompt+sampling combo
    combos: dict[str, list] = {}
    for r in results:
        if "error" in r:
            continue
        key = f"{r['prompt']} | {r['sampling']}"
        combos.setdefault(key, []).append(r)

    print(f"\n{'Combo':<45} {'Avg Overlap':>12} {'Avg Length':>12} {'Avg Issues':>12} {'Avg Latency':>12}")
    print("─" * 93)

    ranked = []
    for combo, rs in sorted(combos.items()):
        avg_overlap = sum(r["word_overlap"] for r in rs) / len(rs)
        avg_length = sum(r["length_ratio"] for r in rs) / len(rs)
        avg_issues = sum(r["issue_count"] for r in rs) / len(rs)
        avg_latency = sum(r["latency_s"] for r in rs) / len(rs)
        print(f"  {combo:<43} {avg_overlap:>12.3f} {avg_length:>12.3f} {avg_issues:>12.1f} {avg_latency:>11.2f}s")
        # Composite score: higher is better
        score = avg_overlap * 0.4 + avg_length * 0.3 - avg_issues * 0.1 - (avg_latency / 10) * 0.2
        ranked.append((combo, score, avg_overlap, avg_length, avg_issues, avg_latency))

    ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'RANKING (composite score)':}")
    print("─" * 93)
    for i, (combo, score, overlap, length, issues, latency) in enumerate(ranked, 1):
        print(f"  #{i} {combo:<41} score={score:.3f}  (overlap={overlap:.3f}, length={length:.3f}, issues={issues:.1f}, latency={latency:.1f}s)")

    # Save detailed results
    if save_path:
        out = Path(save_path)
    else:
        out = Path(__file__).parent / "data" / "benchmark_gemma_prompts_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out}")

    # Print best combo recommendation
    if ranked:
        best = ranked[0]
        print(f"\n{'=' * 80}")
        print(f"RECOMMENDATION: {best[0]}")
        print(f"  Composite score: {best[1]:.3f}")
        print(f"  Overlap: {best[2]:.3f}, Length ratio: {best[3]:.3f}, Issues: {best[4]:.1f}, Latency: {best[5]:.1f}s")
        print(f"{'=' * 80}")


def start_llama_server() -> tuple[subprocess.Popen, int]:
    """Start llama-server and return (process, port)."""
    import os
    appdata = os.environ.get("APPDATA", "")
    llm_dir = Path(appdata) / "com.chirp.app" / "llm"
    server = llm_dir / "llama-server.exe"
    model = llm_dir / "gemma-4-E2B-it-Q4_K_M.gguf"

    if not server.exists():
        print(f"llama-server not found at {server}")
        sys.exit(1)
    if not model.exists():
        print(f"Model not found at {model}")
        sys.exit(1)

    port = 18080
    n_threads = os.cpu_count() or 4

    cmd = [
        str(server),
        "--model", str(model),
        "--port", str(port),
        "--ctx-size", "4096",
        "--n-predict", "2048",
        "--threads", str(n_threads),
        "--gpu-layers", "99",
        "--flash-attn", "on",
        "--batch-size", "512",
        "--parallel", "1",
        "--reasoning", "off",
        "--log-disable",
    ]

    print(f"Starting llama-server on port {port}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=0x08000000 if sys.platform == "win32" else 0,
    )

    # Wait for health check
    for i in range(60):
        time.sleep(0.5)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.json().get("status") == "ok":
                print(f"llama-server ready after {(i+1)*0.5:.1f}s")
                return proc, port
        except Exception:
            pass

    proc.kill()
    print("llama-server failed to start within 30s")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gemma cleanup prompts")
    parser.add_argument("--port", type=int, default=18080, help="llama-server port")
    parser.add_argument("--start-server", action="store_true", help="Auto-start llama-server")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    server_proc = None
    port = args.port

    if args.start_server:
        server_proc, port = start_llama_server()

    try:
        run_benchmark(port, args.output)
    finally:
        if server_proc:
            print("Stopping llama-server...")
            server_proc.kill()
            server_proc.wait()
