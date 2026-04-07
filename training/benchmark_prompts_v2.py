"""
Benchmark system prompt variants for Gemma 4 E2B cleanup.

Uses real transcription inputs from production logs.
Tests multiple prompt variants and compares outputs side-by-side.

Usage:
    python benchmark_prompts_v2.py
    python benchmark_prompts_v2.py --port 54279   # if server already running
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import requests

LLAMA_SERVER = os.path.join(
    os.environ.get("APPDATA", ""),
    "com.chirp.app", "llm", "llama-server.exe"
)
GEMMA_MODEL = os.path.join(
    os.environ.get("APPDATA", ""),
    "com.chirp.app", "llm", "gemma-4-E2B-it-Q4_K_M.gguf"
)

# Real inputs from production logs
TEST_CASES = [
    {
        "id": "tax_opener",
        "input": "Now this might be a silly question, but if I don't I don't know, I'm just trying to bring my tax liability down as far as possible. So I suppose we should probably start with seeing what that looks like and then finding all of our eligible write-offs.",
        "notes": "Should keep 'Now this might be a silly question' -- it's meaningful framing, not filler"
    },
    {
        "id": "cookbooks",
        "input": "How can these numbers be so uh far off and like we just have both of our accounts linked um and cookbooks and then it just like auto approves every transaction and we only ever spend money through the debit card or transfer money to our personal account to spend it for college tuition. But how can I be so far off?",
        "notes": "Should remove um/uh/like fillers, keep meaning and tone"
    },
    {
        "id": "coursera",
        "input": "There's a transaction in here where I bought a subscription to Coursera to learn how to make websites better, and then I realized that this wasn't the correct tool for me. And then I refunded it. So how do I make sure that that's properly accounted for?",
        "notes": "Should NOT rephrase 'correct tool' to 'right tool', should keep 'There's a transaction in here where'"
    },
    {
        "id": "owners_draws",
        "input": "Bro you gotta tell the um owner's draws were I think they are logged as owners draws aren't they also um withdrawal home I believe was a transfer of my savings account",
        "notes": "Messy input with stutters -- should clean but preserve 'Bro' and casual tone"
    },
    {
        "id": "chill_out",
        "input": "You have to chill out, bro. I promise it's not that big of a deal. I'm just seeing the 103 discrepancy between posted amount and like the actual amount. But uh 5614 Ivy Groceries was for gas going to a client meeting. All the other things are like, bro, those are valid, I promise.",
        "notes": "Casual tone with 'bro' -- should keep personality, just remove fillers"
    },
    {
        "id": "alright_fixed",
        "input": "Alright, we fixed um the stuff. Now we're looking for what is the next step before we um go to file our taxes.",
        "notes": "Should keep 'Alright' -- it's not filler, it's a transition word"
    },
    {
        "id": "deductions_long",
        "input": "Yeah, we need to do some fucking deductions here. This is insane. Yeah the Apple store purchase was for an iPad Pro that I use to um like draw wireframes for client sites with. Or do like pitches on the go. I do not pay for my own phone or internet. I did not drive to any client meetings really. I can go measure my home office though. But I only moved in here in August.",
        "notes": "Should keep profanity (user's words), keep 'Yeah', keep 'really', don't change 'measure' to 'go to'"
    },
    {
        "id": "i_mean_taxes",
        "input": "I mean that was just taxes we paid for twenty twenty four. Is that not a business expense there?",
        "notes": "Should keep 'I mean' -- it's emphasis, not filler here"
    },
    {
        "id": "yeah_short",
        "input": "Yeah.",
        "notes": "Should pass through unchanged"
    },
    {
        "id": "oh_brother",
        "input": "Oh brother.",
        "notes": "Should pass through unchanged"
    },
    {
        "id": "cooked",
        "input": "This is so cooked.",
        "notes": "Should pass through unchanged"
    },
    {
        "id": "numbered_list",
        "input": "For the release we need to number one finish the migration number two update the docs and number three notify the customers",
        "notes": "Should format as numbered list"
    },
    {
        "id": "not_a_list",
        "input": "We sold one of the hair washes and two hair colors today",
        "notes": "Should NOT become a list -- numbers are quantities, not ordinals"
    },
    {
        "id": "self_correction",
        "input": "So um I think we should launch on Tuesday actually no Wednesday because um the design team needs like one more day to basically finish the icons",
        "notes": "Should handle self-correction (Tuesday->Wednesday) and remove fillers"
    },
]

# ── Prompt Variants ──────────────────────────────────────────────────

PROMPT_A = """\
You clean up speech-to-text output. Your job is to make it read like the person typed it, while preserving every piece of information they communicated.

Rules:
- Fix ASR errors (misheard words) using context clues
- Remove filler words (um, uh, like, you know, so, basically)
- Remove stutters and word repetitions
- Keep only the corrected version when someone corrects themselves
- When the same idea is stated multiple times, state it once clearly
- Only combine fragments that are clearly the same incomplete sentence -- do NOT merge distinct thoughts
- Do not rephrase sentences that are already clear -- only fix actual errors
- Do not add information the speaker did not say
- Do not summarize or omit meaningful content

Formatting:
- When the speaker uses ordinals (first/second/third, one/two/three, step one/step two) to enumerate items, format as a numbered list. Do NOT convert numbers to a list when they are part of normal speech (e.g. "we sold one widget and two gadgets" stays as prose).
- Preserve paragraph breaks (\\n\\n) from the input. If the speaker says "new paragraph", that is a paragraph break.

Example 1 -- filler removal:
Input: We were um looking at the the new model and it's basically it's really fast actually no it's not that fast but it's faster than what we had before
Output: We were looking at the new model. It's faster than what we had before, though not extremely fast.

Example 2 -- self-correction:
Input: So I think we should um we should probably move the the database to actually no not the database the cache to Redis because it's faster
Output: I think we should move the cache to Redis because it's faster.

Example 3 -- numbered list from ordinals:
Input: The steps are first update the API second test it third deploy it
Output: The steps are:
1. Update the API
2. Test it
3. Deploy it

Example 4 -- numbered list from "number one" style:
Input: For the release we need to number one finish the migration number two update the docs and number three notify the customers
Output: For the release we need to:
1. Finish the migration
2. Update the docs
3. Notify the customers

Example 5 -- numbers that are NOT a list (do not convert to list):
Input: We sold one of the hair washes and two hair colors today
Output: We sold one of the hair washes and two hair colors today.

Example 6 -- paragraph break:
Input: The first feature is the new dashboard. It shows all your metrics in one place.

The second feature is notifications. You'll get alerts when something changes.
Output: The first feature is the new dashboard. It shows all your metrics in one place.

The second feature is notifications. You'll get alerts when something changes.

Example 7 -- already clear input (do not rephrase):
Input: I'll be out of office next Monday. Please forward any urgent emails to Sarah.
Output: I'll be out of office next Monday. Please forward any urgent emails to Sarah.

Output only the cleaned text."""

PROMPT_B = """\
You clean up speech-to-text output. Make it read like the person typed it themselves.

Rules:
- Remove ONLY these filler words: um, uh, uh huh, hmm, mmhmm
- Remove stutters (repeated words)
- Fix misheard words using context
- When someone corrects themselves, keep only the correction
- Do NOT rephrase, restructure, or "improve" sentences -- only remove fillers and fix errors
- Do NOT drop words like "alright", "yeah", "I mean", "also", "so" -- these are the speaker's voice, not filler
- Do NOT censor or change profanity
- Keep the speaker's exact wording when it's already clear
- If the speaker lists items using ordinals (first/second/third, number one/two/three), format as a numbered list
- If the speaker says "new paragraph" or "new line", convert to actual line breaks

Output only the cleaned text."""

PROMPT_C = """\
Clean up this dictated text. Remove only filler words (um, uh) and stutters. Fix mishearings. Keep everything else exactly as spoken. Do not rephrase. Format numbered lists when the speaker uses ordinals. Output only the cleaned text."""


PROMPT_D = """\
You clean up speech-to-text output. Make it read like the person typed it themselves. Preserve their voice and tone.

Rules:
- Remove ONLY these filler words: um, uh, uh huh, hmm, mmhmm
- Remove "like" only when it's a filler (e.g. "I was like um going" -> "I was going"), keep it when meaningful ("I like pizza", "something like that")
- Remove stutters (repeated words)
- Fix misheard words using context
- When someone corrects themselves ("actually no", "wait", "I mean", "scratch that"), drop everything before the correction and keep only what comes after
- Do NOT remove or change words like "alright", "yeah", "so", "also", "I mean", "basically" -- these are the speaker's voice
- Do NOT rephrase, restructure, or "improve" clear sentences
- Do NOT censor or change profanity
- Do NOT drop the beginning of what someone said
- If the speaker lists items using ordinals (first/second/third, number one/two/three), format as a numbered list
- If the speaker says "new paragraph" or "new line", convert to a line break

Example -- self-correction:
Input: So I think we should launch on Tuesday actually no Wednesday because um the design team needs one more day to finish the icons
Output: So I think we should launch on Wednesday because the design team needs one more day to finish the icons.

Example -- another self-correction:
Input: Send it to John no I mean send it to Mike
Output: Send it to Mike.

Example -- numbered list:
Input: For the release we need to number one finish the migration number two update the docs and number three notify the customers
Output: For the release we need to:
1. Finish the migration
2. Update the docs
3. Notify the customers

Output only the cleaned text."""

PROMPT_E = """\
You clean up speech-to-text output. Make it read like the person typed it themselves. Preserve their voice and tone.

Only do these things:
1. Remove filler sounds: um, uh, uh huh, hmm, mmhmm
2. Remove stutters (repeated words)
3. Fix misheard words
4. When someone corrects themselves ("actually no", "wait I mean"), keep only the correction
5. If the speaker uses ordinals (first/second/third, number one/two/three) to list items, format as a numbered list
6. If the speaker says "new paragraph" or "new line", insert a line break

Do NOT do any of these:
- Do not remove words like alright, yeah, so, also, like, basically, I mean
- Do not rephrase or restructure sentences
- Do not censor profanity
- Do not drop the beginning of sentences
- Do not add words the speaker didn't say

Output only the cleaned text."""

PROMPT_F = """\
You clean up speech-to-text output. Make it read like the person typed it themselves. Preserve their voice and tone.

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

PROMPTS = {
    "D_hybrid": PROMPT_D,
    "F_structured": PROMPT_F,
}


# ── Server Management ────────────────────────────────────────────────

def start_server(port=9999, ngl=99):
    """Start llama-server if not already running."""
    try:
        r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        if r.status_code == 200:
            print(f"Server already running on port {port}")
            return None
    except:
        pass

    print(f"Starting llama-server on port {port}...")
    proc = subprocess.Popen(
        [
            LLAMA_SERVER,
            "-m", GEMMA_MODEL,
            "--port", str(port),
            "-ngl", str(ngl),
            "-c", "4096",
            "--no-mmap",
            "--reasoning-off",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for ready
    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                print("Server ready!")
                return proc
        except:
            pass
        time.sleep(1)

    raise RuntimeError("Server failed to start")


def query_llm(port, system_prompt, user_text, max_tokens=512):
    """Send a single cleanup request to the server."""
    payload = {
        "model": "gemma",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True,
    }

    start = time.perf_counter()
    r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=30)
    elapsed = time.perf_counter() - start

    r.raise_for_status()
    data = r.json()
    output = data["choices"][0]["message"]["content"].strip()

    return output, elapsed


# ── Main Benchmark ───────────────────────────────────────────────────

def run_benchmark(port):
    results = {}

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"  PROMPT: {prompt_name}")
        print(f"{'='*70}")

        prompt_results = []

        # Warm up KV cache with a throwaway request
        try:
            query_llm(port, prompt_text, "Hello.", max_tokens=16)
        except:
            pass

        for tc in TEST_CASES:
            word_count = len(tc["input"].split())
            max_tokens = max(64, int(word_count * 1.5))

            output, elapsed = query_llm(port, prompt_text, tc["input"], max_tokens)

            # Simple scoring
            input_words = set(tc["input"].lower().split())
            output_words_set = set(output.lower().split())

            result = {
                "id": tc["id"],
                "input": tc["input"],
                "output": output,
                "time_ms": int(elapsed * 1000),
                "notes": tc["notes"],
            }
            prompt_results.append(result)

            # Print compact
            print(f"\n  [{tc['id']}] ({result['time_ms']}ms)")
            print(f"    IN:  {tc['input'][:100]}{'...' if len(tc['input']) > 100 else ''}")
            print(f"    OUT: {output[:100]}{'...' if len(output) > 100 else ''}")
            print(f"    NOTE: {tc['notes']}")

        results[prompt_name] = prompt_results

    return results


def print_comparison(results):
    """Print side-by-side comparison of all prompts."""
    print(f"\n\n{'='*70}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")

    prompt_names = list(results.keys())

    for i, tc in enumerate(TEST_CASES):
        tc_id = tc["id"]
        print(f"\n  [{tc_id}]")
        print(f"    INPUT: {tc['input'][:120]}{'...' if len(tc['input']) > 120 else ''}")
        print(f"    WANT:  {tc['notes']}")

        for pname in prompt_names:
            r = results[pname][i]
            print(f"    {pname:20s} ({r['time_ms']:4d}ms): {r['output'][:120]}{'...' if len(r['output']) > 120 else ''}")

    # Timing summary
    print(f"\n\n  TIMING SUMMARY (avg ms per request, excluding warmup):")
    for pname in prompt_names:
        times = [r["time_ms"] for r in results[pname]]
        avg = sum(times) / len(times)
        print(f"    {pname:20s}: avg={avg:.0f}ms, min={min(times)}ms, max={max(times)}ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    proc = start_server(args.port)

    try:
        results = run_benchmark(args.port)
        print_comparison(results)

        # Save results
        out_path = Path("data/benchmark_prompts_v2_results.json")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    finally:
        if proc:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    main()
