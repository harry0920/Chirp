"""
Benchmark per-segment streaming cleanup vs the previous monolithic cleanup,
using REAL dictation sessions parsed from the user's chirp log file.

Each session in the log gives us:
  - The list of VAD segment transcripts (raw, after Parakeet but before regex)
  - The "After regex+replace+snips" line (the regex-cleaned joined text)
  - The "LLM cleanup" line (what chirp-cleanup-v2 produced in production)

For each session we run THREE experiments against the same model
(chirp-cleanup-v2 via llama-server):

  1. MONO     — feed the joined+regex'd text once (the OLD behavior).
  2. STREAM   — feed each segment individually, join with naive `" ".join`.
  3. STREAM+J — feed each segment individually, join with the smart-join
                logic from cleanup::join_cleaned_segments (the NEW v3
                behavior). Approximated in Python here.

We score each by:
  - word retention: % of input content words preserved in output
  - length ratio:   words(output) / words(input). 1.0 = same length,
                    <1.0 = compressed/paraphrased, >1.0 = expanded
  - latency:        total ms spent in cleanup calls

Higher retention + length-ratio closer to 1.0 = less paraphrasing.

Usage (from repo root):
    py -3 training/benchmark_streaming_cleanup.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

LOG_PATH = Path(os.environ["LOCALAPPDATA"]) / "com.chirp.app" / "logs" / "Chirp.log"
LLM_DIR = Path(os.environ["APPDATA"]) / "com.chirp.app" / "llm"
LLAMA_SERVER = LLM_DIR / "llama-server.exe"
CHIRP_V2_MODEL = LLM_DIR / "chirp-cleanup-0.6b-q4_k_m.gguf"

PORT = 9997  # avoid colliding with the dev app

# Mirror src-tauri/src/llm.rs BASE_SYSTEM_PROMPT exactly so we test the
# same prompt the production app uses.
SYSTEM_PROMPT = (
    "Clean up dictated speech. Remove fillers, fix stutters, "
    "resolve self-corrections (keep only the final version). "
    "Output only the cleaned text."
)

# ─── Log parsing ────────────────────────────────────────────────────────────

VAD_SEGMENT_RE = re.compile(r"VAD segment \d+ transcript: '(.*?)'\s*$")
AFTER_REGEX_RE = re.compile(r"After regex\+replace\+snips: '(.*?)'\s*$")
LLM_CLEAN_RE   = re.compile(r"LLM cleanup: '(.*?)'\s*$")


def parse_log_dictations(log_path: Path) -> list[dict]:
    """Pull every dictation session out of the log file.

    A session starts with a "Hotkey pressed" line and ends with the next
    "LLM cleanup" or "Transcription complete" line.
    """
    sessions: list[dict] = []
    cur: dict | None = None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if "Hotkey pressed" in line:
            # Push any in-progress session that didn't end cleanly
            if cur and cur.get("segments"):
                sessions.append(cur)
            cur = {"segments": [], "after_regex": None, "production_cleaned": None}
            continue
        if cur is None:
            continue

        m = VAD_SEGMENT_RE.search(line)
        if m:
            cur["segments"].append(m.group(1))
            continue

        m = AFTER_REGEX_RE.search(line)
        if m:
            cur["after_regex"] = m.group(1)
            continue

        m = LLM_CLEAN_RE.search(line)
        if m:
            cur["production_cleaned"] = m.group(1)
            if cur.get("segments"):
                sessions.append(cur)
            cur = None
            continue

    if cur and cur.get("segments"):
        sessions.append(cur)
    return sessions


# ─── Server lifecycle ───────────────────────────────────────────────────────

def start_server() -> subprocess.Popen:
    if not LLAMA_SERVER.exists():
        sys.exit(f"missing: {LLAMA_SERVER}")
    if not CHIRP_V2_MODEL.exists():
        sys.exit(f"missing: {CHIRP_V2_MODEL}")

    cmd = [
        str(LLAMA_SERVER),
        "--model", str(CHIRP_V2_MODEL),
        "--port", str(PORT),
        "--ctx-size", "2048",
        "--n-predict", "512",
        "--gpu-layers", "99",
        "--flash-attn", "on",
        "--batch-size", "512",
        "--parallel", "1",
        "--reasoning-budget", "0",
        "--log-disable",
    ]
    print(f"  starting llama-server on :{PORT}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )
    for i in range(60):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if r.json().get("status") == "ok":
                print(f"  ready after {i + 1}s (pid {proc.pid})")
                return proc
        except Exception:
            pass
    proc.kill()
    sys.exit("llama-server failed to start within 60s")


def stop_server(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait()


# ─── Cleanup call ───────────────────────────────────────────────────────────

def cleanup(text: str) -> tuple[str, float]:
    """Single LLM cleanup call. Returns (cleaned_text, elapsed_ms)."""
    word_count = len(text.split())
    # Mirror llm.rs dynamic max_tokens: 1.2x input + 64 floor
    max_tokens = max(int(word_count * 1.2 + 0.5), 64)

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": True,
    }
    start = time.perf_counter()
    try:
        r = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        out = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"    ERR: {e}")
        out = text
    return out, elapsed_ms


# ─── Smart-join (Python port of cleanup::join_cleaned_segments) ────────────

STUB_END_WORDS = frozenset({
    "a", "an", "the",
    "and", "or", "but", "so", "if", "as", "than", "nor",
    "of", "to", "in", "on", "at", "by", "with", "for", "from",
    "into", "onto", "upon", "about", "under", "over", "between",
    "through", "across",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "my", "our", "your", "his", "its", "their",
    "just",
    # fillers — segment ending in "um"/"uh" is mid-sentence by definition
    "um", "umm", "uh", "uhh", "hmm", "mhm", "mmhmm",
})

CONTINUATION_START_WORDS = frozenset({
    "and", "but", "or",
    "because", "since", "while", "though", "although",
    "however", "therefore", "thus", "hence",
    "moreover", "furthermore",
})

SAFE_TO_LOWERCASE = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "by",
    "as", "if", "or", "so", "is", "am", "be", "do", "we",
    "us", "it", "he", "i",
    "and", "but", "for", "nor", "yet", "are", "was", "had",
    "has", "did", "let", "all", "any", "may", "can", "see",
    "use", "get", "got", "her", "his", "you", "our", "way",
    "out", "off", "now", "two", "one", "few", "say", "set",
    "run", "try", "put",
    "this", "that", "with", "from", "have", "will", "make",
    "just", "like", "also", "then", "than", "when", "they",
    "them", "your", "what", "some", "want", "need", "been",
    "were", "more", "into", "give", "take", "feel", "look",
    "tell", "show", "find", "kind", "sort", "well",
    "would", "could", "might", "since", "while", "after",
    "above", "below", "where", "which", "their", "those",
    "these", "going", "doing", "being",
    "really", "though", "because", "however", "before", "during",
    "between", "through", "across", "behind", "should",
    "although", "therefore",
})


def _split_into_sentences(text: str) -> list[str]:
    """Aggressive sentence split — split on .!? followed by whitespace.

    Mirrors src-tauri/src/cleanup.rs split_into_sentences exactly.
    """
    sentences: list[str] = []
    current: list[str] = []
    chars = list(text)
    i = 0
    while i < len(chars):
        current.append(chars[i])
        if chars[i] in (".", "!", "?"):
            j = i + 1
            while j < len(chars) and chars[j].isspace():
                j += 1
            at_boundary = j >= len(chars) or j > i + 1
            next_is_digit = j < len(chars) and chars[j].isdigit() and j == i + 1
            if at_boundary and not next_is_digit:
                trimmed = "".join(current).strip()
                if trimmed:
                    sentences.append(trimmed)
                current = []
                i = j
                continue
        i += 1
    trimmed = "".join(current).strip()
    if trimmed:
        sentences.append(trimmed)
    return sentences


def _smart_merge(sentences: list[str]) -> str:
    if not sentences:
        return ""
    out = sentences[0]
    for sent in sentences[1:]:
        prev_ends_with_terminal = out.rstrip().endswith((".", "!", "?"))
        prev_last_word_lc = (
            out.rstrip(" .!?,:;\"'").split()[-1].lower()
            if out.rstrip(" .!?,:;\"'").split()
            else ""
        )
        prev_ends_in_stub = prev_last_word_lc in STUB_END_WORDS

        head_word_raw = sent.split()[0] if sent.split() else ""
        head_starts_lower = head_word_raw and head_word_raw[0].islower()
        head_word_lc = "".join(c for c in head_word_raw if c.isalnum() or c == "'").lower()
        head_is_continuation = head_word_lc in CONTINUATION_START_WORDS
        head_is_safe = head_word_lc in SAFE_TO_LOWERCASE

        should_merge = (
            head_starts_lower
            or (prev_ends_in_stub and prev_ends_with_terminal)
            or (head_is_continuation and prev_ends_with_terminal)
        )

        if should_merge:
            if prev_ends_with_terminal:
                out = out.rstrip(" .!?")
            out += " "
            if not head_starts_lower and head_is_safe:
                out += sent[0].lower() + sent[1:]
            else:
                out += sent
        else:
            out += " " + sent
    return out


# Filler patterns mirroring src-tauri/src/cleanup.rs `filler_patterns`.
# We only port the unconditional ones — the lookahead-conditional fillers
# (filler "like", "basically,", "actually,", "so,", "i mean,") are more
# context-sensitive and we leave them to the Rust regex in production.
_FILLER_REGEXES = [
    re.compile(r"(?i)\bum+\b"),
    re.compile(r"(?i)\buh+\b"),
    re.compile(r"(?i)\buh huh\b"),
    re.compile(r"(?i)\bmm+ ?hmm+\b"),
    re.compile(r"(?i)\bhmm+\b"),
]
_DANGLING_COMMA = re.compile(r",\s*,")
_LEADING_COMMA = re.compile(r"^\s*,\s*")
_WHITESPACE = re.compile(r"\s{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:)])")


def _remove_fillers(text: str) -> str:
    out = text
    for r in _FILLER_REGEXES:
        out = r.sub("", out)
    out = _DANGLING_COMMA.sub(",", out)
    out = _LEADING_COMMA.sub("", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _WHITESPACE.sub(" ", out.strip())
    return out


def _capitalize_first(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    return t[0].upper() + t[1:]


def smart_join(segments: list[str]) -> str:
    """Python port of cleanup::join_cleaned_segments.

    Pipeline: strip internal paragraph breaks → naive join → split into
    sentences → smart-merge adjacent sentences → re-run filler regex on
    the joined output → capitalize first character.
    """
    cleaned = [
        s.replace("\n\n", " ").replace("\n", " ").strip()
        for s in segments
    ]
    cleaned = [s for s in cleaned if s]
    if not cleaned:
        return ""
    raw = " ".join(cleaned)
    sentences = _split_into_sentences(raw)
    merged = _smart_merge(sentences)
    # Re-run filler removal on the joined output to catch cross-boundary
    # fillers that survived per-segment cleanup. Mirrors Rust cleanup_text.
    after_fillers = _remove_fillers(merged)
    return _capitalize_first(after_fillers)


# ─── Scoring ────────────────────────────────────────────────────────────────

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def words(s: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(s)]


def word_retention(input_text: str, output_text: str) -> float:
    """% of UNIQUE input content words preserved in output (set Jaccard input-side)."""
    inw = set(words(input_text))
    outw = set(words(output_text))
    if not inw:
        return 0.0
    return len(inw & outw) / len(inw)


def length_ratio(input_text: str, output_text: str) -> float:
    iw = len(words(input_text))
    ow = len(words(output_text))
    return ow / iw if iw else 0.0


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"reading log: {LOG_PATH}")
    if not LOG_PATH.exists():
        sys.exit(f"missing log file: {LOG_PATH}")

    sessions = parse_log_dictations(LOG_PATH)
    print(f"  parsed {len(sessions)} dictation sessions")

    multi = [s for s in sessions if len(s["segments"]) >= 2]
    print(f"  {len(multi)} sessions have 2+ VAD segments (interesting for streaming)")

    if not multi:
        sys.exit("no multi-segment dictations to test")

    proc = start_server()
    results: list[dict] = []

    try:
        for i, sess in enumerate(multi, start=1):
            joined_raw = " ".join(sess["segments"])
            joined_for_mono = sess.get("after_regex") or joined_raw
            n_seg = len(sess["segments"])

            print()
            print("=" * 80)
            print(f"Session {i}/{len(multi)}  ({n_seg} segments, {len(words(joined_raw))} words raw)")
            print("=" * 80)
            print(f"  raw joined:   {joined_raw[:200]}")
            if sess.get("production_cleaned"):
                print(f"  production:   {sess['production_cleaned'][:200]}")

            # Method 1: monolithic — single cleanup call on the joined regex'd text
            mono_out, mono_ms = cleanup(joined_for_mono)

            # Method 2: streaming — one cleanup call per VAD segment, naive join
            stream_outs: list[str] = []
            stream_ms_total = 0.0
            per_seg_ms: list[float] = []
            for seg in sess["segments"]:
                cleaned, ms = cleanup(seg)
                stream_outs.append(cleaned.strip())
                per_seg_ms.append(ms)
                stream_ms_total += ms
            stream_naive = " ".join(s for s in stream_outs if s)

            # Method 3: streaming + smart join (the new v3 behavior)
            stream_smart = smart_join(stream_outs)

            # Score against the input
            mono_ret = word_retention(joined_raw, mono_out)
            mono_lr = length_ratio(joined_raw, mono_out)
            naive_ret = word_retention(joined_raw, stream_naive)
            naive_lr = length_ratio(joined_raw, stream_naive)
            smart_ret = word_retention(joined_raw, stream_smart)
            smart_lr = length_ratio(joined_raw, stream_smart)

            print(f"  MONO    ({mono_ms:5.0f}ms ret={mono_ret:.2f} lr={mono_lr:.2f}): {mono_out[:200]}")
            print(f"  NAIVE   ({stream_ms_total:5.0f}ms ret={naive_ret:.2f} lr={naive_lr:.2f}): {stream_naive[:200]}")
            print(f"  SMART-J (    -ms ret={smart_ret:.2f} lr={smart_lr:.2f}): {stream_smart[:200]}")

            results.append({
                "session": i,
                "n_segments": n_seg,
                "raw_joined": joined_raw,
                "after_regex": sess.get("after_regex"),
                "production_cleaned": sess.get("production_cleaned"),
                "mono": {
                    "out": mono_out,
                    "ms": mono_ms,
                    "retention": mono_ret,
                    "length_ratio": mono_lr,
                },
                "stream_naive": {
                    "outs_per_segment": stream_outs,
                    "out_joined": stream_naive,
                    "ms_total": stream_ms_total,
                    "ms_per_segment": per_seg_ms,
                    "retention": naive_ret,
                    "length_ratio": naive_lr,
                },
                "stream_smart": {
                    "out_joined": stream_smart,
                    "retention": smart_ret,
                    "length_ratio": smart_lr,
                },
            })

        # ── summary ──────────────────────────────────────────────────────────
        print()
        print("=" * 80)
        print(f"  SUMMARY across {len(results)} multi-segment sessions")
        print("=" * 80)

        avg_mono_ret = sum(r["mono"]["retention"] for r in results) / len(results)
        avg_mono_lr = sum(r["mono"]["length_ratio"] for r in results) / len(results)
        avg_mono_ms = sum(r["mono"]["ms"] for r in results) / len(results)
        avg_naive_ret = sum(r["stream_naive"]["retention"] for r in results) / len(results)
        avg_naive_lr = sum(r["stream_naive"]["length_ratio"] for r in results) / len(results)
        avg_naive_ms = sum(r["stream_naive"]["ms_total"] for r in results) / len(results)
        avg_smart_ret = sum(r["stream_smart"]["retention"] for r in results) / len(results)
        avg_smart_lr = sum(r["stream_smart"]["length_ratio"] for r in results) / len(results)

        print(f"  {'metric':<28} {'mono':>10} {'naive':>10} {'smart-J':>10} {'smart-mono':>12}")
        print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")
        print(f"  {'word retention (in-side)':<28} {avg_mono_ret:>10.3f} {avg_naive_ret:>10.3f} {avg_smart_ret:>10.3f} {avg_smart_ret - avg_mono_ret:>+12.3f}")
        print(f"  {'length ratio (out/in)':<28} {avg_mono_lr:>10.3f} {avg_naive_lr:>10.3f} {avg_smart_lr:>10.3f} {avg_smart_lr - avg_mono_lr:>+12.3f}")
        print(f"  {'avg ms per session':<28} {avg_mono_ms:>10.0f} {avg_naive_ms:>10.0f} {'(no LLM)':>10} {avg_naive_ms - avg_mono_ms:>+12.0f}")
        print()
        print("  Higher word retention + length ratio closer to 1.0 = less paraphrasing.")
        print("  Streaming ms is total per-segment serial time; in production those")
        print("  calls overlap with the user still talking, so PERCEIVED latency is")
        print("  only the LAST segment's ms after release.")
        print("  Smart-J adds zero LLM time over naive — pure deterministic Rust pass.")

        out_dir = Path(__file__).parent / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "benchmark_streaming_results.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "n": len(results),
                        "mono": {"retention": avg_mono_ret, "length_ratio": avg_mono_lr, "ms": avg_mono_ms},
                        "stream_naive": {"retention": avg_naive_ret, "length_ratio": avg_naive_lr, "ms": avg_naive_ms},
                        "stream_smart": {"retention": avg_smart_ret, "length_ratio": avg_smart_lr},
                    },
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\n  wrote {out_path}")

    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
