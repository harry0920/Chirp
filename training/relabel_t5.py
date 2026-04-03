"""
Relabel existing training inputs with light-polish targets via teacher model.

Takes the 11K real inputs from training_pairs_clean.jsonl, runs regex cleanup,
then sends ALL of them to the teacher in one big vLLM generate() call for
maximum throughput. Validates outputs locally.

Usage:
    python -m modal run relabel_t5.py
    python -m modal run relabel_t5.py --limit 1500
"""

import json
import re
import string
import time
import random
from pathlib import Path
from difflib import SequenceMatcher

import modal

TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
GPU = "L40S"
T5_PREFIX = "Rewrite as typed text: "

app = modal.App("chirp-relabel-t5")

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

MINUTES = 60

# ── Cleanup prompt (applied per-input) ────────────────────────────────

CLEANUP_PROMPT = string.Template("""\
You are a speech-to-text cleanup tool. The text below was spoken aloud and already had filler words removed. Lightly polish it so it reads like it was typed. Make MINIMAL edits:

- Fix spoken grammar: "gonna" → "going to", "wanna" → "want to", "gotta" → "got to", "me and X" → "X and I"
- Resolve self-corrections: when the speaker says "wait", "no", "I mean", "actually", keep ONLY the corrected version
- Remove stutters and repeated words
- Fix punctuation, add missing commas and question marks
- Normalize spoken numbers to digits when natural

DO NOT summarize, restructure, reorder, or shorten. Keep the speaker's words and voice. If the text is already clean, return it unchanged.

Input: $input_text

Output ONLY the cleaned text:""")


# ── Regex cleanup (Python port of cleanup.rs) ─────────────────────────

I_FLAG = re.IGNORECASE
FILLER_PATTERNS = [
    re.compile(r"\bum+\b", I_FLAG), re.compile(r"\buh+\b", I_FLAG),
    re.compile(r"\buh huh\b", I_FLAG), re.compile(r"\bmm+ ?hmm+\b", I_FLAG),
    re.compile(r"\bhmm+\b", I_FLAG), re.compile(r"\byou know\b(?=\s*,?\s)", I_FLAG),
    re.compile(r"\blike\b(?=\s+(the|a|an|i|we|they|he|she|it|my|our|this|that)\b)", I_FLAG),
    re.compile(r"\bbasically\b(?=\s*,)", I_FLAG), re.compile(r"\bactually\b(?=\s*,)", I_FLAG),
    re.compile(r"\bso\b(?=\s*,\s)", I_FLAG), re.compile(r"\bi mean\b(?=\s*,)", I_FLAG),
]
DANGLING_COMMA_RE = re.compile(r",\s*,")
LEADING_COMMA_RE = re.compile(r"^\s*,\s*")
WHITESPACE_RE = re.compile(r"\s{2,}")
SENTENCE_END_RE = re.compile(r"([.!?])\s+([a-z])")
STANDALONE_I_RE = re.compile(r"\bi\b")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:)])")


def cleanup_regex(text):
    if not text:
        return ""
    result = text
    for filler in FILLER_PATTERNS:
        result = filler.sub("", result)
    result = DANGLING_COMMA_RE.sub(",", result)
    result = LEADING_COMMA_RE.sub("", result)
    result = WHITESPACE_RE.sub(" ", result.strip())
    result = SPACE_BEFORE_PUNCT_RE.sub(r"\1", result)
    result = result.strip()
    if result:
        result = result[0].upper() + result[1:]
    trimmed = result.rstrip()
    if trimmed and trimmed[-1] not in '.!?:;")\n':
        result = trimmed + "."
    result = SENTENCE_END_RE.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}", result)
    result = STANDALONE_I_RE.sub("I", result)
    return result


# ── Validation ────────────────────────────────────────────────────────

def validate_pair(inp, target):
    if not inp or not target:
        return False, "empty"
    inp_words = inp.lower().split()
    target_words = target.lower().split()
    if len(target_words) < 3:
        return False, "too short"

    sim = SequenceMatcher(None, inp.lower(), target.lower()).ratio()
    # Scale threshold by length — longer texts naturally diverge more
    min_sim = 0.55 if len(inp_words) > 50 else 0.60
    if sim < min_sim:
        return False, f"too different (sim={sim:.2f})"

    length_ratio = len(target_words) / len(inp_words)
    if length_ratio < 0.50:
        return False, f"target too short (ratio={length_ratio:.2f})"
    if length_ratio > 1.5:
        return False, f"target too long (ratio={length_ratio:.2f})"

    if re.search(r"(\*\*|^#{1,3}\s|^[-*]\s)", target, re.MULTILINE):
        return False, "has markdown"

    return True, "ok"


# ── vLLM teacher ──────────────────────────────────────────────────────

@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
class TeacherModel:
    @modal.enter()
    def load(self):
        import vllm
        print(f"Loading {TEACHER_MODEL}...")
        self.llm = vllm.LLM(
            model=TEACHER_MODEL,
            tensor_parallel_size=1,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            trust_remote_code=True,
        )
        print("Model loaded.")

    @modal.method()
    def cleanup_batch(self, prompts):
        """Process a big batch of cleanup prompts at once."""
        import vllm
        outputs = self.llm.generate(
            prompts,
            vllm.SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512),
        )
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            # Strip quotes if wrapped
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]
            results.append(text)
        return results

    @modal.exit()
    def stop(self):
        del self.llm


# ── Main ──────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    input_file: str = "data/training_pairs_clean.jsonl",
    output: str = "data/training_t5_v2.jsonl",
    limit: int = 2000,
    seed: int = 42,
    batch_size: int = 200,
):
    random.seed(seed)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing inputs
    print(f"Loading inputs from {input_file}...")
    with open(input_file) as f:
        all_pairs = [json.loads(l) for l in f]
    print(f"  {len(all_pairs)} pairs loaded")

    # Shuffle and take more than we need (we'll lose some to validation)
    random.shuffle(all_pairs)
    candidates = all_pairs[:min(limit * 2, len(all_pairs))]

    # Run regex cleanup on inputs
    print(f"Running regex cleanup on {len(candidates)} inputs...")
    cleaned_inputs = []
    for pair in candidates:
        inp = pair["input"]
        cleaned = cleanup_regex(inp)
        if cleaned and len(cleaned.split()) >= 5:
            cleaned_inputs.append(cleaned)

    print(f"  {len(cleaned_inputs)} valid after regex")

    # Build prompts
    prompts = [
        CLEANUP_PROMPT.substitute(input_text=inp)
        for inp in cleaned_inputs
    ]

    # Send to teacher in batches
    teacher = TeacherModel()
    all_outputs = []
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} prompts)...", flush=True)

        results = teacher.cleanup_batch.remote(batch)
        all_outputs.extend(results)

        elapsed = time.time() - start_time
        rate = len(all_outputs) / elapsed if elapsed > 0 else 0
        remaining = len(prompts) - len(all_outputs)
        eta = remaining / rate if rate > 0 else 0
        print(f"    {len(all_outputs)}/{len(prompts)} done ({rate:.1f}/sec, ETA {eta/60:.1f}min)", flush=True)

    # Save raw outputs for re-validation
    raw_path = output_path.with_suffix(".raw.jsonl")
    with open(raw_path, "w") as f:
        for inp, target in zip(cleaned_inputs, all_outputs):
            f.write(json.dumps({"input": inp, "target": target}) + "\n")
    print(f"Saved {len(all_outputs)} raw pairs to {raw_path}")

    # Validate and write
    print(f"\nValidating {len(all_outputs)} pairs...")
    valid = 0
    rejected_counts = {}

    with open(output_path, "w") as f:
        for inp, target in zip(cleaned_inputs, all_outputs):
            ok, reason = validate_pair(inp, target)
            if ok and valid < limit:
                t5_pair = {
                    "input": f"{T5_PREFIX}{inp}",
                    "target": target,
                }
                f.write(json.dumps(t5_pair) + "\n")
                valid += 1
            elif not ok:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Done! {valid} pairs in {output_path}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Cost estimate: ~${elapsed / 3600 * 1.40:.2f} (L40S @ $1.40/hr)")

    # Word count distribution
    with open(output_path) as f:
        pairs = [json.loads(l) for l in f]
    word_counts = [len(p['input'].replace(T5_PREFIX, '').split()) for p in pairs]
    print(f"\nWord count distribution:")
    print(f"  <25 words: {sum(1 for w in word_counts if w < 25)}")
    print(f"  25-50 words: {sum(1 for w in word_counts if 25 <= w < 50)}")
    print(f"  50-100 words: {sum(1 for w in word_counts if 50 <= w < 100)}")
    print(f"  100+ words: {sum(1 for w in word_counts if w >= 100)}")

    if rejected_counts:
        print(f"\nRejection reasons:")
        for reason, count in sorted(rejected_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
