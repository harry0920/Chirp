"""
Generate training data for FLAN-T5-small fine-tuning via distillation on Modal.

Three-pass approach (same as v2):
  Pass 1: 72B teacher generates raw speech (as Parakeet would hear it)
  Pass 2: Python port of cleanup.rs processes it (exact same regex pipeline)
  Pass 3: 72B teacher generates the ideal clean output

Output format for T5:
  {"input": "Fix the grammar: <post-regex text>", "target": "<clean output>"}

Categories weighted heavily toward what T5-small can't do yet:
  - self_correction (30%) — the biggest gap
  - sentence_merging (20%) — second biggest gap
  - spoken_grammar (20%) — "I was gonna" -> "I was going to", etc.
  - mixed (20%) — realistic combinations
  - stutter/passthrough/question/proper_nouns (10%) — reinforce existing skills

Usage:
    python -m modal run generate_data_t5.py --pairs 2000
    python -m modal run generate_data_t5.py --pairs 5000 --resume
"""

import json
import random
import re
import time
from pathlib import Path
from difflib import SequenceMatcher

import modal

# Reuse the regex pipeline from v2
from generate_data_v2 import (
    cleanup_text_python,
    CORRECTION_SIGNALS,
    MARKDOWN_RE_VAL,
)

# ── Modal setup ────────────────────────────────────────────────────────

TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
GPU = "L40S"

app = modal.App("chirp-t5-training-data")

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

# ── T5 prompt prefix (must match what we use at inference) ────────────
T5_PREFIX = "Fix the grammar: "

# ── Categories (weighted toward T5-small's weaknesses) ────────────────

CATEGORIES = [
    {
        "name": "self_correction",
        "weight": 6,
        "description": (
            "The speaker corrects themselves mid-speech using signal words like "
            "'wait', 'I mean', 'actually', 'no', 'sorry', 'scratch that', 'never mind', 'or rather'. "
            "Include the signal words in the speech. Vary the position. Include corrections of names, "
            "numbers, times, places, and general statements. "
            "IMPORTANT: the correction signal word MUST appear in the speech."
        ),
    },
    {
        "name": "sentence_merging",
        "weight": 4,
        "description": (
            "Choppy, disconnected speech with many short sentences connected by 'and', "
            "'and then', 'also', 'plus'. The kind of rambling where someone strings "
            "together thoughts one at a time instead of forming a proper sentence. "
            "These should have 3-6 short clauses that could be merged into 1-2 sentences."
        ),
    },
    {
        "name": "spoken_grammar",
        "weight": 4,
        "description": (
            "Speech that is grammatically acceptable when spoken but incorrect in writing. Examples:\n"
            "- 'gonna' instead of 'going to'\n"
            "- 'wanna' instead of 'want to'\n"
            "- 'gotta' instead of 'got to' or 'have to'\n"
            "- 'kinda' instead of 'kind of'\n"
            "- 'shoulda/coulda/woulda' instead of 'should have/could have/would have'\n"
            "- Dangling prepositions ('who did you talk to' is fine spoken, 'whom did you speak with' in writing)\n"
            "- Subject-verb disagreement ('there's three options')\n"
            "- Run-on sentences with no clear boundary\n"
            "- Missing articles ('need to update database' vs 'need to update the database')\n"
            "- Informal contractions and fragments\n"
            "- 'me and John' instead of 'John and I'\n"
            "- 'less' when it should be 'fewer'\n"
            "Do NOT use filler words (um, uh) — those are already removed by the regex stage."
        ),
    },
    {
        "name": "mixed",
        "weight": 4,
        "description": (
            "Realistic speech that combines 2-3 issues: a self-correction AND choppy sentences, "
            "or spoken grammar AND a stutter, or a question with poor grammar AND a self-correction. "
            "Real speech is messy and rarely has just one issue. "
            "MUST include at least one self-correction signal word AND at least one grammar issue."
        ),
    },
    {
        "name": "stutter_repetition",
        "weight": 1,
        "description": (
            "Speech with repeated words ('we we need'), repeated phrases "
            "('the thing is the thing is'), or verbal echoes. Keep these short, 8-20 words."
        ),
    },
    {
        "name": "passthrough",
        "weight": 1,
        "description": (
            "Clean, well-formed speech that needs little or no modification. "
            "Short, clear statements. The ideal output should be nearly identical to input. "
            "Include a mix of statements and questions."
        ),
    },
    {
        "name": "question_detection",
        "weight": 1,
        "description": (
            "Questions that need question marks added. Include direct questions "
            "(what, when, where, who, how, do you, are you, can you, is it, will you) "
            "ending with a period instead of question mark."
        ),
    },
    {
        "name": "proper_nouns",
        "weight": 1,
        "description": (
            "Speech mentioning people's names, cities, companies, products — "
            "all lowercase because Parakeet doesn't capitalize. "
            "Keep these short and focused on the capitalization task."
        ),
    },
]

# ── Prompts ────────────────────────────────────────────────────────────

RAW_SPEECH_PROMPT = """Generate exactly {batch_size} examples of raw speech as a JSON array. Each example is what someone might say out loud while dictating into a microphone.

Category: {category_name}
{category_description}

The speech should be REALISTIC — lowercase, no punctuation, includes natural hesitations, and the patterns described above. This is what a speech-to-text engine like Parakeet would output.

RULES:
1. All lowercase (Parakeet outputs lowercase)
2. No punctuation at all (Parakeet doesn't add punctuation)
3. 10-60 words each, mix of lengths, lean toward 15-35
4. Vary domains: work, personal, technical, casual, meetings
5. Sound like REAL people talking, not written text read aloud
6. Output ONLY a JSON array of strings, nothing else

[
  "um so i was thinking we should meet at two no wait i mean three pm",
  "the server went down and then we had to restart it and then we had to check the logs and then we found the issue",
  ...
]"""

CLEANUP_PROMPT = """You are a speech-to-text cleanup tool. The following text has been partially cleaned by a regex stage (capitalized, periods added, fillers removed) but still reads like transcribed speech.

Clean it up so it reads like the person TYPED it instead of said it. Rules:
1. Merge choppy sentences into flowing prose — combine "And then... And then..." chains into single sentences with commas
2. If there are self-corrections (wait, I mean, actually, no, sorry, scratch that), discard the WRONG part and keep ONLY the corrected version
3. Remove stutters and repeated words
4. Fix grammar for WRITTEN text:
   - "gonna" -> "going to", "wanna" -> "want to", "gotta" -> "have to"
   - "me and John" -> "John and I" (when subject)
   - "there's three" -> "there are three"
   - "less items" -> "fewer items"
   - Add missing articles where needed
   - Fix subject-verb agreement
5. Fix capitalization of proper nouns (names, places, companies)
6. Fix question marks where needed
7. Preserve the speaker's vocabulary and meaning — don't add words they didn't say
8. Output PLAIN TEXT only — no markdown, no lists, no headers
9. If the text is already clean, return it as-is

Input: {post_regex_text}

Output ONLY the cleaned text, nothing else. No quotes, no explanation."""


# ── Validation ─────────────────────────────────────────────────────────

def validate_pair(raw_speech, post_regex, clean_output, category_name):
    if not post_regex or not clean_output:
        return False, "empty"

    if len(clean_output) > len(post_regex) * 1.8:
        return False, "output too long"

    if MARKDOWN_RE_VAL.search(clean_output):
        return False, "has markdown"

    if re.search(r"\[.*?\]", clean_output):
        return False, "has brackets"

    if category_name == "self_correction":
        raw_lower = raw_speech.lower()
        has_signal = any(sig in raw_lower for sig in CORRECTION_SIGNALS)
        if not has_signal:
            return False, "no correction signal"

    if category_name == "passthrough":
        similarity = SequenceMatcher(None, post_regex.lower(), clean_output.lower()).ratio()
        if similarity < 0.75:
            return False, f"passthrough too different ({similarity:.2f})"

    return True, "ok"


# ── Helpers ────────────────────────────────────────────────────────────

def format_as_t5_pair(post_regex, clean_output):
    """Format as T5 input/target pair."""
    return {
        "input": f"{T5_PREFIX}{post_regex}",
        "target": clean_output,
    }


def weighted_choice(categories):
    total = sum(c["weight"] for c in categories)
    r = random.uniform(0, total)
    cumulative = 0
    for cat in categories:
        cumulative += cat["weight"]
        if r <= cumulative:
            return cat
    return categories[-1]


# ── vLLM teacher model ─────────────────────────────────────────────────

@app.cls(
    image=vllm_image,
    gpu=GPU,
    timeout=10 * MINUTES,
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
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        print("Model loaded.")

    @modal.method()
    def generate_raw_speech(self, category_name, category_description, batch_size=15):
        import vllm
        prompt = RAW_SPEECH_PROMPT.format(
            batch_size=batch_size,
            category_name=category_name,
            category_description=category_description,
        )
        outputs = self.llm.generate(
            [prompt],
            vllm.SamplingParams(temperature=0.9, top_p=0.95, max_tokens=4096),
        )
        text = outputs[0].outputs[0].text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            raw_speeches = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []
        return [s for s in raw_speeches if isinstance(s, str) and 4 <= len(s.split()) <= 100]

    @modal.method()
    def generate_clean_outputs(self, post_regex_texts):
        import vllm
        prompts = [CLEANUP_PROMPT.format(post_regex_text=t) for t in post_regex_texts]
        outputs = self.llm.generate(
            prompts,
            vllm.SamplingParams(temperature=0.1, top_p=0.95, max_tokens=512),
        )
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]
            results.append(text)
        return results

    @modal.exit()
    def stop(self):
        del self.llm


# ── Main generation loop ───────────────────────────────────────────────

@app.local_entrypoint()
def main(
    pairs: int = 2000,
    batch_size: int = 15,
    output: str = "data/training_t5.jsonl",
    resume: bool = False,
):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = 0
    if resume and output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming from {existing} existing pairs")

    remaining = pairs - existing
    if remaining <= 0:
        print(f"Already have {existing} pairs, target is {pairs}. Done!")
        return

    file_mode = "a" if resume else "w"
    total_weight = sum(c["weight"] for c in CATEGORIES)

    print(f"=== Chirp T5 Training Data Generator ===")
    print(f"Target: {remaining} pairs ({pairs} total)")
    print(f"Teacher: {TEACHER_MODEL} on {GPU}")
    print(f"Format: T5 (input/target pairs)")
    print(f"Prefix: '{T5_PREFIX}'")
    print(f"\nCategory weights:")
    for cat in CATEGORIES:
        pct = cat["weight"] / total_weight * 100
        print(f"  {cat['name']}: {pct:.0f}%")
    print()

    teacher = TeacherModel()
    total_generated = existing
    category_counts = {c["name"]: 0 for c in CATEGORIES}
    rejected_counts = {}
    start_time = time.time()

    with open(output_path, file_mode) as f:
        while total_generated < pairs:
            category = weighted_choice(CATEGORIES)

            # Pass 1: Generate raw speech
            raw_speeches = teacher.generate_raw_speech.remote(
                category_name=category["name"],
                category_description=category["description"],
                batch_size=batch_size,
            )
            if not raw_speeches:
                continue

            # Pass 2: Run through cleanup.rs regex pipeline
            post_regex_texts = []
            raw_for_validation = []
            for raw in raw_speeches:
                processed = cleanup_text_python(raw)
                if processed and len(processed.split()) >= 3:
                    post_regex_texts.append(processed)
                    raw_for_validation.append(raw)
            if not post_regex_texts:
                continue

            # Pass 3: Generate clean outputs
            clean_outputs = teacher.generate_clean_outputs.remote(post_regex_texts)

            # Validate and write pairs
            batch_valid = 0
            batch_rejected = 0
            for raw, post_regex, clean in zip(raw_for_validation, post_regex_texts, clean_outputs):
                ok, reason = validate_pair(raw, post_regex, clean, category["name"])
                if ok:
                    t5_pair = format_as_t5_pair(post_regex, clean)
                    f.write(json.dumps(t5_pair) + "\n")
                    category_counts[category["name"]] += 1
                    batch_valid += 1
                else:
                    batch_rejected += 1
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            f.flush()

            total_generated += batch_valid
            elapsed = time.time() - start_time
            rate = (total_generated - existing) / elapsed if elapsed > 0 else 0
            eta = (pairs - total_generated) / rate if rate > 0 else 0

            rej_str = f", {batch_rejected} rejected" if batch_rejected > 0 else ""
            print(
                f"  [{category['name']}] {batch_valid} valid{rej_str} | "
                f"Total: {total_generated}/{pairs} "
                f"({rate:.1f}/sec, ETA {eta/60:.1f}min)",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Done! Generated {total_generated} pairs in {output_path}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Cost estimate: ~${elapsed / 3600 * 1.40:.2f} (L40S @ $1.40/hr)")
    print(f"\nCategory distribution:")
    for name, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    if rejected_counts:
        print(f"\nRejection reasons:")
        for reason, count in sorted(rejected_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
