"""
Generate spoken grammar training pairs for FLAN-T5 fine-tuning via Modal.

Generates pairs where input is post-regex ASR output with spoken grammar issues,
and output is clean written text.

Usage:
    python -m modal run generate_spoken_grammar.py
    python -m modal run generate_spoken_grammar.py --pairs 500
"""

import json
import re
import time
from pathlib import Path
from difflib import SequenceMatcher

import modal


# ── Inlined Python port of cleanup.rs (must match Rust exactly) ───────
I = re.IGNORECASE

_FILLER_PATTERNS = [
    re.compile(r"\bum+\b", I), re.compile(r"\buh+\b", I),
    re.compile(r"\buh huh\b", I), re.compile(r"\bmm+ ?hmm+\b", I),
    re.compile(r"\bhmm+\b", I), re.compile(r"\byou know\b(?=\s*,?\s)", I),
    re.compile(r"\blike\b(?=\s+(the|a|an|i|we|they|he|she|it|my|our|this|that)\b)", I),
    re.compile(r"\bbasically\b(?=\s*,)", I), re.compile(r"\bactually\b(?=\s*,)", I),
    re.compile(r"\bso\b(?=\s*,\s)", I), re.compile(r"\bi mean\b(?=\s*,)", I),
    re.compile(r"\bkind of\b(?=\s+(like|a|the)\b)", I),
    re.compile(r"\bsort of\b(?=\s+(like|a|the)\b)", I),
    re.compile(r"\bright\s*\?\s*(?=\b)", I),
]

_SPOKEN_PUNCTUATION = [
    (re.compile(r"\bperiod\b", I), "."), (re.compile(r"\bcomma\b", I), ","),
    (re.compile(r"\bquestion mark\b", I), "?"),
    (re.compile(r"\bexclamation (?:mark|point)\b", I), "!"),
    (re.compile(r"\bcolon\b", I), ":"), (re.compile(r"\bsemicolon\b", I), ";"),
    (re.compile(r"\bdash\b", I), " —"), (re.compile(r"\bhyphen\b", I), "-"),
    (re.compile(r"\bopen (?:paren|parenthesis)\b", I), "("),
    (re.compile(r"\bclose (?:paren|parenthesis)\b", I), ")"),
    (re.compile(r"\bnew line\b", I), "\n"), (re.compile(r"\bnew paragraph\b", I), "\n\n"),
]

_NUMBER_WORDS = [
    (r"\bzero\b", "0"), (r"\bone\b", "1"), (r"\btwo\b", "2"), (r"\bthree\b", "3"),
    (r"\bfour\b", "4"), (r"\bfive\b", "5"), (r"\bsix\b", "6"), (r"\bseven\b", "7"),
    (r"\beight\b", "8"), (r"\bnine\b", "9"), (r"\bten\b", "10"),
]
_NUMERIC_CONTEXTS = [
    r"\b(number|step|item|option|version|v|chapter|page|line|row|column|level|grade|score|count|total)\s+",
    r"\b(is|are|was|were|equals?|=)\s+",
    r"\b(about|around|approximately|roughly|nearly|over|under)\s+",
]
_NUMERIC_COMPILED = []
for _ctx in _NUMERIC_CONTEXTS:
    for _wp, _d in _NUMBER_WORDS:
        _NUMERIC_COMPILED.append((re.compile(f"({_ctx})({_wp})", I), _d))

_PCTG_MAP = {"twenty":"20","thirty":"30","forty":"40","fifty":"50","sixty":"60","seventy":"70","eighty":"80","ninety":"90"}
_PERCENTAGE_RE = re.compile(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+percent\b", I)
_HUNDRED_PCT_RE = re.compile(r"\b(one )?hundred percent\b", I)
_DANGLING_COMMA_RE = re.compile(r",\s*,")
_LEADING_COMMA_RE = re.compile(r"^\s*,\s*")
_WHITESPACE_RE = re.compile(r"\s{2,}")
_SENTENCE_END_RE = re.compile(r"([.!?])\s+([a-z])")
_STANDALONE_I_RE = re.compile(r"\bi\b")
_I_CONTRACTION_RE = re.compile(r"\bI'([msdtv])")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:)])")
_NO_SPACE_AFTER_RE = re.compile(r"([.,!?;:])([A-Za-z])")
_EMAIL_RE = re.compile(r"\b(\w+)\s+at\s+(\w+)\s+dot\s+(com|org|net|io|dev|co)\b", I)

def cleanup_text_python(text):
    if not text:
        return ""
    result = text
    for filler in _FILLER_PATTERNS:
        result = filler.sub("", result)
    result = _DANGLING_COMMA_RE.sub(",", result)
    result = _LEADING_COMMA_RE.sub("", result)
    result = _WHITESPACE_RE.sub(" ", result.strip())
    # smart_format
    for ctx_re, digit in _NUMERIC_COMPILED:
        result = ctx_re.sub(lambda m, d=digit: f"{m.group(1)}{d}", result)
    result = _PERCENTAGE_RE.sub(lambda m: f"{_PCTG_MAP.get(m.group(1).lower(), m.group(1))}%", result)
    result = _HUNDRED_PCT_RE.sub("100%", result)
    for pattern, replacement in _SPOKEN_PUNCTUATION:
        result = pattern.sub(replacement, result)
    result = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", result)
    result = _NO_SPACE_AFTER_RE.sub(r"\1 \2", result)
    result = _EMAIL_RE.sub(r"\1@\2.\3", result)
    result = result.strip()
    if result:
        result = result[0].upper() + result[1:]
    trimmed = result.rstrip()
    if trimmed and trimmed[-1] not in '.!?:;")\n':
        result = trimmed + "."
    result = _SENTENCE_END_RE.sub(lambda m: f"{m.group(1)} {m.group(2).upper()}", result)
    result = _STANDALONE_I_RE.sub("I", result)
    result = _I_CONTRACTION_RE.sub(lambda m: f"I'{m.group(1)}", result)
    return result

TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
GPU = "L40S"

app = modal.App("chirp-spoken-grammar")

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
T5_PREFIX = "Rewrite as typed text: "

RAW_SPEECH_PROMPT = """Generate exactly {batch_size} examples of raw speech as a JSON array. Each example is what someone might say out loud while dictating into a microphone.

Category: {category_name}
{category_description}

The speech should be REALISTIC — lowercase, no punctuation, natural speech patterns. This is what a speech-to-text engine outputs.

RULES:
1. All lowercase, no punctuation
2. 10-50 words each, lean toward 15-30
3. Vary domains: work emails, personal tasks, technical discussions, casual messages
4. Sound like REAL people talking
5. Output ONLY a JSON array of strings

[
  "i was gonna send the report but i gotta finish the charts first",
  "me and john are gonna meet up at the coffee shop around three",
  ...
]"""

CLEANUP_PROMPT = """You are a speech-to-text cleanup tool. The following text has been partially cleaned by a regex stage (capitalized, periods added, fillers removed).

Clean it so it reads like properly written text. Rules:
1. Fix spoken grammar for written text:
   - "gonna" -> "going to"
   - "wanna" -> "want to"
   - "gotta" -> "got to" or "have to"
   - "kinda" -> "kind of"
   - "shoulda/coulda/woulda" -> "should have/could have/would have"
   - "me and X" -> "X and I" (when subject)
   - "there's three" -> "there are three"
   - "less items" -> "fewer items"
   - Add missing articles where needed ("need to update database" -> "need to update the database")
   - Fix informal fragments
2. If there are self-corrections (wait, I mean, actually, no, sorry, scratch that), discard the WRONG part, keep ONLY the corrected version
3. Merge choppy "And then..." chains into flowing sentences
4. Remove stutters and repeated words
5. Preserve meaning — don't add words the speaker didn't say
6. Output PLAIN TEXT only, no markdown

Input: {post_regex_text}

Output ONLY the cleaned text, nothing else."""

CATEGORIES = [
    {
        "name": "spoken_grammar",
        "weight": 5,
        "description": (
            "Speech with informal grammar that's acceptable when spoken but wrong in writing:\n"
            "- 'gonna/wanna/gotta/kinda/shoulda/coulda/woulda'\n"
            "- 'me and john went' instead of 'john and i went'\n"
            "- 'there's three options' instead of 'there are three options'\n"
            "- 'less people' instead of 'fewer people'\n"
            "- Missing articles: 'need to update database' instead of 'need to update the database'\n"
            "- 'who did you talk to' (dangling preposition)\n"
            "- Sentence fragments and run-ons\n"
            "IMPORTANT: Do NOT include filler words (um, uh) — they're already removed. "
            "Focus purely on grammar issues."
        ),
    },
    {
        "name": "self_correction_grammar",
        "weight": 3,
        "description": (
            "Speech that combines self-corrections WITH spoken grammar issues. "
            "Must include a correction signal word (wait, I mean, actually, no, sorry, scratch that, never mind) "
            "AND at least one spoken grammar issue (gonna, wanna, missing article, etc). "
            "Example: 'me and sarah are gonna meet at two no wait three pm'"
        ),
    },
    {
        "name": "merging_grammar",
        "weight": 2,
        "description": (
            "Choppy speech with 'and then' chains that ALSO has spoken grammar issues. "
            "Example: 'i gotta go to the store and then i gotta pick up the kids and then we're gonna go home' "
            "The output should merge the sentences AND fix the grammar."
        ),
    },
]


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
        return [s for s in raw_speeches if isinstance(s, str) and 4 <= len(s.split()) <= 80]

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


MARKDOWN_RE = re.compile(r"(\*\*|^#{1,3}\s|^[-*]\s|^\d+\.\s)", re.MULTILINE)
CORRECTION_SIGNALS = [
    "wait", "i mean", "actually", "no,", "no ", "sorry", "scratch that",
    "never mind", "nevermind", "or rather", "or actually",
]


def validate_pair(raw_speech, post_regex, clean_output, category_name):
    if not post_regex or not clean_output:
        return False, "empty"
    if len(clean_output) > len(post_regex) * 1.8:
        return False, "output too long"
    if MARKDOWN_RE.search(clean_output):
        return False, "has markdown"
    if re.search(r"\[.*?\]", clean_output):
        return False, "has brackets"
    # For grammar categories, output should actually differ from input
    if category_name == "spoken_grammar":
        sim = SequenceMatcher(None, post_regex.lower(), clean_output.lower()).ratio()
        if sim > 0.98:
            return False, "no grammar fix applied"
    return True, "ok"


def weighted_choice(categories):
    total = sum(c["weight"] for c in categories)
    import random
    r = random.uniform(0, total)
    cumulative = 0
    for cat in categories:
        cumulative += cat["weight"]
        if r <= cumulative:
            return cat
    return categories[-1]


@app.local_entrypoint()
def main(
    pairs: int = 500,
    batch_size: int = 15,
    output: str = "data/training_spoken_grammar.jsonl",
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
        print(f"Already have {existing} pairs. Done!")
        return

    file_mode = "a" if resume else "w"
    total_weight = sum(c["weight"] for c in CATEGORIES)

    print(f"=== Spoken Grammar Data Generator ===")
    print(f"Target: {remaining} pairs")
    print(f"Teacher: {TEACHER_MODEL} on {GPU}")
    print(f"T5 prefix: '{T5_PREFIX}'")
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

            raw_speeches = teacher.generate_raw_speech.remote(
                category_name=category["name"],
                category_description=category["description"],
                batch_size=batch_size,
            )
            if not raw_speeches:
                continue

            post_regex_texts = []
            raw_for_validation = []
            for raw in raw_speeches:
                processed = cleanup_text_python(raw)
                if processed and len(processed.split()) >= 3:
                    post_regex_texts.append(processed)
                    raw_for_validation.append(raw)
            if not post_regex_texts:
                continue

            clean_outputs = teacher.generate_clean_outputs.remote(post_regex_texts)

            batch_valid = 0
            batch_rejected = 0
            for raw, post_regex, clean in zip(raw_for_validation, post_regex_texts, clean_outputs):
                ok, reason = validate_pair(raw, post_regex, clean, category["name"])
                if ok:
                    t5_pair = {
                        "input": f"{T5_PREFIX}{post_regex}",
                        "target": clean,
                    }
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
    print(f"\nCategory distribution:")
    for name, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    if rejected_counts:
        print(f"\nRejection reasons:")
        for reason, count in sorted(rejected_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
