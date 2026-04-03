"""
Generate high-quality training pairs for FLAN-T5 cleanup model v2.

Single-pass approach: teacher generates both input and target in one call,
anchored by few-shot examples showing the exact edit level we want.

Usage:
    python -m modal run generate_t5_v2.py --pairs 5
    python -m modal run generate_t5_v2.py --pairs 100 --resume
    python -m modal run generate_t5_v2.py --pairs 1500 --resume
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

app = modal.App("chirp-t5-datagen-v2")

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

# ── Few-shot prompt ──────────────────────────────────────────────────

GENERATE_PROMPT_TEMPLATE = string.Template("""\
You generate training pairs for a speech-to-text cleanup model. Each pair has an "input" (what a speech-to-text engine outputs after basic regex cleanup — capitalized, periods added, filler words like "um/uh" already removed) and a "target" (lightly polished version).

The target should read like the person TYPED it, not like someone else rewrote it. Make MINIMAL edits:
- Fix spoken grammar: "gonna" → "going to", "wanna" → "want to", "gotta" → "got to", "me and X" → "X and I"
- Resolve self-corrections: when the speaker says "wait", "no", "I mean", "actually", discard the wrong part, keep ONLY the corrected version
- Remove stutters and repeated words ("we we need" → "we need")
- Fix punctuation and add question marks where needed
- Normalize spoken numbers to digits when natural (twenty three → 23, but "a couple" stays as-is)
- Add missing articles where clearly needed ("need to update database" → "need to update the database")

DO NOT:
- Summarize or shorten the text
- Restructure or reorder sentences
- Change the speaker's word choices or voice
- Add words the speaker didn't say
- Merge multiple sentences into one
- Remove hedging or qualifiers ("I think", "probably", "kind of")
- Make it sound "more professional" — keep it natural

Here are examples of correct input/target pairs:

{"input": "I think we should probably move the meeting to Thursday, it's gonna work better for everyone I think.", "target": "I think we should probably move the meeting to Thursday. It's going to work better for everyone, I think."}
{"input": "Me and Sarah are gonna go to the store and then we're gonna pick up the kids after that.", "target": "Sarah and I are going to go to the store and then pick up the kids after that."}
{"input": "The budget is about twenty three thousand and we need to I mean we should allocate at least five thousand for marketing.", "target": "The budget is about $$23,000 and we should allocate at least $$5,000 for marketing."}
{"input": "So I was talking to the client yesterday and they said that they want the project done by Friday no wait they said Monday. So we need to make sure we're ready by then.", "target": "I was talking to the client yesterday and they said they want the project done by Monday. So we need to make sure we're ready by then."}
{"input": "Can you send me the report when you get a chance.", "target": "Can you send me the report when you get a chance?"}
{"input": "The deployment went smoothly and everything is working as expected.", "target": "The deployment went smoothly and everything is working as expected."}
{"input": "I was thinking that we could probably set up the CI pipeline to run the tests automatically and then it would deploy to staging if everything passes and then we could just review it there before pushing to production.", "target": "I was thinking that we could probably set up the CI pipeline to run the tests automatically, and then it would deploy to staging if everything passes. Then we could just review it there before pushing to production."}
{"input": "The the problem is that we don't have enough data to make a decision yet so I think we should wait until next week when we get the results back from the survey.", "target": "The problem is that we don't have enough data to make a decision yet, so I think we should wait until next week when we get the results back from the survey."}

IMPORTANT: Do NOT copy or closely paraphrase the examples above. Generate completely original content across diverse domains — work, personal life, technical discussions, casual messages, planning, explaining, requesting, storytelling.

Generate exactly $batch_size pairs as a JSON array. Category: $category_name
$category_description

Length distribution: $length_guidance
The inputs should sound like real post-regex speech-to-text output. Targets must be MINIMAL edits — preserve the speaker's voice and all their meaning.
Output ONLY a JSON array of objects with "input" and "target" fields.""")

# ── Categories ────────────────────────────────────────────────────────

CATEGORIES = [
    {
        "name": "self_correction",
        "weight": 5,
        "description": (
            "Speech where the speaker corrects themselves using words like "
            "'wait', 'I mean', 'actually', 'no', 'sorry', 'scratch that'. "
            "The correction can be for names, numbers, times, places, or general statements. "
            "The input includes the mistake AND the correction. The target keeps ONLY the corrected version. "
            "Include the surrounding context — don't just generate the correction part."
        ),
    },
    {
        "name": "spoken_grammar",
        "weight": 4,
        "description": (
            "Speech with informal grammar that's fine when spoken but should be fixed in writing: "
            "'gonna/wanna/gotta/kinda', 'me and X' as subject, missing articles, "
            "dangling prepositions, run-on sentences needing punctuation. "
            "The target fixes grammar but keeps the same words and meaning."
        ),
    },
    {
        "name": "punctuation_flow",
        "weight": 3,
        "description": (
            "Longer speech that's grammatically OK but needs better punctuation — "
            "missing commas, run-on sentences that need periods or semicolons, "
            "questions without question marks. The content is fine, just needs "
            "punctuation to read naturally. Target should add punctuation only."
        ),
    },
    {
        "name": "stutter_repetition",
        "weight": 2,
        "description": (
            "Speech with repeated words ('we we need'), repeated starts "
            "('the thing is the thing is'), or false starts where the speaker "
            "begins a thought and restarts. Target removes the repetition."
        ),
    },
    {
        "name": "passthrough",
        "weight": 3,
        "description": (
            "Clean, well-formed speech that needs NO modification at all. "
            "These are clear statements, questions, or instructions that the "
            "regex already handled perfectly. Target is IDENTICAL to input. "
            "This is critical — the model must learn that doing nothing is correct."
        ),
    },
    {
        "name": "mixed",
        "weight": 4,
        "description": (
            "Realistic speech combining 2-3 issues: a self-correction with spoken grammar, "
            "stutters with punctuation problems, etc. Real dictation is messy and "
            "rarely has just one issue. Include longer examples (60-120 words)."
        ),
    },
    {
        "name": "numbers_and_formatting",
        "weight": 2,
        "description": (
            "Speech with spoken numbers ('twenty three'), dollar amounts ('five hundred dollars'), "
            "times ('two thirty pm'), dates ('march fifteenth'), percentages ('thirty percent'). "
            "Target normalizes to digits/symbols where appropriate."
        ),
    },
]

LENGTH_GUIDANCE = {
    "short": "Generate short examples (10-30 words). 1-2 sentences.",
    "medium": "Generate medium examples (30-60 words). 2-4 sentences.",
    "long": "Generate longer examples (60-120 words). 4-6 sentences. This is the most important length — real dictation is often this long.",
    "very_long": "Generate very long examples (120-250 words). 6-12+ sentences. Extended dictation like composing an email, explaining a plan, giving instructions, or telling a story. Real users dictate paragraphs this long regularly.",
}

# Weight toward longer: 15% short, 25% medium, 40% long, 20% very_long
LENGTH_WEIGHTS = [
    ("short", 15),
    ("medium", 25),
    ("long", 40),
    ("very_long", 20),
]


def pick_length():
    total = sum(w for _, w in LENGTH_WEIGHTS)
    r = random.uniform(0, total)
    cumulative = 0
    for name, weight in LENGTH_WEIGHTS:
        cumulative += weight
        if r <= cumulative:
            return name, LENGTH_GUIDANCE[name]
    return "long", LENGTH_GUIDANCE["long"]


def weighted_choice(categories):
    total = sum(c["weight"] for c in categories)
    r = random.uniform(0, total)
    cumulative = 0
    for cat in categories:
        cumulative += cat["weight"]
        if r <= cumulative:
            return cat
    return categories[-1]


# ── Validation ────────────────────────────────────────────────────────

LENGTH_MINIMUMS = {
    "short": 10,
    "medium": 25,
    "long": 50,
    "very_long": 120,
}


def validate_pair(inp, target, category_name, length_bucket="medium"):
    """Validate a single training pair. Returns (ok, reason)."""
    if not inp or not target:
        return False, "empty"

    inp_words = inp.lower().split()
    target_words = target.lower().split()

    if len(inp_words) < 3 or len(target_words) < 3:
        return False, "too short"

    # Length bucket check — reject if input is too short for requested bucket
    min_words = LENGTH_MINIMUMS.get(length_bucket, 10)
    if len(inp_words) < min_words:
        return False, f"too short for {length_bucket} (need {min_words}+, got {len(inp_words)})"

    # Similarity check — target shouldn't be too different from input
    sim = SequenceMatcher(None, inp.lower(), target.lower()).ratio()
    if sim < 0.60:
        return False, f"too different (sim={sim:.2f})"

    # Length ratio — target shouldn't be much shorter than input
    length_ratio = len(target_words) / len(inp_words)
    if length_ratio < 0.55:
        return False, f"target too short (ratio={length_ratio:.2f})"
    if length_ratio > 1.3:
        return False, f"target too long (ratio={length_ratio:.2f})"

    # Word preservation — most content words should survive
    # (exclude common stop words from the check)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "and", "or", "but", "not", "so",
        "if", "then", "that", "this", "it", "i", "we", "you", "they", "he",
        "she", "my", "our", "your", "their", "its",
    }
    inp_content = set(w for w in inp_words if w not in stop_words and len(w) > 2)
    target_content = set(w for w in target_words if w not in stop_words and len(w) > 2)

    if inp_content:
        preserved = len(inp_content & target_content) / len(inp_content)
        if preserved < 0.60:
            return False, f"too many content words lost ({preserved:.0%})"

    # No-hallucination check — target shouldn't add significant new words
    if target_content:
        new_words = target_content - inp_content
        # Allow small additions (articles, connecting words that got added)
        # but flag if many new content words appeared
        new_ratio = len(new_words) / max(len(target_content), 1)
        if new_ratio > 0.25:
            return False, f"too many new words added ({new_ratio:.0%})"

    # Reject markdown/formatting
    if re.search(r"(\*\*|^#{1,3}\s|^[-*]\s)", target, re.MULTILINE):
        return False, "has markdown"

    # Passthrough check — passthrough pairs should be near-identical
    if category_name == "passthrough" and sim < 0.95:
        return False, f"passthrough too different (sim={sim:.2f})"

    return True, "ok"


# ── vLLM teacher ──────────────────────────────────────────────────────

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
    def generate_pairs(self, category_name, category_description, length_guidance, batch_size=15):
        import vllm
        prompt = GENERATE_PROMPT_TEMPLATE.substitute(
            batch_size=batch_size,
            category_name=category_name,
            category_description=category_description,
            length_guidance=length_guidance,
        )
        outputs = self.llm.generate(
            [prompt],
            vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=3500),
        )
        text = outputs[0].outputs[0].text.strip()

        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            pairs = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []

        return [p for p in pairs if isinstance(p, dict) and "input" in p and "target" in p]

    @modal.exit()
    def stop(self):
        del self.llm


# ── Main ──────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    pairs: int = 5,
    batch_size: int = 15,
    output: str = "data/training_t5_v2.jsonl",
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
    actual_batch = min(batch_size, remaining)

    total_weight = sum(c["weight"] for c in CATEGORIES)
    print(f"=== Chirp T5 Data Generator v2 (Single-Pass) ===")
    print(f"Target: {remaining} new pairs ({pairs} total)")
    print(f"Teacher: {TEACHER_MODEL} on {GPU}")
    print(f"Batch size: {actual_batch}")
    print(f"\nCategory weights:")
    for cat in CATEGORIES:
        pct = cat["weight"] / total_weight * 100
        print(f"  {cat['name']}: {pct:.0f}%")
    print()

    teacher = TeacherModel()
    total_generated = existing
    category_counts = {}
    rejected_counts = {}
    length_counts = {}
    start_time = time.time()

    with open(output_path, file_mode) as f:
        while total_generated < pairs:
            category = weighted_choice(CATEGORIES)
            length_name, length_guide = pick_length()
            # Scale batch size down for longer examples to fit in 4096 context
            length_batch_limits = {"short": batch_size, "medium": batch_size, "long": 5, "very_long": 3}
            actual_batch = min(length_batch_limits.get(length_name, batch_size), pairs - total_generated)

            raw_pairs = teacher.generate_pairs.remote(
                category_name=category["name"],
                category_description=category["description"],
                length_guidance=length_guide,
                batch_size=actual_batch,
            )

            if not raw_pairs:
                print(f"  [{category['name']}] No pairs returned, retrying...")
                continue

            batch_valid = 0
            batch_rejected = 0
            for pair in raw_pairs:
                inp = pair["input"].strip()
                target = pair["target"].strip()

                ok, reason = validate_pair(inp, target, category["name"], length_name)
                if ok:
                    t5_pair = {
                        "input": f"{T5_PREFIX}{inp}",
                        "target": target,
                    }
                    f.write(json.dumps(t5_pair) + "\n")
                    batch_valid += 1
                    category_counts[category["name"]] = category_counts.get(category["name"], 0) + 1
                    length_counts[length_name] = length_counts.get(length_name, 0) + 1
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
                f"  [{category['name']}/{length_name}] {batch_valid} valid{rej_str} | "
                f"Total: {total_generated}/{pairs} "
                f"({rate:.1f}/sec, ETA {eta/60:.1f}min)",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Done! {total_generated} pairs in {output_path}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Cost estimate: ~${elapsed / 3600 * 1.40:.2f} (L40S @ $1.40/hr)")
    print(f"\nCategory distribution:")
    for name, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print(f"\nLength distribution:")
    for name, count in sorted(length_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    if rejected_counts:
        print(f"\nRejection reasons:")
        for reason, count in sorted(rejected_counts.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
