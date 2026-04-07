"""
Curate the final T5 training dataset from existing data + generated spoken grammar.

Assembles ~2000 pairs:
  - 500 self-correction (best from existing 1,219)
  - 500 spoken grammar (from Modal generation)
  - 300 sentence merging (sampled from existing 3,826)
  - 300 mixed (sampled from existing)
  - 200 passthrough (identity examples — model should not change these)
  - 100 questions (sampled from existing)
  - 50 stutters (sampled from existing)
  - ~50 misc (whatever remains)

Usage:
    python curate_t5_dataset.py
    python curate_t5_dataset.py --spoken-grammar data/training_spoken_grammar.jsonl
"""

import json
import random
import re
import argparse
from pathlib import Path
from difflib import SequenceMatcher

T5_PREFIX = "Rewrite as typed text: "

CORRECTION_SIGNALS = [
    "wait", "i mean", "actually no", "actually,", "no,", "no wait",
    "sorry", "scratch that", "never mind", "nevermind", "or rather",
    "or actually", "oh wait",
]
MERGE_SIGNALS = [" and then ", " and i ", " and we ", " and she ", " and he ", " and they "]
STUTTER_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
QUESTION_STARTS = [
    "what ", "when ", "where ", "who ", "how ", "why ",
    "do you ", "are you ", "can you ", "is it ", "will you ",
    "have you ", "does ", "did ", "could you ", "would you ",
]


def categorize(inp, out):
    """Categorize a training pair."""
    inp_lower = inp.lower()
    out_lower = out.lower()
    sim = SequenceMatcher(None, inp_lower, out_lower).ratio()

    has_correction = any(sig in inp_lower for sig in CORRECTION_SIGNALS)
    has_merge = any(sig in inp_lower for sig in MERGE_SIGNALS)
    has_stutter = bool(STUTTER_RE.search(inp_lower))
    has_question = any(inp_lower.strip().lower().startswith(q) for q in QUESTION_STARTS)
    is_passthrough = sim > 0.95

    if has_correction:
        return "self_correction"
    if is_passthrough:
        return "passthrough"
    if has_merge:
        return "sentence_merging"
    if has_stutter:
        return "stutter"
    if has_question:
        return "question"
    return "mixed"


def quality_score(inp, out):
    """Score pair quality. Higher = better. Returns (score, reason)."""
    sim = SequenceMatcher(None, inp.lower(), out.lower()).ratio()
    length_ratio = len(out) / max(len(inp), 1)

    # Reject over-rewrites
    if length_ratio < 0.3:
        return -1, "output too short"
    if sim < 0.3:
        return -1, "too different"
    if length_ratio > 1.5:
        return -1, "output too long"

    # Reject markdown/formatting
    if re.search(r"(\*\*|^#{1,3}\s|^[-*]\s|^\d+\.\s)", out, re.MULTILINE):
        return -1, "has markdown"
    if re.search(r"\[.*?\]", out):
        return -1, "has brackets"
    # Reject email-style formatting
    if "\n\n" in out and "\n\n" not in inp:
        return -1, "added formatting"

    # Score: prefer pairs where output is meaningfully different but not too different
    score = 0
    if 0.5 < sim < 0.95:
        score += 2  # Good transformation
    elif sim >= 0.95:
        score += 1  # Passthrough (still useful)

    # Prefer shorter examples (more focused)
    word_count = len(inp.split())
    if word_count <= 30:
        score += 1

    return score, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spoken-grammar", default="data/training_spoken_grammar.jsonl")
    parser.add_argument("--existing", default="data/training_pairs_clean.jsonl")
    parser.add_argument("--output", default="data/training_t5_final.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load existing data
    print("Loading existing training data...")
    with open(args.existing) as f:
        existing = [json.loads(l) for l in f]
    print(f"  {len(existing)} pairs loaded")

    # Categorize and quality-filter
    categorized = {}
    rejected = 0
    for pair in existing:
        inp, out = pair["input"], pair["output"]
        score, reason = quality_score(inp, out)
        if score < 0:
            rejected += 1
            continue
        cat = categorize(inp, out)
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append((pair, score))

    print(f"  {rejected} rejected for quality")
    for cat, items in sorted(categorized.items()):
        print(f"  {cat}: {len(items)}")

    # Sort each category by quality score (best first)
    for cat in categorized:
        categorized[cat].sort(key=lambda x: x[1], reverse=True)

    # Select from each category
    TARGETS = {
        "self_correction": 500,
        "sentence_merging": 300,
        "mixed": 300,
        "passthrough": 200,
        "question": 100,
        "stutter": 50,
    }

    selected = []

    for cat, target in TARGETS.items():
        available = categorized.get(cat, [])
        n = min(target, len(available))
        # Take top-scored, then shuffle to avoid ordering bias
        chosen = [pair for pair, score in available[:n]]
        random.shuffle(chosen)
        selected.extend([(cat, pair) for pair in chosen])
        print(f"  Selected {n}/{target} for {cat}")

    # Load spoken grammar supplement
    spoken_grammar_path = Path(args.spoken_grammar)
    if spoken_grammar_path.exists():
        print(f"\nLoading spoken grammar data from {spoken_grammar_path}...")
        with open(spoken_grammar_path) as f:
            grammar_pairs = [json.loads(l) for l in f]
        print(f"  {len(grammar_pairs)} pairs loaded")

        # These are already in T5 format (input has prefix, target is clean)
        for pair in grammar_pairs:
            selected.append(("spoken_grammar", {
                "input": pair["input"].replace(T5_PREFIX, ""),  # strip prefix, we'll re-add
                "target": pair["target"],
            }))
    else:
        print(f"\n  WARNING: {spoken_grammar_path} not found. Run generate_spoken_grammar.py first.")
        print(f"  Continuing without spoken grammar examples.")

    # Shuffle everything
    random.shuffle(selected)

    # Write final dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cat_counts = {}
    with open(output_path, "w") as f:
        for cat, pair in selected:
            # Format as T5 pair
            if "target" in pair:
                inp = pair.get("input", "")
                target = pair["target"]
            else:
                inp = pair["input"]
                target = pair["output"]

            t5_pair = {
                "input": f"{T5_PREFIX}{inp}",
                "target": target,
            }
            f.write(json.dumps(t5_pair) + "\n")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    total = sum(cat_counts.values())
    print(f"\n=== Final Dataset: {output_path} ===")
    print(f"Total: {total} pairs")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/total*100:.0f}%)")


if __name__ == "__main__":
    main()
