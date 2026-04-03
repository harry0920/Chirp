"""
Generate long training pairs by concatenating 2-4 shorter pairs.

Takes the validated training pairs and stitches them together to create
realistic long-form dictation examples (75-250 words). No GPU needed.

Usage:
    python concat_long_pairs.py
    python concat_long_pairs.py --input data/training_t5_v2.jsonl --count 300
"""

import json
import random
import argparse
from pathlib import Path

T5_PREFIX = "Rewrite as typed text: "


def concat_pairs(pairs, target_words=150):
    """Concatenate random pairs until we hit target word count."""
    random.shuffle(pairs)

    combined_input = []
    combined_target = []
    word_count = 0
    used = 0

    for p in pairs:
        inp = p["input"].replace(T5_PREFIX, "")
        target = p["target"]

        inp_words = len(inp.split())
        if word_count + inp_words > target_words * 1.3:
            if word_count >= target_words * 0.7:
                break
            continue

        combined_input.append(inp)
        combined_target.append(target)
        word_count += inp_words
        used += 1

        if word_count >= target_words:
            break

    if word_count < 50:
        return None, None, 0

    return " ".join(combined_input), " ".join(combined_target), used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_t5_v2.jsonl")
    parser.add_argument("--output", default="data/training_t5_v2_long.jsonl")
    parser.add_argument("--count", type=int, default=300, help="Number of long pairs to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        pairs = [json.loads(l) for l in f]
    print(f"Loaded {len(pairs)} source pairs")

    # Length targets: mix of medium-long and very-long
    length_targets = []
    for _ in range(args.count):
        r = random.random()
        if r < 0.3:
            length_targets.append(random.randint(75, 100))
        elif r < 0.6:
            length_targets.append(random.randint(100, 150))
        elif r < 0.85:
            length_targets.append(random.randint(150, 200))
        else:
            length_targets.append(random.randint(200, 250))

    output_path = Path(args.output)
    generated = 0
    word_counts = []

    with open(output_path, "w") as f:
        for target_words in length_targets:
            # Shuffle source pairs each time for variety
            pool = pairs.copy()
            random.shuffle(pool)

            inp, target, used = concat_pairs(pool, target_words)
            if inp is None:
                continue

            t5_pair = {
                "input": f"{T5_PREFIX}{inp}",
                "target": target,
            }
            f.write(json.dumps(t5_pair) + "\n")
            generated += 1
            word_counts.append(len(inp.split()))

    print(f"\nGenerated {generated} long pairs in {output_path}")
    print(f"Word counts: min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts)/len(word_counts):.0f}")
    print(f"  75-100 words:  {sum(1 for w in word_counts if 75 <= w < 100)}")
    print(f"  100-150 words: {sum(1 for w in word_counts if 100 <= w < 150)}")
    print(f"  150-200 words: {sum(1 for w in word_counts if 150 <= w < 200)}")
    print(f"  200+ words:    {sum(1 for w in word_counts if w >= 200)}")


if __name__ == "__main__":
    main()
