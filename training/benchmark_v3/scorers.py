"""
Cleanup-quality metrics for the benchmark_v3 corpus.

Five metrics implemented (BERTScore deferred — heavy dep, weight 0.10):

  1. category_success      rule-based per-category success (must_contain /
                           must_not_contain / max_length_ratio)            w=0.30
  2. edit_f05              ERRANT-style edit F0.5 against the reference   w=0.25
                           (precision weighted 2x recall)
  3. wrr                   Word Retention Rate vs input minus fillers     w=0.15
  4. chrf                  sacrebleu chrF++ vs reference                  w=0.15
  5. length_penalty        len_words(out)/len_words(in) penalty curve     w=0.05

Composite (with BERTScore deferred) = sum * (1 / 0.90) so the score is
still on a 0..1 scale. The 0.10 BERTScore weight gets redistributed
proportionally across the others.

All metrics return a float in [0, 1] where 1 = perfect.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any

# Filler/stopword list used by WRR. We do NOT subtract regex-pre-pass fillers
# (they're already gone before the LLM runs) but we DO subtract the residual
# "soft" fillers the LLM is supposed to remove, plus stopwords.
SOFT_FILLERS = {
    "well", "honestly", "literally", "frankly", "obviously", "anyway",
    "basically", "actually", "i", "mean", "guess", "kinda", "kind",
    "sort", "of", "like", "right", "you", "know", "so",
}
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "to", "of", "in", "on", "at", "for", "by", "with", "from", "as",
    "that", "this", "these", "those", "it", "its", "we", "they", "them",
    "he", "she", "his", "her", "our", "their", "my", "me", "us",
    "if", "then", "than", "so", "no", "not",
}


def _tokens(text: str) -> List[str]:
    # Period is in the class so multi-dot tokens like "package.json" / "4.2.1"
    # stay intact, but rstrip "." removes sentence-final periods that would
    # otherwise wreck set equality between "world" and "world.".
    raw = re.findall(r"[A-Za-z0-9_$%/.+#-]+", text.lower())
    return [t.rstrip(".") for t in raw if t.rstrip(".")]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokens(text) if t not in STOPWORDS and t not in SOFT_FILLERS]


# ── 1. category success ─────────────────────────────────────────────────────

def category_success(case: Dict[str, Any], output: str) -> float:
    """Rule-based per-case binary check.

    For identity_clean: exact match (case-insensitive, normalized).
    For all other categories: must_contain present, must_not_contain absent,
    length ratio under bound.
    """
    out = output.strip()
    out_lower = out.lower()

    # must_contain
    for needle in case.get("must_contain", []):
        if needle.lower() not in out_lower:
            return 0.0

    # must_not_contain
    for needle in case.get("must_not_contain", []):
        if needle.lower() in out_lower:
            return 0.0

    # length ratio
    in_words = max(len(case["input"].split()), 1)
    out_words = len(out.split())
    if out_words / in_words > case.get("max_length_ratio", 1.5):
        return 0.0

    # identity: exact match required
    if case["category"] == "identity_clean":
        norm_out = re.sub(r"\s+", " ", out_lower).rstrip(".!?")
        norm_ref = re.sub(r"\s+", " ", case["reference"].lower()).rstrip(".!?")
        if norm_out != norm_ref:
            return 0.0

    return 1.0


# ── 2. edit F0.5 (ERRANT-lite) ──────────────────────────────────────────────

def _diff_edits(src: List[str], tgt: List[str]) -> set:
    """Return a set of (op, src_span, tgt_span) edit triples between two
    token lists, using SequenceMatcher. ERRANT does smarter linguistic
    alignment but for cleanup the SequenceMatcher approximation is fine —
    the metric is still order-of-magnitude correct for ranking models."""
    sm = SequenceMatcher(None, src, tgt, autojunk=False)
    edits = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits.add((tag, tuple(src[i1:i2]), tuple(tgt[j1:j2])))
    return edits


def edit_f05(case: Dict[str, Any], output: str) -> float:
    """ERRANT-style edit F0.5 — precision weighted 2x recall."""
    src = _tokens(case["input"])
    ref = _tokens(case["reference"])
    sys = _tokens(output)
    if not src:
        return 1.0 if not sys else 0.0

    gold = _diff_edits(src, ref)
    pred = _diff_edits(src, sys)

    if not gold and not pred:
        return 1.0  # both no-op (identity case, perfect)
    if not pred:
        return 0.0  # missed every edit
    if not gold:
        return 1.0 if not pred else 0.0  # gold says don't edit; any edit = wrong

    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    if p + r == 0:
        return 0.0
    beta2 = 0.25  # F0.5 — precision weighted 2x recall
    return (1 + beta2) * p * r / (beta2 * p + r)


# ── 3. Word Retention Rate ──────────────────────────────────────────────────

def wrr(case: Dict[str, Any], output: str) -> float:
    """Faithfulness guardrail: precision of output content words against input.

    WRR = |content(out) ∩ content(in)| / |content(out)|

    A WRR near 1.0 means every content word in the output also appeared in
    the input — the model isn't hallucinating new words. A WRR < 0.85 is
    the documented hallucination red flag in the plan §A.2.

    Note: an earlier version of this metric computed RECALL of input
    content instead of PRECISION of output content, which incorrectly
    penalized models for correctly dropping self-correction text. The
    plan's stated intent ("output is a subset of input") matches this
    precision-of-output formulation.
    """
    in_content = set(_content_tokens(case["input"]))
    out_content = set(_content_tokens(output))
    if not out_content:
        return 1.0  # empty output can't hallucinate
    overlap = len(in_content & out_content)
    return overlap / len(out_content)


# ── 4. chrF++ via sacrebleu ─────────────────────────────────────────────────

def chrf(case: Dict[str, Any], output: str) -> float:
    """chrF++ (char 6-gram + word 2-gram) F-score against the reference,
    rescaled to [0, 1]."""
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        return 0.0
    metric = CHRF(word_order=2)  # chrF++
    score = metric.sentence_score(output, [case["reference"]])
    return score.score / 100.0


# ── 5. Length penalty ───────────────────────────────────────────────────────

def length_penalty(case: Dict[str, Any], output: str) -> float:
    """LR = len_words(out) / len_words(in). Returns 1.0 inside [0.5, 1.2],
    linearly decays to 0.0 at LR=0.2 or LR=2.0, clamped to 0 outside."""
    in_w = max(len(case["input"].split()), 1)
    out_w = len(output.split())
    lr = out_w / in_w
    if 0.5 <= lr <= 1.2:
        return 1.0
    if lr < 0.5:
        return max(0.0, (lr - 0.2) / 0.3)
    # lr > 1.2
    return max(0.0, (2.0 - lr) / 0.8)


# ── Composite score ─────────────────────────────────────────────────────────

# Weights from the plan, with BERTScore (0.10) redistributed proportionally
# across the other metrics so the composite still totals 1.0.
WEIGHTS = {
    "category_success": 0.30 / 0.90,
    "edit_f05":         0.25 / 0.90,
    "wrr":              0.15 / 0.90,
    "chrf":             0.15 / 0.90,
    "length_penalty":   0.05 / 0.90,
}


def score_case(case: Dict[str, Any], output: str) -> Dict[str, float]:
    """Score a single case across all metrics. Returns per-metric and composite."""
    s = {
        "category_success": category_success(case, output),
        "edit_f05":         edit_f05(case, output),
        "wrr":              wrr(case, output),
        "chrf":             chrf(case, output),
        "length_penalty":   length_penalty(case, output),
    }
    s["composite"] = sum(s[k] * w for k, w in WEIGHTS.items())
    return s


# ── Hard disqualification gates (per plan §A.2) ─────────────────────────────

def disqualify(per_case_scores: List[Dict[str, float]], cases: List[Dict[str, Any]]) -> List[str]:
    """Return list of disqualification reasons (empty = passes all gates)."""
    reasons = []
    n = len(per_case_scores)
    if n == 0:
        return ["no scores"]

    # WRR < 0.85 on >5% of cases
    halluc_count = sum(1 for s in per_case_scores if s["wrr"] < 0.85)
    if halluc_count / n > 0.05:
        reasons.append(f"hallucinator: WRR<0.85 on {halluc_count}/{n} ({halluc_count/n*100:.1f}%)")

    # Length penalty == 0 (LR outside [0.2, 2.0]) on >5%
    paraphrase_count = sum(1 for s in per_case_scores if s["length_penalty"] == 0.0)
    if paraphrase_count / n > 0.05:
        reasons.append(f"paraphraser: LR-out-of-bounds on {paraphrase_count}/{n} ({paraphrase_count/n*100:.1f}%)")

    # must_not_contain violations on self-correction cases > 10%
    sc_cats = {"explicit_self_correction", "implicit_self_correction", "cross_sentence_self_correction"}
    sc_indices = [i for i, c in enumerate(cases) if c["category"] in sc_cats]
    if sc_indices:
        sc_fails = sum(1 for i in sc_indices if per_case_scores[i]["category_success"] == 0.0)
        if sc_fails / len(sc_indices) > 0.10:
            reasons.append(
                f"changes-meaning: self-correction failures on {sc_fails}/{len(sc_indices)} ({sc_fails/len(sc_indices)*100:.1f}%)"
            )

    return reasons
