"""
context_pack.py — build the constraints + benchmark digest + current-state
blob that every debate participant sees.

The blob is plain markdown, intended to be slotted into the system or first
user message of every model turn. Targets ~3\u20134 KB so it doesn't blow ctx.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
BENCHMARK_DIR = ROOT / "training" / "benchmark_v3"
MATRIX_SUMMARY = BENCHMARK_DIR / "results" / "matrix_summary.json"


CONSTRAINTS_MD = """\
## Hard constraints (non-negotiable)

1. **Quality floor:** must hit composite \u2265 0.92 on the v3 benchmark corpus and pass ALL
   three hard disqualification gates. The current Ministral 3 8B Q4_K_M baseline scores
   0.922 composite, 92.8% category success, and is the only top-tier candidate to pass
   every gate. The replacement must match or beat that.
2. **Size:** download \u2264 500 MB (ideal), \u2264 1 GB (hard ceiling). Current: 4.9 GB.
3. **Latency:** p95 \u2264 1 second on integrated GPU / 8 GB Apple Silicon.
   Current Ministral 8B p95 on RTX 4080 is ~388 ms; on integrated GPUs it is 5\u201315 sec
   per dictation, which is the unshippable failure mode we are trying to fix.
4. **100% local. No cloud, ever.** No API fallback, no telemetry, no hybrid local+cloud
   tier. Chirp's positioning is on-device only.
5. **License:** Apache 2.0 or MIT only. Anything else (CC BY-NC, custom non-commercial,
   gated weights, research-only) is disqualified.
6. **No accept-the-floor.** Shipping a known-worse model (e.g. Ministral 3B at 0.881
   composite or Qwen 2.5 3B at 0.890) is NOT a valid answer. The whole point is to
   preserve Ministral 8B-class intelligence in a smaller package.
"""


CURRENT_STATE_MD = """\
## Current state (not foregone conclusions \u2014 attack or extend)

- **Shipping interim:** Ministral 3 8B Instruct 2512 Q4_K_M, with the `v2-fewshot-hard`
  prompt strategy (8 chat-turn fewshot pairs, JSON output, anti-paraphrase guardrails).
  Wired into v1.3.0 dev branch but uncommitted, pending laptop hardware path.
- **Currently planned long-term answer:** distill Ministral 8B \u2192 FLAN-T5-base (250M, ~250 MB
  Q-quantized) via Arcee DistillKit. Logit-level KD with optional sequence-level supervision
  on a hand-curated adversarial subset. ~6,000 training pairs after filtering. Cost estimate
  ~\\$25 cloud compute. This is ONE path picked from a small set the team thought of \u2014
  treat it as an existing proposal to attack, extend, or replace.
- **Why a previous Chirp T5-small (77M) fine-tune failed:** the training data taught
  summarization, not minimal edits. The architecture itself worked (58 ms p50 on GPU,
  103 ms on CPU via CTranslate2 INT8). Re-benchmarked at 0.78 composite with paraphrase
  pathology. The lesson: training data > architecture.
- **No real user history:** only ~280 dictations from the developer's own use exist. Not
  enough or distributed enough for supervised fine-tuning on real Chirp inputs.
"""


def _format_matrix_row(row: Dict) -> str:
    name = row["candidate"]
    comp = row["composite_mean"]
    cat = row["category_success_mean"]
    p50 = row["ttlt_p50_ms"]
    p95 = row["ttlt_p95_ms"]
    dqs = row.get("disqualifications") or []
    dq = "PASS" if not dqs else f"DQ ({'; '.join(dqs)})"
    return f"- **{name}**: composite {comp:.3f}, category-success {cat:.0%}, p50 {p50:.0f}ms, p95 {p95:.0f}ms \u2014 {dq}"


def benchmark_digest_md() -> str:
    if not MATRIX_SUMMARY.exists():
        return "## v3 benchmark digest\n\n_(matrix_summary.json not found)_\n"
    rows: List[Dict] = json.loads(MATRIX_SUMMARY.read_text())
    rows = sorted(rows, key=lambda r: r["composite_mean"], reverse=True)
    lines = [
        "## v3 benchmark Phase C results (250-case English corpus, all candidates, v2-fewshot-hard prompt)",
        "",
        "Sorted by composite. Latency is on RTX 4080. **DQ** = failed at least one hard gate.",
        "",
    ]
    for row in rows:
        lines.append(_format_matrix_row(row))
    lines.append("")
    lines.append(
        "_The two only-no-DQ models are Ministral 3 8B (0.922) and Ministral 3 3B (0.881). "
        "Everything else either underperforms on composite or trips the changes-meaning gate "
        "on self-correction inputs._"
    )
    return "\n".join(lines) + "\n"


def build_context_pack() -> str:
    """Return the full markdown blob handed to every debate participant."""
    parts = [
        "# Cleanup-model laptop-shippability debate \u2014 shared context",
        "",
        "Chirp is a local-only voice-to-text desktop app. After speech transcription, a small",
        "instruction-tuned LLM does cleanup: removes filler words, resolves spoken self-corrections,",
        "expands stutters, normalizes spoken numbers/dates, and applies dictionary/snippets. The",
        "current shipping cleanup model is too large for laptop hardware and we need a way to",
        "deliver the same task-specific intelligence in a much smaller, faster package.",
        "",
        CONSTRAINTS_MD,
        "",
        CURRENT_STATE_MD,
        "",
        benchmark_digest_md(),
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    print(build_context_pack())
