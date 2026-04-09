"""
orchestrator.py — main debate loop.

Round structure (full debate):
  Round 0: bootstrap research dispatch (no model speaks)
  Round 1: each advocate writes an open proposal
           -> dispatch research between rounds
  Round 2: each advocate refines; skeptic attacks
           -> dispatch research between rounds
  Round 3: rebuttals + cross-pollination
           -> dispatch research between rounds
  Round 4: final positions + skeptic memo
  Round 5: moderator synthesis (Ministral 8B in moderator hat)

Smoke run skips rounds 2-5 and uses 2 advocates only.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import roles
from .context_pack import build_context_pack
from .llama_session import LlamaSession
from .research import ResearchResult, claude_research, research_batch
from .transcript import DebateRun


# ------------- bootstrap research question -------------

BOOTSTRAP_QUESTION = (
    "Survey 2025-2026 techniques for compressing or distilling instruction-tuned LLMs "
    "from ~8B parameters down to <=500M parameters while preserving narrow task-specific "
    "capability (specifically: text cleanup / minimal-edit transformation). Be exhaustive: "
    "list every distinct technique with citation, including (but not limited to) "
    "knowledge distillation variants (logit, sequence, on-policy, MiniLLM, DistillKit), "
    "more aggressive quantization (Q3, Q2, IQ2, AWQ, GPTQ, SmoothQuant, sub-1-bit), "
    "structured/unstructured pruning, LoRA/QLoRA task-specific fine-tuning of small "
    "bases, speculative decoding (Medusa, EAGLE, draft model variants), grammar-constrained "
    "decoding for token budget reduction, encoder-decoder vs decoder-only tradeoffs for "
    "rewriting tasks, recent small instruct models <1B params with strong instruction "
    "following, retrieval-augmented small models, prompt compression / context distillation, "
    "and any 2025-2026 techniques I am unlikely to have heard of. For each technique give: "
    "name, source paper or repo (with link), what tradeoff it makes, and any reported "
    "results on tasks similar to text rewriting / cleanup. Aim for 1500-3000 words."
)


# ------------- RESEARCH_REQUEST parser -------------

_TAG = "RESEARCH_REQUEST:"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 reasoning mode leakage even with
    reasoning_budget=0). Also tolerate an unclosed leading <think> by dropping
    everything up to the first </think> if a closing tag exists with no opener
    above it."""
    if not text:
        return text
    cleaned = _THINK_RE.sub("", text)
    # Unclosed <think> at start: drop up to and including first </think>
    if "<think>" not in cleaned and "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned.strip()


def _scan_balanced_json(s: str, start: int) -> Optional[Tuple[str, int]]:
    """Find the first '{' at or after start, then return (substring, end_index)
    where the substring is a brace-balanced JSON object. Naive — does not honor
    string-literal braces, but adequate for our well-formed-or-garbage inputs."""
    i = s.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i : j + 1], j + 1
    return None


def parse_research_request(text: str) -> Tuple[str, List[str]]:
    """Strip the RESEARCH_REQUEST line and return (body_without_line, questions).

    Tolerant of: missing line, malformed JSON, trailing junk after the JSON, missing
    questions field, non-string list items (coerced to str), more than 5 questions
    (truncated), <think>...</think> reasoning leakage. Anchors on the literal
    "RESEARCH_REQUEST:" tag, scans forward for a brace-balanced JSON object.
    """
    if not text:
        return text, []
    text = _strip_think_tags(text)
    tag_idx = text.rfind(_TAG)
    if tag_idx < 0:
        return text, []
    scan = _scan_balanced_json(text, tag_idx + len(_TAG))
    body = text[:tag_idx].rstrip()
    if scan is None:
        return body, []
    raw_json, _ = scan
    try:
        obj = json.loads(raw_json)
        questions = obj.get("questions", [])
        if not isinstance(questions, list):
            return body, []
        questions = [str(q).strip() for q in questions if str(q).strip()]
        return body, questions[:5]
    except Exception:
        return body, []


# ------------- one model turn -------------

def _run_turn(
    role: roles.Role,
    user_prompt: str,
    *,
    max_tokens: int = 1024,
) -> Tuple[str, float]:
    """Spin up llama-server for this role, send one chat completion, tear down."""
    messages = [
        {"role": "system", "content": roles.system_prompt_for(role)},
        {"role": "user", "content": user_prompt},
    ]
    t0 = time.time()
    with LlamaSession(role.cand) as sess:
        text = sess.chat(messages, role.sampling, max_tokens=max_tokens)
    return text, time.time() - t0


# ------------- bootstrap -------------

SMOKE_BOOTSTRAP_QUESTION = (
    "In 300-500 words, list the top 5-7 distinct techniques from 2025-2026 for compressing "
    "an instruction-tuned LLM from ~8B parameters down to ~500M while preserving narrow "
    "task-specific capability (text cleanup / minimal-edit rewriting). For each technique "
    "give: name, one source link, and one sentence on the tradeoff. This is a smoke test \u2014 "
    "be fast, do not aim for exhaustive coverage."
)


def run_bootstrap(run: DebateRun, *, smoke: bool = False) -> str:
    print("\n=== Round 0: bootstrap research ===", flush=True)
    t0 = time.time()
    question = SMOKE_BOOTSTRAP_QUESTION if smoke else BOOTSTRAP_QUESTION
    timeout = 300.0 if smoke else 900.0
    res = claude_research(question, timeout_s=timeout)
    print(f"  bootstrap: {'ok' if res.ok else 'FAILED ' + (res.error or '')} ({time.time()-t0:.0f}s)", flush=True)
    run.write_bootstrap_research(question, res.answer, res.ok, res.error)
    if res.ok:
        return res.answer
    return f"_(bootstrap research failed: {res.error}; advocates proceed without it)_"


# ------------- per-round helpers -------------

def _dispatch_research_for_seat(
    run: DebateRun,
    seat: str,
    round_num: int,
    questions: List[str],
) -> None:
    if not questions:
        return
    print(f"  research dispatch: {seat} r{round_num} -> {len(questions)} questions (sequential)", flush=True)
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        res = claude_research(q, timeout_s=600.0)
        elapsed = time.time() - t0
        status = "ok" if res.ok else f"FAIL ({res.error})"
        print(f"    q{i}/{len(questions)} {status} ({elapsed:.0f}s)", flush=True)
        run.write_research(round_num, seat, i, res.question, res.answer, res.ok, res.error)


def _advocate_turn(
    run: DebateRun,
    role: roles.Role,
    round_num: int,
    user_prompt: str,
    *,
    max_tokens: int = 1024,
) -> List[str]:
    """Run one advocate's turn. Writes turn to transcript, returns list of research questions."""
    print(f"\n--- Round {round_num} {role.seat} ({role.candidate_key}) ---", flush=True)
    error: Optional[str] = None
    body = ""
    raw = ""
    questions: List[str] = []
    elapsed = 0.0
    try:
        raw, elapsed = _run_turn(role, user_prompt, max_tokens=max_tokens)
        body, questions = parse_research_request(raw)
        # If parsing left an empty body, fall back to the raw model output so the
        # transcript still records what the model actually said. This helps debug
        # parser/model edge cases without losing content.
        if not body.strip() and raw.strip():
            body = (
                "_(parser produced empty body; falling back to raw model output below)_\n\n"
                "```\n" + raw.strip() + "\n```"
            )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(f"  TURN FAILED: {error}", flush=True)
    run.write_turn(round_num, role.seat, role.candidate_key, body, research_request=questions, elapsed_s=elapsed, error=error)
    return questions


# ------------- top-level driver -------------

def run_fast_debate(out_dir: Path, *, reuse_bootstrap: Optional[Path] = None) -> DebateRun:
    """Fast mode: reuse a prior bootstrap.md, run R1 (6 advocates, no research
    dispatch, no skeptic), then jump directly to R5 moderator synthesis.

    Designed for ~5 minute wall time when you want a real transcript fast.
    """
    run = DebateRun.create(out_dir, label="fast")
    print(f"\nDebate output: {run.out_dir}", flush=True)

    context = build_context_pack()
    advocate_roles = roles.build_roles(smoke=False)
    advocates = [r for r in advocate_roles if r.role_type == "advocate"]

    # Round 0: reuse bootstrap if a path is supplied; otherwise run a fresh one.
    run.write_round_header(0, "Bootstrap research")
    if reuse_bootstrap and reuse_bootstrap.exists():
        text = reuse_bootstrap.read_text(encoding="utf-8")
        # Strip the first markdown header section (preamble) if present, keep the answer body.
        if "## Answer" in text:
            bootstrap_text = text.split("## Answer", 1)[1].strip()
        else:
            bootstrap_text = text
        # Copy into our research dir so the run is self-contained.
        (run.research_dir / "bootstrap.md").write_text(text, encoding="utf-8")
        run.write_section(
            "Round 0 \u2014 reused bootstrap",
            f"Reused prior bootstrap research from `{reuse_bootstrap}`. Copied into `research/bootstrap.md`.",
        )
        print(f"  reused bootstrap: {reuse_bootstrap} ({len(bootstrap_text)} chars)", flush=True)
    else:
        bootstrap_text = run_bootstrap(run, smoke=False)

    # Round 1: open proposals (no research dispatch in fast mode)
    run.write_round_header(1, "Open proposals (fast mode \u2014 no research dispatch between rounds)")
    for adv in advocates:
        prompt = roles.round1_advocate_prompt(context, bootstrap_text)
        _advocate_turn(run, adv, 1, prompt, max_tokens=1200)

    # Round 5: moderator synthesis directly off Round 1
    run.write_round_header(5, "Moderator synthesis (fast mode)")
    moderator = roles.moderator_role()
    digest = run.full_transcript_digest(max_chars_per_turn=1500)
    prompt = roles.round5_moderator_prompt(context, digest)
    print(f"\n--- Round 5 moderator ({moderator.candidate_key}) ---", flush=True)
    error: Optional[str] = None
    text = ""
    elapsed = 0.0
    try:
        text, elapsed = _run_turn(moderator, prompt, max_tokens=1500)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(f"  MODERATOR FAILED: {error}", flush=True)
    run.write_turn(5, moderator.seat, moderator.candidate_key, text, elapsed_s=elapsed, error=error)
    print(f"\nDone. Transcript: {run.transcript_path}", flush=True)
    return run


def run_full_debate(out_dir: Path, *, smoke: bool = False) -> DebateRun:
    label = "smoke" if smoke else "debate"
    run = DebateRun.create(out_dir, label=label)
    print(f"\nDebate output: {run.out_dir}", flush=True)

    context = build_context_pack()
    advocate_roles = roles.build_roles(smoke=smoke)
    advocates = [r for r in advocate_roles if r.role_type == "advocate"]
    skeptics = [r for r in advocate_roles if r.role_type == "skeptic"]

    # Round 0
    run.write_round_header(0, "Bootstrap research")
    bootstrap_text = run_bootstrap(run, smoke=smoke)
    run.write_section(
        "Round 0 \u2014 bootstrap pointer",
        "Bootstrap research findings live in `research/bootstrap.md` and are summarised "
        "into every advocate's Round 1 prompt.",
    )

    # Round 1: open proposals
    run.write_round_header(1, "Open proposals")
    for adv in advocates:
        prompt = roles.round1_advocate_prompt(context, bootstrap_text)
        questions = _advocate_turn(run, adv, 1, prompt, max_tokens=1200)
        _dispatch_research_for_seat(run, adv.seat, 1, questions)

    if smoke:
        print("\n[smoke mode] stopping after Round 1.", flush=True)
        return run

    # Round 2: refined proposals + skeptic
    run.write_round_header(2, "Refined proposals + first skeptic attack")
    for adv in advocates:
        own_r1 = run.round_text(1, adv.seat)
        own_findings = run.research_findings_for(adv.seat, 1)
        others_digest = run.round_digest(1, exclude_seat=adv.seat)
        prompt = roles.round2_advocate_prompt(
            context, bootstrap_text, own_r1, own_findings, others_digest
        )
        questions = _advocate_turn(run, adv, 2, prompt, max_tokens=1200)
        _dispatch_research_for_seat(run, adv.seat, 2, questions)

    # Skeptic speaks after seeing all R2 advocate proposals
    for sk in skeptics:
        all_r2 = run.round_digest(2, max_chars=2000)
        prompt = roles.round2_skeptic_prompt(context, all_r2)
        questions = _advocate_turn(run, sk, 2, prompt, max_tokens=1200)
        _dispatch_research_for_seat(run, sk.seat, 2, questions)

    # Round 3: rebuttals + cross-pollination
    run.write_round_header(3, "Rebuttals + cross-pollination")
    skeptic_r2_digest = "\n\n".join(
        f"**{sk.seat}:**\n{run.round_text(2, sk.seat)}" for sk in skeptics
    ) or "_(no skeptic round 2 output)_"
    for adv in advocates:
        own_r2 = run.round_text(2, adv.seat)
        others_r2 = run.round_digest(2, exclude_seat=adv.seat, max_chars=800)
        prompt = roles.round3_advocate_prompt(context, own_r2, skeptic_r2_digest, others_r2)
        questions = _advocate_turn(run, adv, 3, prompt, max_tokens=1200)
        _dispatch_research_for_seat(run, adv.seat, 3, questions)

    # Round 4: final positions
    run.write_round_header(4, "Final positions")
    for adv in advocates:
        digest = run.full_transcript_digest(max_chars_per_turn=900)
        findings = run.all_findings_for_seat(adv.seat)
        prompt = roles.round4_advocate_prompt(context, digest, findings)
        _advocate_turn(run, adv, 4, prompt, max_tokens=1024)

    for sk in skeptics:
        digest = run.full_transcript_digest(max_chars_per_turn=900)
        prompt = roles.round4_skeptic_prompt(context, digest)
        _advocate_turn(run, sk, 4, prompt, max_tokens=1024)

    # Round 5: moderator synthesis
    run.write_round_header(5, "Moderator synthesis")
    moderator = roles.moderator_role()
    full_md = run.full_transcript_text()
    # Truncate transcript if it gets too big to fit in moderator ctx (8K).
    # Keep it under ~28 KB of prompt content total. Moderator gets a tighter digest.
    digest = run.full_transcript_digest(max_chars_per_turn=1500)
    prompt = roles.round5_moderator_prompt(context, digest)
    print(f"\n--- Round 5 moderator ({moderator.candidate_key}) ---", flush=True)
    try:
        text, elapsed = _run_turn(moderator, prompt, max_tokens=1500)
        run.write_turn(5, moderator.seat, moderator.candidate_key, text, elapsed_s=elapsed)
    except Exception as e:
        run.write_turn(5, moderator.seat, moderator.candidate_key, "", error=f"{type(e).__name__}: {e}")

    print(f"\nDone. Transcript: {run.transcript_path}", flush=True)
    return run
