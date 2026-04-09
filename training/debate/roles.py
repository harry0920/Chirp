"""
roles.py — role assignments and per-round prompt builders.

Three role types:
- ADVOCATE: proposes a technique to solve the constraint, defends it across rounds.
- SKEPTIC:  permanent red-team. Never proposes; only attacks. Demands evidence.
- MODERATOR: speaks only in the final synthesis round. Neutral hat.

Models are loaded from training/benchmark_v3/candidates.yaml. The same Ministral 8B
candidate plays Moderator in the final round only; it does NOT also advocate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
CANDIDATES_PATH = ROOT / "training" / "benchmark_v3" / "candidates.yaml"


# ---------- role assignments ----------

# Map debate seat -> candidates.yaml key.
ADVOCATE_KEYS = [
    "ministral-3-3b-2512",
    "qwen2.5-3b",
    "qwen3-1.7b",
    "gemma-4-e4b-it",
    "gemma-4-e2b-it",
    "eurollm-9b",
]
SKEPTIC_KEY = "qwen3-4b-instruct-2507"   # second-strongest reasoner, not the incumbent
MODERATOR_KEY = "ministral-3-8b-2512"    # strongest reasoner, incumbent — synthesis only

# Smoke-test subset (smallest+fastest two advocates)
SMOKE_ADVOCATE_KEYS = ["ministral-3-3b-2512", "qwen3-1.7b"]


@dataclass
class Role:
    seat: str                  # display name in transcript ("advocate-A", "skeptic", "moderator")
    role_type: str             # "advocate" | "skeptic" | "moderator"
    candidate_key: str         # candidates.yaml key
    cand: Dict                 # full candidate config (binary, model, sampling, etc.)
    sampling: Dict             # effective sampling params (with advocate temp bump applied)


def load_candidates() -> Dict[str, Dict]:
    with CANDIDATES_PATH.open() as f:
        return yaml.safe_load(f)["candidates"]


def build_roles(*, smoke: bool = False) -> List[Role]:
    """Return the list of seats in the order they speak each round.

    Order each round: skeptic last (so it sees all advocates first). Moderator
    is excluded from this list and dispatched separately for the synthesis round.
    """
    cands = load_candidates()
    advocate_keys = SMOKE_ADVOCATE_KEYS if smoke else ADVOCATE_KEYS
    roles: List[Role] = []
    for i, key in enumerate(advocate_keys):
        cand = cands[key]
        sampling = dict(cand.get("sampling", {}))
        # +0.1 temperature bump for advocates (more divergent ideation)
        sampling["temperature"] = round(float(sampling.get("temperature", 0.7)) + 0.1, 2)
        roles.append(
            Role(
                seat=f"advocate-{chr(ord('A') + i)}",
                role_type="advocate",
                candidate_key=key,
                cand=cand,
                sampling=sampling,
            )
        )
    if not smoke:
        sk = cands[SKEPTIC_KEY]
        roles.append(
            Role(
                seat="skeptic",
                role_type="skeptic",
                candidate_key=SKEPTIC_KEY,
                cand=sk,
                sampling=dict(sk.get("sampling", {})),
            )
        )
    return roles


def moderator_role() -> Role:
    cands = load_candidates()
    cand = cands[MODERATOR_KEY]
    return Role(
        seat="moderator",
        role_type="moderator",
        candidate_key=MODERATOR_KEY,
        cand=cand,
        sampling=dict(cand.get("sampling", {})),
    )


# ---------- system prompts ----------

RESEARCH_TAIL = """\

After your main response, on a NEW line, output exactly one line in this format:
RESEARCH_REQUEST: {"questions": ["...", "..."]}

List up to 5 specific, falsifiable research questions you want answered before your
next turn. Good questions name a paper, a model, a benchmark, or a concrete number to
verify. Bad questions are vague ("how does distillation work"). If you have no requests,
output:
RESEARCH_REQUEST: {"questions": []}

Do not output anything after that line. The orchestrator parses it.
"""

ADVOCATE_SYSTEM = """\
You are an advocate in a structured technical debate among local LLMs about how to make
Chirp's cleanup model laptop-shippable. Your job: propose ONE technique you believe is
most likely to deliver Ministral-8B-quality cleanup in a much smaller, faster package
that respects every hard constraint in the shared context. You may invent, combine, or
extend techniques from any 2025-2026 work. You are NOT limited to a fixed list. Do not
hedge. State your technique, name what makes it likely to clear the quality floor, and
acknowledge what you do not yet know.

Style: technical, dense, concrete. No throat-clearing. **Hard cap: 350 words for your
main response**, then the RESEARCH_REQUEST line. Cite specific papers, models, or repos
by name when you can. If you are wrong about a citation the skeptic will catch it.
""" + RESEARCH_TAIL

SKEPTIC_SYSTEM = """\
You are the red-team skeptic in a structured technical debate among local LLMs. Your
single job: assume every advocate's proposal will fail, and find the specific failure
mode. Demand evidence. Pick the weakest claim in each proposal and attack it with a
concrete prediction of what will go wrong (which v3 benchmark category will regress,
which constraint will be violated, which step of the pipeline is hand-waved). Never
propose your own technique. Never concede unless an advocate produces hard evidence.

Style: surgical, brief, adversarial but not snarky. **Hard cap: 300 words for your main
response**, then the RESEARCH_REQUEST line. If you cite a paper or benchmark, be precise.
""" + RESEARCH_TAIL

MODERATOR_SYSTEM = """\
You are the moderator in the final synthesis round of a structured technical debate
among local LLMs. You have read every prior round in the transcript. Your job is NEUTRAL
synthesis. You do not have a stake. Output four sections:

1. **Convergence:** approaches or sub-claims multiple advocates landed on independently.
2. **Contested:** the disagreements that did NOT resolve, and what each side actually
   believed.
3. **Empirical resolution:** for each contested point, the specific experiment that would
   resolve it (dataset, metric, runtime budget).
4. **Recommendation:** the single approach you would put in front of the user, with your
   confidence (low/medium/high) and the one piece of evidence that would change your mind.

Be specific. Name techniques by name. 600-1000 words. Do NOT advocate; describe what the
debate showed.

You do not need to output a RESEARCH_REQUEST line. Just write the synthesis.
"""


def system_prompt_for(role: Role) -> str:
    if role.role_type == "advocate":
        return ADVOCATE_SYSTEM
    if role.role_type == "skeptic":
        return SKEPTIC_SYSTEM
    if role.role_type == "moderator":
        return MODERATOR_SYSTEM
    raise ValueError(f"unknown role type: {role.role_type}")


# ---------- per-round user-message builders ----------

def round1_advocate_prompt(context_pack: str, bootstrap_research: str) -> str:
    return f"""\
{context_pack}

---

## Bootstrap research (broad survey of 2025-2026 LLM compression / distillation techniques)

{bootstrap_research}

---

## Round 1: Open proposal

This is your opening turn. No other advocate has spoken yet. Write your proposal for how
to solve the laptop-shippability problem. State the technique in 2-3 sentences, then
explain (a) why it is likely to clear the >=0.92 composite floor, (b) the size and
latency math, (c) the biggest unknown. End with up to 5 research questions for the
orchestrator to resolve before round 2.
"""


def round2_advocate_prompt(
    context_pack: str,
    bootstrap_research: str,
    own_round1: str,
    own_research_findings: str,
    others_round1_digest: str,
) -> str:
    return f"""\
{context_pack}

---

## Bootstrap research

{bootstrap_research}

---

## Your Round 1 proposal

{own_round1}

---

## Findings from your Round 1 research questions

{own_research_findings}

---

## Other advocates' Round 1 proposals (digest)

{others_round1_digest}

---

## Round 2: Refined proposal

Refine your proposal in light of (a) what you learned from your research and (b) what
the other advocates proposed. You may sharpen, pivot, or absorb a stronger idea. If you
pivot, name the advocate whose proposal influenced you. End with up to 5 NEW research
questions for the next dispatch.
"""


def round2_skeptic_prompt(context_pack: str, all_round2_proposals: str) -> str:
    return f"""\
{context_pack}

---

## All advocates' Round 2 refined proposals

{all_round2_proposals}

---

## Round 2: Skeptic attack

Pick the THREE proposals you consider most fragile. For each one, write a targeted
attack: name the advocate, name the specific claim or step, and predict the concrete
failure mode (which v3 category will regress, which constraint will be violated, which
piece is hand-waved). Be specific enough that an advocate could refute you with a
single experiment.
"""


def round3_advocate_prompt(
    context_pack: str,
    own_round2: str,
    skeptic_attacks: str,
    others_round2_digest: str,
) -> str:
    return f"""\
{context_pack}

---

## Your Round 2 proposal

{own_round2}

---

## Skeptic's Round 2 attacks (read all; respond if you were named)

{skeptic_attacks}

---

## Other advocates' Round 2 proposals (digest)

{others_round2_digest}

---

## Round 3: Rebuttal + cross-pollination

If the skeptic attacked you by name, respond directly to the specific charge (do not
deflect). If you were not attacked, write a brief defense of why you were not. Then,
optionally, borrow one idea from another advocate's Round 2 proposal and explain how it
strengthens yours. End with up to 5 NEW research questions.
"""


def round4_advocate_prompt(
    context_pack: str,
    full_prior_transcript_digest: str,
    own_research_findings_so_far: str,
) -> str:
    return f"""\
{context_pack}

---

## Full debate so far (digest of rounds 1-3 and all research)

{full_prior_transcript_digest}

---

## Your accumulated research findings

{own_research_findings_so_far}

---

## Round 4: Final position

This is your final turn. Write your honest final recommendation. You do NOT have to
match your earlier proposals. You can adopt another advocate's idea, propose a hybrid,
or concede the skeptic was right and shift entirely. State (a) the technique you would
ship, (b) why you believe it clears the quality floor, (c) the single experiment that
would make you change your mind, (d) your confidence (low/medium/high).
"""


def round4_skeptic_prompt(context_pack: str, full_prior_transcript_digest: str) -> str:
    return f"""\
{context_pack}

---

## Full debate so far (digest of rounds 1-3)

{full_prior_transcript_digest}

---

## Round 4: Skeptic final memo

Write a memo titled "What would still need to be true for any of these to actually
work." Pick the 2-3 proposals you consider least implausible. For each one, list the
empirical conditions that would have to hold for it to clear the >=0.92 composite floor
and the >=1 sec p95 latency target on integrated GPU. Be ruthless about distinguishing
"plausible" from "demonstrated."
"""


def round5_moderator_prompt(context_pack: str, full_transcript: str) -> str:
    return f"""\
{context_pack}

---

## Full debate transcript (rounds 1-4, all advocates and skeptic)

{full_transcript}

---

## Round 5: Synthesis

Read every prior round and write the four-section synthesis described in your role
prompt: Convergence, Contested, Empirical resolution, Recommendation.
"""
