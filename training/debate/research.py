"""
research.py — dispatch web research questions to Claude Code subagents.

Each question is answered by a fresh `claude --print` subprocess with
WebSearch + WebFetch + Read enabled. The orchestrator collects findings
into a shared research log that all subsequent debate rounds can see.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


CLAUDE_BIN = shutil.which("claude") or "claude"

RESEARCH_SYSTEM_PROMPT = (
    "You are a research assistant for a technical debate among local LLMs about how "
    "to compress an 8B parameter cleanup model into a laptop-shippable form (\u2264500 MB, "
    "\u22641 sec p95 latency on integrated GPU, no quality regression). Your job is to "
    "answer ONE question with concrete, current evidence: cite ArXiv papers (with IDs), "
    "HuggingFace model pages (with org/repo), GitHub repos, blog posts, benchmark "
    "results. Use WebSearch and WebFetch aggressively. Prefer 2025\u20132026 sources. "
    "Format: 200\u2013500 words, no preamble, no hedging. If the answer doesn't exist on "
    "the public web, say so plainly in one sentence and stop. If the question is "
    "vague, answer the most useful interpretation and note the interpretation in one "
    "line at the top."
)


@dataclass
class ResearchResult:
    question: str
    answer: str
    ok: bool
    error: Optional[str] = None


def claude_research(question: str, *, timeout_s: float = 600.0) -> ResearchResult:
    """Run a single research question through `claude --print` with web tools.

    Prompt is passed via stdin to avoid the variadic --allowed-tools argument
    swallowing the trailing positional prompt.
    """
    cmd = [
        CLAUDE_BIN,
        "--print",
        "--append-system-prompt", RESEARCH_SYSTEM_PROMPT,
        "--allowed-tools", "WebSearch,WebFetch,Read",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=question,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ResearchResult(question, "", False, error=f"timeout after {timeout_s:.0f}s")
    except FileNotFoundError as e:
        return ResearchResult(question, "", False, error=f"claude binary not found: {e}")
    except Exception as e:
        return ResearchResult(question, "", False, error=f"{type(e).__name__}: {e}")

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()[:500]
        return ResearchResult(question, "", False, error=f"exit {proc.returncode}: {err}")

    answer = (proc.stdout or "").strip()
    if not answer:
        return ResearchResult(question, "", False, error="empty stdout")
    return ResearchResult(question, answer, True)


def research_batch(
    questions: List[str],
    *,
    max_parallel: int = 4,
    timeout_s: float = 600.0,
) -> List[ResearchResult]:
    """Run a batch of questions in parallel (capped). Returns results in input order."""
    if not questions:
        return []
    results: List[Optional[ResearchResult]] = [None] * len(questions)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as ex:
        future_to_idx = {
            ex.submit(claude_research, q, timeout_s=timeout_s): i
            for i, q in enumerate(questions)
        }
        for fut in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = ResearchResult(
                    questions[idx], "", False, error=f"{type(e).__name__}: {e}"
                )
    return [r for r in results if r is not None]


def claude_available() -> bool:
    """Quick preflight: confirm `claude --version` returns 0."""
    try:
        proc = subprocess.run(
            [CLAUDE_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return proc.returncode == 0
    except Exception:
        return False
