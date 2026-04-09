"""
transcript.py — incremental crash-safe markdown writer for debate runs.

Every append flushes + fsyncs so a Ctrl-C in the middle of a long round still leaves
a valid markdown file on disk up to the interruption point. The file's directory also
holds a `research/` subdir for individual research findings and a `research_log.md`
index that all subsequent rounds read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DebateRun:
    """Owns the on-disk artifacts for one debate run."""

    out_dir: Path
    transcript_path: Path
    research_dir: Path
    research_log_path: Path
    started_at: str
    # In-memory copies kept so the orchestrator can build per-round digests
    # without reparsing the markdown file.
    rounds: Dict[int, Dict[str, str]] = field(default_factory=dict)
    research_by_seat_round: Dict[tuple, List[Dict]] = field(default_factory=dict)

    @classmethod
    def create(cls, base_dir: Path, *, label: str = "debate") -> "DebateRun":
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_dir = base_dir / f"{label}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=False)
        research_dir = out_dir / "research"
        research_dir.mkdir(exist_ok=True)
        run = cls(
            out_dir=out_dir,
            transcript_path=out_dir / "transcript.md",
            research_dir=research_dir,
            research_log_path=out_dir / "research_log.md",
            started_at=ts,
        )
        run._init_files(label)
        return run

    def _init_files(self, label: str) -> None:
        header = (
            f"# Cleanup-model laptop-shippability debate ({label})\n\n"
            f"Started (UTC): {self.started_at}\n\n"
            f"Output dir: `{self.out_dir}`\n\n"
            f"Research findings are saved per-question under `research/` and indexed in `research_log.md`.\n\n"
            "---\n\n"
        )
        self._atomic_append(self.transcript_path, header)
        self._atomic_append(
            self.research_log_path,
            f"# Research log ({label})\n\nEach entry links to a markdown file under `research/`.\n\n",
        )

    @staticmethod
    def _atomic_append(path: Path, text: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    # ---------- transcript writes ----------

    def write_round_header(self, round_num: int, title: str) -> None:
        self._atomic_append(self.transcript_path, f"\n## Round {round_num}: {title}\n\n")

    def write_turn(
        self,
        round_num: int,
        seat: str,
        candidate_key: str,
        body: str,
        research_request: Optional[List[str]] = None,
        elapsed_s: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        timing = f" (took {elapsed_s:.1f}s)" if elapsed_s is not None else ""
        head = f"### Round {round_num} \u2014 {seat} ({candidate_key}){timing}\n\n"
        if error:
            head += f"_ERROR: {error}_\n\n"
        body_block = (body or "_(no output)_") + "\n\n"
        rq_block = ""
        if research_request:
            rq_lines = "\n".join(f"- {q}" for q in research_request)
            rq_block = f"**Research requested:**\n\n{rq_lines}\n\n"
        self._atomic_append(self.transcript_path, head + body_block + rq_block)
        # Cache for digest building
        self.rounds.setdefault(round_num, {})[seat] = body or ""

    def write_section(self, title: str, body: str) -> None:
        self._atomic_append(self.transcript_path, f"\n## {title}\n\n{body}\n\n")

    # ---------- research writes ----------

    def write_research(
        self,
        round_num: int,
        seat: str,
        question_idx: int,
        question: str,
        answer: str,
        ok: bool,
        error: Optional[str] = None,
    ) -> Path:
        slug = f"r{round_num}_{seat}_{question_idx:02d}.md"
        path = self.research_dir / slug
        status = "OK" if ok else f"FAILED ({error or 'unknown'})"
        content = (
            f"# {seat} \u2014 round {round_num} \u2014 question {question_idx} \u2014 {status}\n\n"
            f"## Question\n\n{question}\n\n"
            f"## Answer\n\n{answer or '_(empty)_'}\n"
        )
        path.write_text(content, encoding="utf-8")
        # Index entry
        idx_line = (
            f"- **{seat}** r{round_num} q{question_idx} \u2014 [{slug}](research/{slug})"
            f" \u2014 {status}\n"
            f"  - Q: {question.strip()[:200]}\n"
        )
        self._atomic_append(self.research_log_path, idx_line)
        self.research_by_seat_round.setdefault((seat, round_num), []).append(
            {"question": question, "answer": answer, "ok": ok, "path": str(path)}
        )
        return path

    def write_bootstrap_research(self, question: str, answer: str, ok: bool, error: Optional[str] = None) -> Path:
        path = self.research_dir / "bootstrap.md"
        status = "OK" if ok else f"FAILED ({error or 'unknown'})"
        path.write_text(
            f"# Round 0 bootstrap research \u2014 {status}\n\n"
            f"## Question\n\n{question}\n\n"
            f"## Answer\n\n{answer or '_(empty)_'}\n",
            encoding="utf-8",
        )
        self._atomic_append(
            self.research_log_path,
            f"- **bootstrap** r0 \u2014 [bootstrap.md](research/bootstrap.md) \u2014 {status}\n",
        )
        return path

    # ---------- digest helpers (read in-memory cache) ----------

    def round_text(self, round_num: int, seat: str) -> str:
        return self.rounds.get(round_num, {}).get(seat, "")

    def all_seats_in_round(self, round_num: int) -> List[str]:
        return list(self.rounds.get(round_num, {}).keys())

    def research_findings_for(self, seat: str, round_num: int) -> str:
        items = self.research_by_seat_round.get((seat, round_num), [])
        if not items:
            return "_(no research requested or no findings returned)_"
        out = []
        for i, it in enumerate(items, 1):
            head = f"### Q{i}: {it['question'].strip()}"
            if not it["ok"]:
                out.append(head + "\n\n_(research failed)_\n")
            else:
                out.append(head + "\n\n" + it["answer"].strip() + "\n")
        return "\n".join(out)

    def all_findings_for_seat(self, seat: str) -> str:
        chunks = []
        for (s, rnd), items in sorted(self.research_by_seat_round.items()):
            if s != seat:
                continue
            chunks.append(f"### From round {rnd}\n")
            for i, it in enumerate(items, 1):
                if it["ok"]:
                    chunks.append(f"**Q{i}: {it['question'].strip()}**\n\n{it['answer'].strip()}\n")
        return "\n".join(chunks) if chunks else "_(no findings yet)_"

    def round_digest(self, round_num: int, *, exclude_seat: Optional[str] = None, max_chars: int = 600) -> str:
        """Compact summary of every seat's turn in a given round, for use in
        cross-pollination prompts."""
        chunks = []
        for seat, body in self.rounds.get(round_num, {}).items():
            if seat == exclude_seat:
                continue
            snippet = body.strip().replace("RESEARCH_REQUEST:", "[research_request line]")
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars].rstrip() + " ..."
            chunks.append(f"**{seat}:**\n{snippet}\n")
        return "\n".join(chunks) if chunks else "_(no other seats spoke)_"

    def full_transcript_digest(self, *, max_chars_per_turn: int = 1200) -> str:
        chunks = []
        for rnd in sorted(self.rounds.keys()):
            chunks.append(f"## Round {rnd}\n")
            for seat, body in self.rounds[rnd].items():
                snippet = body.strip().replace("RESEARCH_REQUEST:", "[research_request line]")
                if len(snippet) > max_chars_per_turn:
                    snippet = snippet[:max_chars_per_turn].rstrip() + " ..."
                chunks.append(f"### {seat}\n{snippet}\n")
        return "\n".join(chunks)

    def full_transcript_text(self) -> str:
        return self.transcript_path.read_text(encoding="utf-8")
