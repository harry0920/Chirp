"""
run_debate.py — CLI entry for the local-LLM debate harness.

Usage (run from repo root):
    py -3 -m training.debate.run_debate --preflight
    py -3 -m training.debate.run_debate --smoke-test
    py -3 -m training.debate.run_debate              # full debate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import roles
from .orchestrator import run_fast_debate, run_full_debate
from .research import claude_available

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "training" / "debate" / "output"

PRODUCTION_PID_FILE = Path(
    "C:/Users/dutch/AppData/Roaming/com.chirp.app/llm/llama-server.pid"
)


def _pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check for the production llama-server pid."""
    if os.name == "nt":
        # tasklist exit code is always 0; check the output instead
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return str(pid) in (out.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def preflight() -> int:
    print("Preflight checks:")
    ok = True

    # 1. Production llama-server pid file (only block if the pid is actually live)
    if PRODUCTION_PID_FILE.exists():
        try:
            pid = int(PRODUCTION_PID_FILE.read_text().strip())
        except Exception:
            pid = -1
        if pid > 0 and _pid_is_alive(pid):
            print(f"  [FAIL] production llama-server is running (pid {pid}). Stop chirp.exe before running the debate.")
            ok = False
        else:
            print(f"  [ok] production pid file present but pid {pid} is not alive (stale).")
    else:
        print("  [ok] no production pid file.")

    # 2. claude --print available
    if claude_available():
        print("  [ok] `claude --version` returns 0.")
    else:
        print("  [FAIL] `claude` CLI not on PATH or `--version` failed.")
        ok = False

    # 3. All candidate GGUFs exist
    cands = roles.load_candidates()
    seats = [
        *roles.ADVOCATE_KEYS,
        roles.SKEPTIC_KEY,
        roles.MODERATOR_KEY,
    ]
    missing = []
    for key in seats:
        cand = cands.get(key)
        if not cand:
            missing.append(f"{key} (not in candidates.yaml)")
            continue
        binp = Path(cand["binary"])
        modp = Path(cand["model"])
        if not binp.exists():
            missing.append(f"{key}: binary {binp}")
        if not modp.exists():
            missing.append(f"{key}: model {modp}")
    if missing:
        ok = False
        print("  [FAIL] missing files:")
        for m in missing:
            print(f"    - {m}")
    else:
        print(f"  [ok] all {len(seats)} candidate GGUFs + binaries present.")

    # 4. Output dir is creatable
    try:
        DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
        print(f"  [ok] output dir writable: {DEFAULT_OUT}")
    except Exception as e:
        print(f"  [FAIL] cannot create output dir {DEFAULT_OUT}: {e}")
        ok = False

    print()
    print("PREFLIGHT: PASS" if ok else "PREFLIGHT: FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="run preflight checks and exit")
    ap.add_argument("--smoke-test", action="store_true", help="2 advocates, round 0+1 only")
    ap.add_argument("--fast", action="store_true", help="6 advocates R1 + R5 moderator only, no inter-round research")
    ap.add_argument(
        "--reuse-bootstrap",
        type=Path,
        default=None,
        help="path to a prior bootstrap.md to reuse (skip the 60-90s bootstrap research call)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output base dir")
    args = ap.parse_args()

    if args.preflight:
        return preflight()

    rc = preflight()
    if rc != 0:
        print("\nAborting: preflight failed. Re-run with --preflight to inspect.")
        return rc

    args.out.mkdir(parents=True, exist_ok=True)
    if args.fast:
        run_fast_debate(args.out, reuse_bootstrap=args.reuse_bootstrap)
    else:
        run_full_debate(args.out, smoke=args.smoke_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
