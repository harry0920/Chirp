"""
Benchmark runner: spawn llama-server per candidate, run the corpus,
score per-case, write per-candidate result files.

Usage:
    python runner.py --candidate qwen3-0.6b
    python runner.py --candidate qwen3-0.6b --limit 20  # smoke run
    python runner.py --candidate qwen3-0.6b --greedy    # cross-cand mode
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import scorers
import prompts

CORPUS_PATH = ROOT / "corpus" / "english_gold.jsonl"
CANDIDATES_PATH = ROOT / "candidates.yaml"
RESULTS_DIR = ROOT / "results"

DEFAULT_STRATEGY = "prod-v13"


def load_corpus() -> List[Dict]:
    cases = []
    with CORPUS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_candidates() -> Dict[str, Dict]:
    with CANDIDATES_PATH.open() as f:
        return yaml.safe_load(f)["candidates"]


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def spawn_server(cand: Dict, port: int) -> subprocess.Popen:
    """Spawn llama-server for one candidate. Wait for /health to return ok."""
    binary = Path(cand["binary"])
    model = Path(cand["model"])
    if not binary.exists():
        raise FileNotFoundError(f"binary not found: {binary}")
    if not model.exists():
        raise FileNotFoundError(f"model not found: {model}")

    args = [
        str(binary),
        "--model", str(model),
        "--port", str(port),
        "--ctx-size", str(cand.get("ctx_size", 2048)),
        "--n-predict", str(cand.get("n_predict", 512)),
        "--gpu-layers", str(cand.get("gpu_layers", 99)),
        "--flash-attn", "on",
        "--batch-size", "512",
        "--parallel", "1",
        "--reasoning-budget", str(cand.get("reasoning_budget", 0)),
        "--jinja",
        "--log-disable",
    ]
    print(f"  spawning {binary.name} :{port} ...", flush=True)
    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    health = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=1) as r:
                body = json.loads(r.read())
                if body.get("status") == "ok":
                    print(f"  server ready", flush=True)
                    return proc
        except (urllib.error.URLError, ConnectionResetError, json.JSONDecodeError):
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server died (exit {proc.returncode})")
        time.sleep(0.5)

    proc.kill()
    raise RuntimeError("llama-server failed to start within 60s")


def build_messages(strategy: Dict, text: str) -> List[Dict]:
    """Build the chat messages for a single case using a prompt strategy."""
    msgs = [{"role": "system", "content": strategy["system"]}]
    for u, a in strategy.get("fewshot", []):
        msgs.append({"role": "user", "content": strategy["wrap_input"](u)})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": strategy["wrap_input"](text)})
    return msgs


def cleanup_text(port: int, text: str, sampling: Dict, model_id: str, strategy: Dict) -> tuple[str, str]:
    """Send a single cleanup request to the running llama-server.

    Returns (parsed_output, raw_response) so the per_case file can record
    both — useful when debugging why a JSON parser fell back."""
    payload = {
        "model": model_id,
        "messages": build_messages(strategy, text),
        "stream": False,
        "cache_prompt": True,
        **sampling,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    raw = body["choices"][0]["message"]["content"]
    parsed = strategy["parse_output"](raw)
    return parsed, raw


def dynamic_max_tokens(text: str) -> int:
    # Mirror llm.rs logic: 1.2x word count * 2 (rough token-per-word) with floor 64
    word_count = len(text.split())
    return max(int(word_count * 2.0 * 1.2), 64)


def run_candidate(name: str, limit: int | None, greedy: bool, strategy_name: str = DEFAULT_STRATEGY) -> Path:
    cand = load_candidates()[name]
    strategy = prompts.get(strategy_name)
    cases = load_corpus()
    if limit:
        cases = cases[:limit]

    if greedy:
        sampling_base = {"temperature": 0.0, "top_k": 1}
    else:
        sampling_base = cand.get("sampling", {})

    port = free_port()
    proc = spawn_server(cand, port)
    results = []

    try:
        t_start = time.time()
        for i, case in enumerate(cases):
            sampling = dict(sampling_base)
            sampling["max_tokens"] = dynamic_max_tokens(case["input"])
            t0 = time.time()
            try:
                output, raw = cleanup_text(port, case["input"], sampling, cand.get("model_id", name), strategy)
                err = None
            except Exception as e:
                output = ""
                raw = ""
                err = str(e)
            ttlt_ms = (time.time() - t0) * 1000

            score = scorers.score_case(case, output)
            results.append({
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "reference": case["reference"],
                "output": output,
                "raw": raw,
                "ttlt_ms": ttlt_ms,
                "scores": score,
                "error": err,
            })
            if (i + 1) % 25 == 0 or i + 1 == len(cases):
                avg = sum(r["scores"]["composite"] for r in results) / len(results)
                print(f"  [{i+1}/{len(cases)}] composite={avg:.3f} ({ttlt_ms:.0f}ms)", flush=True)

        elapsed = time.time() - t_start
    finally:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    # Save results — strategy name in the directory so multiple
    # strategies on the same candidate don't collide.
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / name / f"{strategy_name}-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-candidate metadata
    meta = {
        "candidate": name,
        "strategy": strategy_name,
        "model_path": cand["model"],
        "sampling": sampling_base,
        "greedy": greedy,
        "n_cases": len(results),
        "elapsed_seconds": elapsed,
        "system_prompt": strategy["system"],
        "timestamp": ts,
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults: {out_dir}", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY, help=f"prompt strategy (default {DEFAULT_STRATEGY})")
    ap.add_argument("--limit", type=int, default=None, help="run first N cases only")
    ap.add_argument("--greedy", action="store_true", help="cross-candidate greedy mode")
    args = ap.parse_args()
    run_candidate(args.candidate, args.limit, args.greedy, args.strategy)


if __name__ == "__main__":
    main()
