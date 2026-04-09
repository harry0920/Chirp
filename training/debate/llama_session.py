"""
LlamaSession: spin-up / shut-down a llama-server for one local model and
send chat-completion requests against it.

Adapted from training/benchmark_v3/runner.py:55-145. Drops the per-case
scoring; this module only knows how to run a chat. Used as a context
manager so the orchestrator gets clean teardown on exception.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class LlamaSession:
    """Context-managed llama-server lifecycle for a single GGUF.

    Usage:
        with LlamaSession(cand) as sess:
            text = sess.chat(messages, sampling, max_tokens=1024)
    """

    def __init__(
        self,
        cand: Dict,
        *,
        ctx_size: int = 8192,
        n_predict: int = 1024,
        startup_timeout_s: float = 90.0,
        request_timeout_s: float = 240.0,
    ) -> None:
        self.cand = cand
        self.ctx_size = ctx_size
        self.n_predict = n_predict
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.port: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self.model_id: str = cand.get("model_id", "model")

    def __enter__(self) -> "LlamaSession":
        self._spawn()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _spawn(self) -> None:
        binary = Path(self.cand["binary"])
        model = Path(self.cand["model"])
        if not binary.exists():
            raise FileNotFoundError(f"binary not found: {binary}")
        if not model.exists():
            raise FileNotFoundError(f"model not found: {model}")

        self.port = _free_port()
        args = [
            str(binary),
            "--model", str(model),
            "--port", str(self.port),
            "--ctx-size", str(self.ctx_size),
            "--n-predict", str(self.n_predict),
            "--gpu-layers", str(self.cand.get("gpu_layers", 99)),
            "--flash-attn", "on",
            "--batch-size", "512",
            "--parallel", "1",
            "--reasoning-budget", str(self.cand.get("reasoning_budget", 0)),
            "--jinja",
            "--log-disable",
        ]
        print(f"  spawning {binary.name} :{self.port} (model={model.name}) ...", flush=True)
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

        health = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + self.startup_timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health, timeout=1) as r:
                    body = json.loads(r.read())
                    if body.get("status") == "ok":
                        print("  server ready", flush=True)
                        return
            except (urllib.error.URLError, ConnectionResetError, json.JSONDecodeError):
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server died during startup (exit {self.proc.returncode})"
                )
            time.sleep(0.5)

        self.close()
        raise RuntimeError(
            f"llama-server failed to start within {self.startup_timeout_s:.0f}s"
        )

    def chat(
        self,
        messages: List[Dict],
        sampling: Dict,
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a /v1/chat/completions request and return the assistant text.

        Raises on HTTP/JSON errors — the orchestrator decides how to recover.
        """
        if self.port is None:
            raise RuntimeError("LlamaSession not started")
        payload: Dict = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "cache_prompt": True,
            **sampling,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as r:
            body = json.loads(r.read())
        return body["choices"][0]["message"]["content"]

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None
        self.port = None
