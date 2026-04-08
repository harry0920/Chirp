"""
Download all 7 candidate GGUFs to ~/chirp-bench/models/.

Skips files that are already present and the right size. Uses
huggingface_hub.hf_hub_download with parallel jobs (3 at a time —
HF rate-limits aggressive parallelism). Total ~24 GB.

Usage:
    python download_candidates.py
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

# (candidate_name, repo, filename)
CANDIDATES = [
    ("qwen3-4b-instruct-2507", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    ("gemma-4-e4b-it",         "unsloth/gemma-4-E4B-it-GGUF",          "gemma-4-E4B-it-Q4_K_M.gguf"),
    ("ministral-3-3b-2512",    "mistralai/Ministral-3-3B-Instruct-2512-GGUF", "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
    ("gemma-4-e2b-it",         "unsloth/gemma-4-E2B-it-GGUF",          "gemma-4-E2B-it-Q4_K_M.gguf"),
    ("qwen3-1.7b",             "unsloth/Qwen3-1.7B-GGUF",              "Qwen3-1.7B-Q4_K_M.gguf"),
    ("eurollm-9b",             "bartowski/EuroLLM-9B-Instruct-GGUF",   "EuroLLM-9B-Instruct-Q4_K_M.gguf"),
    ("ministral-3-8b-2512",    "mistralai/Ministral-3-8B-Instruct-2512-GGUF", "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"),
]

BENCH_DIR = Path.home() / "chirp-bench" / "models"


def download_one(name: str, repo: str, filename: str) -> tuple[str, Path, str]:
    target = BENCH_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    size_mb = Path(path).stat().st_size // 1024 // 1024
    return (name, Path(path), f"{size_mb} MB")


def main():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading 7 candidate GGUFs to {BENCH_DIR}", flush=True)
    print("(parallel x3, skips already-present files)", flush=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(download_one, n, r, f): n for n, r, f in CANDIDATES}
        for fut in as_completed(futures):
            try:
                name, path, size = fut.result()
                print(f"  [DONE] {name:<28} {size:>10}  {path.name}", flush=True)
            except Exception as e:
                print(f"  [FAIL] {futures[fut]:<28} {e}", flush=True)
                raise

    print("\nAll downloads complete.", flush=True)


if __name__ == "__main__":
    main()
