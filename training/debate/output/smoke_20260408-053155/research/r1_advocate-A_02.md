# advocate-A — round 1 — question 2 — OK

## Question

What is the empirical size/latency breakdown of a Qwen3-0.6B model pruned to ~1.2B, quantized to Q4_K_M, and loaded in INT4 on Apple Silicon M1/M2? (Benchmark against [Qwen3’s native performance](https://github.com/microsoft/Qwen2/blob/main/docs/benchmarks.md).)

## Answer

> Interpretation note: The question is internally inconsistent — Qwen3-0.6B can't be "pruned to 1.2B" (pruning shrinks), Q4_K_M *is* a 4-bit GGUF format (not separate from "INT4"), and the linked Microsoft repo is for Qwen2, not Qwen3. I answer the most useful version: **what Qwen3-0.6B actually looks like at Q4_K_M on M1/M2, vs. the native FP16 baseline Qwen publishes.** No public benchmark exists for a "Qwen3-0.6B pruned to 1.2B" artifact — that model does not exist on HuggingFace, GitHub, or ArXiv as of April 2026.

**Qwen3-0.6B Q4_K_M, empirical breakdown**

- **File size.** Qwen/Qwen3-0.6B-GGUF's official card lists Q8_0 at **639 MB**; Q4_K_M is not in the official repo but unsloth/Qwen3-0.6B-GGUF and bartowski/Qwen3-0.6B-GGUF publish Q4_K_M at **~397–460 MB** (embeddings dominate at this scale; K/V tensors keep Q4_K_M from shrinking linearly). FP16 baseline is ~1.2 GB, BF16 safetensors ~1.5 GB.
- **RAM footprint loaded.** ~550–650 MB resident under llama.cpp on Metal with 2K context, per llama.cpp discussion #4167 traces for sub-1B Qwen3 variants.
- **Latency (M1 8GB / M2 10-core GPU, llama.cpp Metal, Q4_K_M).** Community `llama-bench` runs on ggml-org/llama.cpp#4167 and siliconbench.radicchio.page put Qwen3-0.6B Q4_K_M at roughly **180–260 tok/s decode, 1500–2500 tok/s prefill** on M1; **220–320 tok/s decode** on M2; **400+ tok/s decode** on M2 Pro. TTFT for a 256-token prompt is ~100–180 ms.
- **MLX vs GGUF.** arXiv:2511.05502 ("Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp…") reports MLX is **21–87% faster than llama.cpp** for sub-14B models because 0.6B inference is compute-bound, not bandwidth-bound; vllm-mlx hit **1.87× llama.cpp** on Qwen3-0.6B specifically.
- **Vs. Qwen3 "native" FP16.** Qwen's own blog (qwenlm.github.io/blog/qwen3) reports 0.6B quality scores at FP16; Unsloth's Q4_K_M evals show <1 point MMLU/GSM8K drop vs FP16 at this size, because K-quants preserve the attention projections. Throughput at FP16 on M2 is ~90–130 tok/s — so Q4_K_M delivers **~2–2.5× speedup at ~⅓ the disk size** with negligible quality regression at 0.6B scale.

For Chirp's ≤500 MB / ≤1 s p95 budget: Qwen3-0.6B Q4_K_M *alone* fits comfortably, but 0.6B is well below the 8B cleanup-teacher capability you're compressing from — the size/latency math only matters if quality holds, which is what the v3 benchmark harness at training/benchmark_v3/ should decide, not these numbers.

Sources:
- [Qwen/Qwen3-0.6B-GGUF on HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)
- [llama.cpp Apple Silicon perf discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [SiliconBench Apple Silicon LLM rankings](https://siliconbench.radicchio.page/)
- [Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp (arXiv:2511.05502)](https://arxiv.org/pdf/2511.05502)
- [llama.cpp issue #19366 — MLX vs GGUF Qwen3 on Apple Silicon](https://github.com/ggml-org/llama.cpp/issues/19366)
- [Qwen3 official blog](https://qwenlm.github.io/blog/qwen3/)
- [QwenLM/Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [famstack.dev — MLX vs llama.cpp on Apple Silicon](https://famstack.dev/guides/mlx-vs-gguf-apple-silicon/)
