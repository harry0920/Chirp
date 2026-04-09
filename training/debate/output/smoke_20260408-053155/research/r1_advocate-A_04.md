# advocate-A — round 1 — question 4 — OK

## Question

Is GGUF’s Q4_K_M quantization stable for INT4 inference on Apple Silicon? (Check for latency spikes or accuracy drop on [GGUF’s benchmarks](https://github.com/abetlen/ggml#latency).)

## Answer

**Interpretation:** Answering "is Q4_K_M production-stable on Apple Silicon Metal, and are there documented latency spikes or accuracy drops?" The link in the question (`abetlen/ggml#latency`) does not exist — abetlen maintains `llama-cpp-python`, not a `ggml` repo, and no `#latency` benchmark anchor is published there. Pulling from the actual llama.cpp/ggml-org sources instead.

**Stability verdict: yes, Q4_K_M is the de-facto default on Apple Silicon Metal and has been since mid-2024.** It is explicitly listed as "recommended" in llama.cpp's quantize help output and is the format HuggingFace GGUF uploads default to (TheBloke, bartowski, unsloth, lmstudio-community).

**Accuracy.** On the ggml-org/llama.cpp Discussion #2094 perplexity table (LLaMA-2-7B, wiki.test), Q4_K_M lands at roughly +0.0535 ppl vs F16, versus Q4_0 at +0.2499 and Q5_K_M at +0.0142 — i.e. ~0.9% relative degradation, well inside the "moderate" band. The arXiv unified evaluation on Llama-3.1-8B-Instruct (arXiv:2601.14277, "Which Quantization Should I Use?") reports Q4_K_M retaining ≥98% of F16 scores on MMLU/ARC/HellaSwag, with Q5_K_M only ~0.2 pp ahead. No Apple-Silicon-specific accuracy delta exists — quant math is backend-agnostic; Metal matches CUDA bit-for-bit on the quant kernels.

**Latency.** On M3 Max, 8B Q4_K_M runs 60–75 tok/s decode (Discussion #4167, Discussion #12985 container benchmarks from 2025). **Known spike sources, none of them Q4_K_M-specific:**
1. **First-run Metal shader compilation** — several seconds on cold start, cached to `~/Library/Caches` after. Mitigate with warm-up inference at load.
2. **Unified-memory pressure** — if the model + KV cache exceeds the GPU's wired limit (`iogpu.wired_limit_mb`), you get swap stalls. An 8B Q4_K_M is ~4.8 GB, safe on 16 GB Macs.
3. **IQ-quants** (IQ2/IQ3/IQ4_XS) are the ones with documented Metal slowness (Discussion #5617) — K-quants including Q4_K_M are not affected.

**For your 500 MB budget, Q4_K_M is the wrong knob anyway** — an 8B at Q4_K_M is ~4.8 GB, ~10× over budget. Q4_K_M's stability story only matters as the teacher-inference format; the shippable student needs a <1B model at Q4_K_M (~400–600 MB) or a 1–2B at Q2_K/IQ2_XXS, where accuracy drops become the real question.

Sources:
- [llama.cpp Discussion #2094 — quantization perplexity table](https://github.com/ggml-org/llama.cpp/discussions/2094)
- [llama.cpp Discussion #4167 — Apple Silicon M-series perf](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [llama.cpp Discussion #5617 — slow IQ quants on Apple Silicon](https://github.com/ggml-org/llama.cpp/discussions/5617)
- [llama.cpp Discussion #12985 — M-series GPU container benchmarks](https://github.com/ggml-org/llama.cpp/discussions/12985)
- [arXiv 2601.14277 — Unified eval of llama.cpp quantization on Llama-3.1-8B](https://arxiv.org/html/2601.14277v1)
- [llama.cpp Issue #10982 — Metal vs Vulkan perf](https://github.com/ggml-org/llama.cpp/issues/10982)
