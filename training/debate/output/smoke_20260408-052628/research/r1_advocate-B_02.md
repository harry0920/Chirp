# advocate-B — round 1 — question 2 — OK

## Question

Can the 500M student achieve p95 <100 ms on iGPU?

## Answer

**Interpretation:** "500M student" = ~500M-param causal LM (Qwen2.5-0.5B class) or encoder-decoder (FLAN-T5-base 250M / T5-large 770M), doing sentence-level cleanup (~30–80 input tokens, ~30–80 output tokens) on a laptop iGPU (Intel Arc/Iris Xe, Radeon 780M, Apple M-series).

**Short answer: No, not with vanilla autoregressive decoding. Borderline yes only with aggressive tricks (speculative decoding, non-autoregressive head, or static-shape ONNX graphs).**

### Concrete numbers from current benchmarks

**Autoregressive decode (Qwen2.5-0.5B-class):**
- llama.cpp Vulkan on Intel Arc **discrete** A750 hits ~150 tok/s TG on Qwen2.5-Coder-0.5B-Q8 ([llama.cpp discussion #12805](https://github.com/ggml-org/llama.cpp/discussions/12805)). That's discrete, not iGPU.
- On **integrated** GPUs (Iris Xe, Radeon 780M, Arrow Lake iGPU), Vulkan llama.cpp benchmarks land at **pp512 ≈ 60–150 t/s and tg128 ≈ 30–80 t/s for sub-1B models**, per [llama.cpp discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879) and [TechHara's CPU-vs-iGPU bench](https://medium.com/@techhara/llama-cpp-benchmark-cpu-vs-igpu-93b3cc40ece5). Arrow Lake iGPU also has Vulkan stability regressions ([issue #19327](https://github.com/ggml-org/llama.cpp/issues/19327), [#20201](https://github.com/ggml-org/llama.cpp/issues/20201)).
- Math for a 50-in/50-out cleanup at 50 tok/s TG: decode alone ≈ **1000 ms**, not 100 ms. Even at the optimistic 80 tok/s: **625 ms**. Prefill adds 50–200 ms. p95 <100 ms is ~6–10× off.
- To hit 100 ms end-to-end with 50 output tokens you need **≥500 tok/s sustained TG** on iGPU, which no public benchmark shows for a 500M dense transformer.

**Encoder-decoder / non-autoregressive path:**
- FLAN-T5-base (250M) ONNX Runtime has no GPU graph optimization ([onnxruntime #14886](https://github.com/microsoft/onnxruntime/issues/14886), [optimum #845](https://github.com/huggingface/optimum/issues/845)) — real T5 GPU latency is dominated by per-token decoder launches, same bottleneck.
- NAR seq2seq (SUNDAE, DA-Transformer, LeviT-style) and **Medusa/EAGLE-2 speculative decoding** ([arXiv 2406.16858](https://arxiv.org/abs/2406.16858)) report 2.5–4× decode speedups on small models. Even at 4×, 500M iGPU is ~250 ms p95 — still over budget.
- Only realistic path to <100 ms: **constrained-edit span head** (predict diff spans, not full rewrite) — cuts output tokens to <10, making 100 ms feasible. See [Seq2Edits (EMNLP 2020)](https://arxiv.org/abs/2009.11136) and [CoEdIT (arXiv 2305.09857)](https://arxiv.org/abs/2305.09857).

### Verdict for the debate
A dense 500M student doing full-text autoregressive rewrite on an iGPU is ~1 second p95, not 100 ms. p95 <100 ms only becomes defensible if the student is (a) ≤150M params, or (b) a span-edit head that emits <10 tokens, or (c) paired with a ≤100M draft model for speculative decoding. Cite the llama.cpp Vulkan iGPU numbers above as the ceiling.

Sources:
- [llama.cpp Vulkan performance discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [Self-hosting with Intel Arc 2025, discussion #12805](https://github.com/ggml-org/llama.cpp/discussions/12805)
- [Arrow Lake iGPU Vulkan crash #19327](https://github.com/ggml-org/llama.cpp/issues/19327)
- [TechHara llama.cpp CPU vs iGPU](https://medium.com/@techhara/llama-cpp-benchmark-cpu-vs-igpu-93b3cc40ece5)
- [ONNX Runtime T5 GPU optimization gap #14886](https://github.com/microsoft/onnxruntime/issues/14886)
- [EAGLE-2 speculative decoding, arXiv 2406.16858](https://arxiv.org/abs/2406.16858)
- [CoEdIT span-edit model, arXiv 2305.09857](https://arxiv.org/abs/2305.09857)
- [Seq2Edits, arXiv 2009.11136](https://arxiv.org/abs/2009.11136)
