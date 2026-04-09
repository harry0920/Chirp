# advocate-B — round 1 — question 3 — OK

## Question

What is the expected latency on integrated GPUs for the distilled model?

## Answer

*Interpretation: "the distilled model" = FLAN-T5-base-class (~250M params) student from your Ministral-8B → T5 distillation plan, Q4/int8, on typical laptop iGPUs (Iris Xe, Arc, Radeon 780M, Apple M-series) for ~30-token-in / ~40-token-out cleanup calls.*

**No public benchmark hits this exact config**, so the answer is an extrapolation from adjacent measurements plus the memory-bandwidth ceiling.

**Memory-bandwidth ceiling (the dominant factor for decode on iGPU):**
- FLAN-T5-base Q4 ≈ 125 MB weights. Autoregressive decode = (model bytes read) × (tokens out) / bandwidth.
- DDR5-5600 dual-channel ≈ 90 GB/s usable; LPDDR5x-7500 ≈ 120 GB/s; Apple M3 ≈ 100 GB/s.
- Theoretical decode ceiling: 700–950 tok/s. Practical iGPU realization on llama.cpp Vulkan is ~15–25% of ceiling for sub-1B models → **100–200 tok/s decode**.

**Anchors from measured iGPU runs:**
- Radeon 780M, Llama-3.2-1B Q4, llama.cpp Vulkan: ~30–50 tok/s decode ([AMD Ryzen AI blog](https://www.amd.com/en/blogs/2024/accelerating-llama-cpp-performance-in-consumer-llm.html), [llama.cpp Vulkan discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)). 250M is ~4× smaller and bandwidth-bound, so scale to ~120–200 tok/s.
- Iris Xe, ~1B Q4, Vulkan: ~15–25 tok/s (TechHara's [CPU vs iGPU benchmark](https://medium.com/@techhara/llama-cpp-benchmark-cpu-vs-igpu-93b3cc40ece5)). Scale to ~60–100 tok/s for 250M.
- Radeon 780M ROCm on 7B Q4: 19.6 tok/s decode, 263 tok/s prefill ([llama.cpp ROCm #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)). Prefill at this scale means your ~30-token encoder pass is essentially free (<20 ms).

**Expected p95 latency for a 40-token cleanup output:**
| iGPU class | Decode tok/s | 40-tok generate | +prefill/overhead | **Total p95** |
|---|---|---|---|---|
| Radeon 780M / Arc 140V / M3 | 120–200 | 200–330 ms | +100 ms | **~0.3–0.5 s** |
| Iris Xe (Tiger/Alder Lake) | 60–100 | 400–670 ms | +150 ms | **~0.6–0.8 s** |
| Low-end UHD / older iGPU | 30–50 | 800–1300 ms | +200 ms | **~1.0–1.5 s** ⚠️ |

**Caveats:** (1) T5 encoder-decoder adds cross-attention overhead versus decoder-only anchors — budget +15–20%. (2) ONNX Runtime DirectML on T5 historically underperforms llama.cpp Vulkan for same-size models ([ORT issue #10807](https://github.com/microsoft/onnxruntime/issues/10807)); for your ≤1s target, ship via llama.cpp Vulkan or convert the student to a decoder-only backbone during distillation. (3) First-token latency includes Vulkan shader compile (~500 ms cold) — warm the model at app start.

**Bottom line: 300–800 ms p95 on modern iGPUs (2023+), 1s+ on pre-2021 UHD.** The ≤500 MB / ≤1s target is comfortably achievable on your stated hardware class *if* you ship via Vulkan, not DirectML, and warm the model at launch.

Sources:
- [llama.cpp Vulkan discussion #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)
- [llama.cpp ROCm discussion #15021](https://github.com/ggml-org/llama.cpp/discussions/15021)
- [AMD Ryzen AI 300 llama.cpp blog](https://www.amd.com/en/blogs/2024/accelerating-llama-cpp-performance-in-consumer-llm.html)
- [TechHara CPU vs iGPU llama.cpp benchmark](https://medium.com/@techhara/llama-cpp-benchmark-cpu-vs-igpu-93b3cc40ece5)
- [ORT T5 GPU issue #10807](https://github.com/microsoft/onnxruntime/issues/10807)
- [arXiv:2601.03874 — Small Decoder-Only LMs for Grammar Correction](https://arxiv.org/abs/2601.03874)
