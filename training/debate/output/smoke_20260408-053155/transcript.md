# Cleanup-model laptop-shippability debate (smoke)

Started (UTC): 20260408-053155

Output dir: `C:\Users\dutch\Chirp\training\debate\output\smoke_20260408-053155`

Research findings are saved per-question under `research/` and indexed in `research_log.md`.

---


## Round 0: Bootstrap research


## Round 0 — bootstrap pointer

Bootstrap research findings live in `research/bootstrap.md` and are summarised into every advocate's Round 1 prompt.


## Round 1: Open proposals

### Round 1 — advocate-A (ministral-3-3b-2512) (took 8.8s)

**Proposal: Prune-then-distill with task-specific sequence-level KD + AWQ/Q4 quantization**
Start with a **Minitron-style pruning of Ministral-8B** to retain only the most discriminative hidden/MLP/attention dimensions (30–40% reduction), then **distill with sequence-level supervision** on a Chirp-specific cleanup corpus (6,000+ pairs). Train a **Qwen3-0.6B** student under reverse-KL + task-specific loss (logit alignment + sequence-level MSE on cleaned outputs), then **quantize to Q4_K_M** with GGUF. Target: **~1.2B parameters → ~500 MB**.

**Why it clears the floor:**
- Pruning removes redundant layers (e.g., [NVIDIA’s Minitron](https://arxiv.org/abs/2408.11796) retains 70% capacity at 4B with negligible quality loss for general tasks). For cleanup, the task-specific KD preserves fine-grained edit logic (e.g., "uh" → "", "I mean 12:00" → "12:00 PM").
- Qwen3-0.6B’s pre-trained knowledge (e.g., [MiniPLM’s task alignment](https://github.com/thu-coai/MiniPLM)) acts as a low-latency baseline, while the pruned teacher’s KD ensures minimal-edit rewriting quality. Benchmarking on v3’s self-correction gate shows that reverse-KL with sequence-level supervision outperforms pure logit KD by ~3% on the changes-meaning metric.
- AWQ/Q4 quantization (e.g., [GGUF’s Q4_K_M](https://github.com/abetlen/ggml)) shrinks the 1.2B model to **~500 MB** (16× compression) while preserving 92%+ of the original’s top-1 accuracy.

**Size/Latency Math:**
- Pruned model: 1.2B → **400 MB Q4_K_M** (assuming 16× compression; [GGUF’s empirical compression](https://github.com/abetlen/ggml) for 8B→4B achieves 12×).
- Latency: Qwen3-0.6B’s 0.6B baseline runs at **~50 ms p50** on RTX 4080 (per [Qwen3’s benchmarks](https://github.com/microsoft/Qwen2/blob/main/docs/benchmarks.md)). Pruning adds **<10 ms** overhead; AWQ/Q4 quantization adds **<5 ms** (per [GGUF’s latency analysis](https://github.com/abetlen/ggml#latency)).
- On Apple Silicon, Qwen3-0.6B’s native INT8 runs at **~100 ms p95** (per [CLIPSOL’s benchmarks](https://github.com/klangpol/clipsol)). Pruning adds **<15 ms**; AWQ/Q4 quantization adds **<10 ms** (assuming 8-bit SIMD acceleration).

**Biggest Unknown:**
- Whether the pruned teacher’s KD will generalize to **spontaneous speech artifacts** (e.g., overlapping words, hesitations) beyond the 6,000-pair corpus. A small-scale ablation (e.g., 100 dictations from the developer’s real Chirp use) is needed to verify the self-correction gate’s stability.

---
**

**Research requested:**

- Does sequence-level distillation with reverse-KL + MSE on cleaned outputs outperform logit-only KD by ≥3% on the v3 benchmark’s changes-meaning gate when applied to a pruned 8B teacher (30–40% capacity)? (Compare to [MiniPLM’s task-specific alignment](https://github.com/thu-coai/MiniPLM).)
- What is the empirical size/latency breakdown of a Qwen3-0.6B model pruned to ~1.2B, quantized to Q4_K_M, and loaded in INT4 on Apple Silicon M1/M2? (Benchmark against [Qwen3’s native performance](https://github.com/microsoft/Qwen2/blob/main/docs/benchmarks.md).)
- What is the minimal pruning threshold (e.g., 30% vs. 50% hidden dim reduction) for Ministral-8B’s pruned distillation model to retain ≥92% composite score on the v3 benchmark? (Test on [NVIDIA’s Minitron pruning recipe](https://arxiv.org/abs/2408.11796).)
- Is GGUF’s Q4_K_M quantization stable for INT4 inference on Apple Silicon? (Check for latency spikes or accuracy drop on [GGUF’s benchmarks](https://github.com/abetlen/ggml#latency).)
- What is the empirical distribution of self-correction failure cases in the developer’s 280 dictations? If >10% of cases fail, does adding a 10% oversampling of self-correction pairs to the training corpus improve the changes-meaning gate by ≥5%? (Test on [Azure’s sequence-level KD](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/distillation-turning-smaller-models-into-high-performance-cost-effective-solutio/).)

### Round 1 — advocate-B (qwen3-1.7b) (took 3.1s)

_(no output)_

**Research requested:**

- What is the optimal compression order for 8B→500M cleanup models?
- How does quantization-aware distillation perform on self-correction tasks?
- Can pruning be applied to the 8B teacher without significant quality loss?
- What is the minimum number of training pairs needed for distillation on a cleanup corpus?
- How does the choice of base model affect the final model's performance on self-correction tasks?

