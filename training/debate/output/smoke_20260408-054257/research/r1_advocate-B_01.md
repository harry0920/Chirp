# advocate-B — round 1 — question 1 — OK

## Question

Is the 4-bit AWQ quantization sufficient to maintain cleanup accuracy on real-world data?

## Answer

# Is AWQ 4-bit sufficient for an 8B cleanup model?

**Interpretation:** "Sufficient" = keeps instruction-following and edit-fidelity within noise of FP16 on text-rewriting tasks like Chirp's v3 benchmark.

**Short answer: probably yes for perplexity/semantics, but AWQ specifically has a documented soft spot on instruction-following that matters for cleanup.**

## Evidence (2025)

- **Jin et al., arXiv:2409.11055v6 (Jun 2025)** — 13-benchmark sweep across Llama-3.1-8B, Qwen-2.5, Mistral, etc. AWQ-int4 averaged **~1.8% drop** on long-context/general tasks vs FP16, beating GPTQ-int4 (2.7%) and BnB-nf4 (6.9%). For 7–8B, AWQ is the best-quality 4-bit weight-only method they tested.
- **"Quantization Methods, Task Difficulty, and Model Size" — IJCAI 2025 (paper 0902)** — Finding specific to your use case: **instruction-following (IFEval) degrades fastest under 4-bit**, and AWQ is called out as suffering more on IFEval than GPTQ-int4, attributed to group-wise quant disrupting alignment even when perplexity looks fine. For small/mid models (≤8B) this effect is more pronounced.
- **Red Hat / Neural Magic, "Half a million evaluations on quantized LLMs"** — W4A16 (AWQ-style) on Llama-3-8B-Instruct recovered **>99% of FP16 on OpenLLM v1** and was statistically indistinguishable on Arena-Hard. They recommend W4A16 as the default for ≤8B instruct models.
- **COLM 2025, arXiv:2504.04823** — Confirms AWQ ≈ FP16 on generation quality for 7–8B, but flags format-sensitive tasks (strict JSON, exact-match rewrites) as the failure mode, not open-ended generation.

## Implication for Chirp's cleanup model

Cleanup = strict instruction-following + near-verbatim rewrite — exactly the AWQ soft spot. Expect ≤2% perplexity/semantic drop, but a real risk on edge cases already visible in your v3 benchmark (JSON-fallback, spoken-punctuation, over-paraphrasing). **Sufficient in the average case; not sufficient as a drop-in without re-running the 250-case v3 harness on the AWQ build.** If IFEval-style regressions appear, GPTQ-int4 with a cleanup-domain calibration set, or W4A16 with per-channel (not group) quant, are the documented fallbacks. 500 MB budget for an 8B at 4-bit (~4.5 GB) is the bigger blocker — AWQ quality is not your binding constraint.

Sources:
- [arXiv:2409.11055 — Comprehensive Evaluation of Quantized Instruction-Tuned LLMs](https://arxiv.org/pdf/2409.11055)
- [IJCAI 2025 — Quantization Methods, Task Difficulty, and Model Size](https://www.ijcai.org/proceedings/2025/0902.pdf)
- [Red Hat — Half a million evaluations on quantized LLMs](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)
- [arXiv:2504.04823 — COLM 2025 quantization eval](https://arxiv.org/pdf/2504.04823)
- [AutoAWQ (casper-hansen/AutoAWQ)](https://github.com/casper-hansen/AutoAWQ)
- [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
