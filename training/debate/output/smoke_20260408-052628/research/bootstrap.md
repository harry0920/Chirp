# Round 0 bootstrap research — OK

## Question

In 300-500 words, list the top 5-7 distinct techniques from 2025-2026 for compressing an instruction-tuned LLM from ~8B parameters down to ~500M while preserving narrow task-specific capability (text cleanup / minimal-edit rewriting). For each technique give: name, one source link, and one sentence on the tradeoff. This is a smoke test — be fast, do not aim for exhaustive coverage.

## Answer

**Interpretation:** "Compression" read broadly — any technique that can move an 8B instruction-tuned teacher toward a ~500M narrow-task student.

### Top techniques (2025–2026)

1. **Structured width+depth pruning then distillation (Minitron/Compact)** — Prune the 8B teacher along hidden/attention/MLP and depth axes, then retrain the pruned student via logit distillation on the teacher. NVIDIA's recipe compressed Llama-3.1 8B → 4B with <2% regression using ~100B tokens, orders of magnitude cheaper than training from scratch. [arXiv:2408.11796](https://arxiv.org/abs/2408.11796). *Tradeoff:* still requires substantial retraining compute and a good calibration corpus; pruning past ~4× tends to break instruction-following.

2. **Targeted structured pruning with dynamic batch loading (Sheared LLaMA)** — Prunes LLaMA2-7B to 1.3B/2.7B with only ~3% of from-scratch compute; the target shape is chosen explicitly. [arXiv:2310.06694](https://arxiv.org/abs/2310.06694). *Tradeoff:* best when your target shape matches a known-good small architecture; weaker than Minitron at preserving generative fidelity.

3. **On-policy / reverse-KL distillation (MiniLLM, DistiLLM-2)** — Student samples its own rollouts and learns from teacher scoring; reverse-KL avoids mode-averaging that hurts generation quality. [arXiv:2306.08543](https://arxiv.org/abs/2306.08543), [DistiLLM-2 arXiv:2503.07067](https://arxiv.org/abs/2503.07067). *Tradeoff:* more expensive per step than SFT distillation, but decisively better for generative tasks like rewriting/cleanup.

4. **Task-specific SFT distillation on teacher-labeled data** — Generate ~50k–500k (input, teacher-output) pairs for exactly your cleanup distribution and SFT a small base (Qwen3-0.6B, Gemma-3-270M, SmolLM2-360M). Microsoft's Phi/orca-style recipes and the Azure distillation pipeline formalize this. [Azure distillation guide](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/distillation-turning-smaller-models-into-high-performance-cost-effective-solutio/4355029). *Tradeoff:* narrow — the student collapses to the task distribution; no generalization headroom, but that is exactly what you want for a cleanup model.

5. **Quantization-Aware Training at 4-bit / sub-4-bit (EfficientQAT, ParetoQ)** — Apply after distillation: a 500M student at W4A8 is ~250 MB; ParetoQ shows 2–4 bit is on the Pareto frontier for quality/size. [EfficientQAT arXiv:2407.11062](https://arxiv.org/abs/2407.11062), [ParetoQ arXiv:2502.02631](https://arxiv.org/abs/2502.02631). *Tradeoff:* integrated GPU runtimes (DirectML, Vulkan llama.cpp) have uneven kernel support below 4-bit; W4 is the safe ship target.

6. **Seq-level / rationale distillation ("Distilling Step-by-Step")** — Teacher emits both outputs and short rationales; student trained multitask converges with 10–100× less data. [arXiv:2305.02301](https://arxiv.org/abs/2305.02301). *Tradeoff:* rationales add noise for a minimal-edit rewriter where the teacher has no real "reasoning" to transfer — marginal for pure cleanup.

7. **Speculative/self-distillation from a frozen teacher into a tiny draft model (DistillSpec, SD-style)** — Train a <500M draft that matches the 8B's rewrites exactly on accepted tokens. [DistillSpec arXiv:2310.08461](https://arxiv.org/abs/2310.08461). *Tradeoff:* only useful if you keep the 8B around at inference; does not meet your "no teacher at runtime" constraint unless you ship draft-only.

**Ship stack for your constraints:** Minitron-style prune of an 8B teacher → on-policy distillation on cleanup corpus → EfficientQAT W4. That's the 2025 consensus path to ≤500 MB on-disk, sub-second on iGPU.

Sources:
- [Minitron / Compact LLMs via Pruning and KD — arXiv:2408.11796](https://arxiv.org/abs/2408.11796)
- [Sheared LLaMA — arXiv:2310.06694](https://arxiv.org/abs/2310.06694)
- [MiniLLM — arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
- [DistiLLM-2 — arXiv:2503.07067](https://arxiv.org/abs/2503.07067)
- [EfficientQAT — arXiv:2407.11062](https://arxiv.org/abs/2407.11062)
- [ParetoQ — arXiv:2502.02631](https://arxiv.org/abs/2502.02631)
- [Distilling Step-by-Step — arXiv:2305.02301](https://arxiv.org/abs/2305.02301)
- [DistillSpec — arXiv:2310.08461](https://arxiv.org/abs/2310.08461)
- [Azure AI Foundry distillation guide](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/distillation-turning-smaller-models-into-high-performance-cost-effective-solutio/4355029)
