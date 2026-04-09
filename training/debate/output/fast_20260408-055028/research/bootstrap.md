# Round 0 bootstrap research — OK

## Question

In 300-500 words, list the top 5-7 distinct techniques from 2025-2026 for compressing an instruction-tuned LLM from ~8B parameters down to ~500M while preserving narrow task-specific capability (text cleanup / minimal-edit rewriting). For each technique give: name, one source link, and one sentence on the tradeoff. This is a smoke test — be fast, do not aim for exhaustive coverage.

## Answer

# Top 2025–2026 techniques for compressing an 8B instruction-tuned LLM toward ~500M

*Interpretation: techniques that actually shrink parameter/footprint count (not just runtime), usable on a narrow task like minimal-edit text cleanup.*

**1. Structured width + depth pruning → distillation (Minitron recipe)**
NVIDIA's published recipe for Llama-3.1-8B → 4B: score importance, prune hidden/MLP/attention dims and layers, then retrain with KD from the uncompressed teacher. Gets 2–3× speedup at near-teacher quality but stops well short of 16× shrink — going 8B→500M with this alone loses too much capacity.
[NVIDIA blog, 2024–25](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/) · [arXiv:2408.11796](https://arxiv.org/abs/2408.11796)

**2. On-policy / reverse-KL distillation into a small student (MiniLLM, MiniPLM)**
Train a pre-existing small base (e.g. Qwen3-0.6B, SmolLM2-360M) against the 8B teacher using reverse-KL on student-sampled sequences, optionally narrowed to your task corpus. Best fit for narrow tasks; tradeoff is you need a good small base and lots of teacher rollouts.
[arXiv:2306.08543](https://arxiv.org/abs/2306.08543) · [MiniPLM, ICLR 2025](https://github.com/thu-coai/MiniPLM)

**3. Task-specific sequence-level distillation (teacher-labeled SFT)**
Have the 8B teacher relabel a cleanup corpus, then SFT a sub-1B student on (noisy→cleaned) pairs. Simplest and most reliable path for minimal-edit rewriting; gives up generality entirely, which for cleanup is fine.
[Azure AI Foundry, 2025](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/distillation-turning-smaller-models-into-high-performance-cost-effective-solutio/4355029)

**4. Low-rank decomposition of weights (SVD-LLM / SVD-LLM v2)**
ICLR 2025 truncation-aware SVD with LoRA-style recovery fine-tuning. Stacks with pruning/quant but alone rarely exceeds ~2× compression without noticeable quality loss.
[arXiv:2403.07378 (ICLR 2025)](https://arxiv.org/abs/2403.07378)

**5. Aggressive weight-only quantization (AWQ / GPTQ / GGUF Q4/Q3)**
4-bit is essentially free quality-wise; 3-bit and 2-bit (QuIP#, AQLM) push an 8B into ~2–3 GB but still miss your 500 MB budget unless combined with pruning/distillation. Orthogonal to the above — apply last.
[Awesome-LLM-Compression index](https://github.com/HuangOwen/Awesome-LLM-Compression)

**6. Quantization-Aware Distillation (LLM-QAT / QuaRot-QAT, 2025)**
Train the student under simulated low-bit weights with teacher logits as supervision, so accuracy loss from int4/int3 is absorbed during distillation rather than bolted on after. Tradeoff: extra training complexity and needs a QAT-capable trainer.
[TensorRT Model Optimizer blog, 2025](https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/)

**7. Compression ordering as a pipeline: Prune → Distill → Quantize (P-KD-Q)**
2025 empirical result: this ordering dominates other sequences at equal budgets. For an 8B→500M cleanup model the realistic recipe is Minitron-style prune to ~1–1.5B, KD from the 8B teacher on a cleanup corpus, then AWQ/GGUF Q4 to land under 500 MB.
[Frontiers 2025 survey](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1518965/full) · [ACM CSUR 2025](https://dl.acm.org/doi/10.1145/3728636)
