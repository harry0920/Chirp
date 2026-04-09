# Round 0 bootstrap research — OK

## Question

In 300-500 words, list the top 5-7 distinct techniques from 2025-2026 for compressing an instruction-tuned LLM from ~8B parameters down to ~500M while preserving narrow task-specific capability (text cleanup / minimal-edit rewriting). For each technique give: name, one source link, and one sentence on the tradeoff. This is a smoke test — be fast, do not aim for exhaustive coverage.

## Answer

**Note on interpretation:** 8B→500M is ~16× parameter reduction, so pure post-training pruning/quantization can't get there alone — the techniques below mostly combine, with distillation into a small student as the backbone.

1. **Pruning + distillation (Minitron recipe).** Width/depth prune a large teacher, then distill the teacher's logits/hidden states into the pruned student. NVIDIA's Llama-3.1-Minitron 4B used ~40× fewer tokens than training from scratch; Minitron-SSM (NeurIPS 2025) extends it to hybrids. Tradeoff: great at 2–4× shrink, but 16× ratios need either an already-small target arch or aggressive layer dropping that hurts quality. [arXiv 2408.11796](https://arxiv.org/abs/2408.11796)

2. **Sequence-level / on-policy distillation to a small dedicated student (GKD, MiniLLM-style).** Train a ~500M base (Qwen3-0.5B, SmolLM2-360M, Gemma-3-270M) on teacher-generated rewrites of cleanup inputs; on-policy sampling from the student closes exposure-bias gap. Tradeoff: narrow task generalization — fine for minimal-edit cleanup, terrible if the distribution shifts. [arXiv 2306.13649 (GKD)](https://arxiv.org/abs/2306.13649)

3. **Depth pruning / layer dropping (ShortGPT, LaCo, Qwen3 depth-pruned 6B).** Drop entire transformer blocks by block-influence score, then light distillation. Tradeoff: biggest wall-clock speedup per parameter removed, but accuracy falls off a cliff past ~30% layers; stack with #1 rather than use alone. [arXiv 2403.03853 (ShortGPT)](https://arxiv.org/abs/2403.03853)

4. **Low-rank / activation-aware decomposition (SliceGPT, SVD-LLM, ASVD).** Apply an orthogonal transform and slice rows/columns; no retraining needed. Tradeoff: cheapest (hours on one GPU), but plateaus around 20–30% compression before adapters are needed — a complement to distillation, not a path to 16×. [arXiv 2401.15024 (SliceGPT)](https://arxiv.org/abs/2401.15024)

5. **Extreme low-bit quantization (AWQ, GPTQ, QuaRot, BitNet b1.58, QAT int4/int2).** Push weights to 4/3/1.58-bit post-training or via quantization-aware finetuning. An 8B at 2-bit ≈ 2GB — still over budget, but 500M at int4 ≈ 250MB fits perfectly as the final step after distillation. Tradeoff: int4 is nearly free; sub-3-bit needs QAT or it breaks instruction following. [arXiv 2306.00978 (AWQ)](https://arxiv.org/abs/2306.00978), [arXiv 2402.17764 (BitNet 1.58)](https://arxiv.org/abs/2402.17764)

6. **Unstructured sparsity (SparseGPT, Wanda) + 2:4 sparse kernels.** One-shot magnitude/Hessian pruning to 50% sparsity. Tradeoff: needs hardware sparse kernels (Ampere+) to actually get latency wins — on an integrated laptop GPU it's mostly a memory saving, not a speedup. [arXiv 2306.11695 (Wanda)](https://arxiv.org/abs/2306.11695)

7. **LoRA-distillation / task adapters on a pretrained small base (DistillKit, TRL GKDTrainer).** Freeze a small pretrained model, train a LoRA purely on teacher outputs for the cleanup task, then merge. Tradeoff: fastest to iterate and matches your existing Ministral-8B→FLAN-T5 plan; quality ceiling is set by the student base, not the teacher. [arcee-ai/DistillKit](https://github.com/arcee-ai/DistillKit)

**Practical combo for your target:** pick a ≤500M instruction-tuned base → sequence-level distillation from Ministral-8B on your cleanup corpus (#2/#7) → int4 AWQ/QAT (#5). Pruning the 8B teacher first (#1) only helps if you need faster data generation.
