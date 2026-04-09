# Round 0 bootstrap research — OK

## Question

In 300-500 words, list the top 5-7 distinct techniques from 2025-2026 for compressing an instruction-tuned LLM from ~8B parameters down to ~500M while preserving narrow task-specific capability (text cleanup / minimal-edit rewriting). For each technique give: name, one source link, and one sentence on the tradeoff. This is a smoke test — be fast, do not aim for exhaustive coverage.

## Answer

**Interpretation:** Techniques that could plausibly take an 8B instruction-tuned model to ~500M (16×) while keeping narrow cleanup-rewrite quality. Ordered rough compression ratio first, then ease of use.

1. **Width + depth pruning with KD retrain (Minitron recipe)** — NVIDIA's joint hidden/MLP/attention pruning followed by logit+feature distillation from the unpruned parent. Demonstrated 15B→8B→4B with ~40× fewer tokens than training from scratch. [arxiv 2407.14679](https://arxiv.org/abs/2407.14679), [arxiv 2408.11796](https://arxiv.org/abs/2408.11796). Tradeoff: still needs billions of distillation tokens and a working training rig; width pruning > depth pruning on accuracy, depth > width on latency.

2. **Targeted structured pruning + dynamic batch loading (Sheared-LLaMA lineage)** — end-to-end pruning to a target shape (layers, heads, dims) with a loss-balancing data sampler; LLaMA2-7B→1.3B/2.7B at ~3% of from-scratch compute. [arxiv 2310.06694](https://arxiv.org/abs/2310.06694). Tradeoff: architecture-aware, 16× ratio is aggressive — closer to the bottom of what's been shown to work.

3. **Sequence-level / on-policy KD on task data (DistillM, DistiLLM-2)** — student generates, teacher scores; better than token-level KL for instruction-following students. [arxiv 2402.03898](https://arxiv.org/abs/2402.03898). Tradeoff: cheapest and most task-aligned path for a narrow task like cleanup, but student architecture must already be capable at 500M (pick a strong small backbone).

4. **Two-stage curriculum KD (openPangu-Embedded-1B-KD, 2025)** — Stage 1 reasoning-CoT SFT from teacher, Stage 2 standard prompt→response; SOTA among ~1B instruct models. [arxiv 2509.26497](https://arxiv.org/abs/2509.26497). Tradeoff: recipe-heavy, designed for 1B not 500M, but the curriculum idea transfers.

5. **Distilling Step-by-Step (rationale distillation)** — teacher emits rationales, small student trained multi-task on (label, rationale); smaller student beats bigger teacher with far less data on narrow tasks. [arxiv 2305.02301](https://arxiv.org/abs/2305.02301). Tradeoff: ideal for deterministic tasks like minimal-edit cleanup; rationale quality caps student quality.

6. **Sub-4-bit PTQ (AWQ/GPTQ + 2025 successors: BTC-LLM, ROSAQ, BAQ)** — weight-only quantization to 3–2 bits with saliency/rotation tricks. Surveyed in [arxiv 2511.19495](https://arxiv.org/abs/2511.19495) (compression ordering study). Tradeoff: largest standalone compression, cheapest, but below ~3 bits instruction-following degrades; stack *on top of* pruning/KD rather than alone.

7. **One-shot unstructured pruning (Wanda / SparseGPT)** — calibration-only, no retraining, 50% sparsity roughly free on 7B+. [arxiv 2306.11695 (Wanda)](https://arxiv.org/abs/2306.11695). Tradeoff: gets you ~2× not 16×, and integrated-GPU runtimes rarely accelerate unstructured sparsity — useful as a pre-step for pruning+KD, not as a ship format.

**Practical stack for your constraints:** pick a strong ~500M backbone (or Minitron/Sheared a 1.5–3B candidate), sequence-level KD from your 8B teacher on cleanup data, then 4-bit AWQ at the end. Pure PTQ of an 8B will not hit 500 MB without crossing the sub-3-bit cliff.

Sources:
- [Compact Language Models via Pruning and KD (2407.14679)](https://arxiv.org/abs/2407.14679)
- [Minitron in Practice (2408.11796)](https://arxiv.org/abs/2408.11796)
- [Sheared LLaMA (2310.06694)](https://arxiv.org/abs/2310.06694)
- [DistiLLM (2402.03898)](https://arxiv.org/abs/2402.03898)
- [openPangu-Embedded-1B-KD (2509.26497)](https://arxiv.org/abs/2509.26497)
- [Distilling Step-by-Step (2305.02301)](https://arxiv.org/abs/2305.02301)
- [Compression Ordering Study (2511.19495)](https://arxiv.org/abs/2511.19495)
- [Wanda (2306.11695)](https://arxiv.org/abs/2306.11695)
- [Awesome-LLM-Compression (index)](https://github.com/HuangOwen/Awesome-LLM-Compression)
