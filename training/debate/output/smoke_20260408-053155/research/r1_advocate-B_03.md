# advocate-B — round 1 — question 3 — OK

## Question

Can pruning be applied to the 8B teacher without significant quality loss?

## Answer

**Interpretation:** "Applied to the teacher" reads two ways — (a) shrink the 8B to a smaller *pruned* checkpoint you then distill from, or (b) prune the 8B en route to the ≤500 MB target. Both are answered below; the published evidence is strongest for pruning-then-distillation as a single pipeline.

**Yes — but only when paired with distillation retraining. Pruning alone loses quality; pruning + short distill recovers it.**

The canonical result is NVIDIA's **Llama-3.1-Minitron 4B** (arXiv:2408.11796, "LLM Pruning and Distillation in Practice"). They structurally pruned Llama-3.1-8B to 4B via joint width pruning (hidden/attention/MLP) and recovered with knowledge distillation on **94B–380B tokens**. Width-pruned 4B hits **MMLU 60.5** vs. the 8B teacher's ~65, using **40× fewer tokens than training from scratch**, and beats Phi-2 2.7B, Gemma2 2.6B, and Qwen2-1.5B at similar size. Depth pruning was worse (58.7 MMLU) — width > depth at this scale. A "teacher correction" step (light FT of the 8B on your target distribution *before* distillation) is reported as critical for closing the gap.

**Princeton's Sheared LLaMA** (arXiv:2310.06694, ICLR 2024, `princeton-nlp/LLM-Shearing`) shows the same pattern at 1.3B/2.7B targets: targeted structured pruning + dynamic batch loading recovers SOTA quality at **3% of from-scratch compute**. Models on HF: `princeton-nlp/Sheared-LLaMA-1.3B`, `-2.7B`.

**Caveats specific to your constraint (≤500 MB, narrow cleanup task):**
1. Even the width-pruned Minitron-4B is ~8 GB in FP16 / ~2 GB at INT4 — **pruning alone won't hit 500 MB from 8B**. You need pruning→distill→quantize, or distill into a much smaller student (T5-small-class) directly. The published pruning work stops at 4B because below that, distillation-into-a-small-student outperforms prune-and-recover (see Minitron paper §5).
2. The 94B-token recovery budget is for general-capability retention. For a **single-task cleanup model** the recovery corpus can be much smaller — there's no public number, but DistillKit/MiniLLM-style task distillation typically uses 1–10M task examples.
3. Unstructured/semi-structured pruning (Wanda, SparseGPT, arXiv:2306.11695 / 2301.00774) can reach 50% sparsity at near-zero quality loss but **doesn't shrink the file** unless your runtime supports 2:4 sparse kernels — integrated GPUs generally don't, so it's irrelevant for your deploy target.

**Bottom line for your debate:** Pruning the 8B *as a preprocessing step* before distillation into a small student is not the standard recipe — the literature distills directly from the full 8B. Pruning is the right tool when the target is 2–4B, not 500 MB. For your size budget, skip pruning the teacher and spend the compute on distillation + aggressive quantization (INT4/INT3) of a smaller student.

**Sources:**
- [LLM Pruning and Distillation in Practice: The Minitron Approach (arXiv:2408.11796)](https://arxiv.org/html/2408.11796v4)
- [How to Prune and Distill Llama-3.1 8B to Llama-3.1-Minitron 4B (NVIDIA blog)](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)
- [nvidia/Llama-3.1-Minitron-4B-Width-Base (HuggingFace)](https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base)
- [Compact Language Models via Pruning and Knowledge Distillation (NeurIPS 2024)](https://papers.nips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
- [Sheared LLaMA (arXiv:2310.06694)](https://arxiv.org/abs/2310.06694)
- [princeton-nlp/LLM-Shearing (GitHub)](https://github.com/princeton-nlp/LLM-Shearing)
- [NVlabs/Minitron (GitHub)](https://github.com/NVlabs/Minitron)
