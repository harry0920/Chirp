# advocate-B — round 1 — question 1 — OK

## Question

What is the optimal compression order for 8B→500M cleanup models?

## Answer

**Interpretation:** "Compression order" = sequence of pruning (P), knowledge distillation (KD), and quantization (Q) applied to shrink an 8B → ~500M encoder-style cleanup model.

## Answer: **Prune → Distill → Quantize (P-KD-Q)**

This is the empirically dominant ordering in both the 2025 systematic study and NVIDIA's Minitron production pipeline.

**1. Systematic evidence.** Yesmin et al., *"A Systematic Study of Compression Ordering for Large Language Models"* (arXiv:2511.19495, Nov 2025), tested all 6 permutations of P/KD/Q on LLMs and found **P-KD-Q** gave the best compression-vs-quality frontier. Key mechanistic result: *"sequences featuring early quantization exhibited severe degradation due to irreversible quantization noise affecting downstream training."* Quantize last, always.

**2. Production evidence — Minitron.** NVIDIA's `Llama-3.1-Minitron-4B` (HF: `nvidia/Llama-3.1-Minitron-4B-Width-Base`) took Llama-3.1-8B → 4B via **width pruning then distillation**, using ~100B tokens (40× less than pretraining) and beating from-scratch training by +16% MMLU (arXiv:2408.11796, "LLM Pruning and Distillation in Practice: The Minitron Approach"). Two specifics that matter for your 8B→500M case:
  - **Width pruning beats depth pruning for targets <15B** — prune hidden/attention/MLP dims jointly, not whole layers.
  - **Teacher correction**: before pruning, fine-tune the 8B teacher on *your* distribution (the v3 benchmark corpus + upstream cleanup data). Skipping this causes distribution mismatch that poisons distillation. Given your Ministral 8B teacher was selected on the v3 benchmark, this step is essentially free — and required.

**3. Caveat for your 16× ratio.** Minitron published 2× ratios (8B→4B, 12B→8B). 8B→500M is 16×, outside the validated envelope. The Minitron paper and NVIDIA's TensorRT Model Optimizer blog both recommend **iterative** prune-distill (e.g., 8B→2B→500M) rather than one-shot, because single-step aggressive pruning blows past the recoverable-capacity threshold. Sheared-LLaMA (arXiv:2310.06694) corroborates this for sub-billion targets.

**4. Quantization last.** After a distilled ~500M FP16 checkpoint (~1 GB), apply INT4 weight-only (AWQ or GPTQ) to hit your ≤500 MB budget. `llama.cpp` Q4_K_M or `autoawq` are the standard endpoints; both are post-training and don't interfere with the distilled weights.

**Recommended pipeline for Chirp:**
`Ministral-8B (teacher-corrected on v3 corpus) → width-prune to ~2B → distill on v3 + synthetic cleanup pairs → width-prune to ~500M → distill again → AWQ INT4 → GGUF`. Validate each checkpoint on the 250-case v3 benchmark before advancing.

**Don't do:** QAT before pruning, depth pruning at this scale, or one-shot 8B→500M.

Sources:
- [A Systematic Study of Compression Ordering for LLMs (arXiv:2511.19495)](https://arxiv.org/html/2511.19495v1)
- [LLM Pruning and Distillation in Practice: The Minitron Approach (arXiv:2408.11796)](https://arxiv.org/abs/2408.11796)
- [How to Prune and Distill Llama-3.1 8B to Minitron 4B (NVIDIA blog)](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)
- [Pruning and Distilling LLMs Using TensorRT Model Optimizer (NVIDIA blog)](https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/)
- [NVIDIA/Model-Optimizer (GitHub)](https://github.com/NVIDIA/Model-Optimizer)
