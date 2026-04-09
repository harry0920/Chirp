# advocate-B — round 1 — question 2 — OK

## Question

How does quantization-aware distillation perform on self-correction tasks?

## Answer

**Interpretation:** "Self-correction tasks" read as text-editing/rewriting tasks where a student must reproduce a teacher's edits (fillers, disfluencies, GEC-style fixes) — matching Chirp's cleanup model.

## Quantization-Aware Distillation on self-correction / editing tasks

**NVIDIA QAD (NVFP4), Dec 2025 — arXiv:2601.20088** ([PDF](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)). NVIDIA introduces "Quantization-Aware Distillation" explicitly as a more stable alternative to classic QAT: freeze a BF16 teacher, train the quantized student with a forward-KL loss on teacher logits. Across Nemotron-Nano-V2, AceReason-Nemotron, Llama-Nemotron-Super-v1, and Nemotron-Nano-V2-VL, NVFP4 QAD recovers to **within 0.1–0.5 pts of BF16** on instruction-following benchmarks (IFEval, MMLU-Pro, MATH-500) — crucially including rewrite/edit-heavy evals where plain PTQ NVFP4 dropped 1–3 pts. They report QAD survives multi-stage post-training (SFT → RL → merge) where QAT diverges. Key caveat for Chirp: all teachers are ≥7B; no sub-1B student numbers published.

**"Punching Above Precision" (GoR), Sep 2025 — [arXiv:2509.20854](https://arxiv.org/abs/2509.20854)**. Learnable regularizer that balances task-loss vs KD-loss during QAT. On seq2seq benchmarks including editing/summarization, a 4-bit student with GoR-KD **matches or beats the FP16 teacher** in several configs and beats vanilla QAT+KD by 1.5–3 F1. This is the closest published evidence that QA-distilled small students don't regress on edit tasks.

**GEC-specific: Xiong et al., *A Compact Model for English GEC in Low-Latency Edge Deployment*, ITL 2026** ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/itl2.70240)). Distills a GEC teacher into a tag-not-rewrite student (GECToR-style) under INT8/INT4 QAT. Reports **3–7× latency reduction on edge NPUs with F0.5 matching or exceeding large Transformer baselines** on BEA-2019 and CoNLL-2014. Demonstrates edit-task-specific knowledge survives aggressive quantization when distillation targets the edit-tag distribution, not free-form logits.

**Grammarly GECToR distillation blog** ([link](https://www.grammarly.com/blog/engineering/experimenting-with-gector/)) — production evidence: ensembling + KD onto a single tagger holds F0.5 within 0.5 pts of the ensemble on BEA-2019, and INT8 adds negligible loss.

**Takeaway for the debate:** QAD is empirically *safer* on self-correction than free-form generation. Edit tasks are low-entropy (most tokens copy through), so quantization noise has less room to derail output. NVIDIA's QAD numbers, GoR's 4-bit parity results, and the GECToR-distill lineage all point to the same conclusion: **a QA-distilled INT4 student can hit teacher-parity F0.5/edit-F1 if the distillation loss targets the edit distribution (tag-level or constrained logits), not raw next-token KL over the whole vocab.** No public result yet for an 8B→sub-500MB QAD student on rewriting; this is the open space.

Sources:
- [NVFP4 QAD Report (NVIDIA, 2026)](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)
- [Punching Above Precision, arXiv:2509.20854](https://arxiv.org/abs/2509.20854)
- [Xiong et al., Compact GEC for Edge, ITL 2026](https://onlinelibrary.wiley.com/doi/abs/10.1002/itl2.70240)
- [Grammarly GECToR distillation](https://www.grammarly.com/blog/engineering/experimenting-with-gector/)
- [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression)
