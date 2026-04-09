# advocate-B — round 1 — question 4 — OK

## Question

Can the 1.5B T5 backbone support the required cleanup task without sacrificing quality?

## Answer

**Interpretation:** No standard T5 checkpoint is exactly 1.5B; I read this as "the T5-Large-to-XL bracket" (FLAN-T5-Large 783M ↔ FLAN-T5-XL 2.92B), which spans your 1.5B target. Also relevant: Google's 2025 **T5Gemma 2B** encoder-decoder.

**Short answer:** Yes, quality-wise an encoder-decoder in this size class is well-proven for ASR cleanup/GEC. The binding constraint is your 500 MB / 1 s-p95 budget, not capability.

**Quality evidence (2025):**

- **FLANEC** (arXiv 2501.12979) fine-tunes FLAN-T5 on HyPoradise for ASR n-best correction. FLAN-T5-Large (783M) hits **9.0% avg WER**; FLAN-T5-XL (2.92B) hits **8.5%**. The paper explicitly concludes the 783M model is "the best tradeoff between performance and computational complexity" — the 3B model gains only ~0.5 WER points. A 1.5B interpolation would sit near the knee of the curve with no meaningful quality loss vs XL for your task. ([FLANEC](https://arxiv.org/html/2501.12979v1))
- **pszemraj/flan-t5-xl-grammar-synthesis** (3B, Apache-2.0) and the **flan-t5-large** sibling are the de-facto JFLEG-tuned GEC baselines; the XL card itself warns it's slow and recommends the Large variant for interactive use. ([XL card](https://huggingface.co/pszemraj/flan-t5-xl-grammar-synthesis), [Large card](https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis))
- **ACL BEA 2025** ("Adapting LLMs for Minimal-edit GEC") reports FLAN-T5-XXL (11B) at 75.0/78.8 F0.5 on standard GEC sets — showing the T5 family scales cleanly, and the sub-3B tier is already above the "no quality regression" bar relative to Ministral 8B on cleanup-shaped tasks. ([BEA 2025 PDF](https://aclanthology.org/2025.bea-1.9.pdf))
- **T5Gemma** (Google, July 2025) provides modern encoder-decoder checkpoints including a ~2B tier distilled from Gemma 2, Apache-2.0, which is the strongest current backbone in this size range if you don't want vanilla FLAN-T5. (HF: `google/t5gemma-2b-2b-prefixlm`)

**The real problem is the budget, not the backbone.** A 1.5B-class T5 in fp16 is ~3 GB; even int4/AWQ lands around **800–1000 MB**, blowing the 500 MB cap. Encoder-decoder int4 tooling is also thinner than decoder-only — sherpa-onnx/llama.cpp T5 int4 paths exist but aren't as battle-tested as Qwen/Llama. For 500 MB you realistically need ≤770M params at int4/int8, i.e. **FLAN-T5-Large or T5Gemma-small-tier**, not 1.5B. Your existing distillation plan (Ministral 8B → FLAN-T5-base) is pointed at the right size class; jumping the student to 1.5B would regress on size/latency without buying measurable quality over Large on a narrow cleanup task.

**Sources:**
- [FLANEC: Exploring Flan-T5 for Post-ASR Error Correction (arXiv 2501.12979)](https://arxiv.org/html/2501.12979v1)
- [Adapting LLMs for Minimal-edit GEC, BEA 2025](https://aclanthology.org/2025.bea-1.9.pdf)
- [pszemraj/flan-t5-xl-grammar-synthesis](https://huggingface.co/pszemraj/flan-t5-xl-grammar-synthesis)
- [pszemraj/flan-t5-large-grammar-synthesis](https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis)
