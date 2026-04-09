# advocate-B — round 1 — question 5 — OK

## Question

How does the choice of base model affect the final model's performance on self-correction tasks?

## Answer

**Interpretation:** "Self-correction" here = ASR cleanup tasks where the model must undo filler words, false starts, and speaker restarts (not multi-turn reasoning self-correction). Evidence on how encoder-decoder vs decoder-only vs pretraining mix affects the student's ceiling.

**1. Encoder-decoder T5 family still leads on edit-style rewriting.** The 2024 "Pillars of GEC" survey (arxiv 2404.14914) reports T5-11B and UL2-20B hitting F0.5 = 72.8 / 81.4 on CoNLL-14 / BEA, still the highest non-ensemble numbers. T5-base fine-tuned for minimal-edit GEC reaches 65.5 / 71.3 F0.5 (arxiv 2506.13148). The Unbabel `gec-t5_small` (77M) hits 60.7 F0.5 on CoNLL — the same order as your re-benchmarked T5-small cleanup at 0.78. The span-preserving pretraining objective transfers directly to "copy most, edit a few spans."

**2. Decoder-only bases underperform at small scale, catch up above ~4B.** Arxiv 2601.03874 ("Evaluating Small Decoder-Only LMs for Grammar Correction and Simplification") fine-tunes Qwen2.5-0.5B/1.5B, Llama-3.2-1B, SmolLM2 on GEC and finds they trail T5-small/base at equal params, because causal LMs waste capacity on left-context prediction the task doesn't need. The gap closes at 4B+: a 2025 distillation run used Qwen3-235B → Qwen3-4B and matched Qwen3-14B on grammar correction (Nebius blog). **Implication for your 500MB budget: decoder-only is the wrong base; T5/ByT5/BART is.**

**3. Instruction-tuning of the base matters less than pretraining objective.** FLANEC (arxiv 2501.12979) fine-tunes Flan-T5 sizes for post-ASR error correction on HyPoradise; Flan-T5-large beats Flan-T5-base by ~15% WERR but the jump from T5→Flan-T5 at fixed size is small. One practitioner report (Medium, Sultanov) found vanilla T5 slightly beat Flan-T5 after full fine-tuning — instruction priors get overwritten.

**4. Self-correction-specific distillation.** SuperCorrect (ICLR 2025, arxiv 2410.09008) and SCD (arxiv 2511.07998) show that when the student base has weak error-detection priors, you must distill *error templates*, not just outputs — otherwise the student memorizes corrections without learning to localize edits. Relevant if your Ministral→T5 distillation data is output-only.

**Bottom line for the debate:** pick the base by pretraining objective, not by parameter count or "modernness." For span-edit cleanup at ≤500MB, T5-small/base (or ByT5-small for robustness to ASR artifacts) will beat any sub-2B decoder-only base, and the Ministral-8B teacher's job is to generate edit-rich training data, not to donate its architecture.

Sources:
- [Pillars of GEC survey (arxiv 2404.14914)](https://arxiv.org/html/2404.14914v1)
- [Adapting LLMs for Minimal-edit GEC (arxiv 2506.13148)](https://arxiv.org/html/2506.13148v1)
- [Evaluating Small Decoder-Only LMs for GEC (arxiv 2601.03874)](https://arxiv.org/html/2601.03874)
- [FLANEC: Flan-T5 for Post-ASR Error Correction (arxiv 2501.12979)](https://arxiv.org/html/2501.12979)
- [SuperCorrect, ICLR 2025 (arxiv 2410.09008)](https://arxiv.org/pdf/2410.09008)
- [Self-Correction Distillation (arxiv 2511.07998)](https://arxiv.org/abs/2511.07998)
- [Unbabel/gec-t5_small model card](https://huggingface.co/Unbabel/gec-t5_small)
- [pszemraj/flan-t5-large-grammar-synthesis](https://huggingface.co/pszemraj/flan-t5-large-grammar-synthesis)
- [Nebius: distillation intro (Qwen3-235B→4B GEC case)](https://nebius.com/blog/posts/model-distillation-intro)
- [FlanT5 from scratch for grammar correction (Sultanov)](https://medium.com/@akhmat-s/flant5-from-scratch-for-the-grammar-correction-tool-deadba9a6778)
