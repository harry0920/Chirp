# advocate-A — round 1 — question 1 — OK

## Question

Does sequence-level distillation with reverse-KL + MSE on cleaned outputs outperform logit-only KD by ≥3% on the v3 benchmark’s changes-meaning gate when applied to a pruned 8B teacher (30–40% capacity)? (Compare to [MiniPLM’s task-specific alignment](https://github.com/thu-coai/MiniPLM).)

## Answer

**Interpretation:** No public paper runs this exact combo against a "changes-meaning" gate on your internal v3 benchmark, so I answer what the literature *does* support for each component and what transfers.

**Direct answer: no public evidence of a ≥3% gain on your specific gate exists.** The v3 semantic-faithfulness gate is internal, and no paper reports on it. What the literature supports is weaker: reverse-KL-family + hidden-state MSE beats logit-only KD by roughly **1–3 points** on instruction-following/summarization benchmarks, with the delta shrinking as student capacity grows — which is exactly your pruned-8B regime.

**Component evidence (2024–2026):**

- **Reverse/Skew KL vs forward KL.** DistiLLM (ICLR 2024, [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)) shows Skew-RKL beats forward KL on 3/5 instruction benchmarks, avg ~+1.5 ROUGE-L. DistiLLM-2 ([arXiv:2503.07067](https://arxiv.org/pdf/2503.07067)) argues the loss must match the data source: SKL for teacher-generated, SRKL for student-generated — a direct recommendation for seq-level KD on *cleaned teacher outputs* like yours. Diversity-Aware RKL ([arXiv:2604.00223](https://arxiv.org/html/2604.00223)) patches RKL's mode-collapse, relevant because semantic-drift errors are a mode-collapse symptom.
- **Hidden-state MSE on pruned teachers.** NVIDIA's Minitron ([arXiv:2407.14679](https://arxiv.org/pdf/2407.14679), [2408.11796](https://arxiv.org/html/2408.11796)) prunes Llama-3.1 8B → 4B and recovers with logit + intermediate-state KD; the L2/MSE hidden-state term "consistently leads to the largest performance gains" and is the single most important recovery ingredient at 30–50% pruning. This is the strongest signal for your 30–40% pruning target.
- **Post-pruning recovery comparison.** [arXiv:2507.20749](https://arxiv.org/pdf/2507.20749) and [arXiv:2511.20702](https://arxiv.org/html/2511.20702) explicitly compare hidden-state KD vs logit-only KD on pruned LLMs and report hidden-state KD recovers >95% of original quality with ~5% of training data, while logit-only plateaus lower. Reported gaps are ~2–4 points on MMLU-class benchmarks — borderline for your 3% gate.
- **Seq-KD on cleaned teacher outputs.** This is the oldest piece (Kim & Rush 2016) and still the base layer; DistiLLM-2 shows stacking seq-KD + SRKL is additive, not substitutive.

**MiniPLM comparison.** MiniPLM ([arXiv:2410.17215](https://arxiv.org/abs/2410.17215), [thu-coai/MiniPLM](https://github.com/thu-coai/MiniPLM)) is *not* a loss-function method — it's offline Difference Sampling that reshapes the *pre-training corpus* using teacher-vs-small-ref logprob gaps. It's orthogonal to seq-KD+RKL+MSE and targets pre-training, not task distillation from a pruned teacher. For a 250-case cleanup task it's the wrong tool; its "task-specific alignment" framing in the README refers to data selection, not loss design. Cite it as the data-curation axis, not a competitor to your loss stack.

**Practical read for your debate:** the ≥3% claim is plausible but *not* guaranteed — Minitron and the 2025 recovery papers show hidden-state MSE is the biggest lever, RKL adds ~1 point, and seq-KD on cleaned outputs adds another ~0.5–1. Whether that clears a hard 3% semantic-drift gate depends entirely on how tight your gate is; the literature does not promise it.

Sources:
- [DistiLLM (arXiv:2402.03898)](https://arxiv.org/html/2402.03898)
- [DistiLLM-2 (arXiv:2503.07067)](https://arxiv.org/html/2503.07067v1)
- [Diversity-Aware RKL (arXiv:2604.00223)](https://arxiv.org/html/2604.00223)
- [Minitron pruning+KD (arXiv:2407.14679)](https://arxiv.org/pdf/2407.14679)
- [Minitron in Practice (arXiv:2408.11796)](https://arxiv.org/html/2408.11796)
- [Structural Pruning Recovery (arXiv:2507.20749)](https://arxiv.org/pdf/2507.20749)
- [Post-Pruning Data-Free KD (arXiv:2511.20702)](https://arxiv.org/html/2511.20702)
- [MiniPLM paper (arXiv:2410.17215)](https://arxiv.org/abs/2410.17215)
- [thu-coai/MiniPLM repo](https://github.com/thu-coai/MiniPLM)
- [Self-Data Distillation for Pruned LLMs (arXiv:2410.09982)](https://arxiv.org/html/2410.09982v2)
