# advocate-B — round 1 — question 2 — OK

## Question

Can the sequence-level KD on cleanup data effectively preserve the 8B model's cleanup quality?

## Answer

**Short answer:** Yes for narrow, well-scoped text-rewrite tasks like cleanup — but only with known mitigations for SeqKD's two failure modes (exposure bias / hallucination drift and extractive memorization). Vanilla SeqKD alone will leave quality on the table; augmented variants close most of the gap.

**What the evidence says**

- **SeqKD baseline (Kim & Rush, arXiv:1606.07947)** established that training a student on teacher beam-search outputs recovers most of the BLEU gap on NMT, and for narrow seq2seq tasks — which cleanup essentially is (monotonic, high input-output overlap, bounded edit distance) — students often match teachers within ~1 BLEU. Cleanup is *easier* than NMT because the output is near-copy of the input; SeqKD's weakness (mode collapse on diverse outputs) barely applies.
- **Failure modes are real and measured.** "Don't Throw Away Data: Better Sequence Knowledge Distillation" (arXiv:2407.10456, ICLR 2025) shows standard SeqKD discards the gold target and hurts students; mixing teacher-generated and gold sequences recovers quality. A 2025 NMT study found SeqKD students show **+57% extractive memorization and +31% oscillatory hallucination** vs. baselines — but a light post-finetune on clean data cuts both by ~33% with negligible loss.
- **On-policy / GKD beats vanilla SeqKD.** Agarwal et al., "On-Policy Distillation of Language Models" (arXiv:2306.13649) and DistiLLM (arXiv:2402.03898) report **1.7×–2.1× relative gains over SeqKD** on T5 students across summarization and MT — because sampling from the student's own distribution fixes the train/test distribution mismatch that causes exposure-bias drift. Thinking Machines' 2025 "On-Policy Distillation" post reinforces this as current best practice.
- **T5 as a cleanup student is viable.** Your own v3 benchmark already showed 77M FLAN-T5-small at 0.78 is training-data limited, not architecture limited (MEMORY: project_t5_finetuning.md). A FLAN-T5-base (~250M) student distilled from Ministral 8B on cleanup-specific data, using **SeqKD + gold interpolation + a short on-policy fine-tune pass**, is the configuration with the most public evidence of preserving teacher quality on narrow rewrite tasks.

**Caveat:** No public paper specifically benchmarks SeqKD on *dictation cleanup*. Closest analogs are GEC and style transfer, where distilled T5 students routinely land within 1–2 F0.5 of teachers. You will need your own v3 benchmark to confirm — don't assume the NMT numbers transfer exactly.

**Sources:**
- [Sequence-Level Knowledge Distillation (Kim & Rush)](https://arxiv.org/abs/1606.07947)
- [Don't Throw Away Data: Better Sequence Knowledge Distillation (arXiv:2407.10456)](https://arxiv.org/html/2407.10456)
- [On-Policy Distillation of Language Models (arXiv:2306.13649)](https://arxiv.org/pdf/2306.13649)
- [DistiLLM (arXiv:2402.03898)](https://arxiv.org/abs/2402.03898)
- [On-Policy Distillation — Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Mitigating exposure bias in LLM distillation (Springer 2025)](https://link.springer.com/article/10.1007/s00521-025-11162-0)
