# advocate-B — round 1 — question 1 — OK

## Question

Is the distilled model's instruction-following fidelity comparable to Ministral 3 8B?

## Answer

**Interpretation:** "Instruction-following fidelity" on the single cleanup task (one fixed prompt + few-shot), not general IFEval-style open-ended instruction following.

**Short answer:** Yes, for a narrow, single-prompt task like text cleanup. No, for broad instruction following.

**Evidence for task-specific fidelity being preserved:**

- Arcee's DistillKit v0.1 paper reports that logit-based and hidden-state distillation both beat plain SFT on every benchmark tested when the student's training corpus matches the teacher's task distribution; SuperNova-Medius (Qwen2.5-14B distilled from Llama-3.1-405B via DistillKit) actually *surpasses* models much closer to its own size on IFEval and BBH, showing the ceiling isn't student capacity — it's data coverage ([DistillKit blog](https://www.arcee.ai/blog/distillkit-v0-1-by-arcee-ai), [GitHub](https://github.com/arcee-ai/DistillKit)).
- The ACL 2025 "Generalization vs Fidelity Paradox" paper (aclanthology.org/2025.findings-acl.923) measures student-to-teacher agreement directly and finds self-fidelity only ~2% below KD-teacher fidelity on instruction-following tasks when the distillation set covers the target distribution — and that dataset *coverage* matters more than *size*.
- Thinking Machines' on-policy distillation post (Nov 2025) shows that on-policy KD can match teacher behavior on narrow domains at a fraction of the student's parameter budget, specifically because the loss is computed on the student's own rollouts against teacher logits — the exact setup that matters for a deterministic rewrite task ([thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)).
- For text rewriting specifically: the arxiv.org/abs/2409.11282 document-understanding distillation work gets FLAN-T5-Large to 75–80% of ChatGPT-3.5 on narrow DocQA, and a 77M FLAN-T5-small variant stays "competitive" — this is the closest public analogue to your Ministral→T5-base plan.

**Caveats that matter for your ship target:**

1. FLAN-T5-base is encoder-decoder and was never instruction-tuned on chat-style system prompts. It will *not* match Ministral on held-out instructions it didn't see during distillation — expect zero robustness to prompt changes. If you later want to add a second task (e.g., summary), you must re-distill.
2. Fidelity is bounded by the teacher. Ministral 3 8B scores roughly in the mid-50s on IFEval strict-prompt (public Mistral release notes); the student won't exceed that, and narrow-task distillation typically holds ~95–98% of teacher accuracy on the trained distribution but drops sharply off-distribution.
3. The v3 benchmark (250-case corpus) is exactly the right gate: if the distilled T5-base matches Ministral's v2-fewshot-hard score on that corpus, "instruction-following fidelity" for your use case is solved by definition — the model doesn't need to follow instructions, it needs to produce the same output on the same inputs.

**Practical takeaway:** score the student against Ministral on v3 and stop worrying about IFEval; it's the wrong metric for a single-prompt rewriter.

Sources:
- [DistillKit v0.1 technical paper (Arcee)](https://www.arcee.ai/blog/distillkit-v0-1-by-arcee-ai)
- [DistillKit GitHub](https://github.com/arcee-ai/DistillKit)
- [On the Generalization vs Fidelity Paradox in KD (ACL 2025)](https://aclanthology.org/2025.findings-acl.923.pdf)
- [On-Policy Distillation — Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Leveraging Distillation for Document Understanding (arXiv 2409.11282)](https://arxiv.org/pdf/2409.11282)
- [KD/Dataset Distillation survey (arXiv 2504.14772)](https://arxiv.org/pdf/2504.14772)
