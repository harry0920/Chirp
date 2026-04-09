# advocate-B — round 1 — question 3 — OK

## Question

What is the impact of pruning on the teacher's generative fidelity?

## Answer

**Interpretation:** The question reads two ways — (a) does pruning the teacher degrade its own generative quality before distillation, and (b) how does that cascade into a distilled student. I'll cover both, since both matter for the 8B→laptop path.

**Direct pruning damage (pre-recovery).** One-shot structured pruning without retraining produces sharp perplexity spikes in generative LLMs. LLM-Pruner (Ma et al., arXiv:2305.11627) reports WikiText-2 PPL roughly doubling on LLaMA-7B at 20% width pruning before any recovery, and collapsing at 50%. SoLA (arXiv 2025) and SlimLLM (arXiv:2505.22689) both benchmark against this baseline and show LLM-Pruner-style methods "sharply decline as compression ratio increases" on open-ended generation metrics, while their own methods hold PPL flatter. SDMPrune (arXiv:2506.11120) makes the generative-fidelity point explicitly: one-hot importance scoring "ignores the potential predictions of other words, thus missing vital information for the preservation of the original generative capability" — and folds a self-distillation loss *into the pruning step* to fix it.

**Recovery closes most, not all, of the gap.** NVIDIA's Minitron work (arXiv:2408.11796; Llama-3.1-Minitron-4B, Mistral-NeMo-Minitron-8B) is the load-bearing evidence here. Pruning Llama-3.1-8B → 4B and then distilling from the *unpruned* 8B as teacher recovers quality with ~94B tokens (vs. 15T from scratch), and the pruned+distilled 4B beats from-scratch 4Bs. Critically, they recommend lightly fine-tuning the teacher on the distillation corpus *before* distilling ("teacher correction") to counter distribution shift — they do **not** prune the teacher. Width pruning consistently beats depth pruning for generation fidelity. Mistral-NeMo-Minitron-8B (pruned from 12B) actually exceeds the 12B parent on GSM8k and HumanEval after distillation — suggesting the pruned network, once retrained through KL on teacher logits, can match or exceed the original on targeted generative tasks even though the raw pruned checkpoint was degraded.

**Implication for your 8B→laptop plan.** Don't prune Ministral-8B and use the pruned artifact as the teacher — its logit distribution is corrupted exactly where generative nuance lives. Either (1) distill from the unpruned 8B into a smaller architecture (your FLAN-T5-base path), or (2) prune-then-distill *and* use the original unpruned 8B as teacher during recovery, Minitron-style.

Sources:
- [LLM-Pruner (arXiv:2305.11627)](https://arxiv.org/pdf/2305.11627)
- [Minitron Approach (arXiv:2408.11796)](https://arxiv.org/abs/2408.11796)
- [NVIDIA: Prune & Distill Llama-3.1-8B → Minitron-4B](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)
- [Mistral-NeMo-Minitron-8B blog](https://developer.nvidia.com/blog/mistral-nemo-minitron-8b-foundation-model-delivers-unparalleled-accuracy/)
- [SDMPrune (arXiv:2506.11120)](https://arxiv.org/html/2506.11120v1)
- [SlimLLM (arXiv:2505.22689)](https://arxiv.org/abs/2505.22689)
- [NVlabs/Minitron GitHub](https://github.com/NVlabs/Minitron)
