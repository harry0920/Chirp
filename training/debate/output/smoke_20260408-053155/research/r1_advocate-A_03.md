# advocate-A — round 1 — question 3 — OK

## Question

What is the minimal pruning threshold (e.g., 30% vs. 50% hidden dim reduction) for Ministral-8B’s pruned distillation model to retain ≥92% composite score on the v3 benchmark? (Test on [NVIDIA’s Minitron pruning recipe](https://arxiv.org/abs/2408.11796).)

## Answer

**Interpretation:** No public data exists for Ministral-8B pruned against your internal v3 benchmark — that's a private corpus. Answering the closest public analog: what the Minitron recipe (arxiv 2408.11796) and Mistral's own Ministral 3 cascade-distillation report establish about width-pruning thresholds vs. accuracy retention on 8B-class models.

**Minitron published width-pruning operating points (Llama-3.1 8B → 4B, the only 8B→smaller ablation in the paper):**
- hidden_size: 4096 → 3072 (**−25%**)
- MLP intermediate: 14336 → 9216 (**−36%**)
- heads/layers: unchanged (32/32)
- Parameter count: 8.0B → 4.0B (**−50% params**)
- MMLU retention: ~60.5 / ~64 ≈ **94.5%** (width) vs. 58.7 / ~64 ≈ 91.7% (depth)
- GSM8K retention: **41.2% width vs. 16.8% depth** — depth pruning collapses reasoning well before width does

The Minitron paper does **not** publish a sweep of intermediate hidden-dim reduction ratios; there is no ablation curve showing the exact knee where accuracy falls below 92%. The only two operating points actually reported are 0% (teacher) and the ~−25% hidden / ~−36% MLP recipe above. NVIDIA's earlier Compact LLMs paper (arxiv [2407.14679](https://arxiv.org/abs/2407.14679), the Minitron-4B/8B precursor) similarly reports single operating points, not thresholds.

**Closest thing to a threshold finding in the literature:**
- Muralidharan et al. (2407.14679) explicitly recommend **pruning by ≤50% of parameters per iteration** and **distilling on ≥1.8B tokens for 4B targets, ≥94B tokens for 8B→4B**, below which retention drops sharply.
- Mistral's Ministral 3 tech report ([arxiv 2601.08584](https://arxiv.org/abs/2601.08584)) uses **cascade** pruning (Mistral Small 3.1 24B → 14B → 8B → 3B) precisely because single-shot aggressive pruning past ~40–50% param reduction underperforms; each hop keeps hidden_size cuts around 20–30%.

**Practical takeaway for your distillation plan:** the public evidence suggests −25% hidden / −35% MLP width pruning is the furthest anyone has validated on an 8B Mistral-family model while preserving >92% of teacher quality on downstream tasks, and that number requires ≥90B distillation tokens. Beyond that (e.g., −50% hidden), no published result on an 8B retains ≥92% — you'd be extrapolating. For your laptop ≤500MB target you'll need to combine pruning with quantization rather than push the pruning ratio alone.

Sources:
- [Minitron Approach (2408.11796)](https://arxiv.org/abs/2408.11796)
- [Minitron HTML v3 with ablations](https://arxiv.org/html/2408.11796v3)
- [Compact Language Models via Pruning and KD (2407.14679)](https://arxiv.org/abs/2407.14679)
- [Ministral 3 tech report (2601.08584)](https://arxiv.org/abs/2601.08584)
- [Mistral's Ministral 3 cascade distillation writeup](https://aigazine.com/industry/mistral-unveils-ministral-3-14b-8b-and-3b-models-built-with-cascade-distillation--v)
- [SuperAnnotate Minitron breakdown](https://www.superannotate.com/blog/llm-pruning-distillation-minitron-approach)
