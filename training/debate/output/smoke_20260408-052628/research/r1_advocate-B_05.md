# advocate-B — round 1 — question 5 — OK

## Question

What is the cost of on-policy distillation for cleanup tasks?

## Answer

**Interpretation:** "Cost" = GPU hours / FLOPs / data volume needed to train a small student via on-policy distillation (OPD), with a note on what transfers to a non-reasoning cleanup task like Chirp's.

## Headline numbers (Thinking Machines, Oct 2025)

On the closest public reference point — Qwen3-8B student distilled from Qwen3-32B teacher — OPD reached **74.4% AIME'24 in ~1,800 GPU-hour-equivalents vs. 17,920 GPU hours for RL to 67.6%** (≈**10×** less compute). vs. off-policy SFT, the blog reports a **9–30× FLOPs reduction**, because OPD needed **77K prompts (150 steps × 4 samples × 128 batch)** instead of the extrapolated **~2M prompts** SFT required to match. Reported FLOPs: teacher 8.4×10¹⁹, student 8.2×10¹⁹. ([thinkingmachines.ai](https://thinkingmachines.ai/blog/on-policy-distillation/))

## The cleanup-relevant data point

Buried in the same post is a **non-reasoning** experiment that maps almost directly onto a cleanup-style task: recovering instruction-following after domain mid-training. Qwen3-8B's IF-eval dropped **85% → 45%** after document training, and OPD brought it back to **83%** — essentially free recovery of surface-behavior fidelity, the exact regime a text-cleanup student lives in. No separate GPU-hour figure given, but it used the same recipe and was cheaper than the math run. ([thinkingmachines.ai](https://thinkingmachines.ai/blog/on-policy-distillation/))

## Other 2025 reference costs

- **Black-Box OPD / GAD** (arXiv [2511.10643](https://arxiv.org/abs/2511.10643), Ye & Dong, Nov 2025): distilling Qwen2.5-14B-Instruct from GPT-5-Chat took **~30 hours on 16×H100 ≈ 480 H100-hours** (~$1K at commodity rates). Target is a chat-assistant student, not reasoning — the nearest published analogue to a cleanup distill.
- **MiniLLM** (arXiv [2306.08543](https://arxiv.org/abs/2306.08543)) and **GKD** (arXiv [2306.13649](https://arxiv.org/abs/2306.13649)) — the original on-policy recipes; both report that on-policy sampling adds ~2–3× wallclock per step vs. SFT but converges in far fewer steps, net win on small students.
- **Tinker SDK** (Thinking Machines) open-sources the exact OPD loop used above, so the 1,800-GPU-hour recipe is reproducible.

## Bottom line for Chirp's 8B→small cleanup

Expect the cleanup distill to sit closer to the **GAD number (~500 H100-hours, ~$1K)** than the AIME run, because cleanup is a narrow non-reasoning behavior with short outputs and a small student (sub-1B). The dominant cost is **teacher forward passes on student-sampled rollouts**, not training steps — so teacher size (Ministral 8B vs. a 32B) is the single biggest knob. 77K–100K prompts × 4 samples is a realistic budget.

Sources:
- [On-Policy Distillation — Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Black-Box On-Policy Distillation of LLMs (arXiv 2511.10643)](https://arxiv.org/abs/2511.10643)
- [MiniLLM (arXiv 2306.08543)](https://arxiv.org/abs/2306.08543)
- [GKD: On-Policy Distillation from Self-Generated Mistakes (arXiv 2306.13649)](https://arxiv.org/abs/2306.13649)
- [Thinking Machines 30× cost-cut coverage — Maginative](https://www.maginative.com/article/thinking-machines-claims-30x-cost-cut-for-training-ai-models/)
