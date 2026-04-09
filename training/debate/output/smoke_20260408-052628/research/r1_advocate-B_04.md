# advocate-B — round 1 — question 4 — OK

## Question

Can the distilled model pass the changes-meaning gate on self-correction inputs?

## Answer

**Interpretation:** "Changes-meaning gate" = rejecting outputs that over-delete or rewrite fluent tokens on self-correction inputs (e.g., "send it to John — I mean Jane" → must become "send it to Jane", not "send it to John"). I read the question as: is there public evidence that a distilled sub-1B model can hit that gate without regression?

**Short answer:** Partially. Disfluency-specific distilled/small models clear the precision bar on Switchboard-style self-corrections, but no public 2025 benchmark shows a distilled <1B *instruction* model passing a meaning-preservation gate on self-correction inputs without over-deletion or generalization loss.

**Evidence:**

1. **DRES (ArXiv 2509.20321, Sept 2025)** is the current canonical benchmark for this exact question — word-level precision (E_P), recall (E_R), F1 on Switchboard self-corrections, explicitly designed as a "semantic upper bound." Key findings for your debate:
   - **Llama-3-8B and o4-mini over-delete** (high recall, low precision) — i.e., they fail the changes-meaning gate by stripping fluent content along with the reparandum. Scaling alone does not fix it; gains are "non-linear and not solely dependent on parameter count."
   - **Fine-tuning works for the gate but breaks generalization**: `gpt-4o-mini_ft` hits E_P=96.6, but regresses on GSM8K/MMLU/CoQA. Directly relevant to a distilled ship target: you can pass DRES or you can stay general, not both, with naive SFT.
   - Small models (≤1B) cluster in the "Over-Deletion or Poor" regimes in DRES's 2D error map.

2. **Rocholl & Zayats, "Disfluency Detection with Unlabeled Data and Small BERT Models" (ArXiv 2104.10769)** — a distilled small BERT reaches F1 90.4 on Switchboard disfluency detection, vs 92.3 for BERT-base. This is the existence proof that *task-specialized* sub-100M distillation can hold self-correction precision — but it's a tagger, not a rewriter, so there's no "changes meaning" risk to begin with.

3. **VocalBench-DF (ArXiv 2510.15406, Oct 2025)** evaluated 22 Speech-LLMs and found "substantial performance degradation" across disfluency types, with long-context modeling as a primary bottleneck — another warning that small generative models don't get this for free.

**Bottom line for the debate:** The public literature supports tagger-style distillation passing the gate, and supports SFT-on-DRES pushing precision to ~96. It does *not* yet contain a published result showing a generative distilled sub-1B model passing a changes-meaning gate on self-corrections *without* either over-deletion (Llama-8B failure mode) or generalization collapse (gpt-4o-mini_ft failure mode). If Chirp wants that evidence, it will have to produce it on its own v3 benchmark.

Sources:
- [DRES: Benchmarking LLMs for Disfluency Removal (ArXiv 2509.20321)](https://arxiv.org/abs/2509.20321)
- [DRES Literature Review (Moonlight)](https://www.themoonlight.io/en/review/dres-benchmarking-llms-for-disfluency-removal)
- [Disfluency Detection with Unlabeled Data and Small BERT Models (ArXiv 2104.10769)](https://ar5iv.labs.arxiv.org/html/2104.10769)
- [VocalBench-DF (ArXiv 2510.15406)](https://arxiv.org/abs/2510.15406)
- [DiDOTS: Knowledge Distillation from LLMs (PoPETs 2025)](https://petsymposium.org/popets/2025/popets-2025-0012.pdf)
