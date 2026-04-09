# advocate-B — round 1 — question 4 — OK

## Question

What is the minimum number of training pairs needed for distillation on a cleanup corpus?

## Answer

*Interpretation: minimum labeled (input → cleaned-output) pairs to distill an 8B teacher into a small student (T5-base scale) for voice-transcript cleanup, i.e. sequence-to-sequence rewriting.*

There is no universal floor — it depends on student capacity, task diversity, and whether pairs are hard-labeled or soft-label/logit distillation. Concrete data points from 2024–2026 work:

**Empirical floor: ~1k–5k pairs, sweet spot 10k–50k.**

- Predibase's "12 Best Practices for Distilling Small LMs from GPT" (2024) reports that task-specific distillation of <1B students plateaus around **~10k examples**; gains from 1k→5k are large, 5k→10k marginal, and >10k shows diminishing returns for narrow tasks ([predibase.com](https://predibase.com/blog/graduate-from-openai-to-open-source-12-best-practices-for-distilling-smaller)).
- Vennify's T5-base grammar corrector (analogous seq2seq rewriting task) was trained on ~**3k JFLEG + 3k C4_200M** pairs and ships as the widely-used `vennify/t5-base-grammar-correction` HF model ([vennify.ai](https://www.vennify.ai/fine-tune-grammar-correction/), [HF](https://huggingface.co/vennify/t5-base-grammar-correction)).
- Gramformer (T5-base, grammar correction) was trained on ~**1M synthetic pairs** but the author notes most quality is captured in the first ~50k ([github.com/PrithivirajDamodaran/Gramformer](https://github.com/PrithivirajDamodaran/Gramformer)).
- GECToR / "Efficient GEC via Unsupervised Generation" ([arxiv 2311.11813](https://arxiv.org/pdf/2311.11813)) shows **~10k–20k** high-quality synthetic pairs can match models trained on 100k+ noisy pairs when the teacher generates the targets.
- ICLR 2025 "Speculative Knowledge Distillation" ([paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/a2747a3844ca1e4667fbff3f558eb39b-Paper-Conference.pdf)) and "Influence Distillation" ([arxiv 2505.19051](https://arxiv.org/html/2505.19051v1)) both use **10k-sample curated subsets** selected from 200k pools, matching full-data performance at 3× speed.
- DistillKit (Arcee AI) recipes recommend **≥10k pairs** for logit-level distillation to stabilize KL loss; below ~2k, logit distillation degenerates and plain SFT wins ([github.com/arcee-ai/DistillKit](https://github.com/arcee-ai/DistillKit)).

**Operational rule for your cleanup corpus:**
- **1k pairs**: proof-of-life SFT, no logit distillation — expect brittle behavior on unseen filler patterns.
- **5k pairs**: first usable T5-small/base, covers the 250-case v3 benchmark distribution if stratified.
- **10k–20k pairs**: recommended floor for shipping, especially with logit/KL distillation from Ministral 8B.
- **50k+**: only needed if you add heavy synonym/dictionary/domain diversity.

Because your teacher (Ministral 3 8B) can generate pairs cheaply, the binding constraint is **input diversity**, not label count. Mine 10k–20k real ASR transcripts (Parakeet outputs from LibriSpeech/CommonVoice + your own recordings), run the teacher once, and that's your corpus.

Sources:
- [Predibase: 12 Best Practices for Distilling Small LMs](https://predibase.com/blog/graduate-from-openai-to-open-source-12-best-practices-for-distilling-smaller)
- [Vennify T5 grammar correction tutorial](https://www.vennify.ai/fine-tune-grammar-correction/)
- [Efficient GEC via Unsupervised Generation (arXiv 2311.11813)](https://arxiv.org/pdf/2311.11813)
- [Influence Distillation (arXiv 2505.19051)](https://arxiv.org/html/2505.19051v1)
- [Speculative Knowledge Distillation, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/a2747a3844ca1e4667fbff3f558eb39b-Paper-Conference.pdf)
- [KD & Dataset Distillation Survey 2025 (Springer)](https://link.springer.com/article/10.1007/s10462-025-11423-3)
- [HF seq2seq-distillation reference impl](https://github.com/huggingface/transformers/blob/main/examples/research_projects/seq2seq-distillation/finetune.py)
