# Research log (smoke)

Each entry links to a markdown file under `research/`.

- **bootstrap** r0 — [bootstrap.md](research/bootstrap.md) — OK
- **advocate-A** r1 q1 — [r1_advocate-A_01.md](research/r1_advocate-A_01.md) — OK
  - Q: Does sequence-level distillation with reverse-KL + MSE on cleaned outputs outperform logit-only KD by ≥3% on the v3 benchmark’s changes-meaning gate when applied to a pruned 8B teacher (30–40% capacit
- **advocate-A** r1 q2 — [r1_advocate-A_02.md](research/r1_advocate-A_02.md) — OK
  - Q: What is the empirical size/latency breakdown of a Qwen3-0.6B model pruned to ~1.2B, quantized to Q4_K_M, and loaded in INT4 on Apple Silicon M1/M2? (Benchmark against [Qwen3’s native performance](http
- **advocate-A** r1 q3 — [r1_advocate-A_03.md](research/r1_advocate-A_03.md) — OK
  - Q: What is the minimal pruning threshold (e.g., 30% vs. 50% hidden dim reduction) for Ministral-8B’s pruned distillation model to retain ≥92% composite score on the v3 benchmark? (Test on [NVIDIA’s Minit
- **advocate-A** r1 q4 — [r1_advocate-A_04.md](research/r1_advocate-A_04.md) — OK
  - Q: Is GGUF’s Q4_K_M quantization stable for INT4 inference on Apple Silicon? (Check for latency spikes or accuracy drop on [GGUF’s benchmarks](https://github.com/abetlen/ggml#latency).)
- **advocate-A** r1 q5 — [r1_advocate-A_05.md](research/r1_advocate-A_05.md) — OK
  - Q: What is the empirical distribution of self-correction failure cases in the developer’s 280 dictations? If >10% of cases fail, does adding a 10% oversampling of self-correction pairs to the training co
- **advocate-B** r1 q1 — [r1_advocate-B_01.md](research/r1_advocate-B_01.md) — OK
  - Q: What is the optimal compression order for 8B→500M cleanup models?
- **advocate-B** r1 q2 — [r1_advocate-B_02.md](research/r1_advocate-B_02.md) — OK
  - Q: How does quantization-aware distillation perform on self-correction tasks?
- **advocate-B** r1 q3 — [r1_advocate-B_03.md](research/r1_advocate-B_03.md) — OK
  - Q: Can pruning be applied to the 8B teacher without significant quality loss?
- **advocate-B** r1 q4 — [r1_advocate-B_04.md](research/r1_advocate-B_04.md) — OK
  - Q: What is the minimum number of training pairs needed for distillation on a cleanup corpus?
- **advocate-B** r1 q5 — [r1_advocate-B_05.md](research/r1_advocate-B_05.md) — OK
  - Q: How does the choice of base model affect the final model's performance on self-correction tasks?
