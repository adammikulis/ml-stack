# gpt-oss -- gpt-oss-120b, gpt-oss-20b

From the 120b header (`ggml-org/gpt-oss-120b-GGUF`, mxfp4). Memory figures are
`ml-stack-serve fit` records from 2026-09-02; answering and draft figures are the runs in
[`docs/report-2026-09-23.md`](../report-2026-09-23.md), which names its command and store.

- 36 layers, 64 heads, 8 KV heads, head size 64; `attention.sliding_window 128` with no
  pattern key: llama.cpp alternates, even layers sliding (period 2). Measured **72K a token**
  f16, 38K q8_0, a 27M fixed cost a sequence; 22 users at 32k with a 110G wired limit (42 at
  q8_0).
- Harmony chat template: no `enable_thinking`; the family adapter sends `reasoning_effort`.
  On nine questions it answered 24% F1 with thinking off, against 53-75% across the
  nine-question runs with it on; no run turned thinking off at a hundred questions.
- EAGLE3 heads (`eagle3-gpt-oss-*-{BF16,Q8_0}.gguf`) load on mainline and make it slower
  (120b against the same run with the head taken out: 0.76-0.82x at length 2, 0.72-0.75x at 4;
  20b 0.71-0.77x at length 2).
- Over 100 questions with a q8_0 cache: 120b 60% F1 / 58% recall at 6.3 s/q; 20b 38% F1 at
  3.7 s/q.
