# gpt-oss -- gpt-oss-120b, gpt-oss-20b

From the 120b header (`ggml-org/gpt-oss-120b-GGUF`, mxfp4) and today's measurements.

- 36 layers, 64 heads, 8 KV heads, head size 64; `attention.sliding_window 128` with no
  pattern key: llama.cpp alternates, even layers sliding (period 2). Measured **72K a token**
  f16, 38K q8_0, a 27M fixed cost a sequence; 22 users at 32k on 110G (42 at q8_0).
- Harmony chat template: no `enable_thinking`; the family adapter sends `reasoning_effort`.
  **It needs its thinking**: 60% F1 with, 24% without (100 questions / 9 questions).
- EAGLE3 heads (`eagle3-gpt-oss-*-{BF16,Q8_0}.gguf`) load on mainline and make it slower
  (0.82x at length 2, 0.75x at 4) -- decode is already fast, the head's verification costs
  more than it saves.
- The fast fallback: 60% F1 / 58% recall at 6 s/q; the 20b is 37% with the most invented ids
  of any model measured.
