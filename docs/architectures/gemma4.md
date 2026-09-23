# gemma4 -- gemma-4 E2B, E4B, 26B-A4B

From the E4B header (`unsloth/gemma-4-E4B-it-qat-GGUF`, UD-Q4_K_XL). Memory figures are
`ml-stack-serve fit` records from 2026-09-02; answering and draft figures are the runs in
[`docs/report-2026-09-23.md`](../report-2026-09-23.md), which names its command and store.

- 42 layers; `attention.sliding_window 512` with `sliding_window_pattern` true on five layers
  in six (the sixth is full attention); `key_length 512` on full layers, `key_length_swa 256`
  on sliding ones; **`shared_kv_layers 18`**: the last 18 layers reuse earlier layers' cache and
  own none. Per-layer input embeddings (`embedding_length_per_layer_input 256`).
- The header formula (every layer full attention) said 168K a token; **measured 32K a token
  and a 40M fixed cost a sequence** (E4B), 12K a token (E2B). Users at 32k with a 110G wired
  limit: 101 (E4B, f16), 191 (q8_0). The 26B-A4B stores `attention.head_count_kv` as an
  array per block.
- MTP heads (`mtp-gemma-4-*-{Q4_0,Q8_0,BF16}.gguf`) load on mainline. Against the same run
  with the head taken out: E4B 0.99-1.09x at length 2, 0.85-1.09x at 4 and 0.56-0.75x at 8
  and 16; E2B 1.03-1.29x at length 2 and 0.99-1.30x at 4. The fastest on each model is
  inside the noise of its baseline's seconds per question. Head precision (Q4_0, Q8_0, BF16)
  moved E4B by at most 0.06x.
- Its card asks for temperature 1.0 "across all use cases"; the runs above are greedy.
- Over 100 questions, served with the head at length 2 and a q8_0 cache: E4B 40% F1 at
  3.1 s/q, E2B 30% at 1.5 s/q.
