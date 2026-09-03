# gemma4 -- gemma-4 E2B, E4B, 26B-A4B

From the E4B header (`unsloth/gemma-4-E4B-it-qat-GGUF`, UD-Q4_K_XL) and today's measurements.

- 42 layers; `attention.sliding_window 512` with `sliding_window_pattern` true on five layers
  in six (the sixth is full attention); `key_length 512` on full layers, `key_length_swa 256`
  on sliding ones; **`shared_kv_layers 18`**: the last 18 layers reuse earlier layers' cache and
  own none. Per-layer input embeddings (`embedding_length_per_layer_input 256`).
- The header formula (every layer full attention) said 168K a token; **measured 32K a token
  and a 40M fixed cost a sequence** (E4B), 12K a token (E2B). Users at 32k on 110G: 101 (E4B,
  f16), 191 (q8_0). The 26B-A4B stores `attention.head_count_kv` as an array per block, which
  once crashed the preflight.
- MTP heads (`mtp-gemma-4-*-{Q4_0,Q8_0,BF16}.gguf`) load on mainline; only length 2 pays
  (E4B 1.09x, E2B 1.03x); head precision made no difference.
- Its card asks for temperature 1.0 "across all use cases"; on tool calls greedy measured
  better (E4B 70% vs 55% on the invented community, 2026-09-01).
- The cheap tier: E4B 40% F1 at 3 s/q over 100 questions; E2B 30% at 1.5 s/q.
