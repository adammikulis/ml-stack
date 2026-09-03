# qwen4exp -- Qwen3.8-Flash-Next

Read off the GGUF header (`unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL, 2026-09-02) and
measured on an M4 Max, 128G, wired limit 110G. Anything not marked measured is the header.

## What it is

- **48 layers, full attention on 1 in 4** (`full_attention_interval 4`): 24 heads, 2 KV heads,
  head size 256, `compress_ratios` of 4 on those layers and a top-k 2048 `indexer` (4 heads,
  key 128) -- the attention layers are compressed and sparse, not plain full attention.
- **The other 36 layers are gated-deltanet recurrent** (`ssm.*`: inner 6144, state 128, 16
  groups, conv kernel 4). A fixed state per sequence, no cache that grows with context.
- **512 experts, 10 used, 640 wide** (`expert_count 512`, `expert_used_count 10`), plus a
  shared expert of the same width. The `512x56B` size label counts experts.
- **Hyper-connections** (`hyper_connection.count 4`, low rank 320) and **per-layer input
  embeddings** (`embedding_length_per_layer_input 160`, `ple.layers [1]`).
- **A 3-gram lookup table as one tensor**: `per_layer_token_embd.weight`, shape
  160 x 320,001,536 = **51.2B parameters**, IQ4_NL, **26.8G on disk** in the UD-Q4_K_XL build.
  `ple.ngram_size 3`, `ple.heads_per_ngram 8`, 16 hash heads of ~20M rows each
  (`ple.head_vocab_sizes`, `ple.head_offsets`). It is gathered per token, never multiplied.
- 262,144 context (`context_length`), rope base 1e7 with dimension sections `[11, 11, 10, 0]`.
- Chat template with an `enable_thinking` switch (the family adapter's `think_kwargs`);
  `--reasoning-budget 0` alone does NOT stop it thinking -- measured.

## What that means when serving (measured)

- **Memory is not the file size.** The build is 103.7G on disk: ~77G of everything else (the
  experts are Q8_0 -- `blk.N.ffn_down_exps` is 0.8G a layer) plus the 26.8G table, which is
  mmapped and paged in row by row as n-grams are seen. Real Mem sat at ~90G through a day of
  questions. Measured split in the serving shape (`fit`, q8_0 cache, 32k x 2, 2026-09-02
  evening): 106.3G on disk with the head, 78.8G of it in GPU memory and 27.5G mapped on the
  CPU -- the table -- so the GPU holds ~79G of weights and the resident peak over a hundred
  questions was 99G. That split is llama.cpp's own placement, not a flag of ours: an
  input-side table gathered per token stays host-mapped the way token embeddings do. On
  unified memory the two halves are one pool of RAM and nothing is gained by forcing it
  either way, so ml-stack passes no override on a Mac; `--on-cpu per_layer_token_embd=CPU`
  is for a discrete GPU whose VRAM the table would not fit beside the weights. Capacity
  planning starts from the GPU-mapped weights plus a measured resident peak, never the
  file size (`ml-stack-serve fit --tensors`, `fit --measure`).
- **The cache is tiny**: 48K bytes a token on the 12 attention layers at f16 (26K at q8_0),
  plus a fixed ~257M a sequence for the recurrent state and sliding cells (~594M with the
  MTP head's own cache). At 32k a user costs 1.8G (f16) or ~1.4G (q8_0). Users at 32k on
  110G: 12 at f16 when the whole file is counted; 22 at q8_0 once the table is counted on
  the CPU side where it lives, 31 at 16k a slot. (`fit`, 2026-09-02.)
- **Take a K-quant, not an IQ quant, on Metal**: UD-Q4_K_XL answered in 44 s a question at
  64% F1 where UD-IQ4_XS took 70 s at 54% -- the IQ lookup-table kernels are slow on Metal.
  (The table tensor itself is IQ4_NL in both builds; it is a gather, so that does not matter.)
- **The MTP head loads only on the unsloth fork** (`b10715-mix-86bd2d3`+; mainline's
  qwen4exp MTP graph is PR #27836, open). Shared-Q8_0 head at `--spec-draft-n-max 4` and
  `--spec-draft-p-min 0.5`: 1.47x, 73-79% accepted; length 8 loses; head precision is
  irrelevant. See `docs/research/qwen38-flash-next-mtp.md`.
- **It thinks unless told not to through the template**; with thinking off it answered
  better and faster (tight asking: 81-85% F1 at 29 s/q on ten questions, 80% at 27 s/q on
  a hundred). `-ub 2048` helps prefill (13%); q8_0 cache is free; 16k a slot answers like
  32k (peak use 6.5k). Its measured serving shape is `ml-stack-serve profile`'s record.
- **Recall is its strength (85-95%), precision its weakness** until asked tightly; it
  calls tools 5-9 times a question; half the wall clock is reading tool results.

## What to check when a new build appears

`ml-stack-models files <repo>` for the quant types per build, `ml-stack-serve fit --tensors`
for the table's size in that build, `ml-stack-bench drafts` for the head, and whether
mainline has merged the MTP graph (`gh pr view 27836 -R ggml-org/llama.cpp`).
