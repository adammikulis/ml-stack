# qwen4exp -- Qwen3.8-Flash-Next

Read off the GGUF header (`unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL, 2026-09-02) and
measured on an M4 Max, 128G, wired limit 110G. Anything not marked measured is the header.
Memory figures are `ml-stack-serve fit` records from 2026-09-02; answering and draft figures
are the runs in [`docs/report-2026-09-23.md`](../report-2026-09-23.md), which names its
command and store.

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
  mmapped and paged in row by row as n-grams are seen. Measured split in the serving shape (`fit`, q8_0 cache, 32k x 2, 2026-09-02
  evening): 106.3G on disk with the head, 78.8G of it in GPU memory and 27.5G mapped on the
  CPU -- the table -- so the GPU holds ~79G of weights, and the hundred-question run of
  2026-09-02 peaked at 99G resident. That split is llama.cpp's own placement, not a flag of ours: an
  input-side table gathered per token stays host-mapped the way token embeddings do. On
  unified memory the two halves are one pool of RAM and nothing is gained by forcing it
  either way, so ml-stack passes no override on a Mac; `--on-cpu per_layer_token_embd=CPU`
  is for a discrete GPU whose VRAM the table would not fit beside the weights. Capacity
  planning starts from the GPU-mapped weights plus a measured resident peak, never the
  file size (`ml-stack-serve fit --tensors`, `fit --measure`).
- **The cache is tiny**: 48K bytes a token at f16 (26K at q8_0), measured. The 12 attention
  layers' K/V come to 24K of that (2 KV heads x 256 x K+V x 2 bytes, x12); the other 24K
  is the sparse indexer's cache, one per attention layer and the same size, which the
  build keeps beside the K/V (`llama_memory_hybrid_idx`). Not the MTP head: the record
  without it measures the same 48K. `preflight._kv_estimate_bytes` counts both. On top
  of that a fixed ~257M a sequence for the recurrent state and sliding cells (~594M with
  the MTP head's own cache). At 32k a user costs 1.8G (f16) or ~1.4G (q8_0). Users at 32k on
  110G: 12 at f16 when the whole file is counted; 22 at q8_0 once the table is counted on
  the CPU side where it lives, 31 at 16k a slot. (`fit`, 2026-09-02.)
- **UD-Q4_K_XL against UD-IQ4_XS on Metal**, plain asking, thinking on, no head: 43.7 s a
  question at 64% F1 on nine questions against 70.1 s at 54% on ten. (The table tensor itself
  is IQ4_NL in both builds.)
- **The MTP head loads only on the unsloth fork** (`b10715-mix-86bd2d3`+; mainline's
  qwen4exp MTP graph is PR #27836, open). The shared-Q8_0 head against the same run with the
  head taken out, UD-Q4_K_XL, thinking off, q8_0 cache: 1.27-1.32x at length 4 (72-78%
  accepted), 1.25-1.51x at 2, 1.14-1.74x at 7, 0.99-1.07x at 8. On UD-IQ4_XS, thinking on:
  1.46-1.73x at 4, 0.73-0.95x at 8; the shared and unshared Q8_0 heads are within 0.01x of
  each other. No run served with `--spec-draft-p-min 0.5` has a baseline without the head.
  See `docs/research/qwen38-flash-next-mtp.md`.
- **It thinks unless told not to through the template.** Thinking off, q8_0 cache, head at
  length 4: 81-85% F1 at 25-33 s/q on nine questions, 80% at 26.7 s/q on a hundred. Thinking
  on, no head, f16 cache: 64% and 81% on nine questions (43.7 and 27.6 s/q); no
  hundred-question run had thinking on. On nine questions with the head at length 4, the run
  labelled `ub2048` took 25.5 s/q against 29.2-29.4 for the same serving without it; the run
  records no `-ub`, so only its label says what differed. 16k a slot answered as 32k did on
  nine questions (81% F1). On UD-IQ4_XS a q8_0 cache answered as f16 did (85% F1 on nine
  questions, 31.5 against 36.6 s/q). Its serving shape is `ml-stack-serve profile`'s record.
- **Recall runs 77-95% across its runs of nine questions or more; precision is lower**, 43-83%
  depending on the asking.

## What to check when a new build appears

`ml-stack-models files <repo>` for the quant types per build, `ml-stack-serve fit --tensors`
for the table's size in that build, `ml-stack-bench drafts` for the head, and whether
mainline has merged the MTP graph (`gh pr view 27836 -R ggml-org/llama.cpp`).
