# Qwen3.8-Flash-Next through the MLX tree engine, 2026-09-17

Machine: M4 Max, 128G, macOS 25.6, mlx 0.32.2, mlx-vlm 0.7.1, ml-stack at `88cde1b`.
Target: `mlx-community/Qwen3.8-Flash-Next-4bit` (snapshot `07b5dc6c`), 4-bit affine,
group 32; MoE 512 experts top-10, 48 layers, 3:1 Gated-DeltaNet to QSA attention, one PLE
layer, indexer budget 2048 over blocks of 4.
Drafters: `ngram` (depth 8); `mtp` = `unsloth/Qwen3.8-Flash-Next-GGUF`
`MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf`, quantized to 4-bit on first load;
`dflash` = `PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash` (DSpark, block 7), 4-bit.

## What the checkpoint costs to load

    ml-stack-bench tree lossless --drafter ngram ...

| | bytes |
|---|---|
| safetensors on disk | 111.5 G |
| PLE n-gram table (mapped, not held) | 37.4 G |
| held after the load, measured `mx.get_active_memory()` | 74.07 G |
| `report_for`'s fit estimate | 74.06 G |

The PLE table is read row by row from the mapped file. A single-token step reads sixteen
rows and costs nothing measurable; a 2,300-token prefill reads about 37,000 rows scattered
over the 37 G table, and on a machine already short of memory that prefill is dominated by
disk (measured: 129-285 MB/s of reads, GPU 70-97%, 11 G of swap in use while other agents
held 25-37 G).

## Lossless witness

Greedy tree decoding against greedy plain decoding of the same prompt, token by token.
A reply that differs is judged against the model's own shape noise: the largest
disagreement between decoding one token at a time and reading the same tokens in one
batched pass, measured over the same positions in the same run.

    ml-stack-bench tree lossless --tokens 128 --drafter ngram --drafter mtp=<gguf> \
        --drafter dflash=PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash

| drafter | prompt | tokens | verdict |
|---|---|---|---|
| ngram | chat | 128 | identical |
| ngram | code | 128 | differs at 89; gap 0.250 against 1.500 of measured noise |
| ngram | math | 128 | differs at 58; gap 0.125 against 2.000 of measured noise |
| ngram | code | 64 | identical |

Every difference so far is a near-tie: the model's own top-two margin at that position is
one or two bfloat16 quanta, and the model disagrees with itself by more than the margin
when it reads the same tokens in a different shape.

## What is not measured yet

The `mtp` and `dflash` witnesses, the 2,300-token prompt that decodes past the indexer
budget, the mutation checks and the speed table are `HANDOFF.md` entries: this machine
had 25-37 G of another agent's servers resident throughout, which is what turns a 74 G
load into a swapping one.
