# Gemma as a decision-model torso: where it stopped (2026-10-03)

Paused on purpose. Nothing here is merged; the work is on branch `feat/decider-gemma4`
(tip `8120e9a1`, a WIP commit). The default decider stays the Strands 2B.

## What exists on the branch
* Loading a Gemma 4 E2B torso for the pointer decider (`Gemma4Model.language_model`), a pin for
  `google/gemma-4-E2B` (10.2 GB, Apache-2.0 per the model card; weights are cached under
  `~/.cache/poolhouse/decide/`), `--base gemma-4-e2b`, `jevbench --decider`, `--max-tokens`.
* Memory-safe training: gradient checkpointing (on by default on GPUs), `--accum`, length-bucketed
  micro-batches, the baseline model freed before training, peak memory logged per step.
  Measured on E2B (bf16, MPS): activations cost about 44 GB per 3072-token sequence without
  checkpointing (the cause of a 140 GB run that swapped), about 14 GB peak with it; 9 s per
  optimiser step at batch 1 x accum 4.
* A fix so pointer prompts are tokenised with the tokenizer's special tokens (a Gemma prompt
  starts with `<bos>`), and a guard test that no test can pin into the real sentinel manifest.
* 2026-10-08: FunctionGemma 270M is dropped (Gemma Terms of Use) in favour of embeddinggemma-2 (Apache-2.0).
* WIP: a pin and a test for `google/functiongemma-270m-it` (gemma3_text, 18 layers, hidden 640,
  536 MB, cached) as a fast test bed.

## Results
| run | what | outcome |
|---|---|---|
| `gemma4-e2b-v1` | E2B, batch 4, 4096 tokens, no checkpointing | stopped: 140 GB, swapping |
| `gemma4-e2b-v2` | E2B, 300 steps (44 min), Strands synthetic train data (4980 cases), eval file as the test split | refused by the baseline gate: accuracy 0.420 vs the Strands 2B's 0.787, Brier 0.612 vs 0.286 (near chance) |
| overfit, 250 steps, with and without `<bos>` | FunctionGemma 270M, small set | both ended at loss about 0.69-0.70, which is the loss of a coin flip on yes/no; BOS did not explain it |
| `g270m-v1` | FunctionGemma 270M, real recipe | stopped at step 296, loss 0.73-0.86, not clearly falling |

The two reference points on JevBench (231 public items): the Strands 2B pointer decider 0.736,
Qwen3.8-27B through the logprob backend 0.874 (Brier 0.166, ECE 0.031).

## What it probably means
The model never learned even a tiny set, so a bug in the Gemma path is more likely than a model
limit. Not yet checked: the learning rate and LoRA targets, the pointer head input, label
alignment per question kind, and the hidden-state positions at the option and answer markers
with Gemma's shared-KV and sliding-window layers. Strands' own recipe is much larger (21 public
datasets, about 11 hours on an RTX 3090) than the synthetic-only data used here.

## To resume
1. Overfit 50 cases for 60 steps on the 270M model; it must reach near-zero training loss. Until
   it does, do not spend GPU time on longer runs.
2. Fix what stops it; add a regression test.
3. Then one real run, judged on the Strands eval file with the baseline gate; JevBench only after
   it passes. Fine-tuned weights never go in the repo. Leftover run directories sit in
   `~/.poolhouse/decide/models/` (`gemma4-e2b-v1`, `-v2`, `g270m-v1`) and can be deleted.
