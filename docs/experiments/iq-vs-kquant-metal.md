# IQ against K-quant builds on Metal: a pre-registered protocol

Status: written before any cell was run. The decision rule below is fixed; changing it after
seeing results means writing a new protocol, not editing this one.

## Question

Is an IQ-family GGUF build slower or less accurate than a K-quant build of the same model on
this Mac's llama.cpp Metal backend? `poolhouse.serve.quant_guard` warns on every IQ lease
(`docs/serving.md`, "IQ quantisations on Apple silicon") on the strength of one pair of runs.
This protocol replaces that pair.

## What exists to test

Pairs of one model in an IQ and a K-quant build, found with `hub.discover` on this machine
(70 GGUF files):

| model | IQ build | K-quant build | used |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | UD-IQ4_XS, 89 GB, 3 shards | UD-Q4_K_XL, 106 GB, 4 shards | yes |
| functiongemma-270m-it | IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_M, IQ2_XXS | Q4_K_M, Q4_0 and 15 more | no |

Nothing else in the Hugging Face cache or `~/.cache` has both kinds (GLM-5.3-Flash and
Falcon-H1 are IQ only; Qwen3.8-27B and Qwen3.6-35B-A3B are K-quant only). functiongemma is
rejected: its `functiongemma-270m-it-IQ4_XS.gguf` has no `general.file_type`, and by tensor
bytes it is 178 MB `q8_0` (the embedding table) against 35 MB `iq4_nl` and 19 MB `iq4_xs`, so
IQ kernels touch about a quarter of it. A real pair of another family is worth getting
(`poolhouse-models find`, then `fetch`) and added to `iq-vs-kquant-metal.json` before running.
Two snapshots of Flash-Next IQ4_XS are in the cache; the runner records the path it used.

## Design

Config: `docs/experiments/iq-vs-kquant-metal.json`. Runner: `scripts/experiments/iq_vs_kquant.py`.
Every cell is one call of the existing machinery, no new framework.

| task | machinery | measures |
| --- | --- | --- |
| `jevbench` | `poolhouse-decide jevbench --backend logprob --gguf MODEL` | accuracy over JevBench's public items (all tiers, `--tier original`); ms per decision (one token out, so close to prefill) |
| `generation` | `poolhouse-bench sweep --serve MODEL --plain-only --sample 20` | F1 over the bench store's question set with its 95% bootstrap band; wall seconds per question |
| `speed` | `poolhouse-bench speed --serve MODEL --prompts 512,4096 --streams 1 --generate 256` | prefill tokens/s, decode tokens/s, time to first token, wall seconds |

- **Thinking.** `generation` runs with thinking on (no reasoning budget) and off
  (`--reasoning-budget 0`); the bench store records it as the run's thinking column.
  `jevbench` never thinks (decisions are always unthinking, `poolhouse.client.thinking`), and
  `speed` runs with thinking off. Flash-Next's template has the switch.
- **Repeats.** 3 per cell, temperature 0, `-s 1234` for the question set. The two builds
  alternate order each repeat (IQ first, then K-quant, then IQ) so drift lands on both.
  Run-to-run spread is reported with every table.
- **Held constant.** No draft head (`--no-draft`), no profile (`--no-profile`), context
  32768, one slot, f16 cache, the same llama.cpp build for every cell; the build id from
  `llama-server --version`, the model file paths and `POOLHOUSE_IQ` are written into the
  results file. The warning stays at its default; it changes nothing about serving.
- **Quiet machine, one model on the GPU.** The runner refuses to start a cell while
  `pgrep -fl 'llama-server|decide_cli train'` prints anything, and each cell serves and takes
  down its own model. Do not start it while another job holds the card.
- **Speed apart from accuracy.** Accuracy columns come from `jevbench` and `generation`;
  decode and prefill rate, time to first token and wall seconds from `speed`; the seconds
  per question of `generation` is reported beside accuracy but judged separately.

## Decision rule (fixed before running)

For each group (pair, task, thinking setting), with three repeats per build:

- **Interval.** For a build, the mean accuracy over the repeats, with a 95% interval whose
  half-width is the larger of (a) the item-level half-width (Wilson interval over the items
  for `jevbench`; the bench's bootstrap band for `generation`) and (b) the t interval of the
  repeats (t = 4.303 at three repeats).
- **IQ is worse on accuracy** when the K-quant interval's lower end is at least **3.0 points**
  above the IQ interval's upper end.
- **IQ is slower** when its seconds per question (`generation`), milliseconds per decision
  (`jevbench`) is at least **20% higher** than the K-quant's in at least **2 of the 3**
  repeats (repeat *k* of IQ against repeat *k* of K-quant); for `speed`, when K-quant decode
  tokens/s is at least 1.20 times IQ's in at least 2 of 3 repeats at either prompt size.
- **Verdict per group.** `worse` when it is worse on accuracy or slower; `no worse` when the
  means differ by less than 3.0 points and it is slower in none of the repeats;
  `inconclusive` otherwise; `incomplete` when fewer than 3 repeats finished.

**What the verdicts mean for the default mode of `POOLHOUSE_IQ`:** `worse` in most groups of
the Flash-Next pair, `warn` stays and the text gets the numbers; `worse` in a few groups,
`warn` stays, the text names which; `no worse` or `inconclusive` everywhere, the default
becomes `off`; `block` is never the default. One model family is one sample: a verdict is
about that pair, and the text says so.

## Running it

```
python scripts/experiments/iq_vs_kquant.py            # the plan and every exact command
python scripts/experiments/iq_vs_kquant.py --check    # validates; loads no model
pgrep -fl 'llama-server|decide_cli train'             # must print nothing
poolhouse-decide jevbench --fetch                      # the public items, once
python scripts/experiments/iq_vs_kquant.py --run      # 24 cells, one model at a time
python scripts/experiments/iq_vs_kquant.py --analyse  # table, intervals, verdicts
```

`--run --resume` skips finished cells. Results go to `~/.poolhouse/bench/` (the config's
`results` path) as JSON, the table as markdown beside it. The runs sit in a store of their
own (`iq-vs-kquant.ladybug`), not the default one.

## Limits written in advance

- One family. Another IQ type or size may behave differently.
- 20 questions per generation cell: its interval is wide, which is why the rule asks for
  3 points beyond the interval and not 3 points of difference.
- JevBench at temperature 0 is deterministic, so its repeats measure the machine's timing
  spread, not accuracy spread; its accuracy interval is the item-level one.
- The 106 GB build and the 89 GB build do not fit the memory pressure the same way; any
  paging shows up in the speed cells and is not separated from kernel cost.
