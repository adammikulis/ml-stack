# Which model answers best

- **Written:** 2026-09-23, at commit 28d13f5.
- **Command:** `ml-stack-bench show --kept ~/.ml-stack/bench/runs.ladybug --rank docs/model-ranking.md`
- **Store:** `~/.ml-stack/bench/runs.ladybug`, opened read-only, the newest run from 2026-09-06.
- **Models:** the five in the table, each named by the file it served.

Measured over the invented community that ships with this package, by
`ml-stack-bench`. A conclusion, not evidence: the runs behind it are not in this
repository. Re-measure after any model release -- none of this survives one.

Accuracy is each model's largest run -- the most questions, the newest on a tie --
since a draft head cannot change an answer, only the clock. Cost is the model's
fastest run of at least 20 questions whose F1 was not separated from that
-- the two 95% bootstrap intervals over their own questions overlapping, or,
for a run carrying no interval, within 5 points -- per question, whatever
head, draft length or build it ran on; the last column says which run that was.
Every F1 below carries the interval its questions put around it: a difference
smaller than the interval is a difference these questions did not measure.

| model | F1 | recall | precision | questions | s/question | load | resident | kv+run | sampling | find | made | cost from |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 80% ±6 | 89% | 77% | 100 | 25.7 | 5s | 86.5G | - | greedy | words | 1.6 | `draft:mtp-Qwen3.8-Flash-Next-shared-Q8_0@n4-rb0` on unsloth (27 q, 1.27x, s/q not separated) |
| `gpt-oss-120b-mxfp4-00001-of-00003.gguf` | 60% ±8 | 58% | 66% | 100 | 6.3 | 27s | 59.7G | 0.7G | greedy | words | 0.9 | its own run |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | 40% ±8 | 43% | 46% | 100 | 3.1 | 3s | 5.4G | 1.5G | greedy | words | 1.2 | its own run |
| `gpt-oss-20b-MXFP4.gguf` | 38% ±8 | 37% | 44% | 100 | 3.7 | 1s | 12.7G | 1.5G | greedy | words | 1.3 | its own run |
| `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | 30% ±7 | 35% | 31% | 100 | 1.5 | 2s | 3.1G | 0.7G | greedy | words | 1.1 | its own run |

Runs whose F1 fell clear of their model's -- separated, their 95% intervals not
overlapping, or, carrying none, more than 5 points down -- so their cost was
not taken. A head cannot change an answer, so look at what else these changed:

- `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` rejected: `Qwen3.8-Flash--plain-kv-q8_0-rb0` F1 -11 pts (100 q, 25.4 s/question)

*289 run(s) not ranked: fewer than 20 questions, which is a smoke run proving the path works rather than a measurement -- it supplies neither accuracy nor cost. 91 run(s) not ranked: not measured over the community that ships with this package.*

## Extraction, on the shipped gold set

`ml-stack-ingest --gold tests/fixtures/extraction-gold.json --model MODEL --fail-under 0.7`:
twenty invented passages with every triple written down, so the number is precision as well
as recall. Measured 2026-09-05, each model in its profile's shape.

| model | F1 | recall | precision | triples | wall clock | passes 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL` | 78% | 74% | 83% | 101 of 137 | 237 s | yes |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL` | 61% | 62% | 61% | 85 of 137 | 113 s | no |
| `gemma-4-E2B-it-qat-UD-Q4_K_XL` | 51% | 45% | 57% | 62 of 137 | 69 s | no |

The gemma models also reach outside the core vocabulary far more (44 of 140 relations on
E4B, 46 of 108 on E2B, against 13 of 122 on Flash-Next), which is what the fold has to
absorb afterwards.
