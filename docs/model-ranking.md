# Which model answers best

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
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 80% ±6 | 89% | 77% | 100 | 25.4 | 5s | 97.7G | - | greedy | words | 159 | `Qwen3.8-Flash--plain-kv-q8_0-rb0` on unsloth/llama-server (100 q, s/q not separated) |
| `gpt-oss-120b-mxfp4-00001-of-00003.gguf` | 60% ±8 | 58% | 66% | 100 | 6.3 | 27s | 59.7G | 0.7G | greedy | words | 86 | its own run on 3466812/llama-server |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | 40% ±8 | 43% | 46% | 100 | 3.1 | 3s | 5.4G | 1.5G | greedy | words | 121 | its own run on 3466812/llama-server |
| `gpt-oss-20b-MXFP4.gguf` | 38% ±8 | 37% | 44% | 100 | 3.7 | 1s | 12.7G | 1.5G | greedy | words | 130 | its own run on 3466812/llama-server |
| `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | 30% ±7 | 35% | 31% | 100 | 1.5 | 2s | 3.1G | 0.7G | greedy | words | 113 | its own run on 3466812/llama-server |

*217 run(s) not ranked: fewer than 20 questions, which is a smoke run proving the path works rather than a measurement -- it supplies neither accuracy nor cost. 91 run(s) not ranked: not measured over the community that ships with this package.*
