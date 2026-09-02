# Fine-tuning a tool caller on its own traces

*Asked 2026-09-02. A plan and a measurement, not a result: nothing here has been trained
yet. Every number about this machine's models comes from `docs/model-ranking.md`; every
number about training is an estimate and says so.*

## What is wrong now

A question is answered by a loop: the model is offered six tools, calls one, reads what came
back, calls another, and eventually says what to light on the page. The score is F1 over
what it lit. Measured over the invented community:

| model | F1 | s/question | resident |
| --- | --- | --- | --- |
| `Qwen3.8-Flash-Next` (Q4_K_XL) | 70% | 25.4 | 97.7G |
| `gpt-oss-120b` (mxfp4) | 60% | 6.3 | 59.7G |
| `gemma-4-E4B-it-qat` (UD-Q4_K_XL) | 40% | 3.1 | 5.4G |
| `gemma-4-E2B-it-qat` (UD-Q4_K_XL) | 30% | 1.5 | 3.1G |

The gap between 70% and 40% is not knowledge. It is calling the right tool with the right
ids first time: E4B looks up words that are not in the graph, shows entries it never read,
and answers before it has searched. Those are all *decisions in a conversation*, and a
decision made well by a 97.7G model on this graph is a decision a 5.4G model can be shown.

What was measured to move it already is exactly of this kind: a worked call written into a
tool's *description* took E4B from 17% to 70% recall on the same weights. Teaching the
weights the same thing is the next step along the same line, and it costs nothing at
inference — the fine-tune has no shortlist to run and no extra context to read.

## The plan

1. **Trace.** The bench now keeps the transcript of a traced question on the row and in the
   store: every message sent, every call with its arguments, every result with its size and
   how many ids it held, and per call the timings `Spent.note` reads. On by default for a
   run of 20 questions or fewer, off for the hundred, `MLSTACK_BENCH_TRACE=1` to force it.
   (`bench.measure.Counting`, `bench.wants_trace`, `Row.trace`.)
2. **Harvest.** `ml-stack-train-tools from-bench --kept STORE --model e4b --min-f1 0.8 --out
   FILE.jsonl` turns every traced question that scored well enough into one example per
   model turn.
3. **Train.** The `tool-calls` recipe over that file — the base model's own chat template,
   loss on the assistant tokens only. See "Which model, and how" below: the target is E4B,
   and E4B needs a LoRA path this repository does not have yet.
4. **Export and serve.** `ml_stack.gguf.export`, then `ml-stack-serve up`.
5. **Measure.** `ml-stack-bench run` the fine-tuned caller against its own base, same
   questions, same graph, same finder, same sampling. Nothing below is believed until this
   says so.

**Whose traces.** The obvious source is the model's own — E4B's traces train E4B. The more
interesting one is Flash-Next's: 70% on the same questions, and its traces are what a
*good* caller looks like on this graph. Both are worth a run, and they are different
experiments (self-distillation against distillation from a larger model); the harvester
takes `--model` precisely so the two are never mixed into one dataset by accident.

## The data shape

One example per model turn. The conversation up to that turn is the input; the turn is the
target. A question of four calls is four examples, each a decision made with strictly more
evidence than the last — which is the thing being taught: not *a* call, but the next one,
given what came back.

```json
{"messages": [{"role": "system", "content": "You are answering a question about a graph…"},
              {"role": "user", "content": "who works on compilers?"},
              {"role": "assistant", "content": null,
               "tool_calls": [{"id": "call_0", "type": "function",
                               "function": {"name": "look_up",
                                            "arguments": {"texts": ["compiler"]}}}]},
              {"role": "tool", "name": "look_up",
               "content": "[{\"id\": \"topic:compiler\"}, …]"},
              {"role": "assistant", "content": null,
               "tool_calls": [{"id": "call_0", "type": "function",
                               "function": {"name": "show",
                                            "arguments": {"ids": ["topic:compiler"]}}}]}],
 "tools": [ …the schemas that were offered on that call… ],
 "tool": "show", "from": "who works on compilers?", "call": 2,
 "run": "bench:e4b-shortlist:20260902T190000", "model": "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
 "split": "train"}
```

Identical to what `synthesise` writes, so the two sources mix in one directory and
`ml-stack-train-run --recipe tool-calls --data` reads either. Deliberate choices in it:

- **`tools` is what was offered on that call, not every tool that exists.** `graph.ask`
  takes tools away as a question goes on; an example that offers a tool the model did not
  have teaches it to reach for something that will not be there.
- **A tool result is cut at 2000 characters**, with its whole length and its id count kept
  beside the cut text. The lesson is the call, not the graph's answer, and a full run's
  results would be the largest thing in the store by an order of magnitude.
- **A turn the ceiling cut off (`finish_reason` of `length`) is dropped.** A truncated call
  has arguments that stop mid-word: the one thing a tool caller must never learn.
- **Only questions at or above `--min-f1`.** A wrong answer's calls are wrong in some
  particular way, and that is precisely what must not be reinforced.
- **Held out one question in ten by hash of the question**, with every turn of it — a
  question split across train and holdout scores memorising, not calling.

### What today's store would yield

Nothing, and that is the finding. `~/.ml-stack/bench/runs.ladybug` on 2026-09-02: **235
runs, 751 scored questions at or above F1 0.8, 0 of them traced.** Those questions made
**4006 model turns** between them — 4006 training examples that no longer exist, recoverable
only by spending the GPU again. Per model, at the same threshold:

| `--model` | questions ≥ 0.8 | turns they made |
| --- | --- | --- |
| `flash` | 301 | 1896 |
| `e4b` | 183 | 894 |
| `oss` | 162 | 742 |
| `e2b` | 64 | 278 |

(`ml-stack-train-tools from-bench --dry-run` prints this for any store: it is `would_yield`,
and it exists so that the answer "0" is never given without the number beside it. Turns per
question run 4.9 for E4B and 6.3 for Flash-Next — a fuller question, and a fuller lesson.)

This is why tracing defaults *on* for a sampled run rather than off: a trace is only useful
if it exists before you knew you wanted it.

## The measure

Same bench, same graph, same questions, one variable:

```
ml-stack-bench sweep --serve <base>.gguf --serve <finetuned>.gguf \
    --sample 100 --shortlist 8 --kept ~/.ml-stack/bench/runs.ladybug
ml-stack-bench show --rates          # accuracy per second, per 1k tokens, per GB
ml-stack-bench show --trace <label>  # and what it actually called, when it was wrong
```

What must move for this to have worked, in order of how much it would mean:

1. **F1 on the full hundred**, against the base at the same sampling and the same finder.
   The bar is not "better than the base" but better by more than a short run's noise — the
   ranking's own 5-point rule, measured on the hundred where one question is worth 2 points.
2. **Calls per question down.** A caller that gets there in three calls instead of five is
   the same answer for 40% less wall clock, and the bench already counts it.
3. **`unread_named` down.** Entries the prose names that no tool call produced — the
   made-up-plausible-name failure F1 cannot see. E4B scored 121 of these on the hundred.
4. **Held-out questions.** The one-in-ten split, and better still a question set the traces
   never came from: a caller trained on this community's traces that only works on this
   community has learned the community, not the calling.

And what must *not* move: `graph.ask`'s tool descriptions. The fine-tune is measured against
the descriptions it was traced under; changing both at once measures neither. (The answer
cache is already fingerprinted on the tool descriptions for the same reason.)

## Which model, and how

**Not Flash-Next.** 97.7G resident to serve, four shards; a full fine-tune needs several
times the weights in optimizer state, and even a LoRA needs the base resident with
activations on top. It is not trainable on this machine and would not be worth it if it
were: it is the model whose *good behaviour is being copied*, not the one to change.

**E4B is the target.** `gemma-4-E4B-it-qat` is 5.4G resident at Q4, ~8B parameters raw with
~4B active; it already runs the whole loop at 3.1 s/question, and the product is plainly
useful: **a fine-tuned E4B that calls tools well enough to route for a larger model** — E4B
decides what to look up and what to light, and Flash-Next (or nothing at all) writes prose.
That splits the 25.4 s/question question in two and puts the expensive model on the part it
is actually better at.

**The gap: this repository has no LoRA.** The `tool-calls` recipe is a full fine-tune
through `transformers`, which is right for `functiongemma-270m-it` (the default base, and
the path that works end to end today) and wrong for E4B: bf16 weights, gradients and Adam
moments for 8B parameters is ~128G of state before activations. What is needed is one of

- **peft LoRA over the existing recipe** — rank 16 on the attention and MLP projections,
  base held in bf16 or 4-bit, ~40M trainable parameters. Smallest change to what exists.
- **MLX-LM's LoRA** — native on this machine, but a second training path beside
  `ml_stack.train`, and the checkpoint/resume guarantees this repository is built on would
  have to be rebuilt around it.

Either ends with a merge before `ml_stack.gguf.export` (llama.cpp can also serve a GGUF
adapter with `--lora`, which is worth measuring separately: it makes A/B on one load
possible).

### What it would cost

Estimates. Nothing here has been run, and the way to replace each with a measurement is on
the right.

| | estimate | how to actually know |
| --- | --- | --- |
| data | 3–5k turns from 2–3 traced hundred-question runs | `from-bench --dry-run` after the runs |
| GPU to gather | ~3 runs × 100 questions × 3–25 s ≈ 0.5–2 h | `ml-stack-bench estimate` before each |
| LoRA, E4B, 1–2 epochs | ~2–5k steps at batch 4, ctx 4096 | `ml-stack-train-run --dry-run` trains 20 steps and writes nothing — that is s/step, measured |
| wall clock to train | 1–4 h on this machine | the above × steps |
| export + serve | minutes | — |
| the deciding bench | 2 × 100 questions ≈ 20 min | `ml-stack-bench estimate`, and the ceiling refuses over it |

The GPU is busy with sweeps; a training run and a bench must not share it. `ml-stack-bench`
already takes the measuring lock — a training run must take it too, or be scheduled after.

### What is unmeasured

- **Whether traces from a 70% model transfer to a 4B one at all.** A call Flash-Next makes
  after reading the whole graph may depend on reasoning E4B cannot do, in which case E4B
  learns to imitate the *form* and get the ids wrong — which is worse than not calling.
- **Whether self-distillation moves anything.** Training E4B on E4B's own successes may
  only sharpen what it already does; 183 questions is not much to sharpen with.
- **Whether it survives a graph it has not seen.** Every trace so far is over the invented
  community. A caller that has memorised its ids is worthless. The world generator
  (`ml-stack-world`) can build a second community to test exactly this, and should.
- **Whether the fine-tune breaks the prose.** Loss on assistant tokens teaches the answer
  turn as well as the call turns. `answer_chars` and a read of a few answers, not just F1.
- **Cost per step on this hardware.** MPS with `transformers` is not fast, and nobody here
  has measured a step of an 8B LoRA on it.
- **Whether a 270m caller is enough.** `functiongemma-270m-it` fine-tuned on the same data
  is hours cheaper and might route acceptably. Untested, and worth testing first because it
  is the cheap experiment that would make the expensive one unnecessary.

## Three telemetries, and the record type they want to be

Nothing was changed here; this is the reading of what already exists, because the three
were built for three questions and now overlap.

| | grain | where it lives | what it answers |
| --- | --- | --- | --- |
| `client.Spent` | **one answer** — every call totalled | rides with the answer to the page (`Spent.public`, `Spent.totals` for a session) | what did answering this cost, and how fast did it feel |
| bench `Row.trace` | **one call** | `Row.trace`, kept in the bench store beside the totals | what did it *do*, call by call — and what can be learned from it |
| `serve.fit` | **one model** | `ml_stack/data/fit.json`, layered with `~/.ml-stack/fit.json` | how many people fit on this machine at this context |

They line up cleanly, and the arithmetic agrees today by construction rather than by
sharing code: `Spent.note` and `Counting._reply` read the *same* fields off the same reply
(`raw.model`, `usage.prompt_tokens`, `usage.completion_tokens`,
`timings.{prompt_ms,predicted_ms,prompt_n,cache_n,predicted_n,draft_n,draft_n_accepted}`,
`finish_reason`, `thinking`, `content`, `tool_calls`) and add them up two different ways.
Two readers of one wire format is one reader too many: the day llama.cpp renames a timing,
one of them silently reports zero.

The record type that would fix it:

```python
@dataclass
class Call:                    # one round trip, as the server reported it
    model: str; seconds: float
    prompt_tokens: int; completion_tokens: int
    read_tokens: int; cached_tokens: int; predicted_tokens: int
    prompt_ms: float; predicted_ms: float
    draft_tokens: int; draft_taken: int
    finish: str; thinking_chars: int; answer_chars: int
    tool: str = ""; args: dict = ...; result_chars: int = 0; result_ids: int = 0

    @classmethod
    def of(cls, reply, took) -> "Call": ...     # the one reader of the wire format
    @property
    def held(self) -> int: ...                  # cached + read + predicted: what fit
```

Then `Spent` becomes `list[Call]` with the existing properties as sums over it — `context_peak`
is `max(c.held)`, `acceptance` is `sum(draft_taken)/sum(draft_tokens)` — and the bench trace
becomes `list[Call]` plus the messages between them. `fit` stays where it is: it is not a
record of what happened but the two constants (`per_token`, `per_seq`) that say what `held`
*costs*, so `Call.held × per_token + per_seq` is the memory a call actually took, which is
the join none of the three can make today.

Worth doing when the three next need touching at once — not before, and not while a fleet
sweep is mid-flight over the current shape.
