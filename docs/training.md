# Training

`Trainer` runs the loop on PyTorch or MLX — the framework is taken from the model, so
the same call works on a Mac and on a CUDA box:

```python
from ml_stack.train import Trainer, warmup_cosine

report = Trainer(model, optimizer, loss, out="runs/small").fit(
    batches, steps=100_000,
    schedule=warmup_cosine(3e-4, total_steps=100_000, warmup_steps=2_000),
    eval_data=holdout, eval_every=1_000, checkpoint_every=1_000)
```

Checkpoints are atomic and resumes are exact — weights *and* optimizer state, or it
refuses. `steps` is a total, so re-running the same call after a crash finishes the run
rather than doubling it. A run that goes non-finite is stopped before it writes a
checkpoint of a model that is already NaN.

Every piece is usable on its own for a loop you write yourself: `CheckpointState`,
`MetricsLog`, `RunLock`, the schedules, the guards, the leak-safe splits.

Or skip the code entirely and use a recipe:

```
ml-stack-train-run --recipe text-lm --data corpus.jsonl --out runs/lm --dry-run
```

`--dry-run` trains twenty steps and writes nothing, so a bad setting costs forty seconds
instead of six hours.

## Fine-tuning a tool caller

```
ml-stack-train-tools --tools python:ml_stack.graph.ask:TOOLS \
    --prompts python:ml_stack.graph.ask:TOOL_PROMPTS --out runs/caller
```

Plug in a project's tools and end with a GGUF that calls them. One command, three stages,
each skipped when `--out` already holds its output:

- **synth** writes `data/train.jsonl`, `data/holdout.jsonl` and `data/manifest.json`. The
  seed is the worked examples the tool descriptions already carry — `for "Which companies
  do people here work for?" call list_kind with {"kind": "org"}`, or `"What does Quenlow
  Robotics do?" → web_search(query="Quenlow Robotics")` — because those are what was
  measured to matter (17% to 70% recall on the same weights -- [Measuring](bench.md)).
  The router's example
  questions come in through `--prompts` (a `chat` key is the messages that want no tool),
  and all of it is templated into conversations: system, user, an assistant turn that
  calls the tool — and a share that call nothing, so the model learns when not to.
  Arguments come from the question where they can (its words, a URL, an enum value it
  names) and from a worked example where they cannot, so an id-shaped argument is always
  one a description showed. One seed question in ten is held out by hash, and every
  paraphrase of it goes with it. `--ask URL` has a served model write more questions per
  tool with the examples as few-shots, which is also where better arguments come from;
  `--per-tool` is how many conversations each tool gets.
- **train** runs the `tool-calls` recipe: `--base` (`google/functiongemma-270m-it` unless
  told otherwise) with every conversation rendered through its own chat template and the
  loss on the assistant tokens only, into `run/` — checkpoints, `metrics.jsonl`, resumable.
  `--set steps=600` and the other recipe fields work as in `ml-stack-train-run`. It is
  torch whatever the machine's default backend is, on the accelerator unless
  `ML_STACK_DEVICE=cpu` says otherwise.
- **export** puts the latest checkpoint back into Hugging Face layout under `model/` and
  hands it to `ml_stack.gguf.export` — llama.cpp's converter and `llama-quantize`, `--quant
  Q8_0` — so the GGUF lands in `--out`, ready for `ml-stack-serve up`.

`--dry-run` prints the plan with counts and loads no model; `--only synth|train|export` runs
one stage. The data is plain JSONL rows of `{"messages", "tools"}`, so `ml-stack-train-run
--recipe tool-calls --data runs/caller/data` trains on it too, and so does anything else
that writes that shape. Whether the fine-tune beats its base is measured, never assumed:
serve the GGUF and `ml-stack-bench run` it beside the model it came from.

### From what a model actually did

```
ml-stack-train-tools from-bench --kept ~/.ml-stack/bench/runs.ladybug \
    --model e4b --min-f1 0.8 --out runs/caller/data
```

The descriptions teach the *shape* of a call. A benchmark's traces teach the calls that
scored — on a real graph, with real ids, which no description can supply. Every question a
run kept a transcript for that scored at least `--min-f1` becomes one training example per
model turn: the conversation up to that turn as the input, the call the model made as the
target. A question of four calls is four examples, each a decision made with strictly more
evidence than the last.

The rows are the shape `synth` writes, so both sources mix in one directory: `--out
FILE.jsonl` writes one file, `--out DIR` writes `train.jsonl`, `holdout.jsonl` and a
manifest. `--model` is a substring of the run's label or of the served model's file, because
two models' turns in one dataset teach the average of two callers. One question in ten is
held out by hash, with every turn of it. A turn the ceiling cut off is dropped — a truncated
call is the one thing a tool caller must never learn — and each example carries only the
tools that were offered on that call, since `graph.ask` takes tools away as a question goes
on.

Runs are traced by default when 20 questions or fewer are asked, and not on the hundred,
where the transcripts would be tens of megabytes in a store nothing backs up;
`MLSTACK_BENCH_TRACE=1` traces a run of any size, `=0` traces none. A trace holds, per call,
the tool and its arguments, how much came back and how many ids were in it, and the timings
`Spent` reads — so the per-call record and the per-answer totals are one measurement added
up two ways. `from-bench --dry-run` says what a store would yield, and what it *would have*
yielded had it been traced: for a store filled before tracing existed the answer is zero,
and zero means nothing without the number beside it (2026-09-02: 751 scored questions, 4006
model turns, none of them kept).

### A model too big to fine-tune whole

```
ml-stack-train-run --recipe tool-calls --size e4b --lora --export-gguf \
    --data runs/caller/data --out runs/caller --set steps=1000 --yes
```

A full fine-tune of an 8B model needs about 128G of optimizer state; `--lora` trains two
small matrices on each attention and MLP projection instead — ~40M parameters with the base
frozen in bf16, ~19G resident for gemma-4 E4B on this machine. `--size e4b` brings the
defaults that suit it (batch 4, context 2048, 1e-4, rank 16), `--lora-rank`,
`--lora-alpha`, `--lora-dropout` and `--lora-targets` override them, and the checkpoints
hold the adapter rather than a copy of the frozen base. Needs peft: `pip install
'ml-stack[train-lora]'`.

What the run will cost is printed before a weight is loaded — parameters, resident
gigabytes, tokens a step, seconds a step, wall clock — and a run estimated past 30 minutes
is refused with exit 5 unless `--yes`, the same ceiling and the same code the bench uses.
`--dry-run` trains 20 real steps, writes nothing, and replaces the estimate with a measured
seconds-per-step. `--export-gguf` then merges the adapter into the base, converts through
the managed llama.cpp checkout the served binary was built from, quantises, and preflights
the file before anyone waits on a load; `manifest.json` records the training data's hash and
example count, so what a fine-tune learned from can be identified afterwards.

A real fine-tune is hours, and a run started with `&` or `nohup` dies with the shell that
started it, so `--detach` re-runs the command in its own session with its output in a log
under `~/.ml-stack/train/logs` and hands the shell straight back. The pid, the argv and the
log are recorded as the `train` job the same way the bench and the ingest record theirs, so
`ml-stack-train-run status` says what is running, `wait` blocks until it has ended — the
next command is `wait && next` rather than a loop written by hand — and `stop` ends it. One
at a time: a second `--detach` beside a run still going is refused, because the two would
share one GPU and neither measurement would be worth having.

`docs/research/tool-caller-finetune.md` is the plan this is the first half of — what to
train, on whose traces, what it would cost, and what is unmeasured.

The next recipe is embeddinggemma for `graph.route`: a contrastive fine-tune on the same
question → tool pairs, so the router that chooses which tools to offer learns the project's
questions as well. It is not built yet.

