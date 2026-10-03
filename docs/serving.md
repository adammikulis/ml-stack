# Finding and serving a model

## Finding a model

The Hub has models newer than anything written down here, and newer than anything an assistant
was trained on. Look rather than remember:

```
$ ml-stack-models find gemma-4 E4B
    588135  unsloth/gemma-4-E4B-it-qat-GGUF
    563542  unsloth/gemma-4-E4B-it-GGUF
    307153  ggml-org/gemma-4-E4B-it-GGUF
$ ml-stack-models files unsloth/gemma-4-E4B-it-qat-GGUF
    4.1G  hf:unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-Q4_K_M.gguf
```

Publishers in `PREFER` are ranked first — the Hub's own ordering puts whatever is popular
at the top, which for a model released last week is somebody's remix rather than the
release. The printed reference is what `ml-stack-serve up` takes; llama-server downloads and
caches it on first use, so there is no separate fetching step.

## Serving a model

What makes this part worth having is the lifecycle, not the launcher. Every model server on
a machine goes through one manager: it is written down *before* the process exists (the
record carries the port, the model and the owner; the pid is filled in when the server
answers, and a start that fails is forgotten), one set of settings is served per port and a
lease that asks for others is refused with the field that differs named, a server already
serving what was asked for is adopted rather than started again, a server the record does
not know is reported as somebody else's and never killed, and the backend launches nothing
without the manager's lease in hand -- so an untracked server cannot come out of the
library at all. What each model scored best with (`ml-stack-serve profile`) is what a
lease is built from, so serving and asking use the numbers that were measured rather than
remembered. The commands below are the surface of that.

From a shell:

```
ml-stack-serve up model.gguf --context 32768
ml-stack-serve up hf:unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-Q4_K_M.gguf
ml-stack-serve status
ml-stack-serve down
```

`status` prints the port, the model, the context each slot gets, how many slots there are
and which process holds the lease. `--json` gives a script the same, and it exits non-zero
when nothing is serving. `up` adopts a server already serving that model with those settings
instead of starting a second one, and prints the base URL. `down` stops only a server
started on this machine.

From Python:

```python
from ml_stack.serve import serve
from ml_stack.client import Client

with serve("model.gguf", port=8899) as server:
    client = Client(server.base_url)
    client.assert_grammar_support()          # fail now if constrained decoding is broken
    print(client.chat([{"role": "user", "content": "hello"}]).content)
```

`serve` adopts a healthy server that is already running rather than starting a second one,
and leaves an adopted server alone on exit. It only stops what it started.

**Embedding it in another program.** Importing `ml_stack.serve`, `ml_stack.client` or
`ml_stack.fleet` prints nothing, opens no socket and writes no file. State goes under
`ML_STACK_HOME` (default `~/.ml-stack`) and the cache under `ML_STACK_CACHE`; set both
before the first call to keep everything inside a directory the program owns. `ServerManager`
reports through the `ml_stack.serve` loggers, or through the `say=` callback `lease` takes. A
`ServerManager` is a context manager: `with ServerManager() as manager:` stops every server
it started when the block ends and leaves adopted ones running, and `manager.close()` does
the same. A manager is safe
to call from several threads (one lock per port).

A server a `ServerManager` starts stops when the program that started it ends: at exit, on
SIGTERM, SIGINT and SIGHUP (the handler a program already had still runs first), and, when the
program is killed outright, through a small watchdog process that stops the server's tree once
its host is gone. Nothing is registered at import; the first server a process starts installs
the exit hook and the handlers. `ServerManager(stop_on_exit=False)` leaves the servers running,
and `ml-stack-serve up` does that itself, since it exits and the server is meant to stay. The
first lease a manager makes also stops every server on the machine whose leasing process has
gone, but only one whose record proves the pid is still that server (its start time and a
digest of its command line). Server logs under `ML_STACK_HOME/logs` are kept to 60 files, 256 MB
and 30 days across every port (`ML_STACK_LOG_FILES`, `ML_STACK_LOG_MB`, `ML_STACK_LOG_DAYS`;
0 is no limit). A server is started without this process's tokens and keys in its
environment; one that downloads weights is given the Hugging Face token the credentials
resolve to and nothing else (`docs/credentials.md`).

**One serving per port, written down once.** llama.cpp serves a model one way at a time, so
two parts of a program that lease it differently are not two clients of one server:
whichever leases second finds a mismatch, stops the first and loads the weights again. A
`Serving` is every server setting in one object and `Serving.lease()` is the only place it becomes
`serve`'s arguments, so the two cannot drift apart. `slot` starts the server on the first
ask, holds it per port for the process, and hands each caller a `Client` pinned to a slot of
its own -- so several conversations at once do not reprocess each other's context.

```python
from dataclasses import replace

from ml_stack.serve import Serving, slot, draft_for, projector_for

model = "hf:unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
serving = Serving(model=model, port=8080, slot_context=131072, cache_type="q8_0",
                  draft=draft_for(model, "auto"),      # the head shipped beside the weights
                  draft_n_max=4, reasoning_budget=0,   # measured, not remembered
                  mmproj=projector_for(model, "auto"), # so the model can see
                  build="unsloth")                     # a head mainline will not load

client = slot(serving, index=request_number, n_predict=16384)

crowded = replace(serving, slots=4, slot_context=32768)   # four conversations at once
```

A serving holds one slot unless it is asked for more, and that slot gets the whole context.
`slots=N` divides the same memory between N conversations, each with its own KV cache. A
lone slot is given the model's own trained window when the room allows it, read off the
GGUF header; a context asked for past the trained window turns on YaRN position scaling by
itself and says so, because past the trained length the positions are the part that goes
wrong first. `ml-stack-serve escalate --port 8080 --add 2` grows a running server's slots
in place and carries its conversations across, so the next person does not cost the ones
already talking a cold reload.

`draft_for` and `projector_for` answer 'auto' the way `ml-stack-serve up` does -- a lease
built by hand has to resolve what the CLI resolves for itself -- and each says out loud why
it found nothing rather than serving undrafted or blind in silence. `release_all()` lets go
of every held server; `on_disk()` says which ports are up.

A port already serving something else is refused, with the field that differs named —
the model, the number of slots, or the context each slot gets. Adopting a server started with
the wrong settings hands back a lease that cannot do what was asked of it.

### IQ quantisations on Apple silicon

A lease for an IQ-family GGUF (IQ1_S, IQ1_M, IQ2_XXS/XS/S/M, IQ3_XXS/XS/S/M, IQ4_NL, IQ4_XS)
on a Mac with an arm64 CPU goes ahead with one warning per process: it MAY be slower and less
accurate than a K-quant of the same model on Metal, the evidence is thin, and the K-quant
builds of the same model found on disk are listed. Each such lease also emits a sentinel event
`serve.iq_warning` (model, quant, who) and marks its lease record, and `ml-stack-serve status`
shows a `WARNING` line for the server while it runs. Linux, Windows, CPU-only leases
(`n_gpu_layers` 0) and engines other than llama.cpp are never affected.

`ML_STACK_IQ=block|warn|off` (default `warn`) and `ml-stack-serve up --iq block|warn|off`
set the mode, and `iq=` on `ServerManager.lease` and `serve` does the same for one lease.
`block` refuses with `BlockedQuant` (`up` exits 3); `off` gives no warning and records nothing.

**Evidence, and its limits.** Qwen3.8-Flash-Next answering `plain` with thinking on and no
draft head on Metal (`docs/architectures/qwen4exp.md`, `docs/report-2026-09-23.md`):
`UD-IQ4_XS` took 70.1 s a question at 54% F1 over ten questions, `UD-Q4_K_XL` 43.7 s at 64%
over nine. That is one model, one machine, one run each. The same report points the other
way or shows the noise: the identical `UD-Q4_K_XL` asking also ran 27.6 s at 81% F1 over nine
questions; `UD-IQ4_XS` `plain` thinking on over 34 questions ran 40.0 s at 59% (65.1 s at 55%
in another run); `UD-IQ4_XS` `plain+tight` with thinking off scored 85% F1 at 36.6 s over nine
questions, the best F1 the K-quant reached too. Run-to-run spread on one configuration is
larger than the gap in the pair. The IQ4_XS build is 87.6G against 104.0G. No other
IQ against K-quant speed measurement exists in `docs/model-ranking.md`, `docs/fit.md` or the
bench store code. `docs/experiments/iq-vs-kquant-metal.md` is the pre-registered protocol
that settles it.

**How a file is judged IQ** (`ml_stack.serve.quant_guard.iq_quant`, header only): its
`general.file_type` is an IQ type, or IQ tensors hold more than half of the weight bytes over
all shards. One IQ tensor does not make a file IQ: an unsloth `UD-Q4_K_XL` carries an IQ4_NL
lookup table of about a quarter of its bytes. The file name decides only for a model that is
not a file here yet (`hf:` references). The check runs in `ServerManager._start_server`, which
every lease passes (`lease`, `serve`, `up`, the Broker's `start` and ask paths, bench, fleet,
ingest, the guard judge).

**The mode is the person's.** No tool, request or model can change it. `serve_up` (MCP and
chat) rejects an `--iq` word in `extra`, reports a strict refusal instead of starting a
process, and `up` takes no abbreviation of its flags. The Broker wire drops an `iq` option, so
a broker uses its own `ML_STACK_IQ`; a client's strict mode is applied in the client before the
call, and a client's `off` or `warn` never reaches a broker. A spec or ask carrying `iq` is
refused. `suggest`, `recommend` and the chat default rank an IQ build after an otherwise equal
build on Apple silicon (a tie-break, never an exclusion) and its note says it may be slower on
Metal.

### Thinking, per use

`ml_stack.client.thinking` decides whether a request asks the model to think, from the
person's `ML_STACK_THINK=off|on|auto` (default `auto`). Decisions (the decide, judge and guard
logprob paths) never think. Agent turns and short answers think only when the person sets
`on`; under `auto` a prompt thinks only when it contains a phrase that asks for reasoning
("think step by step", "show your reasoning") or the use is `reasoning`. The agent loop and
the logprob decider send the family's template flag (`enable_thinking` for Qwen and Gemma,
`reasoning_effort` for gpt-oss) accordingly, and `ml-stack-serve up` prints the policy.
`ml-stack-chat` sends `think=False` on every turn itself; a `/think` command and `--think`
there, and the policy's header line, are not wired (chat.py was out of bounds for this
change). A bench run's thinking is recorded as its thinking column (`--reasoning-budget`).

### The settings a model scored best with, for one kind of work

The `Serving` above was typed out by hand, and every value in it came from a bench run
somebody remembered. A **profile** is those settings written down instead: one record per model
file **and workload** of the serving and the asking that measured best, and the row of the
store that set it. `ml_stack/data/profiles.json` ships them and `~/.ml-stack/profiles.json`
(`$MLSTACK_PROFILES_FILE`) layers this machine's own over them, exactly as `fit.json` does.

There are three workloads, because the best settings depend on what the model is doing:
`ask` (tool-calling over a graph), `ingest` (documents into JSON under a schema) and `chat`
(prose, under no schema). A tool call is mostly JSON skeleton and repeated key names, so a
draft head guesses it right most of the time; free prose it guesses wrong, and every wrong
guess costs a verification pass. `--for` says which record is wanted, and the graph asking
is what it means when nothing is said.

```
ml-stack-serve profile
ml-stack-serve profile Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
ml-stack-serve profile Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --for ingest
ml-stack-serve up model.gguf --profile --for ingest
ml-stack-bench report --profile
```

```
Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
  for         ask -- tool-calling over a graph
  serve with  --context 32768 --build unsloth --draft mtp-…-shared-Q8_0.gguf
              --spec draft-mtp --spec-n-max 4 --kv q8_0 --mmproj auto --reasoning-budget 0
              and -ub 2048 --spec-draft-p-min 0.5 -- llama-server's own, passed by --profile
  measured at --parallel 2, 16384 per slot
  per request draft 4 ahead, greedy -- sent with each call, where the build takes it
  ask with    tight + batch + kinds + summary + greedy
  measured    80% F1 (89% recall, 77% precision) at 26.7 s/question over 100 question(s)
              2026-09-02, Mac, from `Qwen3.8-Flash--all-plain-kv-q8_0-rb0`
```

A record has a **startup half** and a **request half**. The build, the draft head file, the
cache type, the context and the slots are what a server is told once; the draft depth, its
confidence floor and the sampling ride on each call, so one served model can guess four
tokens ahead for a tool call and two for a document. A build without llama.cpp's
per-request speculative override ignores those fields and serves the depth it was started
with, which is the one on the `serve with` line; a server that refuses them is asked again
without them and says so.

Ask for a workload nothing has measured and the graph-asking record is served, with a
`note` saying which workload measured it and which one it was asked for — never silently.

Another record in the same file is asked nothing like it, which is the point. Where a model
measured better on a short offer, more turns and its publisher's own sampling, its `ask with`
line reads:

```
  ask with    tight + few + rounds 20 at temperature 1.0 / top-p 0.95 / top-k 20
```

Because no two models want the same settings. Flash-Next answers well only on a fork build,
with the shared MTP head at four, a q8_0 cache, its thinking off, `-ub 2048`,
`--spec-draft-p-min 0.5` and three ways of asking at once; gemma-4 wants its thinking left
on, a different head at two, and none of those flags. Written as defaults each would be
wrong for the other. Written per model they are what they are — and each record names the
row of the store that measured it, on which machine and on what date, so a person can tell
a measurement from a habit.

From Python, both ends read the same record:

```python
from ml_stack.serve import profile_for, slot
from ml_stack.graph.conversation import converse

found = profile_for("hf:unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf",
                    workload="ask")
config = found.config(port=8080, n_predict=16384)
client = slot(config, index=request_number)
answer = converse(question, graph, client, asking=config.asking)
```

`Profile.config()` is a **`Config`**: everything in one object, in three sections that
different code reads. `config.serving` is the `Serving` the server is leased with,
`config.asking` is an `Asking` — how `converse` is called — and `config.talking` is a
`Talking`, what the `Client` is built from. `config.lease()`, `config.asking` and
`config.client()` are the only places each becomes arguments, and
`config.over(cache_type="f16", few=True, temperature=0.7)` lays a knob over it, routed to
the section that owns it rather than to whichever call takes `**kwargs` next.

Hand the same config to the bench (`bench.served(config, ...)`), to a page
(`AskRoutes.config`, and `client_on_slot()` hands out a slot of it) and to `slot` and they
lease one serving and ask one way by construction. Three places each building their own from the record is how a knob about the
asking reached `Client.__init__` and took an 87G load down with it, and how two of them
could lease one port two ways — which llama.cpp answers by stopping the server and loading
the weights again.

`Asking.for_model(name, workload=...)` is the way that model measured best at that work, and
`Profile.asked()` is the same record's. A model matched only by family (the same weights at
another quantisation) comes back with `note` saying so. A rewrite from a run that predates
asking records keeps the asking of the record for the same file, never another
quantisation's.

Nothing writes a record by hand. `ml-stack-bench report --profile` takes, per model **and
workload**, **the fastest row whose F1 the questions could not tell apart from the best** — among that model's
longest runs, held is `score.held_up`, the two 95% bands overlapping — and writes the build,
head, cache, thinking, context, asking and sampling it was served and asked with, saying in
the record which row it was. Best F1 alone would trade real seconds for a hundredth of a
point the questions cannot see; ranking *models* is still F1, because that is a different
question. The two things a kept run cannot see (llama-server's own extra flags and the
vision projector) are carried from the record already there rather than erased.

The asking a record carries is the whole asking: `tight`, `batch`, `single`, `few`, `kinds`,
`summary`, `rich`, `terse`, `reach` and `rounds`, plus the sampling — so a model measured on
three tools, twenty rounds and its publisher's temperature is served and asked exactly that,
while the next model in the same file is asked the opposite. One asking per model and
workload, and every one of them a number somebody paid for.

**Every load preflights first.** Before a process starts, `LlamaServerBackend.start` checks
that every shard of the GGUF is present and complete (an `hf:` reference is resolved through
the Hub cache the way `ml-stack-models files` reports what is already on this machine), that
`general.architecture` is one this build reads, that the weights plus an estimated KV cache
fit what `ml-stack-setup` says this machine may use, and that every flag the spec would emit
is one the build accepts — one fast read of a GGUF's own header, never the tensors, so a
fault that used to surface at the far end of an 87G load surfaces before anything is
spawned. `ml-stack-serve up --preflight-only` runs the same report and exits 0 or 1 without
starting or adopting anything; `ml-stack-models fetch hf:owner/repo/file.gguf` downloads
every shard of a build into the same cache ahead of time, so a benchmark's timed window never
pays for the download. A lease also records `load_s` (and `warmup_s`, from one short
completion sent right after the health check, so the first *measured* question is not the
one paying for shader compilation) — both show up in `ml-stack-serve status --json`, and the
load timeout itself scales with the weights on disk (`60s + 1.5s/GB`, floor 300s) rather than
racing a fixed clock against whichever model is biggest.

**How many people fit** is a measured number, not an estimated one. The preflight's KV
estimate reads the GGUF header, and the header does not say enough: gemma4 slides a
512-token window on some layers and shares one cache across its last eighteen, gpt-oss
slides a 128-token window on every other layer, and Qwen3.8-Flash-Next keeps a token cache
on one layer in four and a fixed state per *sequence* on the other three. Counting every
layer as full attention is wrong by a different multiple for each of them. llama.cpp prints
exactly what it allocated at load, and that is what is recorded:

```
ml-stack-serve fit model.gguf --measure --context 32768
ml-stack-serve fit --room 24G --per-user 8192 --per-user 65536
ml-stack-serve fit --room 110G --room 24G --plot docs/fit.png --write docs/fit.md
```

`--measure` serves the model once with `-lv 4` (the library's own load lines are
LOG_LEVEL_TRACE, so the server's default verbosity of 3 prints none of them), reads the
`llama_kv_cache`, `llama_memory_recurrent` and compute-buffer lines out of the log the
backend already writes, stops the server, and keeps two numbers: **bytes per token of
context** and **bytes fixed per sequence**. Those compose in both directions -- how many
users fit at a context, and the longest context a given number of users can each have.

`--plot FILE.png` (or `.svg`, `.pdf`) draws two panels over the same records. The first is
how many users fit against the context each one gets. The second is the one worth having:
memory against users at one context, each line starting at zero users -- where its height is the
model sitting there with an empty cache -- and climbing by exactly one user's worth of cache
per step. That is what makes a heavy model with a cheap cache comparable to a light one with
an expensive cache: Qwen3.8-Flash-Next is large and its KV is tiny, so it starts high and
barely rises, and the picture shows where it overtakes a small model whose cache costs eight
times as much a token. Familiar card sizes (6, 8, 12, 16, 24, 32, 48, 64, 96, 128 GB) are
drawn faintly behind, and each `--room` in force is drawn across it, so the chart answers
"what would I need" as well as "does it fit here". `--room` is repeatable -- the first is
solid, the second dashed. `--at N` sets the context the second panel charges at (default
32768). The legend carries each model's arithmetic in full: `87.2G + 0.14G/user at 32k`,
which is `Fit.line(context)`, the pair of numbers every line is drawn from. Drawing needs
matplotlib (`pip install 'ml-stack[plot]'`); nothing else here does, and `ml-stack-bench
show --plot` deliberately still writes hand-built SVG with no library so it opens on a
machine with no packages. With `--write` alongside, the page embeds the picture beside it.

`--ui` puts the same two panels up as a page you can move, which the picture cannot be:
a room slider (this machine's, the familiar card sizes, or a number you type), a per-user
context slider from 1k to 256k, a users slider, a checkbox per measured model, a cache-type
select and a drafted toggle where both records exist -- and every drag redraws both panels
and a table saying, per model, what it costs loaded, what a user costs at that context, how
many fit and the longest context the chosen number could each be given. The second panel's
x range follows the models on screen and takes a drag or a wheel to zoom, because the static
picture's did not: one 2B model that fits 250 people flattened every other line in it.
Hovering either panel lights the model's row and says its exact numbers at the cursor. It
serves on loopback and opens a browser at it:

```
ml-stack-serve fit --ui
```

The fleet app shows the same view under **Fit**, at `/ui/fit`, from the same two routes --
`/ui/fit.json` hands over the records worked out for the room and the head count the sliders
stand at, so every number on the screen is the one `ml-stack-serve fit` prints. A sibling
tab, **What it cost to be
right**, draws `ml-stack-bench show --rates` the same way: accuracy against wall clock,
tokens paid for, or KV and runtime, with the Pareto frontier joined -- nothing on it is both
more accurate and cheaper, so choosing among those points is choosing a budget. Both pages
are hand-drawn SVG with no library and no CDN, so they open on a machine that has never been
online.

The records are the single source of truth, in `src/ml_stack/data/fit.json`, keyed by the
model file's basename, the cache type and the guessing-ahead kind; `~/.ml-stack/fit.json`
layers a machine's own measurements over the shipped ones, and `--measure` says which of the
two it wrote to. Without `--measure`, `fit` only reads -- nothing is served and no GPU is
touched.

**Guessing ahead** comes in two shapes and `--spec TYPE` chooses. A *draft* kind runs a
second small model — `--draft auto` finds the `mtp-` head a repository ships, wherever the
publisher put it, and `--draft-ngl` decides how much of it goes on the GPU (without which it
may run on the CPU, and a draft slower than the model it guesses for is a loss). An *n-gram*
kind runs no second model at all: it proposes tokens by looking up sequences already in the
prompt, which suits work that copies from its context and costs no weights and no memory.

Where the n-gram table lives depends on the kind. `ngram-simple`, `ngram-map-k`,
`ngram-map-k4v` and `ngram-mod` keep none — the lookup is over tokens already in memory,
and nothing touches the disk. `ngram-cache` is the exception: `--lookup-cache` is written as
it generates, so what was learnt answering one question can speculate the next.

### Multi-token prediction

A model is served with multi-token prediction (`--spec-type draft-mtp`) whenever it has a
prediction layer and the build can use it, through `ServerManager.lease`, `serve()`, the
Broker and `ml-stack-serve up` alike. The lease says what it did: `ServerInfo.mtp` is
`embedded`, a head's file name, or empty, `ServerInfo.mtp_note` is the one-line reason, the
lease record keeps both, and `ml-stack-serve status` prints the head with the share of
drafted tokens the model kept (read from the server's `/metrics`).

Where the layer comes from, in order:

1. **The weights carry it.** Tensors named `blk.N.nextn.*` in the GGUF (Qwen3.8-27B,
   Qwen3.5-2B-MTP): `--spec-type draft-mtp` and no second file.
2. **A head file beside the weights.** `hub.heads_for` lists the `mtp-` files in the model's
   own Hub repository (any snapshot, `MTP/` folder) or the same folder on disk; a head that
   borrows the target's embeddings (`shared`) counts only when the serving build is a fork
   that loads one. Among those, the `Q8_0` head, else the smallest, whose header passes
   `serve.mtp.mismatch`: same architecture, embedding width, block count (or one more),
   tokenizer model and prefix, vocabulary size, and a prediction layer in the head.
3. **Trust.** A head from any other repository is not used unless sentinel holds a pin a
   download made for it (`source` other than `first-use`). A head the default picks is
   checked against its sentinel pin before the weights load; a mismatch quarantines it as it
   does a model, and the lease goes on without it. A head named with `draft=` is checked the
   same way and a mismatch refuses the lease.

Served without, with the reason in `mtp_note` and the log, never as a failure:

| reason | why |
| --- | --- |
| `ML_STACK_MTP=off` (also `0`, `no`, `false`, `none`) | the machine-wide opt-out |
| `ServerSpec(mtp=False)`, `Serving(mtp=False)`, `up --no-mtp`, `up --draft none`, `up --spec none` | the per-lease opt-out |
| `--spec-type` has no `draft-mtp` in this build's `--help` | the build cannot |
| `draft`, `spec_type` or tree decoding already chosen by the caller | the caller's choice stands |
| `slot_save_path` set, or a lease that may escalate | slot save writes the target's cache and tokens only (`llama_state_seq_save_file(ctx_tgt, ...)`), so a restored slot would draft from nothing |
| embedding server, MLX engine | nothing to draft |
| the server exits while loading with the head | started again without it for this lease, and not tried again for that model and build in this process |

The guard's judge lease (`guard/native.py`) asks for `mtp=False`: it reads the probabilities
of one token (`max_tokens=1`), so there is nothing to draft, and in the build checked
(`b11380`) tokens accepted from a draft carry no `logprobs` (`TODO: set result.probs` in
`server-context.cpp`), so any scoring that reads more than the first token must not be
served drafted. The bench's no-head arms and `ml-stack-draft`'s `none` arm set `mtp=False` too, so
a baseline is a baseline.

#### What has been measured

Draft depth: a model whose architecture is in `serve.mtp.DEPTH` is started at that depth
unless the lease names one; every other model gets the server's own default (3). The table
is `gemma4: 2` only.

Measured on this repo's own runs (`ml-stack-bench` store, 2026-09-01 to 2026-09-06, Mac;
`docs/report-2026-09-23.md`, "Draft heads, per model", and `docs/llama-cpp-per-request-speculative.md`).
Speed is the head's seconds-per-question against the same model's undrafted run on the same
build; accept is the share of drafted tokens kept. A head cannot change an answer, so the F1
columns of the report are not evidence about speed. The report prints the build only for the
fit records (gemma-4: `3466812`, the per-request-depth patch on b10751; Flash-Next:
`92cedc867`, unsloth `b10715-mix-86bd2d3`); the draft rows carry none, and a `shared` head
loads only on the fork.

| model, head | depth | accept | speed against no head |
| --- | --- | --- | --- |
| Flash-Next UD-IQ4_XS, Q8_0 (shared or not) | n2 / n4 / n8 | 74-79% / 58-59% / 39-42% | 1.13-1.43x / 1.46-1.73x / 0.73-0.95x |
| Flash-Next UD-Q4_K_XL, shared Q8_0 | n2 / n3 / n4 | 86-87% / 79-85% / 72-78% | 1.25-1.51x / 1.24-1.48x / 1.27-1.32x |
| same | n5 / n6 / n7 / n8 | 68-73% / 61-62% / 59% / 54% | 1.29-1.51x / 1.28-1.60x / 1.14-1.74x / 0.99-1.07x |
| gemma-4-E2B, Q4_0 | n2 / n4 | 69-82% / 54-67% | 1.15-1.29x / 0.99-1.30x |
| gemma-4-E2B, per-request depth sweep (tokens/s) | n1 / n2 / n4 / n8 | 76 of 150 kept at n8 | 169.6 / 176.3 / 176.6 / 141.3 |
| gemma-4-E4B, Q4_0 / Q8_0 / BF16 | n2 / n4 / n8 / n16 | 66-71% / 50-57% / 30-37% / 16-20% | 0.99-1.09x / 0.85-0.94x / 0.64-0.75x / 0.56-0.66x |
| gemma-4-26B-A4B | n4 | 65-84% | no undrafted run: no speed |
| Qwen3.8-27B, own layer | | | no drafted run: memory records only |
| gpt-oss-120b / 20b, eagle3 (not MTP) | n2 / n4 | 42-65% | 0.71-0.82x, a loss; never selected by the default |

Reading it: depth 8 and above lost or broke even everywhere it was tried; gemma-4 is best at
2 (E2B flat from 2 to 4, E4B loses from 4 up), which is the one row of
`DEPTH`. Flash-Next has no single best depth in these runs (n4 beats n2 on IQ4_XS, n2 to n7
sit inside each other's spread on Q4_K_XL, the same depth varies 1.14-1.74x between runs),
so it keeps the server's 3, which lies inside the good range. The E4B gain at n2 and the E2B
gain are inside the report's own s/q noise. Head precision on gemma-4-E4B (Q4_0, Q8_0, BF16)
is within 3% at equal accept, and shared against non-shared Flash-Next is within 1%, so the
order "Q8_0, else the smallest" is not contradicted. No run compares an embedded layer with a
detached head.

Published or claimed elsewhere, not measured here (`docs/research/qwen38-flash-next-mtp.md`,
2026-09-01):

| source | what it says |
| --- | --- |
| llama.cpp #27836, M3 Max, Flash-Next IQ4_XS | 27.4 -> 37.2 tok/s (+36%, 89% accept) at n2; +42% (86%) at n3 |
| unsloth fork PR #144, B200, Q4_K_XL | 1.67x, 66% accept; bf16 and Q8_0 heads equal; concurrency 8 is 0.81-0.87x |
| a #144 comment, RTX PRO 6000 | 1.94x with the non-shared Q8_0 head, 8% faster than shared |
| unsloth `MTP/README.md` | 1.3-1.7x at low concurrency, n2, not for concurrent serving |
| #27836 thread, M5 Pro / M5 Max | -12% to +5% at n2-n3; +13-68% with n6 and `p-min 0.7` |
| #27836 thread, Vulkan and dual-GPU CUDA | 2-4x slower despite 85-95% accept, until #28123 (rollback) |

No in-repo run shows a multi-GPU or Vulkan loss, so there is no automatic off for them; the
opt-outs above are the remedy. Concurrency above one slot is likewise published only.

**Flash-Next on the managed build.** The tracked build `b11380` (`eec18f5d3`) contains the
qwen4exp MTP graph (`llama_model_qwen4exp::graph_mtp`) and the `blk.N.nextn.hc_head_*`
tensors, and has no `nextn_shared_target_tensors`, so it can run a head but cannot borrow the
target's embeddings: the default offers it the non-shared `mtp-...-Q8_0.gguf` only. Whether it
loads that head-only file as `-md` is not known without loading it. The one check, on a quiet
machine (loads the 87G model once per arm):

    ml-stack-draft Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf --depth 2 --depth 3 --depth 4

An arm that cannot load prints the server's error; a loading head prints accept and speed
against the no-head arm, which are the first numbers for this build.

**A release lags master by an architecture or two.** Checked on this machine: the newest
homebrew bottle (`brew outdated` empty) reads `gemma4` and `qwen3moe` but not `qwen4exp`, so
Qwen3.8-Flash-Next exits with "unknown model architecture" on it. `ml-stack-serve build`
fixes that permanently rather than once: it clones or fast-forwards llama.cpp's own master
and builds it (`--from source`, Metal on macOS, CUDA or Vulkan on Windows/Linux when a
compiler is on PATH), or downloads the newest GitHub release with an asset for this machine
(`--from release`, the default with no compiler — most Windows installs). Either way the new
binary is trusted only once it answers `--help` and reads every architecture the build it is
about to replace did; only then does `~/.ml-stack/llama.cpp/current` — which `find_binary`
checks ahead of PATH and a login shell's `/opt/homebrew/bin`, though never ahead of
`--binary` or `$LLAMA_CPP_SERVER` — point at it. `ml-stack-serve build --check` reports the
installed build's commit and age without building anything; `--rollback` points `current`
back; `--persist` installs a weekly refresh (a LaunchAgent on macOS, a Scheduled Task on
Windows) that reruns it unattended — safe because a refresh that fails verification changes
nothing. `--adopt DIR` registers a flat build that already exists — a hand-built binary, or
a release someone unpacked by hand — as a managed build through the same verification,
without compiling or downloading anything; a compile already running keeps going regardless,
and only switches `current` itself once it goes on to verify. `ml-stack-setup` names the fix
directly when an architecture or a flag is missing. A one-off binary from somewhere else
still works: `ml-stack-serve up --binary /path/to/llama-server`.

`ml-stack-serve llama-cpp status|update|rollback|pin|list|prune` is the flow that follows upstream
with a pinned commit, a sandboxed compile, a smoke test before a build is trusted, and a pin
sentinel verifies every time the binary is found; see [llama-cpp-tracking.md](llama-cpp-tracking.md).

Verifying by architecture *name* is only as precise as the name: measured for real, a
build's `libllama` read `phi4` and looked exactly like a missing architecture next to a
build that had it — but master's own `src/llama-arch.cpp` defines no `LLM_ARCH_PHI4` at
all; `phi4` names a chat template (`llama-chat.cpp`), not a model architecture, and Phi-4
loads through the `phi3` architecture regardless. With a source checkout to read the real
names from, the comparison is restricted to them; without one, it falls back to guessing by
family prefix, the same as before.

**A release also renames flags**, and a flag the build does not have fails at the far end
of the load: `--draft-max` became `--spec-draft-n-max`, and llama.cpp 0.3.0 keeps the old
name only to say it was removed. So `up` asks the build what it accepts (`flags_of` reads
`--help`, cached per binary and mtime) and refuses before loading, one line per flag with
the nearest the build has — `this llama-server has no --draft-max; it has
--spec-draft-n-max`. `ml-stack-setup` lists every flag `ServerSpec` can emit that the
installed build lacks, and offers `ml-stack-serve build` as the fix; it says nothing when
the build answers them all. A build that prints no help is unknown, not empty, and is given
no opinion.

`ServerSpec(draft=...)` serves a small model of the same family alongside the large one: it
guesses several tokens ahead and the large model checks them in one pass, so a run they
agree on costs about what one token used to. It takes the same two forms as the model — a
path, or `hf:owner/repo/file.gguf` — and `--draft auto` finds the one a repository ships
wherever the publisher put it: at the root, under `MTP/`, or in a sibling `-MTP-GGUF`
repository. Beware that a `-MTP-GGUF` repository is not always heads: for
Qwen3.6-35B-A3B it is the whole model rebuilt with the prediction layers in it, 36G of
weights for `--spec draft-mtp`, and `auto` correctly reports no draft rather than
offering it as one. `hub.draft_note(repo)` reads the head's own `MTP/README.md` (or
`README.md`) for the sentence that says why a head needs something more than the model
itself — `ml-stack-models files` prints it under the draft line it already reports, so a
publisher's warning is read before a load, not guessed at after one fails.

**Some heads need a fork, and one chooser — told which binary will serve — decides.**
Measured for real on 2026-09-01 (mainline `3466812`; the managed `b11380` is covered under "What has been measured"): every `mtp-` head under `unsloth/Qwen3.8-Flash-Next-GGUF/MTP/` fails on
mainline llama.cpp master with `check_tensor_dims: tensor 'output_hc_norm.weight' not
found` — mainline loads a draft as a whole model, and those heads carry only the head,
borrowing the trunk's embeddings from the target — and the repository's own `MTP/README.md`
says so: "these do not work on mainline ggml-org/llama.cpp yet". There used to be three
resolvers for `--draft auto`, and the one the bench used chose that head for mainline twice,
paying an 87G load each time to reach the error. Now there is one:
`hub.choose_head(model, binary=...)` returns what to serve, the `--spec-type` it needs, and
one sentence saying why — "shipped beside the weights", "withheld: the repository's README
says it needs a fork and this build is mainline", "no head shipped beside the weights" —
with the build read off the binary itself (`serve.binary.borrows`: a build under
`named/` or whose `BUILD.json` names a fork can borrow; `current`, brew and anything on PATH
cannot). A fork build is given unsloth's recommended `shared-Q8_0` head; mainline avoids a
`shared` head altogether. The model may be an `hf:` reference, a path (the repository is
read off the Hub cache's directory name), or a bare filename; offline, the head already
beside the weights on disk is the answer. `ml-stack-serve up --draft auto` prints the
reason and the README's sentence under it, and `ml-stack-models files` prints the head,
the warning, and what `this build` and each `--build NAME` on this machine would serve —
which build a head needs, before a load rather than after one fails.

**A named build keeps a fork beside `current` instead of replacing it.**
`ml-stack-serve build --repo OWNER/REPO [--ref TAG|BRANCH|SHA] --name NAME` builds a fork
from source the same way `--from source` builds master; `--from release --tag TAG`
downloads a matching release asset instead — and does not need a compiler, or a compile
that would perturb whatever else is on the GPU or the CPU right now. Either way the result
lands at `~/.ml-stack/llama.cpp/builds/<name>-<commit>/`, verified the same way `current`
is (answers `--help`) — except a named build is not required to be a superset of `current`'s
architectures; a fork may read fewer on purpose, or be younger, and that is reported rather
than refused. `~/.ml-stack/llama.cpp/named/<name>` points at it once verified; `current`
is never touched. Select it with `ml-stack-serve up --build NAME` (which resolves the
binary through the named link), `find_binary(build=NAME)` for a direct library caller, or
`$MLSTACK_LLAMA_BUILD=NAME` for one with no `build=` to pass — all three outrank `current`
but never an explicit path or `$LLAMA_CPP_SERVER`. `ml-stack-serve build --list` shows
`current` and every named build with commit, age and repo; `ml-stack-doctor` (or
`ml-stack-setup --checkouts`) lists them too, beside `current`'s own age.

Measured on this machine 2026-09-01, from the newest unsloth release
(`b10715-mix-86bd2d3`, a macOS arm64 asset, `--from release` — no compile):
`ml-stack-serve build --repo unslothai/llama.cpp --from release --tag b10715-mix-86bd2d3
--name unsloth`. `--build unsloth` then preflights Qwen3.8-Flash-Next
(`UD-IQ4_XS`) with `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, `--spec draft-mtp
--spec-draft-n-max 2` cleanly — architecture, shards and every flag check pass — without
ever serving it. The measurement was run later (see "What has been measured"). Unsloth's own
recommended `--spec-draft-n-max` is 2, other reports found 3 or 6 better depending on platform and
`--spec-draft-p-min` (0.7 is the value used alongside them); `--ctx-checkpoints 0` is
needed for a byte-identical comparison at all, because **greedy output with a head on is
not byte-identical on Metal at n-max ≥ 3** (also seen on HIP) — so a bench comparing heads
on this machine must watch its own F1 for a real quality change, not assume decoding stayed
identical just because sampling is greedy.

