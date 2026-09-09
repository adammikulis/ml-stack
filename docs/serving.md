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
answers, and a start that fails is forgotten), one shape is served per port and a lease
that asks for another shape is refused with the field that differs named, a server already
serving what was asked for is adopted rather than started again, a server the record does
not know is reported as somebody else's and never killed, and the backend launches nothing
without the manager's lease in hand -- so an untracked server cannot come out of the
library at all. The measured shape of each model (`ml-stack-serve profile`) is what a
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
when nothing is serving. `up` adopts a server already serving that model in that shape
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

**One shape per port, written down once.** llama.cpp serves a model one way at a time, so
two parts of a program that lease it differently are not two clients of one server:
whichever leases second finds a mismatch, stops the first and loads the weights again. A
`Shape` is the whole shape in one object and `Shape.lease()` is the only place it becomes
`serve`'s arguments, so the two cannot drift apart. `seat` starts the server on the first
ask, holds it per port for the process, and hands each caller a `Client` pinned to a slot of
its own -- so several conversations at once do not reprocess each other's context.

```python
from dataclasses import replace

from ml_stack.serve import Shape, seat, draft_for, projector_for

model = "hf:unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
shape = Shape(model=model, port=8080, seat_context=131072, cache_type="q8_0",
              draft=draft_for(model, "auto"),         # the head shipped beside the weights
              draft_n_max=4, reasoning_budget=0,      # measured, not remembered
              mmproj=projector_for(model, "auto"),    # so the model can see
              build="unsloth")                        # a head mainline will not load

client = seat(shape, index=request_number, n_predict=16384)

crowded = replace(shape, seats=4, seat_context=32768)   # four conversations at once
```

A shape holds one seat unless it is asked for more, and that seat gets the whole context.
`seats=N` divides the same memory between N conversations, each with its own KV cache. A
lone seat is given the model's own trained window when the room allows it, read off the
GGUF header; a context asked for past the trained window turns on YaRN position scaling by
itself and says so, because past the trained length the positions are the part that goes
wrong first. `ml-stack-serve escalate --port 8080 --add 2` grows a running server's seats
in place and carries its conversations across, so the next person does not cost the ones
already talking a cold reload.

`draft_for` and `projector_for` answer 'auto' the way `ml-stack-serve up` does -- a lease
built by hand has to resolve what the CLI resolves for itself -- and each says out loud why
it found nothing rather than serving undrafted or blind in silence. `release_all()` lets go
of every held server; `held()` says which ports are up.

A port already serving something else is refused, with the field that differs named —
the model, the number of slots, or the context each slot gets. Adopting a server of the
wrong shape hands back a lease that cannot do what was asked of it.

### The shape a model measured best in, for one kind of work

The `Shape` above was typed out by hand, and every value in it came from a bench run
somebody remembered. A **profile** is that shape written down instead: one record per model
file **and workload** of the serving and the asking that measured best, and the row of the
store that set it. `ml_stack/data/profiles.json` ships them and `~/.ml-stack/profiles.json`
(`$MLSTACK_PROFILES_FILE`) layers this machine's own over them, exactly as `fit.json` does.

There are three workloads, because the best shape depends on what the model is doing:
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
  measured at --parallel 2, 16384 per seat
  per request draft 4 ahead, greedy -- sent with each call, where the build takes it
  ask with    tight + batch + kinds + summary + greedy
  measured    85% F1 (89% recall, 83% precision) at 26.0 s/question over 10 question(s)
```

A record has a **startup half** and a **request half**. The build, the draft head file, the
cache type, the context and the seats are what a server is told once; the draft depth, its
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

Because no two models want the same shape. Flash-Next answers well only on a fork build,
with the shared MTP head at four, a q8_0 cache, its thinking off, `-ub 2048`,
`--spec-draft-p-min 0.5` and three ways of asking at once; gemma-4 wants its thinking left
on, a different head at two, and none of those flags. Written as defaults each would be
wrong for the other. Written per model they are what they are — and each record names the
row of the store that measured it, on which machine and on what date, so a person can tell
a measurement from a habit.

From Python, both ends read the same record:

```python
from ml_stack.serve import profile_for, seat
from ml_stack.graph.ask import converse

found = profile_for("hf:unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf",
                    workload="ask")
run = found.run(port=8080, n_predict=16384)
client = seat(run, index=request_number)
answer = converse(question, graph, client, asking=run.asking)
```

`Profile.run()` is a **`Run`**: the whole configuration in one object, in three sections
that different code reads. `run.shape` is the `Shape` the server is leased in, `run.asking`
is an `Asking` — the ways `converse` is called with — and `run.talking` is a `Talking`, what
the `Client` is built from. `run.lease()`, `run.asking` and `run.client()` are the only
places each becomes arguments, and `run.over(cache_type="f16", few=True, temperature=0.7)`
lays a knob over it, routed to the section that owns it rather than to whichever call takes
`**kwargs` next.

Hand the same run to the bench (`bench.served(run, ...)`), to a page (`AskRoutes.run`, and
`seated()` hands out a seat of it) and to `seat` and they lease one shape and ask one way by
construction. Three places each building their own from the record is how a knob about the
asking reached `Client.__init__` and took an 87G load down with it, and how two of them
could lease one port two ways — which llama.cpp answers by stopping the server and loading
the weights again.

`Asking.for_model(name, workload=...)` is the way that model measured best at that work, and
`Profile.asked()` is the same record's. A model matched only by family (the same weights at
another quantisation) comes back with `note` saying so: a shape measured on Q4_K_XL is the
right place to start for IQ4_XS and is not a measurement of it.

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
`/ui/fit.json` hands over the records seated for the room and the head count the sliders
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
Measured for real: every `mtp-` head under `unsloth/Qwen3.8-Flash-Next-GGUF/MTP/` fails on
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
ever serving it. For the later measurement itself (not yet run): unsloth's own recommended
`--spec-draft-n-max` is 2, other reports found 3 or 6 better depending on platform and
`--spec-draft-p-min` (0.7 is the value used alongside them); `--ctx-checkpoints 0` is
needed for a byte-identical comparison at all, because **greedy output with a head on is
not byte-identical on Metal at n-max ≥ 3** (also seen on HIP) — so a bench comparing heads
on this machine must watch its own F1 for a real quality change, not assume decoding stayed
identical just because sampling is greedy.

