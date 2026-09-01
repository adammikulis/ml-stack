# ml-stack

**Run and train models across every machine in your house.**

Install it on each one and type the same passphrase. They find each other on their own —
no addresses, no keys to copy, no config file. Chat with a model from any machine,
whichever one is actually running it. Work goes to whichever machine is free and fastest:
the box with the card trains, the spare CPUs prepare data, and any of them can be taken
back the moment you want it.

```
$ ml-stack-peers ls
NAME             URL                          FREE    STATE      DEVICE
gpubox           http://192.168.2.27:8770     0/1     busy       NVIDIA GeForce RTX 4090  6.2/24.0 GB free
radeon           http://192.168.2.31:8770     1/1     idle       AMD Radeon RX 7900 XTX  22.1/24.0 GB free
mac-studio       http://192.168.2.44:8770     1/2     idle       Apple M2 Ultra  96.0/128.0 GB free
pi-rack          http://192.168.2.51:8770     5/6     busy +2    16 cpu
```

Everything runs on your own hardware. Nothing leaves the network.

![The cluster view](docs/images/cluster.jpg)

[Full list of what it does](docs/FEATURES.md).

- **Chat from any machine.** The one with the card runs the model; the laptop talks to
  it. A machine that installs nothing extra still gets to use it, and conversations are
  kept.
- **Nothing to configure.** A passphrase is the whole setup. Two households on one
  network stay separate without either of them being told to.
- **Work lands where it fits.** Placement is by what a machine reports and how fast it
  has actually been measured, per kind of work. A machine nobody has measured is tried,
  not skipped.
- **Your machine stays yours.** Block out working hours, or hit pause when you start a
  game — the run stops, requeues, and picks up from its last checkpoint.
- **Train without writing code.** Pick what you want it to learn, point it at your
  files, and it runs. Or drive it from Python if you would rather.
- **Mixed hardware is the normal case.** NVIDIA, AMD ROCm, Apple silicon and plain CPUs
  in one cluster, each reporting its own temperature, clocks and throttle state.

## Installing

**One line**, on macOS or Linux:

```
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh
```

On Windows, in PowerShell:

```
irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
```

It works out which machine it is on, fetches the right download, and opens it. You get a
window that asks what to call the machine and which cluster to join, then sets the rest
from the hardware it finds:

![Setting up a machine](docs/images/setup.jpg)

**Or download it yourself** from the [latest release](../../releases/latest):

| | |
|---|---|
| macOS | `ml-stack-macos-arm64-<version>.zip` — Apple silicon (M1 or later) |
| Windows | `ml-stack-windows-x86_64-<version>.zip` |
| Linux | `ml-stack-linux-x86_64-<version>.zip` |

Each download holds the app and `ml-stack-headless`, for a machine with no screen — the
same daemon, serving the interface to a browser on your network.

Do the same on every machine you want to train with, typing the same passphrase. They
find each other on their own.

**If you write Python**:

```
pip install ml-stack            # all of it, and nothing else. No dependencies.
pip install ml-stack[train]     # and numpy and safetensors, to train
pip install ml-stack[all]       # and everything the rest of it can use
```

Building from source needs `pip install build`, then:

```
python packaging/build.py            # wheels into dist/
python packaging/build.py --bundle   # and a standalone app for this platform
```

Everything is adjustable later:

![Settings](docs/images/settings.jpg)

## Driving it from Python

`ml-stack` has no dependencies, so the machine you drive from needs no CUDA, no MLX
and no training stack.

```python
from ml_stack.fleet import Peer, Requires, Unit, run

peers = Peer.discover()
report = run(
    [Unit(id=f"shard{i}",
          argv=["python", "-m", "ml_stack.train.run", "--recipe", "text-lm",
                "--data", f"shards/{i}.jsonl", "--out", f"out/{i}"],
          requires=Requires(labels=("train",), min_vram_gb=8))
     for i in range(8)],
    peers, kind="text-lm")

for place in report:
    print(place.unit_id, place.peer, place.state, f"{place.elapsed_s:.0f}s")
```

Work waits for capacity rather than failing, is retried on a *different* machine if it
fails, and a machine that fails several in a row is set aside rather than draining the
queue. A unit no machine can run fails at once, naming every machine and why:

```
gpubox: has 23.0 GB VRAM, needs 80.0
pi-rack: does not report 'cuda'; has no backends
radeon: this machine is in use (mon tue wed thu fri 09:00-17:00); work resumes Mon 17:00
```

One machine at a time, when that is what you want:

```python
rtx = Peer.find_one(require="cuda")     # refuses to guess between two
rtx.push("data/train.jsonl", "data/train.jsonl")
job = rtx.submit(["python", "-m", "ml_stack.train.run", "--recipe", "text-lm",
                  "--data", "data/train.jsonl", "--out", "out/lm"])
rtx.wait(job["id"], on_metric=print)
rtx.pull("out/lm/step_000010000/model.safetensors", "local/model.safetensors")
```

Uploads and downloads resume and are verified by digest.

## Training

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

## Packages

| Module | What it is |
|---|---|
| `ml_stack.contracts` | Reader for `contracts/`; the RAM→model ladder and the fitting rule |
| `ml_stack.media` | WAV containers, image format sniffing, resumable asset download |
| `ml_stack.client` | HTTP client: chat, completion, embeddings, health, token estimate |
| `ml_stack.fleet` | Find the other boxes, run jobs on them, move files between them |
| `ml_stack.serve` | Start, adopt and tear down a model server |
| `ml_stack.gguf` | Converter/quantiser discovery, export, tokenizer-metadata repair |
| `ml_stack.speech` | ASR / TTS / VAD behind three protocols and one resolver |
| `ml_stack.vision` | Image payloads, and a gate that verifies a model can see |
| `ml_stack.backend` | One array API over MLX and PyTorch, so math is written once |
| `ml_stack.graph` | A graph: stored, searched, asked about, drawn — and as tensors |
| `ml_stack.entities` | Resolving names, planning edits, spelling, paths through a graph |
| `ml_stack.scrape` | Reading a site you are signed in to, with presets to start from |
| `ml_stack.train` | Atomic checkpoints, schedules, guards, metrics, leak-safe splits, tokenizer fertility |
| `ml_stack.testing` | Cross-backend numerical parity harness |

Everything above ships in one package. The extras carry what a module needs beyond the
standard library: `[app] [train] [serve] [gguf] [graph] [store] [scrape] [vision] [testing]
[torch] [mlx] [telemetry]`, and `[all]`.

## Working with a graph

A graph here is a mapping with `nodes` and `edges` and nothing else agreed in advance —
what a project calls its kinds and its relations is the project's business.

```python
from ml_stack.graph import GraphStore, replace, converse, render, hybrid

replace("graph.ladybug", graph)              # safely: see below
with GraphStore("graph.ladybug") as store:
    store.set_embedding("person:ada", vector)
    store.similar(vector)                    # nearest by meaning
    store.search("compiler")                 # stemmed, so it finds "compilers"
    store.shortest_path("person:ada", "person:bea")

hybrid(graph, "who fixes machines", store=store, vector=asked)   # all three at once
converse("how are these two connected?", graph, client)          # the model, with tools
open("page.html", "w").write(render(graph, title="Who knows what"))
```

**A store cannot be lost to a bad rebuild.** A pipeline that read nothing produces an empty
graph, and an empty graph looks exactly like "remove everything". `replace` refuses a write
that would take most of a store, and leaves a verified snapshot when it would take a tenth.
`snapshot` and `roll_back` are there directly, and a restore saves what is there first.

**Two processes cannot corrupt one.** The database's own lock stops the second writer with an
IO error; what `ml_stack.graph.access` adds is knowing whose lock it is, waiting for a turn,
and letting go of a read handle when a writer wants in.

## The commands

| | |
| --- | --- |
| `ml-stack-models find <words>` | search the Hub for a model, unsloth first; `files <repo>` lists the quantisations and prints the `hf:` reference to serve each; `card <repo>` reads the sampler settings its publisher recommends |
| `ml-stack-serve status\|up\|down` | one model per port, in one shape; refuses a mismatched lease; announces to the fleet; `--draft auto` and `--mmproj auto` find the speculative head and the vision projector shipped with the weights; `--spec` chooses draft or n-gram guessing; `--binary` runs a build that reads a newer architecture |
| `ml-stack-bench prepare\|run\|sweep\|show` | time and score a graph's answers — wall clock, calls, cached tokens against read ones, KV cost, draft acceptance, and how much of the expected answer was shown; `show --rates` adds accuracy per second, per 1k tokens and per GB with the Pareto frontier, `--plot` draws it |
| `ml-stack-setup` | what this machine can do — memory a model may use and whether that survives a reboot, which architectures the installed build reads, what is already downloaded — and what the stack does without being asked |
| `ml-stack` | the windowed app; `ml-stack-app`, `ml-stack-traind`, `ml-stack-peers`, `ml-stack-train-run` |

## Finding a model

The Hub has models newer than anything in this README, and newer than anything an assistant
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

From a shell:

```
ml-stack-serve up model.gguf --context 32768 --parallel 2
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

A port already serving something else is refused, with the field that differs named —
the model, the number of slots, or the context each slot gets. Adopting a server of the
wrong shape hands back a lease that cannot do what was asked of it.

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

**A release lags master by an architecture or two.** Checked on this machine: the current
release reads `gemma4` and `qwen3moe` but not `qwen4exp`, so Qwen3.8-Flash-Next exits with
"unknown model architecture" until it is served with a build from master —
`ml-stack-serve up --binary /path/to/llama-server`. The architecture names live in
`libllama`, not in the server binary, so grep the library rather than the executable:

```
strings "$(dirname "$(which llama-server)")"/../lib/libllama*.dylib | grep -x qwen4exp
```

`ServerSpec(draft=...)` serves a small model of the same family alongside the large one: it
guesses several tokens ahead and the large model checks them in one pass, so a run they
agree on costs about what one token used to. It takes the same two forms as the model — a
path, or `hf:owner/repo/file.gguf` — and `--draft auto` finds the one a repository ships
wherever the publisher put it: at the root, under `MTP/`, or in a sibling `-MTP-GGUF`
repository. Beware that a `-MTP-GGUF` repository is not always heads: for
Qwen3.6-35B-A3B it is the whole model rebuilt with the prediction layers in it, 36G of
weights for `--spec-type draft-mtp`, and `auto` correctly reports no draft rather than
offering it as one.

## A hierarchy read out of prose or a picture

```python
from ml_stack.graph.tree import FAMILY, ORG, read, to_graph

rows = read(client, ORG, images=[chart], reader=document_model)   # or text=...
graph = to_graph(rows, ORG)                                       # entries and links
```

An org chart, a family tree, a subject taxonomy and a parts breakdown are one object with
four shapes: named things, and a link from each to the one above it. A `Shape` carries what
an entry is called, what the link means and how many parents it may have — a family tree
keeps two, an org chart one, because a second manager there is a misreading.

`reader` is a second model that reads the picture first, which is how a document model gets
used for what it is good at: it transcribes, `client` structures. A picture needs a server
started with its projector (`--mmproj auto`), and a picture that cannot be prepared raises
rather than being dropped — otherwise a model is asked to read a chart with no chart
attached, and answers confidently about nothing.

## Which tool a question wants

```python
from ml_stack.graph.ask import TOOL_PROMPTS, tools_for
from ml_stack.graph.route import narrow, rank

routed = rank(question, TOOL_PROMPTS, base_url=embedder, model=name)
tools = narrow(tools_for(graph), routed)      # [] when the question wants no graph
```

`graph.route` asks a small embedder which tool a question resembles, by comparing it to
**example questions** rather than to the tools' descriptions. That distinction is the whole
of it: a question against prose describing a capability is comparing unlike things, and a
question against questions is like-to-like. The examples live in `ask.TOOL_PROMPTS` and are
never sent to the chat model, which wants the opposite text — what a tool *does*.

Both sides carry the same embedding prefix, because both are questions. Using the
asymmetric `QUERY`/`DOCUMENT` pair here scored "tell me about Otto Vance" at 0.409 against
an example reading "tell me about Iris Bellweather", which is the same sentence; with the
symmetric prefix it is 0.83.

The useful case is the one that is not a tool at all. `CHAT` collects greetings, jokes and
asides, and a message routed there is offered **no tools whatsoever** — one model call
instead of the six a graph question takes. Without somewhere for those to go, a greeting is
matched against four search tools and wins one of them: "hi" scored 0.900 against
"highlight them on the graph", because everything is close to everything and the only
question is close to *what*.

Nothing narrows unless the routing was clear, `show` survives every narrowing, and an
embedder that will not answer routes nothing rather than defaulting to chat — a real
question mistaken for small talk is answered without looking anything up, which reads as a
confident answer and is about nothing.

## Answering the same question twice

```python
from ml_stack.graph.ask import Answer, converse
from ml_stack.graph.cache import asked, digest, forget

out, again = asked(store, question, lambda: converse(question, graph, client),
                   kind=Answer, graph=graph, model=name, system=SYSTEM, tools=tools)
```

`asked` hands back the answer already given when nothing that shaped it has changed, and
calls the model only on a miss — measured at 27.9s against 0.00s for the repeat. What makes
that safe is the fingerprint, which covers the graph, the model, the system prompt, **the
tool schemas including their descriptions**, the shortlist and whatever `context=` the
caller adds (the turns before this one, most obviously). Rewording a tool changes what the
model does with it, so it misses — which is right: rewording them moved every score in the
bench.

`keep=` refuses an answer that should not be served twice — one whose turn also *did*
something, like filing a change request. Its answer is a receipt, and handing it out again
would tell the next person their request was filed when it was not.

A rebuilt graph misses on its own, but does not sweep on its own: pass
`forget(store, keeping=digest(graph))` after a rebuild or the store keeps every answer it
ever gave. Entries live under keys beginning `_`, which `GraphStore.docs` skips, so a cache
in the same store as a graph never leaks into `read()`.

## What this measured, and what it changed

Answering a graph is the case this was built for, so the numbers are here rather than in
somebody's notes. Ten questions over the invented community, 32k per slot, greedy, one run
at a time on an otherwise idle machine. **F1** over the entries an answer lights, with the
pair behind it, because the pair is what says how a run was wrong:

| run | F1 | recall | precision | lit per question | wall | KV + runtime |
| --- | --- | --- | --- | --- | --- | --- |
| gptoss-shortlist | **62%** | 70% | 59% | 2.0 | 198s | 11.89G |
| gptoss-plain | 61% | 65% | 62% | 2.0 | 127s | 11.89G |
| e4b-plain | 58% | 70% | 51% | 2.0 | 159s | 7.75G |
| e2b-plain | 41% | 55% | 39% | 2.3 | 59s | 3.21G |
| e4b-shortlist | 33% | 70% | 25% | 4.5 | 300s | 8.34G |
| e2b-shortlist | 29% | 80% | 18% | 6.1 | 64s | 3.50G |

A good answer lights about **1.7** entries. Read the last column against that and the table
explains itself.

**Scoring on recall alone said the opposite, and it was wrong.** Under it, `e2b-shortlist`
was the most accurate run there was at 80%, beating the 120B — because showing more costs
nothing under recall, and it lit six entries where fewer than two were wanted. A model that
lit every entry in the graph on every question scored 100%. That metric survived a day and
twenty-four green tests, none of which asked what the degenerate strategy would score. One
does now.

**A shortlist handed to a small model is echoed, not selected from.** It is the single
largest effect here: E4B 58% → 33%, E2B 41% → 29%, precision halving while recall rises.
The same shortlist does nothing for the 120B either way. The idea is sound and the machinery
is worth keeping; what is missing is teaching a model that a shortlist is somewhere to look.

**What the tool descriptions changed was real.** E4B answered 17% before they carried a
worked call and 70% recall after, on the same weights and the same questions. Six of its
nine failures had been the identical shape: two model calls, a hundred characters of prose,
no search at all. The large models never needed telling, which is why this went unseen until
a small one was measured.

Two mistakes cost the earlier numbers their meaning. Timings were taken while other runs
shared the GPU, which is why `sweep` and `run` refuse a busy server now. And every model had
512 tokens for thinking, tool calls and answer together, because `n_predict` defaulted low:
a thinking model fills that with reasoning and returns an empty answer.

None of this survives a new model release. Re-run it.

The runs themselves are in a graph store under `~/.ml-stack/bench`, which nothing backs up.
`ml-stack-bench show --export PATH` writes them out, so a day of GPU time is not on one disk.
**That file does not belong in a repository, and `--export` refuses one**: the numbers
describe one machine and one llama.cpp build, they go stale with the next model release, and
`run --graph` takes any graph, including a real community's. Back it up somewhere outside a
working tree.

What is worth keeping here is the conclusion, not the evidence. `ml-stack-bench show --rank
FILE.md` writes one line per model -- its best run, and what that run cost -- because that is
what the defaults in this library are set from, and a default with no recorded reason is a
default nobody can argue with. Both it and `--export` carry only runs whose recorded graph
fingerprint is the community that ships with this package, and refuse a run from before that
marker existed, because not knowing which graph a run read is not the same as knowing it was
invented. The ranking also ignores anything shorter than a short run, so a `--smoke` run
cannot rank a model on two questions.

## Measuring a change to the asking

```
ml-stack-bench prepare --embed-url http://127.0.0.1:8081 --embed-model embeddinggemma-300M-Q8_0.gguf
ml-stack-bench sweep --on gptoss=http://127.0.0.1:8080 --on e4b=http://127.0.0.1:8083 \
    --embed-url http://127.0.0.1:8081 --embed-model embeddinggemma-300M-Q8_0.gguf
ml-stack-bench show
```

`sweep` measures each model twice — as it is, and with a search run before it — and prints
them all in one table. `run` does one of those on its own, and `show --compare A B` puts two
side by side with the difference.

`sweep` and `run` **refuse a server that is already working**, because a timing taken while
another run has the same GPU is not a timing. It is a real failure and not a hypothetical:
several sweeps left running in the background against one server produced wall clocks that
were two runs sharing a machine, and nothing in the numbers said so. A server that will not
answer `/slots` is reported as unknown rather than assumed idle. `--anyway` proceeds on
purpose.

Serve every model being compared with the **same context and the same number of slots**, or
the comparison is of two configurations rather than two models: a model at 8k per slot is
faster and holds a smaller cache than the same model at 32k. The table prints `ctx` on every
line so a mismatch is visible rather than silent.

The questions are asked of an invented community that ships with this package, so a number
means the same thing on any machine and no real person's details are involved. Each question
may carry the ids a good answer names, which is what makes accuracy measurable rather than
impressionistic. Runs are kept in a graph store under `~/.ml-stack/bench`, so one can be
compared with another a week later.

`show` reports wall clock, model calls, prompt tokens **and how many of them were cached** —
a conversation re-sends itself every turn, so the tokens shown and the tokens actually read
are different numbers and only the second is a cost.

It also reports what the server costs to keep up beyond its weights (`kv+run`: the KV cache
and the runtime around it, measured as resident memory minus the weights on disk) and that
figure per 1k of held context (`per 1k`). The second is the one to compare: it says what one
more conversation costs, whatever context each server happened to be given, and it is what
decides how many conversations a machine can hold at once.

`show --rates` puts accuracy over each of the three scarcities — time, tokens, and the
memory a conversation holds — because a score alone cannot choose between a model that is
better and one that is cheaper. It marks the **Pareto frontier**: the runs nothing else
beats on both accuracy and cost, which are the only ones there is ever a reason to choose.
`--cost seconds|paid_tokens|kv_bytes` redraws it against whichever is scarce, and
`--plot out.html` writes it as a scatter with the frontier joined — hand-built SVG, no
library and no network, so it opens on any machine.

`--n-predict` is a **ceiling, not a budget**: nothing is spent that is not generated, so a
high one costs nothing and a low one truncates. It defaults high on purpose. A thinking
model spends most of a turn reasoning before it writes anything — measured, gemma-4 filled
a 220-token ceiling entirely with thought and returned empty content — so what a low
ceiling cuts is always the answer, never the thinking.

`--card` asks with what the model itself recommends, which is the only place a
recommendation is ever applied. It is read from the served model's **GGUF metadata** where
that exists — `general.sampling.temp`, `.top_k`, `.top_p` are written into the file, so they
cannot drift from the weights and need no prose parsed out of a README — and from the card
otherwise. The two agree where both exist: gemma-4 says temperature 1.0 / top_p 0.95 /
top_k 64 in each. They are per model, not per family: Qwen3.8-Flash-Next asks for top_k 20. A publisher's advice is a hypothesis about a task they have not seen: gemma-4
asks for temperature 1.0 across all use cases, and on this one — calling tools with exact
ids, where sampling noise becomes a wrong argument rather than a livelier sentence — greedy
measured better on the plain path. Read the card with `ml-stack-models card <repo>`, test it
with `--card`, and ship what the measurement favoured.

A score is only worth acting on when you can see which questions made it, so
`show --detail` prints the questions themselves — what each one wanted, what the answer
showed, what it missed, and what it cost — with `--detail LABEL` for one run and `--all` for
every question rather than only the ones that fell short. It is what turns a number into a
diagnosis: a model whose wrong answers took *more* calls than its right ones searched hard
and missed, while one whose wrong answers took *fewer* never reached for the tools at all,
and those two failures are fixed by opposite things.

## `contracts/` is data, not code

`contracts/` holds JSON describing things a runtime and a non-Python host both need to
agree on: the RAM→model tier ladder, the sampler surface, GBNF grammars. It contains no
code, so a native or scripting host can read it directly.

There is exactly one copy on disk. The wheel pulls it in at build time,
so there is no synced duplicate in the source tree to drift.

Resolution order at runtime: `$ML_STACK_CONTRACTS` → the copy inside the installed wheel → a
`contracts/` found by walking up from the source file. The walk-up is last on purpose: if a
wheel is installed *and* a repo happens to be an ancestor, the wheel's own data should win,
because that is what its version was tested against.

## Testing

```
python -m pytest tests/ -q
```

Nothing here mocks the transport. Every client test runs a real `http.server` on a real
socket, because the failures these modules exist to prevent are transport-shaped: a server
that answers `/health` while still loading, one that ignores a `Range` header, one that
returns 500 on a concurrent request. A mocked `urlopen` reproduces none of them.

The fleet tests go further: they boot real `ml-stack-traind` subprocesses and speak real
UDP on a real interface, on randomised ports so a run never answers -- or gets answered
by -- a daemon you actually have running on your LAN. A forged beacon, a replayed reply,
a multicast group a router quietly drops: a fake socket reproduces none of those either.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
