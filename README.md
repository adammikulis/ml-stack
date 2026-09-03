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

One script per platform, four modes. Re-running any of them upgrades in place.

**macOS and Linux:**

```
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh -s -- --headless
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh -s -- --dev
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sudo sh -s -- --system
```

**Windows**, in PowerShell. `iex` runs a piped script with no arguments, so the mode is an
environment variable rather than a scriptblock incantation:

```
irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
$env:ML_STACK_MODE="headless"; irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
$env:ML_STACK_MODE="dev";      irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
$env:ML_STACK_MODE="system";   irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex   # as administrator
```

| | what it installs | what it downloads | how it updates itself |
|---|---|---|---|
| **the app** (default) | the release zip for this machine, and a window | nothing, until you click: the first-run screen shows the models that fit, gemma-4-E2B suggested (2.6G, ~1.5s a question) beside E4B (4.4G, ~3s) and Flash-Next (104G, ~27s), the ones too big for the room greyed out | the newest published release, replacing the whole install — daemon, CLI and window — then restarting |
| `--headless` | a venv under `~/.ml-stack` (Windows: `%LOCALAPPDATA%\ml-stack`), console scripts on PATH, no window | `--models auto`: the best measured model this machine has room for, unless `ML_STACK_MODELS=none` | releases, the same way — or `main`, if you installed from `main` |
| `--dev` | a git checkout with `pip install -e .` | the same as headless | follows `main`: pulls, reinstalls if the packaging moved, restarts |
| `--system` | `--headless`, plus a service that starts at boot with nobody logged in. Needs `sudo` / an administrator | the same as headless | the same as headless |

**Per user or per machine.** The first three need no administrator and the daemon runs
while you are logged in; the fourth is per machine:

| | per user (app, headless, dev) | per machine (`--system`) |
|---|---|---|
| rights | none | `sudo`, or PowerShell as administrator |
| runs | while you are logged in | at boot, before anyone logs in |
| as | you | still you — see below |
| the Windows firewall | one approval prompt, once | the same, in the same step |
| the macOS wired limit | `sudo` once, offered by the app's first run and by `ml-stack-setup` | applied in the same step, since it already has the rights |
| the model cache | yours, `~/.cache/huggingface` | **the same one.** One cache per machine, shared, never duplicated |

That last row is why `--system` installs the service to run **as the account that ran it**
(a LaunchDaemon with `UserName`, a systemd unit with `User=`, a Scheduled Task with `/RU`)
rather than as root or SYSTEM. A service under another account would have its own empty
`~/.cache/huggingface` and download every model a second time; running as you, it opens the
one that is already there, in place, and nothing is copied, linked or fetched twice. If you
do point it at another account, the installer lists the models you have with their sizes and
says they would be downloaded again; `--adopt-cache` *moves* the cache to the shared path and
leaves a symlink behind, so your own tools keep working and every file still exists once.
Declining leaves your cache alone. It never copies.

Unattended, for a machine you are setting up from a script: `ML_STACK_NAME`,
`ML_STACK_PASSPHRASE`, `ML_STACK_CLUSTER`, `ML_STACK_MODE`, `ML_STACK_MODELS`,
`ML_STACK_ADOPT_CACHE`, `ML_STACK_REF` answer every prompt, and a machine with no terminal
is never prompted at all. `ML_STACK_OFFLINE_ZIP` and `ML_STACK_OFFLINE_MODELS` install from
local files and skip every network step. `--uninstall` takes it off and leaves the model
cache where it is.

Past the install, every step is an ml-stack command rather than shell -- `ml-stack-setup`
(what this machine can do), `ml-stack-serve build` (llama.cpp), `ml-stack-models fetch`
(into the one cache, every download checked against its sha256), `ml-stack-fleet join
--persist`, and `ml-stack-doctor` at the end, whose lines it prints.

**Or download it yourself** from the [latest release](../../releases/latest):

| | |
|---|---|
| macOS | `ml-stack-macos-arm64-<version>.zip` — Apple silicon (M1 or later) |
| Windows | `ml-stack-windows-x86_64-<version>.zip` |
| Linux | `ml-stack-linux-x86_64-<version>.zip` |

Each download holds the app and `ml-stack-headless`, for a machine with no screen — the
same daemon, serving the interface to a browser on your network.

![Setting up a machine](docs/images/setup.jpg)

Do the same on every machine you want to train with, typing the same passphrase. They
find each other on their own.

**On Windows** the same daemon runs, with the handful of things Windows does differently
decided in one place (`ml_stack/platform.py`): a job gets its own process group and is
stopped with a `CTRL_BREAK_EVENT` it can catch as `SIGBREAK`, the one-runner lock is
`msvcrt.locking` where POSIX has `flock`, the cluster key is made private with `icacls`
where `chmod 600` would only flip the read-only bit, and `ml-stack-traind --persist`
registers a Scheduled Task at logon (`com.ml-stack.traind.login`) the way `ml-stack-serve
build --persist` registers its weekly refresh. `--system` registers a second one
(`com.ml-stack.traind.system`) at `ONSTART` instead, so the machine is a peer before anyone
logs in -- with `/RU <you>` rather than `/RU SYSTEM`, because SYSTEM has its own profile and
would download every model again into an empty cache. It is a Scheduled Task rather than a
real service because a service needs a wrapper to hold a long-lived Python process, while a
startup task is one line and survives a reboot either way. Two things a Windows machine needs that the
others do not: llama.cpp comes from `ml-stack-serve build --from release` (a release zip,
since most Windows installs have no compiler), and **Windows Defender Firewall blocks the
daemon's TCP 8770 and its UDP 8771 beacons inbound by default**, so `ml-stack-peers ls` on
another machine sees nothing until, in a prompt opened as administrator:

```
netsh advfirewall firewall add rule name="ml-stack traind" dir=in action=allow protocol=TCP localport=8770 && netsh advfirewall firewall add rule name="ml-stack discovery" dir=in action=allow protocol=UDP localport=8771
```

`ml-stack-setup` prints that line as a finding until both rules exist. The first time on a
new Windows machine, in this order, each of which should say what follows it:
`pip install -e .` (an editable install, so `ml-stack-traind` is on PATH);
`ml-stack-setup` (the firewall finding, `!` until the rules are added, then `ok`);
`ml-stack-doctor` (the checkout and hooks);
`ml-stack-serve build --from release` ("current -> ...\builds\bNNNN" after "verifying");
`ml-stack-traind --persist` ("installed to start at login", then a `traind.log` under
`~\.ml-stack` that begins `ml-stack traind on http://0.0.0.0:8770`);
`ml-stack-peers ls` from another machine (the Windows box listed with its GPU). Everything
Windows-specific here was written against a faked `platform.system()` on a Mac -- the
Windows calls themselves run for the first time when that list does.

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

A model sweep goes over the fleet the same way. `ml_stack.fleet.bench` is the fleet side
of `ml-stack-bench sweep --fleet`: `plan(models, peers)` sends each model, largest first,
to the idle peer with the most room for it -- `room_bytes` is what the daemon announces,
`hub.room()` rather than free memory -- spreading models over machines rather than
stacking them, and names every model that fits nowhere and why on each peer. `jobs_from`
turns the plan into one `Job` per peer (the sweep's line with only that peer's `--serve`s),
`dispatch` posts them, `wait` polls until each is `done` or `failed` and prints the log's
tail, and `gather` brings every peer's runs home into one store as `bench:` docs with
`server["host"]` and `server["commit"]` set, never overwriting and skipping what is already
there; `import_runs(FILE.json, into, host=...)` does the same by hand from a
`show --export` file for a peer with no daemon. A daemon refuses a bench job (409, with
`refused` saying which) when its checkout's commit is not the dispatcher's, when its
`measuring.lock` is held or a bench of its own is still running, or when a model's
estimated bytes exceed its room; otherwise it runs `ml-stack-bench ... --detach` on itself
and adopts the pid into its job list, so `ml-stack-peers ls` shows it `measuring` and no
training job starts beside it. The dispatcher counts as a peer through `here()`.

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

### Fine-tuning a tool caller

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
  measured to matter (17% to 70% recall on the same weights, above). The router's example
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

#### From what a model actually did

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

#### A model too big to fine-tune whole

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

**The model is given six things it can do, not the graph.** `look_up` finds entries by
name or by the words attached to them, `look_at` reads what is held on them, `look_around`
reads a whole neighbourhood at once, `path_between`
traces how two connect, `list_kind` reads out everything of one kind, and `show` says what
the answer is about. `list_kind` exists for the question no search reaches: "which companies
do people here work for?" is answered by every `org` in the graph, and no word finds those,
because nothing is *labelled* company — a small model searched "company", then
"organization", found nothing and gave up. A kind the graph does not have comes back as the
kinds it does, with counts, so a wrong guess costs one call rather than a turn. An answer
that comes back empty says why in `steps` — `no answer: finish_reason=stop, thinking 628
chars, answer 0 chars` — because the assumption is always the token budget and, measured,
it almost never is. A tool of the caller's own that returns pictures (`_images` in its
result) has them shown to the model as a message of their own, since a tool result cannot
carry an image; a picture that cannot be prepared is a line in `steps`, not a crash.

**Fewer, fatter calls, for a model that reads far faster than it writes.** Measured
2026-09-02 over the invented community, Qwen3.8-Flash-Next — hybrid recurrent, 256k of
context at 48K bytes a token — found nearly everything (89-95% recall, and 76-83% precision
once the asking was tight), and spent 5-9 tool calls a question at about 2k new tokens each:
half the wall clock reading results back at ~390 tok/s, the other half writing at ~35. For
that model a round trip is the expensive part and reading is nearly free, so the way to make
a question cheaper is to take more per call. `look_around(ids, hops=1)` is the fat call: the
entries you name, and under each of them everything joined to it with the relation, the
neighbour's kind, **its id in brackets** and a line of its own words — so "who could help
with X" is one `look_up` and one `look_around` where it used to be five `look_at`s, and the
answer may select a neighbour it never looked up. `converse(..., reach=N)` is the budget
that makes such a result safe to ask for: N tokens per tool result instead of the flat 6000
characters, with `look_at`, `look_around` and `list_kind` packing **whole entries with their
quotes**, most-mentioned first, rather than every entry with its words clipped — because the
quote is the evidence, and `list_kind`'s fixed forty was only ever a guess at what a result
may cost. Both are off by default (`reach=None`), because the small models have the opposite
profile — E4B and E2B reach about 60% recall with cheap decoding and an expensive cache, and
a fatter result is a worse trade for them. `ml-stack-bench --also reach` measures the fat
asking against the default on one load, and `--reach N` sets the size on every way.

**`look_up`'s first hits are put in the order the vectors mean.** Fusion decides which
entries come back — a reciprocal rank is a vote count, and three ways agreeing beats one way
being certain — but it cannot say which of them the asker meant, because a vote is not a
distance. The vectors can, and they are already in the store, so `hybrid` re-orders its first
six hits by cosine to the question whenever the question arrived embedded. Membership never
changes and a hit the vectors have never seen keeps its place rather than sinking, so an
exact label match cannot be pushed down the page by something the embedder merely likes;
`rerank=False` is the fused order exactly as it was.

**A hit that says why it matched, and who is joined to it, is behind a flag until it is
measured.** `look_up` returns `{id, label, kind}` and nothing else, so a model cannot tell an
exact label from one word in one quote, and a topic it finds is only halfway to the people
who have it — measured against a real graph, every staffing question spent rounds guessing
spellings. `tools_for(graph, rich=True)` (and `converse(..., rich=True)`, `hybrid(...,
rich=True)`) adds `score`, `matched` — which of `label`, `attribute`, `said`, `words`,
`meaning` found it — and, on anything that is not a person, `joined`: the eight
most-mentioned people on any edge to it. The look_up description gains one sentence saying
so, on a copy. With the flag off nothing observable changes, byte for byte, because the
answer cache fingerprints those descriptions and a sweep of the current behaviour has to stay
comparable with the one that measures this.

**Lighting only what answers the question is the asking, not a variant.** A model that
finds nearly everything can then light nearly everything: measured over the invented
community, 34 questions at 32k, Qwen3.8-Flash-Next reached 92% recall at 44% precision, and
named 70 entries its tools never found, read or showed, where a good answer lights about
two. So `converse` and `tools_for` ask tight by default — the words about `show` change, on
a copy: light only the entries that answer the question, the ones the asker would act on,
never what was looked at on the way, usually one to three — the closing nudge says the same,
one sentence is added to the system prompt (name only what a tool returned; say when
something was not found rather than guess a name), `show` is capped at `LIT_TIGHT` (six, the
ids the prose names kept first, `cut N of M lit` in `steps`), and an entry the prose names
that no tool ever returned is dropped (`dropped N unread from show`). `tight=False` is the
loose asking kept as a **control** — the words the ranking runs and the answer cache
fingerprinted, the same schema objects, byte for byte as they were — and
`ml-stack-bench --also loose` measures it against the default on the same load.

**Three more askings, each off until it is measured: `batch`, `kinds`, `summary`.**
Measured 2026-09-02 over the invented community, Qwen3.8-Flash-Next answered at 70% F1 —
85% recall, 65% precision — and spent 25 seconds a question over about seven tool calls.
`converse(..., batch=True)` is for the seconds: the calls were one question asked one entry
at a time, because nothing said the ids are a list, so the system prompt says it, each
searching tool's description gains a worked three-entry call, and a turn that reads one
entry while more are still unread is told once to *read the rest in one call*. What it
should move is `Answer.rounds` — a round is a round trip through the model, and a reply that
asks for three tools at once is one round, all three of them run before the next turn.
`converse(..., kinds=True)` is for the precision: the misses were mostly right-adjacent —
the topic lit beside the people for a question that asked *who* — and the question word
already says what kind the answer is, so `asked_kinds` reads it off the asking clause and
`show` keeps only that kind. It filters nothing when the question named several kinds or
none (`how is X connected to Y`, `tell me about X`), a listing is exempt as it is from the
cap, and a filter that would empty the selection is not applied; over the bench's own 110
questions it filters 72 to the right kind, leaves 31 alone and gets exactly one wrong.
`converse(..., summary_tool=True)` is for the broad question no search reaches — "what is
this group about?" has no name in it to look up — and adds `summarise`: counts per kind, the
ten most-mentioned entries of each kind with a line of their own words and their ids in
brackets, and the busiest relations, computed from the graph with no model call at all.
`routing_prompts(summary=True)` is what to route against when it is offered.
`ml-stack-bench --also batch --also kinds --also summary` measures all three against the
default on one load.

**Two more, pulling the other way: `single` and `few`.** `converse(..., single=True)` is
`batch` turned around, and it is here for the opposite model. A fat tool result is a long
thing to hold in mind: a small model handed a dozen entries in one message answers about the
last one it read, or about none of them, and what comes back is fluent and about nothing. So
the system prompt says to read one entry at a time, each searching tool's description gains a
worked *one*-entry call, and a turn that reads several at once is told once to read them one
at a time. It buys short results and spends rounds — exactly the trade `batch` makes in the
other direction. `converse(..., few=True)` offers three tools — `look_up`, `look_at`, `show`
— and takes away every other way of looking, for the model whose tool choice degrades with
the number of schemas rather than with the question. **Nothing is faked to cover what went.**
There is no path tool and no listing tool in that offer, so look_up's description and the
system prompt say so, and say how to answer those questions with what is there: look both
ends up, read them, and read on to whatever they are joined to. A description telling the
model to "ask for a path as `path A to B`" would be a tool that does not exist, and a model
that believed it would spend every turn it has on a call nothing answers. Anything that does
not search survives `few` — `show`, and a caller's own change request — because it is not a
choice between ways to look. `converse(..., rounds=N)` is the ceiling those two trade
against: `--rounds N` rides on every way a sweep asks, the way `--reach` does.

**One asking per model.** These ways exist to be *chosen per model by measurement*, never
picked once and applied to everything. Flash-Next wants batch, kinds and summary together;
a 2B that loses the thread of a long result wants `single` and more rounds; a model whose
tool choice degrades with the offer wants `few` and more rounds still — and which is which
is a number in a store, not a taste. The same goes for sampling: a model is measured at the
temperature, top-p and top-k that suit *it*, not at one setting shared by all. So the choice
lives in the model's **profile** (`ml_stack/data/profiles.json`,
`ml_stack.serve.profile`, "A model's measured shape" below):
`ml-stack-bench report --profile`
writes, per model, the asking and the sampling of the fastest row whose F1 the questions
could not tell apart from the best — F1 alone would trade real seconds for a hundredth of a
point it cannot see — with the label of the row that set it. `converse(..., profile=MODEL)`
then asks that way, filling in only what the call left unsaid, and `ml-stack-serve profile
MODEL` reads it out: `ask with tight + few + rounds 20 at temperature 1.0 / top-p 0.95 /
top-k 20`.

**The page's routes come with the page.** `graph.html` streams its answers from
`/ask/stream`, falls back to `/ask`, and reopens a conversation from `/thread/<name>`;
`ml_stack.graph.serve.AskRoutes` is that server side for any `http.server` handler. A
subclass says how a question is answered (`asker`) and where conversations are kept
(`threads`), and hangs its own journal or queue off `answered`; the SSE framing, the `done`
frame carrying the whole answer, the history handed back to the model and the turns
remembered with their `steps` are the library's. Who may ask is still the project's policy.

**A store cannot be lost to a bad rebuild.** A pipeline that read nothing produces an empty
graph, and an empty graph looks exactly like "remove everything". `replace` refuses a write
that would take most of a store, and leaves a verified snapshot when it would take a tenth.
`snapshot` and `roll_back` are there directly, and a restore saves what is there first.

**A store checks itself.** Every `put_doc` reads its document back by key and raises
`StoreMismatch` when what comes back is not what went in; a node is read back by id the same
way, an edge from its own `RETURN`, and `replace` counts what it wrote before committing.
Measured 2026-09-01: twelve bench runs read back empty through a scan of `Doc.value` while a
lookup by key returned them whole, so `ml-stack-store check PATH` reads every document, node
and edge by key *and* by scan and prints one line per disagreement (exit 1 on any);
`--fix` rewrites a document the scan lost and checks again rather than announcing a repair,
and `ml-stack-store docs PATH` lists the documents with their sizes.

**Two processes cannot corrupt one.** The database's own lock stops the second writer with an
IO error; what `ml_stack.graph.access` adds is knowing whose lock it is, waiting for a turn,
and letting go of a read handle when a writer wants in.

**The files around a graph are the same in every project.** `ml_stack.files.write_json`
writes beside the file and renames over it, so whatever is reading the graph while a
pipeline rewrites it sees the old one or the new one and never half of either;
`prune_orphans` deletes the per-record files (an extraction per message) whose record the
log has since dropped. `ml_stack.geo.geocode_all` turns the places people write — "Raleigh",
"MD", "sf" — into points through Nominatim, cached to a JSON file, one request a second,
picking the answer that is actually *called* what was asked rather than the county Nominatim
ranks first; pass your own `user_agent`, as its usage policy asks. `ml_stack.redact.names_in`
reads every name a graph and its message log hold, for a `Redactor` to keep out of anything
printed.

**A vocabulary the model coined drifts, and folds back.** A model asked to read prose into
(subject, relation, object) invents the relation as it goes, so one relationship arrives as
`works_at`, `worksat` and `worked_at` and the graph is split three ways.
`ml_stack.entities.fold.fold_edges` keeps whichever spelling the graph already uses more and
folds the rest into it -- `entities.close` decides what counts as the same word -- but only
while one of them is rare: past `ESTABLISHED` weight both are names people keep choosing, and
neither folds without a written entry saying which is right. Every fold is logged *and*
returned, so a wrong one is something a test can point at. `dead_keys` is the other half:
the entries of a hand-written map that nothing produces any more, which fail silently
otherwise.

**An extraction already done is not done again.** `Client.extract(..., cache_dir=...)` keeps
each answer as a file there and reads it back instead of asking the model. The key is
`cache_version` + the schema + the text + `cache_extra` (the rest of the prompt that varies
per record -- the thread a message replies to, the vocabulary offered) and deliberately *not*
the instructions: wording those is iterative, and a pipeline that re-reads its whole corpus
because a sentence was rephrased is one where nobody rephrases anything. `cache_version` is
the knob for a change nobody should be allowed to skip. Only an answer that passed `check` is
kept, so a run that gave up is asked again rather than remembered as settled.

### Message formats

**A corpus is one list, whichever product it came from.** `ml_stack.world.Message` is the
shape every reader returns and every emitter writes: a world id, a `source`, a `channel`, a
`sender`, a Slack-style `ts`, the text, and `thread` naming the root. `ml_stack.sources.read`
looks at a path and reads a Slack export directory, an mbox, a Microsoft Graph
`chatMessage` dump or the rows a Slack scraper writes -- each also there by name
(`sources.slack_export`, `sources.mbox`, `sources.teams`, `sources.rows`). Given the
world's people (`id -> {"label", "email"?, "handle"?}`) a reader puts `person:` ids back on
every `U0…`, address and Graph uuid; without them the product's id stays in `sender` and
`attrs["sender_kind"]` says whose it is.

```python
from ml_stack import sources
from ml_stack.world.emit import slack_export, mbox, teams, rows

slack_export(messages, people, "demo/slack", domain="pellard.example")  # users.json, channels.json, dms.json, <channel>/<day>.json, dms/<D0…>/
mbox(messages, people, "demo/mail.mbox")                              # From/To/Cc/Date/Subject/Message-ID/In-Reply-To/References
teams(messages, people, "demo/teams.json")                            # {"value": [chatMessage, ...], "channels": [...], "chats": [...]}
log = rows(messages, people)                                          # the Slack scraper's rows: channel, channelId, ts, sender, text, threadTs, scrapedAt, permalink

back = sources.read("demo/slack", people)      # sniffed; equal to `messages` up to attrs
```

The emitters write what each product actually exports -- Slack's per-day files cut at
midnight UTC with `thread_ts`, `reply_count` and `reactions`; mail that any client threads,
written through `mailbox.mbox`; Graph's `from.user`, `body.content`, `replyToId` and
`channelIdentity` or `chatId` -- with product ids minted deterministically from the world's,
and written back into each message's `attrs`. The world's id rides in the one slot each
product has for it (`client_msg_id`, Teams' `id`, an `X-World-Id` header beside
`X-World-Ts`, since `Date:` has no fraction of a second); scraper rows have none, so that
reader mints `<channelId>-<ts>` the way the scraper's pipeline does. `rows` exists so a demo
profile drops into a pipeline built on those rows with no adapter between them.

### Days that produce conversations

```python
import random
from ml_stack.world.simulate import model_writer, run, simulate, template_writer
from ml_stack.world.story import calendar

world.calendar = calendar(world, days=20, rng=random.Random(world.seed))
for message in simulate(world, days=20, writer=None, rng=random.Random(1)):   # no model
    ...
run("world/", "out/", days=20, mix=0.1, model_url="http://127.0.0.1:8080", seed=1)
```

An invented organisation is a graph until its people talk, and talk with nothing behind it
reads as noise. `world.story.calendar` lays **arcs** over the days -- for a company a
launch, an incident, a new hire's first week, an escalation, an offsite, a quarterly
review, a reorg, a deadline slip; for a community an introduction, a question that gets
answered, a meetup, a job post, a recommendation, an intro between two members; a
university has paper deadlines, grants, seminars, defences and lab moves; an open-source
project releases, bug fixes, RFCs, first pull requests and advisories; a nonprofit
fundraisers, programme launches, volunteer drives and board meetings. `World.kind` picks
the table. Who is in an arc comes from the graph: a group is any node with people joined
to it, named by the words in its label, so an incident is whoever is in "engineering" and
"support" whatever the graph calls their kinds, and a community with no `reports_to` is
scheduled from `works_with`, `part_of` and `moderates` because the sampler only ever uses
the relations it finds. Each arc says where it happens -- a Slack channel, an email
subject, a Teams chat -- and is deterministic from the seed.

`simulate` is the clock. Each working day it takes the arcs alive that day plus routine
chatter, a Poisson-ish number per person along their real relations: people who work
together talk in their team channel or a DM, a reporting line is a 1:1 DM or an email,
two people with no group in common get email or Teams. Every thread is two to eight
`Message`s with timestamps inside work hours in each sender's office timezone
(`attrs.timezone` on their place, else UTC), monotone within the thread. Who says what is
a **writer**, `(persona, prompt, context) -> str`. `template_writer` needs no model:
sentences per conversation kind and organisation kind, filled with the graph's own names
-- the speaker's project, place and subject, the group, the person they are talking to --
and never the same sentence twice in a thread. `model_writer` has a persona speak through
`converse` over the subgraph it `knows`, with its own `system` prompt, the thread so far as
turns, and its earlier threads of the same arc read back from the store as memory, so what
it said last week is what it says this week. `mix` is the share of threads the model
writes, and the arcs get it first because an arc is where consistency is noticed.

**Outcomes write back.** An arc's end leaves one typed edge in `world.graph` -- `decision`,
`moved_to`, `now_works_with` or `joined` -- carrying `attrs.said_in`, the message it was
said in, so the next conversation and the truth agree, and the graph handed in is the
graph after. **What a message costs** is two model calls: the answer, then `converse`
asking what the answer was about, which is kept because its ids are exactly the `Drew`
edges the thread memory wants. A persona is handed what its thread is about as the
`opening` -- its own entry, the project, the place, the others in the thread -- which
grounds it and stops `converse` sending an answer that touched nothing back to look. `run`
reads `graph.json`, `personas.json` and any `calendar.json` from a directory, writes
`messages.jsonl`, the graph after and the calendar used, holds `simulate.lock` while a
model is in use, and returns the counts, including `messages_per_model_call`.

## The commands

| | |
| --- | --- |
| `ml-stack-models find <words>` | search the Hub for a model, unsloth first; `files <repo>` lists the quantisations and prints the `hf:` reference to serve each; `card <repo>` reads the sampler settings its publisher recommends |
| `ml-stack-serve fit` | how many people fit at a given context, and the longest context one person can have -- from **measured** per-model KV numbers, not a formula: `--measure` serves a model once at `-lv 4` and records what llama.cpp says it allocated; `--room 24G` asks about a machine that is not this one; `--per-user N` sets the contexts in the table; `--plot FILE.png` draws who fits against the context and what the memory costs as the users arrive, with the familiar card sizes behind it, so a large model with a tiny cache can be seen overtaking a small one with a fat cache; `--write FILE` writes the Markdown; `--ui` puts the same two panels up as an interactive page on loopback (also the app's **Fit** view) |
| `ml-stack-serve status\|up\|down\|profile\|build` | one model per port, in one shape; refuses a mismatched lease; announces to the fleet; `--draft auto` and `--mmproj auto` find the speculative head and the vision projector shipped with the weights; `--spec` chooses draft or n-gram guessing; `profile` prints the shape a model measured best in and `up --profile` fills every flag not given from it; `build` compiles or downloads a current llama-server and switches to it once verified, so a release lagging master by an architecture is a permanent fix rather than a one-off `--binary` |
| `ml-stack-bench prepare\|run\|sweep\|show\|report` | time and score a graph's answers — wall clock, calls, cached tokens against read ones, KV cost, draft acceptance, and how much of the expected answer was shown; `show --rates` adds accuracy per second, per 1k tokens and per GB with the Pareto frontier, `--plot` draws it; `report` composes every run, every draft head and the measured memory into one document per model, ending in the line to serve it by (`--text`, `--md FILE`, `--room`, `--at`) |
| `ml-stack-bench queue FILE` | an evening of measurements as a file rather than the ninth zsh script of the night: one `ml-stack-bench` line per step, `#` comments, `set VAR=` with `${VAR}`, and `smoke:`/`then:` pairs where a failed smoke skips the run it guards and says so; every line is checked against the parser before the first model loads (`--dry-run` prints the plan), `--yes` and `--ceiling` are given once at the top, `--resume` skips what the store already holds since the queue started, `--detach` puts the whole evening in one background log and `status` says which step is running and what is left. Each step is its own process, so it takes the measuring lock itself and two steps never share the GPU |
| `ml-stack-claude MODEL [-- claude args]` | Claude Code on a model this machine serves, in its measured shape: the lease is taken, every model variable names the served alias, telemetry and betas a local server lacks are off, and the server goes when claude exits |
| `ml-stack-agent "task" --model MODEL` | one agentic task through the Claude Agent SDK (the `claude` extra) on the same lease; prints what it said and what it spent. `ml_stack.harness.session()` is the same for a program |
| `ml-stack-ingest DOCS --out STORE` | a shelf of documents read section by section into one graph: chapters, sections, figures and key terms out of a PDF (`ml_stack.sources.pdf` -- the publisher's outline when there is one, the way the headings are set when there is not), each section through `Client.extract` against the document contract, folded per book with `entities.fold`, and written with the book, chapter, section and page behind every node; `--images` shows the model the figures, `--chapter` and `--sample` smoke it, `--detach` and `--resume` survive an evening, `status` says how far it has got, and `--gold FILE` scores the extraction against passages with known triples (`--fail-under` makes that a gate) |
| `ml-stack-jobs status\|wait KIND\|stop KIND` | the long commands this machine has recorded -- a detached bench sweep, an ingest reading a shelf -- with the pid, the argv and the log of each: what is running, blocking until one has ended so the next command is `wait && next` rather than a `pgrep` loop written by hand, and ending one |
| `ml-stack-setup` | what this machine can do — memory a model may use and whether that survives a reboot, which architectures the installed build reads and how old it is, what is already downloaded — and what the stack does without being asked |
| `ml-stack-doctor` | what `ml-stack-setup` does not check — the checkouts (hooks installed, working tree clean, how far ahead of origin, a worktree pinned behind HEAD, whether `import ml_stack` lands in the checkout or a copy), the bench store (runs that read back as nothing, a `measuring.json` whose pid is dead, a log with no run kept from it) and the managed llama.cpp (`current` answers `--help`, the named builds, one older than 14 days); `--repo PATH` picks the checkouts, `--bench-home PATH` the store, `--yes` runs the fixes it offers; exit 1 when anything is wrong, and never a push |
| `ml-stack-train-tools` | a project's tool schemas → synthetic conversations → a fine-tuned caller → a GGUF, in one command; `--dry-run` prints the plan with counts, `--only` runs one stage, `--ask` has a served model write more questions; `from-bench` builds the same data out of the traces a bench run kept |
| `ml-stack-fleet join\|status\|leave` | one command makes this machine a peer: the checks serving depends on, a llama-server if there is none, the passphrase, the daemon (`--persist` starts it at logon too), and then what the fleet sees; `status` lists every peer with what it serves, its room, whether it is measuring, the commit it runs and how it updates itself; `leave` undoes it; `--track main` makes this machine follow a branch rather than releases |
| `ml-stack-mcp` | the same functions, as MCP tools over stdio for an agent to drive -- `serve_*`, `models_*`, `bench_*`, `fleet_*`, `world_make`, `setup_look`, `doctor`; anything long detaches and returns its log and pid; `--list` prints the tools |
| `ml-stack` | the windowed app; `ml-stack-app`, `ml-stack-traind`, `ml-stack-peers`, `ml-stack-train-run` |

## Joining the fleet

Three lines on a new machine:

```
pip install ml-stack
ml-stack-fleet join --persist
ml-stack-fleet status
```

`join` runs the checks serving depends on (the memory a model may use, a llama-server --
downloaded if there is none), asks for the passphrase every machine shares (or takes
`--passphrase WORDS`), starts the daemon, installs it at logon with `--persist`, announces on
the discovery port, and prints the peers that answered. `status` is that listing on its own:

```
NAME             URL                          ROOM             STATE        COMMIT       UPDATES      SERVING
studio           http://192.168.2.44:8770     96.0G            idle         0ce5bc5 3h   main 4m      quince-2b.gguf:8099
larch            http://192.168.2.27:8770     20.5/24.0 GB     measuring    0ce5bc5 3h   releases 2h  -
harrowgate       http://192.168.2.31:8770     20.5/24.0 GB     idle         9f2c1ab 6d   off          -
```

COMMIT is what each peer is running and how old that commit is; UPDATES is how it keeps
current and when it last looked. A fleet half on one commit and half on another is the
thing those two columns exist to make visible -- `harrowgate` above is six days behind and
following nothing, which is a machine somebody has to visit.

### Following main

A machine that is a git checkout with an editable install can follow a branch instead of
waiting for a release:

```
ml-stack-fleet join --persist --track main     # or: ml-stack-traind --track main
```

Every five minutes it asks `git ls-remote` for the head of `main`, and when it has moved:
`git pull --ff-only` (never a merge -- a checkout holding commits `main` does not have is
reported and left alone, because resetting somebody's work in progress at three in the
morning is unforgivable), `pip install -e .` only if `pyproject.toml` or a lock file moved,
and then a restart -- `launchctl kickstart` or `systemctl restart` where a login service is
installed, a re-exec where there is not. A pull that fails changes nothing, so the daemon
keeps running the code it started with and says so on the next `ml-stack-fleet status`.

Neither this nor the release update ever interrupts work: both wait for no job running, no
benchmark measuring (the same lock `ml-stack-bench status` reads, so a run started at the
keyboard counts) and no model loaded. A machine part way through a sweep is left alone
until it is not, however new the code is.

Be honest about what this is: **it runs code nobody reviewed, minutes after it is pushed**.
That is the right trade for a machine in the next room that you would otherwise have to walk
over to, and the wrong one for anything else. `--track off` goes back to releases, and it is
off unless asked for. It is remembered in the daemon's settings, so it is asked for once and
survives a reboot.

Discovery is multicast on UDP port **8771** (`239.255.77.70`, TTL 1 -- it never leaves the
segment), one above the daemon's HTTP port 8770 so one firewall rule covers both;
`$ML_STACK_DISCOVERY_PORT` moves it. Beacons are signed with the key the passphrase derives,
so a machine that does not hold it hears nothing and is heard by nobody. There is one
discovery mechanism: `ml-stack-peers ls`, the app's Cluster view and `ml-stack-fleet status`
all read the same beacons.

The app's Cluster view has the same Join button, and a "Run across the fleet" form that
builds `ml-stack-bench sweep --fleet --serve MODEL ...` from the models the peers hold,
starts it detached, and shows `status` and `history` beside it.

**For an agent**, `ml-stack-mcp` serves the same functions as MCP tools over stdio. In
Claude Code:

```
claude mcp add ml-stack -- ml-stack-mcp
```

or in a project's `.mcp.json`: `{"mcpServers": {"ml-stack": {"command": "ml-stack-mcp"}}}`.
The tools are `serve_status`, `serve_up`, `serve_down`, `models_find`, `models_files`,
`models_fetch`, `bench_run`, `bench_status`, `bench_history`, `bench_show`, `fleet_peers`,
`fleet_join`, `world_make`, `setup_look` and `doctor`; a model load, a download and a
measurement never block the call -- each returns a log path and a pid, and `bench_status`
follows it. With `pip install 'ml-stack[mcp]'` the SDK's server is used; without it the
command speaks the protocol itself.

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

**One shape per port, written down once.** llama.cpp serves a model one way at a time, so
two parts of a program that lease it differently are not two clients of one server:
whichever leases second finds a mismatch, stops the first and loads the weights again. A
`Shape` is the whole shape in one object and `Shape.lease()` is the only place it becomes
`serve`'s arguments, so the two cannot drift apart. `seat` starts the server on the first
ask, holds it per port for the process, and hands each caller a `Client` pinned to a slot of
its own -- so several conversations at once do not reprocess each other's context.

```python
from ml_stack.serve import Shape, seat, draft_for, projector_for

model = "hf:unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
shape = Shape(model=model, port=8080, seats=4, seat_context=32768, cache_type="q8_0",
              draft=draft_for(model, "auto"),         # the head shipped beside the weights
              draft_n_max=4, reasoning_budget=0,      # measured, not remembered
              mmproj=projector_for(model, "auto"),    # so the model can see
              build="unsloth")                        # a head mainline will not load

client = seat(shape, index=request_number, n_predict=16384)
```

`draft_for` and `projector_for` answer 'auto' the way `ml-stack-serve up` does -- a lease
built by hand has to resolve what the CLI resolves for itself -- and each says out loud why
it found nothing rather than serving undrafted or blind in silence. `release_all()` lets go
of every held server; `held()` says which ports are up.

A port already serving something else is refused, with the field that differs named —
the model, the number of slots, or the context each slot gets. Adopting a server of the
wrong shape hands back a lease that cannot do what was asked of it.

### A model's measured shape

The `Shape` above was typed out by hand, and every value in it came from a bench run
somebody remembered. A **profile** is that shape written down instead: one record per model
file of the serving and the asking that measured best, and the row of the store that set
it. `ml_stack/data/profiles.json` ships them and `~/.ml-stack/profiles.json`
(`$MLSTACK_PROFILES_FILE`) layers this machine's own over them, exactly as `fit.json` does.

```
ml-stack-serve profile
ml-stack-serve profile Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
ml-stack-serve up model.gguf --profile
ml-stack-bench report --profile
```

```
Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
  serve with  --context 32768 --parallel 1 --build unsloth --draft mtp-…-shared-Q8_0.gguf
              --spec draft-mtp --spec-n-max 4 --kv q8_0 --mmproj auto --reasoning-budget 0
              and -ub 2048 --spec-draft-p-min 0.5 -- llama-server's own, passed by --profile
  ask with    tight + batch + kinds + summary + greedy
  measured    85% F1 (89% recall, 83% precision) at 26.0 s/question over 10 question(s)
```

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

found = profile_for("hf:unsloth/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL.gguf")
run = found.run(port=8080, seats=4, n_predict=16384)
client = seat(run, index=request_number)
answer = converse(question, graph, client, profile=run)     # or profile="model.gguf"
```

`Profile.run()` is a **`Run`**: the whole configuration in one object, in three sections
that different code reads. `run.shape` is the `Shape` the server is leased in, `run.asking`
is an `Asking` — the ways `converse` is called with — and `run.talking` is a `Talking`, what
the `Client` is built from. `run.lease()`, `run.converse()` and `run.client()` are the only
places each becomes arguments, and `run.over(cache_type="f16", few=True, temperature=0.7)`
lays a knob over it, routed to the section that owns it rather than to whichever call takes
`**kwargs` next.

Hand the same run to the bench (`bench.served(run, ...)`), to a page (`AskRoutes.run`, and
`seated()` hands out a seat of it) and to `seat` and they lease one shape and ask one way by
construction. Three places each building their own from the record is how a knob about the
asking reached `Client.__init__` and took an 87G load down with it, and how two of them
could lease one port two ways — which llama.cpp answers by stopping the server and loading
the weights again.

`converse(profile=...)` takes a `Run`, a `Profile` or a model name, and applies the ways
that model measured best *under* anything the call said outright, so overruling one on
purpose still works. A model matched only by family (the same weights at another
quantisation) comes back with `note` saying so: a shape measured on Q4_K_XL is the right
place to start for IQ4_XS and is not a measurement of it.

Nothing writes a record by hand. `ml-stack-bench report --profile` takes, per model, **the
fastest row whose F1 the questions could not tell apart from the best** — among that model's
longest runs, held is `score.held_up`, the two 95% bands overlapping — and writes the build,
head, cache, thinking, context, asking and sampling it was served and asked with, saying in
the record which row it was. Best F1 alone would trade real seconds for a hundredth of a
point the questions cannot see; ranking *models* is still F1, because that is a different
question. The two things a kept run cannot see (llama-server's own extra flags and the
vision projector) are carried from the record already there rather than erased.

The asking a record carries is the whole asking: `tight`, `batch`, `single`, `few`, `kinds`,
`summary`, `rich`, `terse`, `reach` and `rounds`, plus the sampling — so a model measured on
three tools, twenty rounds and its publisher's temperature is served and asked exactly that,
while the next model in the same file is asked the opposite. One asking per model, and every
one of them a number somebody paid for.

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

The fleet app shows the same page under **Fit**, at `/ui/fit`, from the same two routes --
`/ui/fit.json` hands over the records with this machine's room, and the page does the
arithmetic itself so a slider costs no round trip. A sibling tab, **What it cost to be
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
`current` and every named build with commit, age and repo; `ml-stack-setup` lists named
builds under the `llama-server` line.

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

"Which companies are here?" is the other question that is not a search. Nothing in a graph
is labelled company, so `look_up` finds nothing however it is worded, and a question that
asks what *kinds* of thing are represented routes to `list_kind` instead. Its examples are
kept apart from `look_up`'s on purpose: one asks for a particular thing, the other for
everything of one sort.

Nothing narrows unless the routing was clear, `show` survives every narrowing, and an
embedder that will not answer routes nothing rather than defaulting to chat — a real
question mistaken for small talk is answered without looking anything up, which reads as a
confident answer and is about nothing.

When a search has already been run for the question — a shortlist, from the word index and
the vectors — what it found is read to the model *before* the question, as candidates to
check, and never after it as material to answer from. Measured on gemma-4-E4B: eight likely
entries handed over as the last message, phrased "use them if they answer it", took it from
58% F1 to 33%, because it echoed the list rather than selecting from it. What comes last is
what a small model answers about, so the question is the last thing it reads.

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

## A conversation of any length

```python
from functools import partial
from ml_stack.graph.thread import WINDOW, latest_summary, recall, recent, summarise, write_summary

turns = recent(store, thread, turns=WINDOW)                       # the last ten, always
summary = latest_summary(store, thread)                           # one paragraph, rarely changed
recalled = recall(store, thread, question, embedder=embed)        # two or three older turns
out = converse(question, graph, client, turns=turns, summary=summary, recalled=recalled)
...
summarise(store, thread, partial(write_summary, client))          # every EVERY turns
```

What goes back with a question, in this order: the system prompt; the latest **summary**, as
one message reading "Earlier in this conversation: …"; the turns **recalled** for this
question, oldest first, each marked as recalled; the last `WINDOW` (ten) turns, whole and in
order; the shortlist, if any; the question. The window is chosen by recency and nothing
else — a follow-up ("and where is she based?") resolves from it alone — and neither the
summary nor the recall ever takes a turn out of it. With no summary and nothing recalled the
messages are byte for byte what they were, which the ranking runs and the answer cache rest
on; pinned in `tests/test_graph_ask.py`.

`recall` is the word index over what was said, fused with the turn vectors when the same
`embedder` the turns were remembered with is given (`remember_turn(..., embedder=)`, kept
under the thread's own name), the way `search.hybrid` fuses. It never returns a turn inside
the window or a summary. `summarise` rolls the summary forward when `every` (eight) ordinary
turns have been said since the last, handing the writer the previous paragraph and those
turns with what they drew on; the paragraph is a `Turn` of role `"summary"`, joined to every
id it names that those turns rested on, out of `follow`'s ordinary window and read by
`latest_summary`. `AskRoutes` does all of this for a page: `history` returns a `History` —
the window as messages, with `.summary` and `.recalled` on it — and `remember` embeds each
turn and calls `summarise` when the subclass returns a `summariser()`.

What it costs, per question, with everything on: one embedding call for the question and
one per turn written (two), one word-index and one vector query, and every eight turns one
more short model call — over eight turns of text plus the previous paragraph, made after the
answer has gone out. Prompt tokens per question are the summary (a paragraph), up to three
recalled turns, the ten-turn window and the question; because the summary sits ahead of
everything that changes per question, it is inside the cached prefix and re-read for free
until it changes. Measured on the fake model: a fact stated at turn one is in front of the
model at turn two hundred twice, once in the summary and once recalled, and the prefix is
identical across turns 193–200. Measure the `cached` share per turn with `ml-stack-bench
concurrent` after changing `EVERY`; a summary that changes too often shows up there.

A fact stated in conversation reaches the *graph* through the change-request path, not
through any of this. The summary and the recall keep it in the model's view for this
thread; only an entry makes the tools find it next time, in any thread.

## What this measured, and what it changed

Architectures that behave unlike a dense transformer when served have their own notes
under [`docs/architectures/`](docs/architectures/README.md): what the header says, what it
meant when measured. Flash-Next's 51B-parameter n-gram table is there.

**IQ quantisations are the slow choice on Apple silicon.** Measured 2026-09-02, the same
ten questions, the same fork build, the same 32k x 2 slots, Qwen3.8-Flash-Next answering
plain with thinking on: `UD-IQ4_XS` (87 GB) took 70 s a question at 54% F1; `UD-Q4_K_XL`
(104 GB) took 44 s at 64%. The IQ formats decode through lookup tables that Metal runs
markedly slower than the K-quant kernels, so on a Mac the smaller file is the slower
model, and at ten questions the accuracy gap is suggestive rather than settled. Rule: on
macOS take a K-quant (`Q4_K_M`, unsloth's `UD-Q4_K_XL`) and spend the memory; take an IQ
build only when a K-quant does not fit at all. `ml-stack-models files` marks IQ builds on
a Mac so the choice is made knowingly. On a CUDA card the IQ kernels are fine, and the
memory saved is worth having.

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

**The table above was measured with the loose asking, which is no longer how anything
asks.** Every run in it let `show` name what the answer was about, uncapped — what
`tight=False` still does, and what `--also loose` now measures as the control. Telling
`show` instead to light only the entries that answer the question moved Qwen3.8-Flash-Next
from 43% to 83% precision over the invented community (2026-09-02) with nothing about the
searching changed, so tight is what `converse` does when it is asked nothing. Re-measure
before reading a row of that table as current.

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

Every measuring command **estimates itself before it starts** -- after the self-check, before
a download or the lock -- from what is kept: seconds per question from the newest run of
each model (at the same context when one is kept there), else a guess from its weights on
disk, times the questions, the ways one load is asked and the models, plus a load each,
printed as `estimate:` lines that `history` reads back beside the actual. Over `--ceiling`
minutes (30, or `MLSTACK_BENCH_CEILING`) it refuses with exit 5 and says what to shorten;
`--yes` runs it anyway, and a `--smoke` is never refused. No more eight-hour tests.

Serve every model being compared with the **same context and the same number of slots**, or
the comparison is of two configurations rather than two models: a model at 8k per slot is
faster and holds a smaller cache than the same model at 32k. The table prints `ctx` on every
line so a mismatch is visible rather than silent.

`look_up` is measured **as the application ships it**. With a store -- `prepare` builds one,
and `run` and `sweep` take it as their default once it exists -- every `look_up` the model
makes is `ml_stack.graph.search.hybrid`: the characters, the store's word index and, given
`--embed-url`, its vectors, fused by rank. Without one it is character matching alone, which
is what the bench measured for months while the application ran the other thing, so every
ranking it wrote ranked a `look_up` nobody used. The table prints `find` on every line --
`chars`, `words` or `meaning` -- beside `draft`, and for the same reason as `ctx`: a run
with one finder against a run with another is two measurements, not a comparison.

The questions are asked of an invented community that ships with this package, so a number
means the same thing on any machine and no real person's details are involved. Each question
may carry the ids a good answer names, which is what makes accuracy measurable rather than
impressionistic. Runs are kept in a graph store under `~/.ml-stack/bench`, so one can be
compared with another a week later.

There are two question sets, and a ranking should be read on both. The curated set,
`ml_stack.graph.community.QUESTIONS` -- a hundred scored, ten whose right answer is nobody
-- is written for nuance: two people who share a surname, a false premise about a real
person, a role nobody has but one person nearly does, a count scored as the people counted,
an answer two hops away from the person the question describes, and things only somebody's
own words say. Rather than write all hundred by hand, half of the second fifty were drawn by
handing this graph to the generator below and reading what it produced; `ml-stack-bench
prepare --mix` prints how many questions ask for each kind of answer, which is what says
whether it still measures the whole page or has drifted into being about people. The generated set, `ml-stack-world questions --world DIR --n 200`,
is derived from an invented world's truth for breadth -- hundreds of questions over
thousands of people, tagged by `kind`, with `--kinds aggregate,twohop,trap,quote` to draw
only those -- and reaches sizes the hand-written set never will. Both are fed to
`--questions`; a model that is good on one and not the other is telling you which of the two
it was tuned on.

`show` reports wall clock, model calls, prompt tokens **and how many of them were cached** —
a conversation re-sends itself every turn, so the tokens shown and the tokens actually read
are different numbers and only the second is a cost.

`pfx` is that cache per turn. A question is several calls, and the second should pay for
the tool result and the model's reply and nothing before them; when its cached tokens fall
short of the previous call's whole prompt, the system prompt and every tool schema were
read again. The table prints the share of turns that kept the prefix, `--detail` says
`cache 3/4 turns` per question, and a change to the asking that breaks the prefix -- the
cheapest speed lever there is -- shows here where the totals hide it. Blank on a run from
before it was counted, because not counted is not none.

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

`show --rank FILE.md` composes each model's line rather than reading it off one run, because
a draft head cannot change an answer -- the target verifies every token -- only the wall
clock and the memory. Accuracy comes from the model's largest run (the full sweep, undrafted,
on mainline; the newest on a tie), and cost, printed per question so a twenty-question
`drafts` run compares with a hundred, from its fastest run of at least `SHORT` questions
whose F1 held within five points of that -- a head, a draft length and a fork included -- with
the last column naming which run and which build it was (`--noise` widens or tightens the
five). A run that fell outside the noise is listed under the table as `rejected`, so a head
that hurt accuracy is seen rather than skipped, and a smoke run supplies neither accuracy nor
cost. `--rates` and `--plot` carry the same composed point per model, marked `=` and drawn as
a ring, beside the runs themselves. The question that made it was how to rank a model whose
draft head was not yet settled: before this, the ranking took each model's best-F1 run and
reported that run's cost, which ranked a drafted model at its undrafted speed, or not at all
when the drafted run was short.

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

F1 scores what was lit, and nothing else scored the prose. An answer can light the right
entries and still name one the model never found, read or showed -- a plausible name it
made up, or half-remembered from the question before -- and F1 is none the wiser. So every
row also counts the entry labels that appear in the answer's text (whole words, case aside,
as the page matches them) that no tool call produced, and `show` prints the total per run
as `made`, beside the scores; `--detail` names them, and `--rank` carries the column. Blank
on a run from before it was counted, because not counted is not none.

`sweep --serve` asks the same served model in several ways for one load: `--also terse`
describes the tools briefly, `--also card` asks with the model's own sampling, `--also
greedy` at temperature 0, `--also rich` has `look_up` say what matched and why, with a
topic hit bringing the people joined to it, `--also loose` asks the old way — `show`
told to name what the answer is about, uncapped — as a control against the tight asking
every run now uses, `--also reach` gives one tool result a page of neighbourhood rather than
a flat character cut, `--also batch` asks for every read in one call, `--also single` asks
for the opposite (one entry a read, more turns), `--also few` offers three tools and no
other way of looking, `--also kinds` keeps only the kind the question asked for, and `--also
summary` offers the whole graph at a glance. (`--also tight` is what the first way already
does, and says so.) `--reach N` and `--rounds N` are not ways of their own: they ride on
every way, as `--batch`, `--kinds` and `--summary` do. There is no asking every model wants,
which is the point of measuring ten of them on one load — `report --profile` then writes the
winner into that model's record.

```
ml-stack-bench sweep --serve gemma-4-E2B-it --also terse --also card --detach
ml-stack-bench status
ml-stack-bench tail -f
ml-stack-bench stop
ml-stack-bench sweep --serve gemma-4-E2B-it --also terse --also card --resume
```

A measurement is hours, and a child of a shell -- `nohup`, `&`, a redirect into a scratch
directory -- dies with the shell, or with the agent that opened it; a ranking sweep was
killed that way half an hour in. So `--detach` on `run`, `sweep`, `drafts` and `concurrent`
has the command re-run itself in a session of its own, with its output in a log under
`~/.ml-stack/bench/logs/`, and gives the shell back at once. `status` says what is measuring,
since when, and the last line of its log; `tail -f` follows the log; `stop` sends the pid
SIGTERM -- never a name -- which the child takes as an exit, so a model it put up comes
down with it. `sweep --resume` then skips every model and way already kept today with the
same questions, context and slots, so the killed sweep costs the model it died on and not
the ones before it.

### An evening as a file: `ml-stack-bench queue`

A night of measurements is not one command, it is nine — a fairness sample, a knob matrix
smoked one knob at a time, the hundred-question runs, the extraction runs, then the ranking
and the report. That was a zsh script in a scratch directory, rewritten nine times in one
evening (2026-09-02), with `&&` between each smoke and the run it guarded and a `--yes`
typed onto every long line; nothing could say what was running or what was left.
`ml-stack-bench queue FILE` is that evening as a file:

```
# the restart: every improvement smoked, compared on ten, then the hundred
set FX=hf:unsloth/Some-Model-GGUF/UD-Q4_K_XL/Some-Model-UD-Q4_K_XL.gguf
set BEST=--serve ${FX} --serve-draft auto --serve-kv q8_0 --context 65536 --parallel 2

smoke: sweep ${BEST} --label-suffix=-v2 --smoke
then:  sweep ${BEST} --label-suffix=-v2 --sample 10
sweep ${BEST} --label-suffix=-v2
show --rank docs/model-ranking.md
```

```
ml-stack-bench queue docs/examples/flash-next-restart.queue --dry-run
ml-stack-bench queue docs/examples/flash-next-restart.queue --yes --detach
ml-stack-bench status          # step 3/9, what is left, and what it has kept so far
ml-stack-bench stop            # the queue, and the step inside it
```

One `ml-stack-bench` invocation per line, `#` comments, `${VAR}` from a `set` line or from
the environment, and a `smoke:` whose failure skips the `then:` under it and says so — the
`&&` kept, so a shape that will not load is never measured on a hundred questions while the
rest of the evening still happens. Every line is checked against this parser as the file is
read, so `--sampel` on the last line is refused before the first model loads rather than
after the eighth measurement, and an unset `${FX}` is refused rather than expanded to
nothing and measured as the default model for six hours.

It is not a second scheduler. Each step is its own `ml-stack-bench` process, so it brings
the measuring lock, the self-check, the estimate and the smoke it already has, and a step
of a queue and a run started by hand still wait for each other; the queue holds no lock and
is only the thing that waits. `--yes` and `--ceiling` are given once at the top and passed
to every step that takes them (never to `show`), `--resume` skips every step whose label the
runs store already holds since the queue started, `--detach` puts the whole evening in the
background the way one run does — one log, named after the queue file — and `status` grows a
`queue` block naming the step in flight, the tally so far and what is left. A step that
fails on its own does not end the queue, and the exit code is 1 if any step failed. Every
summary line is one shape, so the log can be read by eye or by `grep`:

```
=== 21:41:07 step 3/9: sweep --serve hf:unsloth/Some-Model-GGUF -- ok (612s)
=== 21:41:07 step 4/9: sweep --serve hf:unsloth/Some-Model-GGUF -- skipped (0s): its smoke (step 3) failed
```

`docs/examples/flash-next-restart.queue` is the seventh of those nine scripts, as a file.

`history` answers "how much GPU time did that day cost, and how much of it kept nothing"
from the logs directory alone, one line per detached measurement, oldest first:
`started  sub  model/label  est  actual  exit  kept`. The start comes from the log's header
or its filename stamp, the end from the log's last write, the exit from what the log says --
`done`, `killed`, `crashed: <the exception line>`, `running` while `measuring.json` names a
pid that is alive -- the estimate from an `estimate:` line beside what it actually took,
and `kept` is every run in the store whose `at` falls inside that window. The last line is
the total: runs, GPU hours, and the hours that produced no kept run, which is the number to
be embarrassed by. `--since today|24h|7d|<date>` narrows it, `--json` dumps the entries,
`--home` and `--kept` point it elsewhere.

The same sweep kept twelve runs as nothing: the store took them and gave back an empty
string for each, and the smoke run had passed because the summary was printed from memory.
`save` now reads every run back the way `show` reads it before it returns, and refuses to
if what comes back is not what went in; a `--smoke` run's summary is read from the store
for the same reason. `show` counts any run that still reads back empty, and `forget --empty`
removes them.

**The runner checks itself before it spends the GPU, and smokes before it measures.** Every
measuring command -- `run`, `sweep`, `drafts`, `concurrent`, `extract` -- first drives the
exact command line it was given through the whole path with no server and no GPU: a
scripted model that takes exactly what `Client` takes, a served model that never starts, a
preflight that reads nothing, the invented community and two of its questions, into a
scratch store it reads back. It prints `selfcheck: ok (2.1 s)` and goes on, or refuses with
exit 4 and the traceback -- before the lock is taken and before anything is fetched;
`--no-selfcheck` skips it, for a run you are deliberately repeating. Then, unless the run
*is* a `--smoke` run or is told `--no-smoke`, it smokes for real: two questions (three
messages, for `extract`) through the real server and the real store, read back, before its
own questions -- on the same load where a model is served, so a sweep smokes each model as
it comes up and pays for it once -- and a smoke where every question fails ends the run with
exit 1 and the reason before anything else starts. The day this was written a new `--also
tight` way reached `Client.__init__` as a keyword and took an 87G load down with it, because
the smoke was a step in a plan and the test's fake client accepted anything; now the fake is
the runner's own -- `bench.selfcheck.ScriptedModel`, bound against the real signature, and
the tests use it too -- and the check is not a step anyone has to remember.

**A load is fetched, checked and timed before it is measured.** Every `hf:` reference a
measuring command names -- the models `--serve` puts up, the heads `--serve-draft` and
`--draft` name -- is downloaded through `hub.fetch` *before the measuring lock is taken*,
one line each with its size, because a download inside the timed window is a timing of the
network and holding the lock through it makes the next run wait for the Hub;
`--no-prefetch` skips it. Then each served model is preflighted -- shards present,
architecture read by this build, weights plus an estimated KV cache under what this machine
may wire, every flag one the build accepts -- and the report is printed under the `up in`
line, so the estimate sits beside what `kv+run` then measures; it is kept on the run as
`server.preflight`. A refused preflight prints the reason and moves to the next model
instead of ending the sweep: a sweep of five must not die on the one that does not fit. The
lease's own `load_s` and `warmup_s` are kept on the run too -- not a stopwatch around the
serve, which also holds an adopted server's nothing -- and `show` prints `load` next to
`wall`, blank for a run from before it was recorded; `--detail`'s header names it and
`--rank` carries it.

```
ml-stack-bench sweep --serve gemma-4-E2B-it --serve gemma-4-E4B-it --serve gpt-oss-120b \
    --shortlist-for e2b,e4b --serve-kv q8_0
ml-stack-bench drafts gemma-4-E4B-it --draft '' --draft auto --n-max 4 --n-max 8 --n-max 16
```

`sweep --shortlist-for e2b,e4b` gives the shortlist half only to the models whose name holds
one of those; the rest are measured plain. Either way both halves of a model are asked of
**one load** -- the shortlist is a question about the asking, like `--also terse`, and
loading the model twice to answer it measured nothing about the asking. `--plain-only`
still means no shortlist half for anyone.

`--serve-kv q8_0` quantises the KV cache of every served model. The label ends `-kv-q8_0`
and the `ctx` column reads `32k x1/q8`, because a run with a quantised cache against one at
f16 is a comparison of configurations, and the column is what stops it being read as a
comparison of models. `drafts --n-max N`, repeated, serves each head once per value --
`--spec-draft-n-max` is bound when the server starts, like the head itself -- labelled
`draft:<head>@n8`, so the table shows acceptance and wall clock per (head, n-max). Without
it, once at the build's own default. The baseline with no head is measured once. What a
head was worth is then a number rather than a division done by hand: `show` prints
`speed` beside `draft` -- the newest undrafted run of the same model, build and size,
per question, over this one, as `1.42x` -- `--rank` carries it into `cost from`, and
`drafts` ends with its own table, one row per (head, n-max) with acceptance, speedup
and how far F1 moved, and names the fastest configuration whose F1 held within the noise.

```
ml-stack-bench concurrent e2b-4x3 --conversations 4 --turns 3 --base-url http://127.0.0.1:8080
```

Everything above asks one question at a time, which is right for timing a model and wrong
for the question a server is actually asked: how many people can talk to it at once, and
what that costs each of them. `concurrent` runs N conversations of T turns each on threads
against one server, each a chain of questions with the earlier turns carried, and records
per turn the wall clock, the time until the server began generating, and what the turn
spent waiting -- its wall clock less what the server itself reports reading and generating,
which is the queueing once N exceeds the slots `/slots` reports. For the run it keeps the
wall clock over all of them (not the sum of the turns), the most the server held while they
were in flight, sampled rather than read afterwards, and F1 as usual, so a setting that
answers faster by answering worse is visible. `show` marks such a run `4x3` in the `conc`
column; the flags a build has for holding conversations -- `--kv-unified`, `--cache-ram`,
`--cache-idle-slots`, `--slot-prompt-similarity`, `--slot-save-path` -- are typed on
`ServerSpec`, so they can be varied and the build asked whether it has them before a load.
It takes the same lock as `run` and `sweep`, and `--smoke` runs two conversations of one
turn to prove the path.

### Measuring the reading, not the asking

```
ml-stack-world make --kind community --size small --seed 3 --out ./world
ml-stack-bench extract flash-next --world ./world --serve Qwen3.8-Flash-Next --smoke
ml-stack-bench extract flash-next --world ./world --serve Qwen3.8-Flash-Next --twice
ml-stack-bench show --extract
```

Everything above measures a graph that already exists. Before it exists it has to be read
out of what people said, one message at a time, and which model reads best was never
measured, because nobody knows the truth behind a real message. An invented world does:
the simulation writes each message *from* the graph, and since 2026-09-02 it writes down
what each message asserts as it goes -- ``attrs["asserts"]``, the ids of the people,
organisations, topics, places and other entries the writer put into that sentence and the
relations it stated -- so the gold is a record, not an inference. The template writer's
record is exact; the model writer's is the opening it was grounded in plus what its answer
drew on, a lower bound, and is scored separately with its coverage read as "against a
lower bound". The asserts ride through ``messages.jsonl`` and the scraper-shaped rows
``emit`` writes as an extra key the readers ignore.

`extract` samples N messages (forty by default, three under `--smoke`), stratified so an
arc's thread and every kind of routine chatter both appear -- an arc is a handful of
threads in a fortnight and a plain draw would miss it -- has the model read each into
`contracts/extraction.schema.json` (people with an optional role, organisation and place;
organisations; topics; places; relations as ``from / rel / to``) under a grammar with the
sender named as context and thinking off, folds the extractions into one graph by name
(case aside, near-spellings joined by `entities.spelling.close`, a first name joined to
its full name), and scores that against the union of what those messages assert. A world
without messages is simulated for a few days with the template writer first, and the
command says so. The estimate is printed before the clock starts: forty messages at the
guessed fifteen seconds each is ten minutes, or whatever an earlier run of the same model
measured.

The table keeps coverage and precision as separate columns per kind, because an F1 alone
cannot say whether a model missed things or made them up and those are fixed by opposite
changes to the asking; `invented` is the count and rate of extracted people and
organisations that match nothing in the gold, the reading-side twin of `made`; relations
match loosely on the name (case, underscores and spaces aside, then near-spellings) and
strictly on the ends. Under each row: the folded graph's connected components and the
share of nodes in the largest against the gold's own, so a model that scores well on
triples and builds a fragmented graph is seen; conformance, how many relations used the
world's own vocabulary; fact survival, the share of each message's assertions still
present after the fold, which catches a fold that merged two people into one; and
resolution, extracted nodes per gold node (`splits`) and gold nodes per extracted node
(`merges`), 1.00 each when perfect. `--twice` reads the sample again with the model's own
card and reports the Jaccard similarity of the two graphs: a model that gives a different graph each
run is a finding. An entry naming something the messages asserted under `others` -- a
project, a department, which the generic schema has no word for -- is neither found nor
invented. The runs sit in the same store as the answering runs, marked `kind: "extract"`;
`show` prints them in their own table under the answering one, `show --extract` alone.
`extract` takes the measuring lock like `run`, and `--detach`, `status`, `tail` and `stop`
work as they do there.

Every run records which machine measured it (`server["host"]`) and which code
(`server["commit"]`, the short sha, `(dirty)` when the tree had changes). `show` adds a
`host` column only when a store holds more than one; the ranking never composes one host's
accuracy with another's cost -- a cost run from another machine is listed as
`rejected: other host` -- and `--rates` and `--plot` name points by host. `sweep --fleet
[--peers NAME,...]` spreads the `--serve` models over the fleet: one job per model with
the same line otherwise, planned, dispatched, waited for and gathered into `--kept`, then
shown; a peer on another commit is refused before anything is dispatched.

## An invented company

A demo of a graph read out of a community needs a community, and a real one cannot be shown.
`ml_stack.world` invents one from a seed: an organised group with people who have reasonable
jobs, a voice each, and a memory -- the graph -- so that when they talk (`world.simulate`)
what they say makes sense. A company is one kind of organised group; anything that
communicates in an organised way is another, and five are built in, all producing the same
schema `ml_stack.graph.community` uses so the store, the bench, the page and the ask loop
take them unchanged:

| kind | structure |
| --- | --- |
| `company` | departments under a CEO, reporting lines with spans of five to nine, customers, partners, products, projects |
| `community` | a Slack community of professionals: day jobs at *different* invented organisations, interest groups with moderators, no reporting lines |
| `university` | departments of labs, each led by a principal investigator who `advises` postdocs and students; grants, seminars |
| `open-source` | one project of many repositories; lead and core maintainers `maintain`, contributors `contribute_to`; releases, sponsors |
| `nonprofit` | programmes under an executive director, a board that `advises`, volunteers, funders |

```sh
ml-stack-world make --kind company --size medium --seed 3 --out ./world --json
ml-stack-world questions --world ./world --n 40 --out questions.jsonl
ml-stack-bench run <model> --graph ./world/graph.json --questions questions.jsonl
ml-stack-world simulate --world ./world --out ./talk --days 20 --mix 0.3 --model-url http://127.0.0.1:8080
ml-stack-world emit --from ./talk --as slack-export --out ./export
```

`make` writes `graph.json`, `personas.json`, an empty `calendar.json` and `world.json`;
`simulate` (`world.simulate.run`) has the people talk for some working days -- arcs from
`world.story` for the kind, launches or defences or releases, and routine chatter along
whatever relations the graph holds -- templated unless `--mix` hands a share of threads to
a model at `--model-url`; `emit` writes `messages.jsonl` the way Slack, a mail client, Teams
or a scraper exports it (`--as slack-export|mbox|teams|rows`), so `ml_stack.sources` reads
the invented corpus exactly as it reads a real one.

```python
from ml_stack.world.organisation import make, summary
from ml_stack.world.questions import questions

world = make("community", "small", seed=0)   # the same world every time for a seed
world.graph                                  # nodes, edges, messages: the community schema
world.personas[world.people[0]]              # {"voice", "system", "knows": [ids]}
questions(world, 40)                         # [{"q", "expect": [ids]}, ...] for the bench
```

Sizes are `small`, `medium` and `large` -- 50, 500 and 5,000 people -- and the large one is
made in well under a second (a test holds it under ten). Every person is `part_of` a unit (a department, group, lab,
repository or programme), `works_at` an organisation, is `experienced_in` a few `topic`s
drawn from the unit's own, is `based_in` a real city or remote, `works_on` a handful of
cross-unit projects, and `works_with` the people on their team and their projects; a few are
mentored. Each has a title with a level (IC1 to IC5, manager, director, VP, C-level for a
company; faculty, postdoc and student for a university; lead, core, maintainer and
contributor for a project), a responsibility in a sentence, a start date and tenure, one to
three things they would say about their work (so `look_up`'s "said" voter has something),
and a persona: a voice in a sentence, a system prompt built from their node, and `knows`,
the graph two hops out stepping through people and projects, plus everything public.

Names are assembled from syllable tables at the moment they are asked for -- six sound
families, so five thousand people do not read as one culture -- and organisations from
word stems, so nothing here is, or can recognise, a real person. The only real things are
the cities.

The questions are generated from the truth that made the world -- who reports to whom, who
works on what, who is where -- spread over the same kinds of answer the bench's own set
covers: people mostly, then organisations, places, subjects, units, paths between two
people, events, work going spare, and a few whose right answer is nobody. A kind that lacks
a relation (a community has no `reports_to`) simply asks no such question. Four more are kinds
of *question* rather than of answer, because the bench's own set was short of them:
`aggregate` (a count, scored as the people counted; the unit or employer with the most
people, only when that is not a tie), `twohop` (who works with whoever knows a subject, or
which units or places those people are in -- the far end, never the middle), `trap` (a
false premise about a real person, whose right answer is the place exactly as the graph
has it) and `quote` (answerable from what a person said and from nothing else). Every
question carries its `kind`, which the bench ignores, and `--kinds` draws only some.

## Documents into a graph

A book is not a document to a model. A thousand-page PDF has no prompt that fits it, and the
obvious cut -- a page, a fixed number of characters -- goes through the middle of a
definition, so the sentence that says what a thing *is* arrives without the name it defines.
The unit that survives being read alone is a **section**: the book itself decided it was one
idea, it names itself, and it carries its own figures.

```sh
ml-stack-ingest textbook.pdf --out ./shelf.ladybug --model Qwen3.8-Flash-Next --chapter 2
ml-stack-ingest ~/books/*.pdf --out ./shelf.ladybug --model Qwen3.8-Flash-Next --resume --detach
ml-stack-ingest status --out ./shelf.ladybug     # how far, what failed, how long is left
ml-stack-ingest show   --out ./shelf.ladybug     # what each book was read as
ml-stack-ingest fold   --out ./shelf.ladybug     # every book so far into the store
ml-stack-ingest stop                             # end the run, after it folds what it read
```

`ml_stack.sources.pdf` does the reading. `read(path)` gives a `Document` of `Chapter`s of
`Section`s: a publisher's PDF carries an outline (`doc.get_toc()`) and that is believed, and
a book printed to PDF by a browser has none, so the headings are found by the way they are
set -- a section heading is numbered `N.M` and set larger than the body, a chapter opens with
`CHAPTER N` above the largest line on its page. Which reading was used is on the document as
`how`, because "the sections look wrong" is answered by knowing which. The text is cleaned
the way reading it aloud would clean it: a word broken across a line is put back together,
and the running head, the running foot and the page number are dropped -- found by
*repetition*, a margin line that says almost the same thing on a fifth of the pages, rather
than by matching any particular wording. The two things worth keeping that are not prose are
kept and labelled: a figure's caption, as `[Figure 2.9] ...`, and the terms the book sets in
bold. `units()` is the last cut, splitting a section over ~2,500 tokens on paragraph
boundaries and never inside one. `is_openstax()` reads the licence page, because a file
renamed by whoever downloaded it says nothing.

`ml-stack-ingest` is the other half. Each unit goes through `Client.extract` against
`contracts/extraction-document.schema.json` -- concepts with a kind and a one-line definition
*in the book's words or empty*, relations from a closed vocabulary of eighteen glossed verb phrases,
what each figure shows and which concepts it illustrates, and the key terms -- and the
extractions are folded into one graph per book with `entities.fold`, so `has_part` and
`haspart` are one relationship and a plural folds into the spelling the book uses more. Nodes
and edges go into one `GraphStore`, every one of them *pointing at* the units it was read
from -- `provenance` is unit ids and nothing else, the unit document holds the book,
chapter, section and pages, and points in turn at the hidden `run` node that read it: the
model, its build and head, sampling, the schema and instructions hashes, the version, the
host, when. `located()` and `origin()` walk the pointers back to a page and a model, so a
claim in a knowledge graph always has a page and a model behind it, without a string
copied onto every node. Each extraction's `ml_stack.telemetry.Call` is kept, so "the shelf
took nine hours" breaks down into which book, which section and how much of it was prompt.

The model is served the way the bench serves one: `--model` takes a lease for the whole run
in the shape its profile measured (`--no-profile` serves it bare), or `--base-url` uses a
server that is already up. `--images` hands the model each section's rendered figures as
pictures rather than only their captions -- the `_images` convention `graph.ask` uses -- and
without a projector the captions are all it gets, which it says rather than pretending
otherwise. A shelf is hours, so `--detach` runs it in its own session with a log under
`~/.ml-stack/ingest/logs`, a progress file beside the store records every unit that finished,
`--resume` skips those, and `status` says how many sections of how many books are done, at
what rate, what is in the store, and how long the rest will take.

### A book is readable before it is finished

A shelf of a few thousand sections is days at eighty-odd seconds a section, and a book that
is only in the store once it is finished is a book nobody can ask about until then. Each
unit's extraction lands in `<store>.<slug>.reads.json` the moment it comes back, and the
book so far is folded and written into the store as the run goes -- at a chapter's end once
twenty-five sections have gone by since the last fold, and inside a chapter longer than
fifty. Writing a book is an upsert and nothing more -- a node the store lacks is added, one
it has takes the fold's mentions, aliases, definition and provenance, an edge likewise, and
nothing is merged or removed: a knowledge graph is updated by adding to it. Joining
duplicates is a separate pass (`ml_stack.graph.tidy`), and `fold --rebuild` -- the book's
own nodes and edges out, then the full fold from its reads -- is the one path that removes
anything, for after a fix that changed what a read means. `fold --dry-run` says what a fold
would add and writes nothing.

The interval is a measured cost rather than a formality. The fold is `entities.fold`
comparing every concept name against every other, so it grows with the square of the
vocabulary: 400 invented sections of a twelve-word vocabulary fold and write in 3.8 s, and
300 sections of a 2,700-word one take 44 s to fold and 9 s to write.

`ml-stack-ingest fold --out STORE [--book SLUG]` does the same from the shelf on demand, and
is idempotent. `show` prints what each book was read as -- concepts with their kind and
definition, relations with their verb and the page behind them, the spellings and plurals
the fold joined, how many figures -- and says which books are partial. `stop` ends a
detached run: it raises inside the section being read, folds the book so far, and exits, and
the command waits for it and says whether the fold landed.

`Shelf` is the same thing for an application:

```python
from ml_stack.ingest import Shelf

shelf = Shelf("./shelf.ladybug")
for book in shelf.books():
    print(book.slug, book.read, "of", book.wanted, "partial" if book.partial else "")

graph = shelf.graph("velthorne-open-texts")   # folded from the reads: no store, no PDF
with shelf.store() as store:                  # read-only, beside the running writer
    store.nodes(kind="concept")
```

A unit that failed contributes nothing to the fold -- what a cut-off reply wrote is kept for
reading, not for believing -- and every file beside the store is written through a rename,
so a kill in the middle of one leaves the file that was there.

### Whether it does a good job

```sh
ml-stack-ingest --gold gold.json --model Qwen3.8-Flash-Next --fail-under 0.7
```

`--gold FILE` is the measurement, not an opinion. The file holds passages with the triples
they state -- `{"subject", "predicate", "object"}` and, for each, the other names a right
answer may use -- and every passage goes through *the same* extraction the shelf is read
with: the same prompt, the same schema, the same sampling. Subjects and objects are matched
through their aliases and `entities.close`, predicates through theirs, and what comes back is
recall, precision, F1 and every triple that was missed or invented, listed. Aliases are the
point: an extractor writing the singular where the gold writes the plural is right, and a
scorer without them reports a failure that is not one. `--fail-under` turns the score into a
gate.

## Searching the web

The graph's tools see the graph and nothing else. `ml_stack.web` adds the web in the same
`(schema, callable)` shape, so a model can be handed both:

```python
from ml_stack.graph.ask import converse, tools_for
from ml_stack.web import PROMPTS, tools as web_tools

converse(question, graph, client, tools=tools_for(graph) + web_tools())
# and, for routing: rank(question, {**TOOL_PROMPTS, **PROMPTS}, ...)
```

`web_search(query)` returns titles, links and a line each; `web_read(url)` returns one
page as text, cut at a sentence, and falls through to a real browser when the plain fetch
came back thin. `web_tools(vision=True)` adds `web_look(url)` — a full-page screenshot and
the page's largest pictures, returned under `_images` for the ask loop to hand to a model
that can see. The descriptions carry worked examples, for the reason measured above.

**A model cannot read the machine it runs on.** `web_read` and `web_look` refuse anything
that is not http(s) and any host that resolves to a loopback, private or link-local
address — `file:`, `localhost`, `127/8`, `10/8`, `192.168/16` and their kin — before a
byte is fetched or a browser navigates. The browser uses its own profile
(`MLSTACK_WEB_PROFILE`, default `~/.ml-stack/web`), never the scraper's signed-in one.

`MLSTACK_SEARCH` picks the engine. `ddgs` (the default; `pip install 'ml-stack[web]'`) is
keyless, fronts several engines, and is rate-limited by them: a refusal comes back to the
model as `{"none": "search unavailable: ..."}` rather than an empty list, so it moves on
instead of asking again. `searxng` is the robust option when the questions are many: a
self-hosted instance at `SEARXNG_URL` with its JSON format enabled, reached through the
stdlib, and nobody rate-limits it but you.

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

## Hooks

`scripts/install-hooks.sh` links the git hooks in `scripts/hooks/` into `.git/hooks`:
`no-real-names` refuses a commit whose staged files carry a person's name, `commit-msg`
refuses one whose message does. A third hook there is for Claude Code rather than git:
`scripts/hooks/claude-bash-guard` is a PreToolUse hook on Bash that refuses the shells
which keep getting written instead of ml-stack commands -- a hand-written `pgrep` waiter,
`nohup`, `llama-server` started directly, `find`-ing for GGUFs, `hf download`, curl probes
at the model, killing llama by name, `SKIP_NAME_CHECK=1` -- and names the command to run
instead. Wire it into a project's `.claude/settings.json` (the docstring shows the JSON);
`MLSTACK_GUARD=off` disables it for a session. Both hooks are tested:
`tests/test_no_real_names.py` and `tests/test_bash_guard.py`.

What `no-real-names` takes for a name, and what stands a name-shaped pair down, is data:
`contracts/name-shapes.json` holds the place prefixes and suffixes (a gazetteer's
"North Carolina", "Colorado River"), the job-title endings (a role catalogue's
"Software Engineer"), the `no-real-names: shapes off` file marker and the data-file
suffixes it implies, the RFC 2606 reserved domains, the `noreply` mailboxes, the uuid
and name patterns, and the file suffixes never read -- each section with a `why` saying
what it is for and when it was learned. A refusal names the rule (`patterns: nameish;
nothing stood it down`), and `NAMES_WHY=1` (or `python -m ml_stack.redact.hook --why`)
prints, for every pair a rule cleared, which section and which word did it
(`'North Carolina' cleared by place_first: north`), so the next exception is a word added
to a known section rather than a code change. `NAMES_SHAPES=path.json` reads another
rules file instead of the shipped one.

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

What *is* faked -- the model's answers, a `serve()` that would load 87G, a preflight that
would read it -- is faked once, in `ml_stack.testing.fakes`, with the real signature.
`FakeClient` is built exactly as `Client` is built, `fake_serve` / `FakeServe` take what
`serve()` takes, `FakePreflight` returns a real `Report`, and `ScriptedModel` replays tool
calls through `Client.chat`'s signature. None takes a `**kwargs` the real one lacks: a fake
that accepts every keyword lets a test pass on a keyword the real thing refuses, which is
how a `--also tight` flag once reached `Client.__init__` in a benchmark and took the load
down with it. `tests/test_testing_fakes.py` diffs every fake's signature against the real
one (`mirrors`, `drift`), so a change to the real one fails the suite until the fake follows.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
