"""Every ml-stack command: the line `ml-stack --list` prints, and the README's table.

`HELP` is one line per command, so the umbrella lists them without importing any.
`TABLE` is the rows of the README's "The commands" section, in its order;
`scripts/reference` prints them and `tests/test_cli_reference.py` holds the file to it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HELP", "TABLE", "Row", "table"]


@dataclass(frozen=True, slots=True)
class Row:
    """One row of the README's command table: how it is invoked, and what it does."""

    invocation: str
    about: str

    @property
    def command(self) -> str:
        """The console script this row is about."""
        return self.invocation.split()[0]


HELP: dict[str, str] = {
    "agent": "One agentic task through the Claude Agent SDK on a model this machine serves.",
    "audit": "Every tracked file of a repository read for a person's details.",
    "bench": "Time a set of questions through a graph, and compare two runs.",
    "claude": "Claude Code on a model this machine serves, in its measured shape.",
    "do": "A task in words, done by a served model with the ml-stack commands as tools.",
    "doctor": "The checkouts, the bench store and the managed llama.cpp, with a fix for each.",
    "fleet": "Make this machine a peer in one command, and see what the fleet sees.",
    "graph": "A rendered graph page, served with a model behind it.",
    "help": "every command with the first line of its help, or one command's own help",
    "ingest": "Read documents into a knowledge graph, section by section.",
    "jobs": "The long commands this machine records: what runs, waiting on one, stopping it.",
    "mcp": "The ml-stack commands as MCP tools over stdio, for an agent to drive.",
    "models": "Find a model that is newer than anything you remember, and serve it.",
    "peers": "Set up a cluster and see who is in it.",
    "serve": "See which model is being served on this machine, put one up, take it down.",
    "setup": "The machine facts serving depends on, and what to do about each.",
    "speech": "Speech recognition, synthesis and voice activity on this machine.",
    "store": "What a graph store holds, read by key and by scan, and whether they agree.",
    "suite": "A measurement run over several seeds, written down whole.",
    "train-run": "Train one recipe from a config file, to an adapter or a served GGUF.",
    "train-tools": "A project's tools into training data, a fine-tuned caller and a GGUF.",
    "traind": "A training daemon: one GPU box, one job at a time, reachable over the LAN.",
    "world": "Invent an organised group as a graph with people who could talk.",
}
"""``ml-stack-<word>`` -> the one line ``ml-stack --list`` prints for it."""


TABLE: tuple[Row, ...] = (
    Row("ml-stack-models find <words>",
        "search the Hub for a model, unsloth first; `files <repo>` lists the quantisations and prints the `hf:` reference to serve each; `card <repo>` reads the sampler settings its publisher recommends; `layout <model>` prints the attention layout off a GGUF header -- which layers hold a full cache, slide, recur or share it, plus experts, indexers and lookup tables"),
    Row("ml-stack-serve fit",
        "how many people fit at a given context, and the longest context one person can have -- from **measured** per-model KV numbers, not a formula: `--measure` serves a model once at `-lv 4` and records what llama.cpp says it allocated; `--room 24G` asks about a machine that is not this one; `--per-user N` sets the contexts in the table; `--plot FILE.png` draws who fits against the context and what the memory costs as the users arrive, with the familiar card sizes behind it, so a large model with a tiny cache can be seen overtaking a small one with a fat cache; `--write FILE` writes the Markdown; `--ui` puts the same two panels up as an interactive page on loopback (also the app's **Fit** view)"),
    Row("ml-stack-serve limits [--memory 90G] [--servers N] [--seats N] [--idle 10m]",
        "how much of this machine ml-stack may take, written down once and read everywhere: the memory cap rides on `hub.room`, so every preflight, fit and lease honours it without being handed it; the server and seat caps refuse a lease **before** a process is started, naming what to stop or what to raise; `--idle` is what `reclaim` and the fleet daemon act on. Every limit is off until one is set, so a machine nobody has told anything about behaves exactly as it did; `--clear` takes them all off"),
    Row("ml-stack-serve reclaim [--idle 10m] [--watch]",
        "stop the model servers nobody is using, so the memory they hold goes back. Idleness is **asked** of each server (`/slots`) rather than remembered, since a written-down 'last used' is only as good as every caller remembering to write it and an adopted server has no caller here at all; a server that does not answer is left alone, and a long single answer is never mistaken for idleness. What each look found is kept, so idleness adds up across looks and a pass can act on what the daemon has been watching -- but only over intervals short enough to be an observation: a gap in the looking is a gap in the evidence, and a pass after a silent afternoon reports nearly nothing and stops nothing"),
    Row("ml-stack-serve status\\|up\\|down\\|profile\\|build\\|escalate\\|memory",
        "one model per port, in one shape; refuses a mismatched lease; announces to the fleet; `--draft auto` and `--mmproj auto` find the speculative head and the vision projector shipped with the weights; `--spec` chooses draft or n-gram guessing; `profile` prints the shape a model measured best in for each workload -- asking, ingesting or chatting -- and `up --profile --for WORKLOAD` fills every flag not given from that record; `build` compiles or downloads a current llama-server and switches to it once verified, so a release lagging master by an architecture is a permanent fix rather than a one-off `--binary`; `escalate --add N` grows a running server's seats in place, carrying its conversations over; `memory` says how much a model may use here and `--persist` makes that survive a reboot"),
    Row("ml-stack-bench prepare\\|run\\|sweep\\|drafts\\|concurrent\\|show\\|report",
        "time and score a graph's answers — wall clock, calls, cached tokens against read ones, KV cost, draft acceptance, and how much of the expected answer was shown; `show --rates` adds accuracy per second, per 1k tokens and per GB with the Pareto frontier, `--plot` draws it; `report` composes every run, every draft head and the measured memory into one document per model, ending in the line to serve it by (`--text`, `--md FILE`, `--room`, `--at`). `--on NAME=URL` measures a server somebody else started -- a llama-server by `http://`, Ollama by `ollama://host:port/model`, an OpenAI-style server by `openai://host/model` -- and every run records what served it (program, version, format, runtime, quant), the process tree's resident peak sampled every second, and None rather than 0 for any figure that program does not report; `sweep --no-draft` serves a model in its measured shape minus the head, `--serve-label` names the runs"),
    Row("ml-stack-bench queue FILE",
        "an evening of measurements as a file rather than the ninth zsh script of the night: one `ml-stack-bench` line per step, `#` comments, `set VAR=` with `${VAR}`, and `smoke:`/`then:` pairs where a failed smoke skips the run it guards and says so; every line is checked against the parser before the first model loads (`--dry-run` prints the plan), `--yes` and `--ceiling` are given once at the top, `--resume` skips what the store already holds since the queue started, `--detach` puts the whole evening in one background log and `status` says which step is running and what is left. Each step is its own process, so it takes the measuring lock itself and two steps never share the GPU"),
    Row("ml-stack-claude MODEL [-- claude args]",
        "Claude Code on a model this machine serves, in its measured shape: the lease is taken, every model variable names the served alias, telemetry and betas a local server lacks are off, and the server goes when claude exits. With no MODEL it lists the servers already up and the models on this disk, best measured first, and takes a number or a name; `--on URL` talks to a server as it stands and leaves it up afterwards; the served model's own chat template is rewritten first, so Claude Code's mid-conversation system messages render rather than fail"),
    Row("ml-stack-agent \"task\" --model MODEL",
        "one agentic task through the Claude Agent SDK (the `claude` extra) on the same lease; prints what it said and what it spent. `ml_stack.harness.session()` is the same for a program"),
    Row("ml-stack-ingest DOCS --out STORE",
        "documents read section by section into one graph: chapters, sections, figures and key terms out of a PDF (`ml_stack.sources.pdf` -- the publisher's outline when there is one, the way the headings are set when there is not), each section through `Client.extract` against the document contract, folded per source with `entities.fold`, and written with the source, chapter, section and page behind every node; `--images` shows the model the figures, `--chapter` and `--sample` smoke it, `--detach` and `--resume` survive an evening, `status` says how far it has got, `ask` answers a question of the store so far, `--gold FILE` scores the extraction against passages with known triples (`--fail-under` makes that a gate), and `retry` re-reads the units that gave up"),
    Row("ml-stack-store check\\|docs\\|doc\\|embed\\|tidy",
        "what a graph store holds and whether it agrees with itself: `check` reads every document, node and edge by key and by scan and prints each disagreement (`--fix` rewrites a document a scan reads empty), `docs` lists the documents with their sizes, `doc PATH KEY` prints one as JSON (`--drop` takes it out), `embed --smooth N` spreads the vectors the store already holds over the graph so an entry with no text of its own is found by what its neighbours mean, and `tidy` runs the hygiene pass -- duplicates merged, inverses folded, doubtful labels flagged, conflicts, orphans and hierarchy cycles reported -- dry unless `--apply`; `--base-url URL` has a served model judge the names a spelling apart, the verbs in conflict and the doubtful labels from their passages, every verdict written to the store, `--rejudge` asks about every held verdict again, and `--gold` scores the judge against pairs whose answers are known"),
    Row("ml-stack-world make\\|questions\\|simulate\\|emit\\|check",
        "an invented organised group -- company, community, university, open-source, nonprofit -- as a graph with people who could talk: `make` writes graph, personas and calendar from a seed; `questions` draws scored questions off its truth; `simulate` has the people talk for N working days, templated unless `--mix` hands a share of threads to a served model at `--model-url`; `emit` writes what was said the way Slack, a mail client, Teams or a scraper exports it; `check` reads an export back against its truth, holds its chain of command to a hierarchy with a top, and runs every generated name through the name detector"),
    Row("ml-stack-jobs status\\|wait KIND\\|stop KIND",
        "the long commands this machine has recorded -- a detached bench sweep, an ingest reading documents -- with the pid, the argv and the log of each: what is running, blocking until one has ended so the next command is `wait && next` rather than a `pgrep` loop written by hand, and ending one"),
    Row("ml-stack-suite run NAME --seed N",
        "a measurement run over several seeds and written down whole: the seed, the commit and whether the tree was edited, the wall clock, the peak memory, how busy the card was when it started, and each metric's mean, spread and how many seeds actually produced it. A suite is a function that measures one seed and returns numbers (`@ml_stack.train.suite.register`); `--import MODULE` brings a project's own in, `--set k=v` reaches its arguments, `--out DIR` writes the result after **every** seed, so a run killed in its third keeps the two it paid for, and two runs of one suite take turns on a lock rather than timing each other"),
    Row("ml-stack-speech providers\\|transcribe\\|say\\|regions",
        "what this machine can hear and say: `providers` lists every engine -- faster-whisper, whisper.cpp, Whisper through transformers, piper, kokoro, the operating system's own voice, Silero and an energy detector that needs nothing installed -- with whether it could run here and which one is picked when none is named; `transcribe FILE` prints the text (`--json` the segments with their times), `say TEXT --out FILE.wav` writes the spoken line, and `regions FILE` the seconds somebody is speaking. `--provider NAME` picks one engine, `--language xx` names the language rather than guessing it. Audio is anything ffmpeg reads"),
    Row("ml-stack-setup",
        "what this machine can do — memory a model may use and whether that survives a reboot, which architectures the installed build reads and how old it is, what is already downloaded, what speech providers answer — and what the stack does without being asked; `--checkouts` adds everything `ml-stack-doctor` checks on its own, in the same command; `--yes` runs the fixes offered; exit 1 when anything is wrong"),
    Row("ml-stack-doctor",
        "the checkouts (hooks installed, working tree clean, how far ahead of origin, a worktree pinned behind HEAD, whether `import ml_stack` lands in the checkout or a copy), the bench store (runs that read back as nothing, a `measuring.json` whose pid is dead, a log with no run kept from it) and the managed llama.cpp (`current` answers `--help`, the named builds, one older than 14 days) — the checks `ml-stack-setup` leaves for a second command; `--repo PATH` picks the checkouts, `--bench-home PATH` the store, `--yes` runs the fixes it offers; exit 1 when anything is wrong, and never a push"),
    Row("ml-stack-train-run --recipe R --data DIR --out DIR",
        "train one recipe from a config file; `--dry-run` is twenty steps that write nothing, `--lora` fits an adapter and `--export-gguf` carries it through to a served file, `--detach` puts the hours in a background log that `status`, `wait` and `stop` follow. `ml-stack-train-run parity` instead of the flags compares this machine's two array backends operation by operation -- reductions, shapes, elementwise maths, every axis argument on more than its default axis -- and prints the largest absolute difference for each, exit 1 if any disagree"),
    Row("ml-stack-train-tools",
        "a project's tool schemas → synthetic conversations → a fine-tuned caller → a GGUF, in one command; `--dry-run` prints the plan with counts, `--only` runs one stage, `--ask` has a served model write more questions; `from-bench` builds the same data out of the traces a bench run kept"),
    Row("ml-stack-fleet join\\|status\\|leave",
        "one command makes this machine a peer: the checks serving depends on, a llama-server if there is none, the passphrase, the daemon (`--persist` starts it at logon too), and then what the fleet sees; `status` lists every peer with what it serves, its room, whether it is measuring, the commit it runs and how it updates itself; `leave` undoes it; `--track main` makes this machine follow a branch rather than releases"),
    Row("ml-stack-mcp",
        "the same functions, as MCP tools over stdio for an agent to drive -- `serve_*`, `models_*`, `bench_*`, `fleet_*`, `world_make`, `speech_*`, `setup_look`, `doctor`; anything long detaches and returns its log and pid; `--list` prints the tools"),
    Row("ml-stack <command>",
        "runs any `ml-stack-<command>` below (`ml-stack do ...`, `ml-stack bench sweep ...`, `ml-stack train run ...`); `--list` prints each with the first line of its help; bare, the daemon and your browser on it"),
    Row("ml-stack-graph geocode --graph FILE --cache FILE.json",
        "every entry that names a place given a point through Nominatim, cached so a place is asked about once, written back onto the node as `lat`/`lon` -- which is what the page's map draws. `--near K` joins each placed entry to its K closest with a `near` edge weighted 1/(1+km), `--out` writes elsewhere than over the graph"),
    Row("ml-stack-graph serve --site FILE",
        "a rendered graph page (`graph.page.render`) served on loopback as a whole document: `GET /` is the page with `window.GRAPH_LIVE` set so it asks this server rather than nothing, `/export/<path>` a file under `--export DIR` (JSON and HTML by their suffix, anything else as bytes, nothing outside the root), and the ask routes -- `/ask`, `/ask/stream`, `/thread/<name>`, `/ask/model`, `/metrics` -- answered over `--graph FILE` with `--model` leased on the first question (`--model-port` is where it is served), conversations kept in `--store PATH`; `--port` picks the listener. `/review`, `/refresh`, `/request` and `/draft` answer once a subclass gives them a queue, stages, a requests file and a drafter"),
    Row("ml-stack-audit",
        "every tracked file of a checkout read for a person's details with the pre-commit hook's reader -- names from `NAMES_GRAPH` and `NAMES_SCRAPE`, the allow-lists, the shape rules and the recogniser, asked about card numbers, IBANs and passports as well as people -- printed as a list to look at, exit 1 when there is one; `--staged` is the hook's own check without a commit, `--all` adds the noisy kinds (places, dates, URLs, addresses, driver licences), `--floor` the recogniser's confidence, `--root` another checkout, `--fixtures` the allow-list, `--json` one object per finding. `scripts/encrypted-volume.sh NAME DIR setup\\|mount\\|unmount` keeps a directory inside an AES-256 sparsebundle on macOS, the passphrase in the login keychain as `NAME-data`; setup with a source directory moves it in and leaves a symlink"),
    Row("ml-stack-bench standard",
        "the standard sets -- GSM8K (`gsm8k_cot_llama`), MMLU-Pro, IFEval and HumanEval (`humaneval_instruct`, which runs the model's code on this machine) -- through lm-evaluation-harness (the `standard` extra) against any OpenAI-compatible chat endpoint, given its `--url` and `--model` name, under the same measuring lock as a run; `--limit N` scores the first N of each set, the thinking switch (`--think` or `--no-think`) rides in every request and the JSON says which was sent (or `server default`), `--dry-run` prints the `simple_evaluate` kwargs per set, and the `--out` file (`--label` names the configuration) is one JSON per configuration -- score, metric, documents scored and seconds per set -- for a comparison to read"),
    Row("ml-stack-do \"TASK\"",
        "a task in words -- \"run benchmarks with quince-2b\" -- done by a served model (`--model` leased in its measured shape, or `--url` for one already up) with every `ml-stack-mcp` tool, the bench subcommands, the jobs, and the GGUFs and Ollama models on this machine to look up: it asks what the task leaves open, one question at a time, confirms the models it found, prints a plan and asks go (`--yes` skips only the go), runs the tools, waits for what detached, and reports what was measured and where it is; `--dry-run` prints the system prompt and the tools as the model sees them, `--rounds` caps the turns, and with no TASK a session reads tasks from stdin until EOF"),
    Row("ml-stack-bench animate COMPARISON.json",
        "a comparison of model builds as an animated graphic written to `--out FILE.mp4`, rendered with manim on a dark ground with one colour per build: a title card, decode and prefill tokens/s per prompt size with counting bars, time to first token, decode as streams are added, peak memory against a line at the machine's memory with load seconds beside, the graph bench as F1 against seconds per question with its Pareto frontier drawn, calls per question, a grid of standard-set scores, and a closing card of headline numbers; anything a build never measured is a blank marked so, never a zero. --png FILE keeps the last frame as a still, --quality l, m or h picks 480p to 1080p, --seconds N sets the length of the whole cut and every scene scales with it, --only renders named scenes alone, --dry-run prints the scene plan; the `viz` extra installs manim"),
    Row("ml-stack-bench speed\\|compare",
        "`speed` is how fast a served model reads and writes -- prefill and decode tokens/s, per stream and summed, and the time to the first token -- for each prompt size (`--prompts 512,4096,16384`, built to the count and measured) and each number asking at once (`--streams 1,2,4`), greedy, thinking off, `--generate` tokens each; `--on NAME=URL` for a standing server, `--serve MODEL` for one put up in its measured shape (`--no-draft` without its head), `--smoke` for one cell; kept as a run of kind `speed`, `show --speed` prints it. `compare --labels A,B,C --export FILE.json` assembles the configurations into one document -- what served each, its newest graph run, its speed grid, the memory either measured, the standard sets `--standard FILE.json` scored under the same label, and the head's acceptance -- with null for anything not measured, refusing a path inside a repository"),
    Row("ml-stack-peers setup\\|init\\|key\\|token\\|ls\\|pause\\|resume\\|when\\|busy",
        "the cluster this machine belongs to and who else is in it: `setup` joins one with the passphrase every machine shares (`init` mints a random key instead), `key` and `token` print the key and the bearer token it derives, `ls` lists the daemons answering on this LAN, `pause` stops taking work now and `resume` starts again, and `when` and `busy WHEN` set the hours this machine keeps to itself -- `busy 'mon-fri 09:00-17:00'`, repeatable, and queued work waits for the window to close rather than failing. `--cluster-key` picks a key file other than `~/.ml-stack/cluster.key`"),
    Row("ml-stack-traind",
        "the daemon behind a peer: one box, one job at a time, reachable over the LAN. It announces this machine (`--no-announce` keeps it invisible), serves the API and the web interface (`--no-web` only the API), runs `--slots` jobs at once, declares `--label` roles work can require, and reports the card through `--report MODULE:CALLABLE`. `--busy WHEN` and `--free WHEN` are the hours it keeps to itself and the exceptions carved out of them, `--on-paused` says whether work already running is stopped or left to finish, `--track BRANCH` follows a branch rather than releases and restarts itself when nothing is running, and `--persist` installs it to start at logon instead of serving now"),
    Row("ml-stack-help [<command>...]",
        "every command with the first line of its help, or one command's own help -- also `ml-stack help bench sweep`"),
)
"""The README's command table, one entry per row, in the order it prints."""


def table() -> str:
    """The command table as Markdown, exactly as the README carries it."""
    head = "| | |\n| --- | --- |"
    rows = "\n".join(f"| `{row.invocation}` | {row.about} |" for row in TABLE)
    return f"{head}\n{rows}"
