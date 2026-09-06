# Handoff

**Every item here is a task.** A finished task is deleted, not marked done — what exists and
why is in `README.md`, `docs/`, the code and `git log`. Each carries the context to pick it
up cold. Rules: invented names only, everywhere (`tests/known-fixtures.txt`, or a rule in
`contracts/name-shapes.json` when the refusal is a code fragment); tests build their own
fixtures and never read `~/.ml-stack`; a measurement is estimated before it runs and smoked
before it is paid for; nothing is pushed without Adam's go-ahead (a push cuts a release).
The app that drives this library is `~/ai_ceo`; its `HANDOFF.md` holds what is
Slack-specific. What was measured on 2026-09-02 and what it settled is
`docs/report-2026-09-02.md`, `docs/model-ranking.md`, `docs/architectures/` and
`src/ml_stack/data/profiles.json` (the shape a model measured best in for one workload --
`ask`, `ingest` or `chat` -- read by `ml-stack-serve up --profile --for WORKLOAD`, `sweep`,
`extract` and `converse`).

Settled: Flash-Next answers (80% F1 at 27 s/q, 100 questions) and extracts (96% node / 76%
relation F1); draft length 4 for both; one slot for extraction; `single` +8 pts on E4B at
ten questions, unconfirmed.

## The store (each needs the GPU; Adam's call)

- [ ] **The store holds APBiology and Biology2e chapter 2, sound; nine textbook PDFs are
  unread.** `~/.ml-stack/sources.ladybug` on ladybug 0.20.2: ~9,500 concepts with
  definitions, page provenance and the run that read each, after the judged pass (362
  merges, 1,489 inverse pairs, 622 conflicts judged with 282 edges dropped, 186 definitions,
  113 suspects), `ml-stack-store check` clean. The reads files beside the store are the
  truth (`ml-stack-ingest fold` rebuilds). Whether the other nine -- about four days of GPU
  at 86 s a unit, one slot -- are worth it is Adam's call; the command is `ml-stack-ingest
  ~/Documents/Textbooks/<pdf> --out ~/.ml-stack/sources.ladybug --model <flash-next>
  --images --resume --serve-port 8080`, one source at a time, and it tidies itself at the
  source's end. Two answers before that: what a question over the store scores
  (`ml-stack-ingest ask --out ... --gold FILE`, no gold questions written yet), and what
  `ml-stack-ingest sources` says once a second full source is in.
- [ ] **Watch for units that still run to the ceiling.** The document schema caps every
  list (`maxItems`), so the grammar closes the array; a unit that still fails is read once
  more, then given up on; `status` counts those, its whole reply is `raw` in the reads
  file beside the store, and `ml-stack-ingest retry --out STORE` frees them after a fix.
  If one still circles under the cap, try DRY sampling for extraction and measure it on
  the gold gate.
- [ ] **Score answering over the store.** Write twenty invented-free but real-source
  questions with expected concept ids (`ml-stack-ingest ask --gold FILE`, the bench's
  scorer) and run them through Flash-Next in its profile; until that number exists, "a
  usable bio graph" means queryable and sound, not scored. The judge's conflict verdicts
  are worth reading first: it kept both edges in cases where its own reason said one was
  a misreading ("Larynx part_of trachea" beside "Larynx precedes trachea") -- the
  instructions now say keep both only when the passages support both (2026-09-03); the
  store's verdicts predate that and are worth `ml-stack-store tidy --rejudge
  ~/.ml-stack/sources.ladybug`, ~an hour.
- [ ] **Fill the store's "between sources" section.** `ml-stack-ingest sources` reads
  `tidy:merges`, written by every fold and tidy from 2026-09-04 on; the store's 210 shared
  concepts were joined before it existed, so the section is empty. `ml-stack-ingest fold
  --out ~/.ml-stack/sources.ladybug` re-folds every source from its reads (no model,
  minutes) and writes the log; it rewrites the store, so take a copy beside it first.

## Per-request draft depth

`docs/llama-cpp-per-request-speculative.md` is the patch written up as a pull request would
put it. `patches/llama.cpp/0001-speculative-per-request.patch` is the patch;
`ml-stack-serve build --from source` applies it and names the build directory after its
digest.

- [ ] **Adam's call: send it to `ggml-org/llama.cpp`.** Nothing has been opened. The write-up
  is what a pull request would say.
- [ ] **Flash-Next's MTP head still needs the unsloth fork, and this machine's `unsloth`
  build is a downloaded release, so it carries no patch.** Serving Flash-Next therefore
  still takes a depth at startup, which is the workload the per-request depth was wanted
  for. Building the fork from source applies the patch the same way
  (`ml-stack-serve build --from source --repo unslothai/llama.cpp --name unsloth --ref
  <tag>`), but the patch has not been tried against that tree and may need rebasing onto
  it.
- [ ] **The dflash and dspark draft implementations still draft the depth the server
  started at.** They size one block up front, so a request asking for less gets the right
  number of tokens (`common_speculative_gen_draft` truncates) at the cost of the deeper
  draft. Depth sweeps over those two read acceptance correctly and latency optimistically.

## Measurements queued (each needs the GPU; sample first)

- [ ] **Nothing has measured any model for `ingest` or `chat`.** The record now holds a
  shape per workload and the five shipped records are all `ask`, so `ml-stack-serve up
  --profile --for ingest` serves the graph-asking shape and says so. What is missing is a
  measurement: `ml-stack-bench extract` records `workload: ingest` and `report --profile`
  writes that slot, so a sampled extraction run over Flash-Next at draft depths 2, 4 and 8
  would fill it. What makes it worth the GPU is that reading and answering accept the
  head at very different rates *at the same depth 4*: reading a textbook unit accepts 96%,
  answering a question over the graph accepts 75%. Both are on this machine and both
  reproduce --

  reading -- `draft_n` and `draft_n_accepted` sit on every call in the reads file beside the
  store, under `<unit>.calls[]`, not on a bench run, which is why `runs.ladybug` has none:

      import json
      from pathlib import Path
      d = json.loads(Path("~/.ml-stack/sources.ladybug.biology2e.reads.json")
                     .expanduser().read_text())
      calls = [c for r in d.values() if isinstance(r, dict) for c in (r.get("calls") or [])]
      sum(c["draft_n_accepted"] for c in calls), sum(c["draft_n"] for c in calls)
      # 30,648 accepted of 31,757 drafted -- 96.5%, biology2e chapter 2.
      # The apbiology file the same way: 1,168,485 of 1,213,939 -- 96.3% over the book.

  answering -- `ml_stack.bench.comparison._acceptance` over the runs in `runs.ladybug` whose
  model is Flash-Next: depth 4, 83 runs, median 75.4%; depth 2, 79.1%; depth 8, 48.1%.

  Neither says what the best *reading* depth is, because every ingest run so far was served
  at the profile's 4 and `--n-max` has never been run at another. A head accepted 96% at 4
  has room a head accepted 75% does not, so the reading depth is likely higher than 4 rather
  than lower -- that is an inference from the acceptance, not a measurement, and it is what
  the sweep would settle. `chat` has no bench of its own -- the world writer and
  `fleet.chat` both point at a server already up -- so measuring it needs a command first.

Adam, 2026-09-04: "we're never going to have that many users, so flash-next is the way to
go (with shared MTP) always." So ranking a second model is not worth GPU: `gpt-oss-20b`
(a profile, no fit record), `Qwen3.8-27B` (a fit record, no profile) and Flash-Next
`IQ4_XS` (a fit record, no profile) stay half-measured on purpose, and `ml-stack-fleet
plan` names them as unplaceable rather than guessing.

- [ ] **Re-read every draft-head speedup taken before 2026-09-05.** The bench pairs a
  drafted run against a "baseline" to say what the head bought. That pairing used to match
  on the label string, so it happily paired runs that also differed in cache type,
  reasoning budget, slot count or asking. On this machine's store: 225 runs carry a head,
  the old rule paired 175 of them, the new rule (identity minus the head) pairs 54, and the
  121 that fell away were comparisons across more than one change. Any head speedup quoted
  from those runs was partly measuring the KV quantisation or a different way of asking.
  `ml-stack-bench show` and `compare` now use the new pairing, so the numbers are already
  right going forward; what is pending is deciding which recorded conclusions rested on the
  old ones. The three-configuration comparison above is the first place it matters.

- [ ] **`single` on E4B at a hundred questions** (`sweep --serve gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
  --profile --plain-only --also single --yes`, ~10 min): the one asking-way change that
  moved a small model, unconfirmed at ten. If it holds, `report --profile` sets it.
- [ ] **Smoothed embeddings on the bench.** `graph.smooth` spreads every vector over the
  graph (`ml-stack-store embed PATH --smooth N`, or `remember(..., smooth_hops=N)`); nothing
  says whether it makes the answers better. The measure: `ml-stack-bench ready --embed-url
  ... --embed-model ...` to build the vectors, `embed --smooth 1` and `--smooth 2` over
  copies of that store, then a sampled `sweep --plain-only --sample 20` on each against the
  unsmoothed one. What to watch is recall on questions about entries nobody wrote about --
  the whole point -- against precision on the rest, since a vector pulled towards its
  neighbours is a vector less about itself.

- [ ] **Constrained ids on E2B and E4B.** `sweep --serve <gemma> --profile --constrain-ids
  --sample 20` against the kept plain runs: precision up, recall held. A profile records
  `constrain_ids` and the answer cache keys on it, so `report --profile` can set it.
- [ ] **Thinking off on the gemma family** (`--reasoning-budget 0` on a sampled sweep each).
- [ ] **Four unified slots decoded slower at one stream; say whether the head is why.**
  Measured 2026-09-05 on E2B (`ml-stack-bench speed --serve gemma-4-E2B... --streams 1
  --prompts 4096 --context 65536 --parallel N --serve-kv-unified|--no-serve-kv-unified`, two
  runs each): one slot 125/153 tok/s, one slot unified 143/152, four slots 152/156, four
  slots unified 121/122 -- and the draft head's acceptance on the four unified slots was 44%
  where every other shape reported 70%. Prefill ~2,600 tok/s in every shape. So one seat
  by default costs nothing at one stream, and the page's 2026-09-04 shape (four unified
  slots) was ~20% slower to decode, but whether the cache or the head under it is the
  reason is one more run: the same four shapes with `--no-draft`.
- [ ] **The stable prefix: reading fell, calls rose; measure again on a quiet machine.**
  Nine sampled questions on the page's Flash-Next against `Qwen3.8-Flash--all-plain-kv-q8_0-rb0`
  (kept as `flashprefix-plain`): the last call of a question reads ~50 tokens with the whole
  prefix cached where it read 3193 with none; prefix hits 100% from ~80%; uncached read
  2.1k a question from 5.2k; F1 81% against 85% inside the ±19 band. But calls went 6.2 to
  7.7 a question -- 22 `show` calls over nine questions where one is the design -- and
  written tokens 5.1k to 7.8k, so the wall clock (27.7 against 25.6 s/q) did not move; three
  agents' suites were running, so the wall clock is unreliable either way (Adam). Next:
  `show --trace flashprefix-plain`, find what invites the second `show` (the show nudge now
  offers every tool), then a quiet `--sample 20` of both.
- [ ] **The fine-tuned tool caller.** `ml-stack-train-tools from-bench` over the traced runs
  (traces are on by default at ≤20 questions; the hundred-question runs before that carry
  none -- rerun Flash-Next's hundred with `--trace` for ~5,000 turns), then
  `ml-stack-train-run --recipe tool-calls --size e4b --lora --export-gguf --yes` (~18 h
  here; Adam's go-ahead), then the measure in `docs/research/tool-caller-finetune.md`.
- [ ] **Watch ggml-org/llama.cpp#27836, then measure it here; nobody has compared it to
  the fork.** A draft, last touched 2026-09-02, checked 2026-09-06. Mainline is b10825,
  the fork b10715, and the profile pins `--build unsloth` so mainline moving changes
  nothing until this lands.

  **The PR's numbers do not answer whether it beats the fork**, and reading them as if
  they did is a mistake already made once here. Its table is MTP against *no speculation*
  on an M3 Max: 27.43 tok/s baseline, 89.2% acceptance and 37.22 tok/s at
  `--spec-draft-n-max 2`, 85.7% and 38.83 at 3. That is **UD-IQ4_XS at depth 2 and 3**,
  in a harness it names but does not describe. This machine serves **UD-Q4_K_XL at depth
  4** against graph tool-calling, and acceptance falls as depth rises, so the PR's own
  trend puts depth 4 below both figures. Nothing in it is a fork comparison.

  So the only way to know what it is worth is `ml-stack-bench drafts` on this machine, on
  this quantisation, at this depth, against the questions this graph asks -- once with the
  fork and once with mainline. Acceptance sets tokens per forward pass, so the difference
  is decode speed on the model this project actually runs, which is worth measuring
  properly rather than estimating. Taking it costs a download: `ml-stack-serve build`,
  then the profile's `--build unsloth` goes.

  One thing the PR does establish and is worth keeping: temperature-0 output is
  byte-identical with the head on and off, so speculation is not changing what the model
  says. Its benchmark runs were produced with an agent, per its own disclosure.

## Flash-Next, two builds: llama.cpp (unsloth GGUF Q4_K_XL, with and without the draft head) against Ollama (MLX, nvfp4)

Adam, 2026-09-03: which is better, faster, or both; a LinkedIn graphic; "if ollama can't
keep up due to lack of drafting head, points in llama.cpp's favor"; "make sure we're
measuring actual max mem usage during the test". The machine has 128 GB; the GGUF serves at
~90 GB and the Ollama model is 104 GB on disk (`qwen3.8-flash-next:125b-mlx`, 1658
safetensors tensors, `file_type nvfp4`, default context 262144), so the halves run one at a
time with the page's server down for the Ollama half.

- [ ] **Run the three configurations and draw them.** Everything is landed and tested on
  fakes; nothing has touched a real server yet, so `--smoke` each line first. Labels carry
  the profile's suffix as `served()` always did; `served_by` is recorded on every run.
  ```
  # 1. llama.cpp + draft head, against the page's server on 8080 (its profile shape)
  ml-stack-bench sweep --on flash=http://127.0.0.1:8080 --plain-only --short
  ml-stack-bench speed --on flash=http://127.0.0.1:8080
  # 2. llama.cpp without the head, served by the bench in the measured shape minus -md
  ml-stack-bench sweep --serve Qwen3.8-Flash-Next-UD-Q4_K_XL --serve-label flash --no-draft --plain-only --short
  ml-stack-bench speed --serve Qwen3.8-Flash-Next-UD-Q4_K_XL --serve-label flash --no-draft
  # 3. Ollama, after `ml-stack-serve down` on 8080 (104 GB does not fit beside 90)
  ml-stack-bench sweep --on flash-ollama=ollama://127.0.0.1:11434/qwen3.8-flash-next:125b-mlx --plain-only --short --context 32768
  ml-stack-bench speed --on flash-ollama=ollama://127.0.0.1:11434/qwen3.8-flash-next:125b-mlx --context 32768
  # the standard sets, per configuration (~40 min each at --limit 200, thinking off)
  ml-stack-bench standard --url http://127.0.0.1:8080/v1 --model flash --label flash-plain --no-think --limit 200 --out ~/.ml-stack/bench/standard/flash-plain.json
  # then
  ml-stack-bench show --speed
  ml-stack-bench compare flash-plain flash-nodraft-plain flash-ollama-plain --standard ~/.ml-stack/bench/standard/*.json --export ~/flash-comparison.json --title "Flash-Next three ways"
  ml-stack-bench animate ~/flash-comparison.json --out ~/flash-comparison.mp4 --png ~/flash-comparison.png
  ```
  Memory is sampled over the serving process tree every second (Ollama: the listener's
  children hold the weights) and kept as `resident_peak`. `speed --serve` defaults
  `--parallel` to the most streams asked (4) and the per-seat context to the largest prompt
  plus the generation, so set `--context` for the 4-stream cells. `ttft_s` is a streamed
  first token on llama.cpp (`ttft_from: stream`) and the server's prompt clock on Ollama
  (`prompt_ms`, marked `*`). Estimate before each: ~45 min the hundred-question graph
  bench, ~10 min speed, ~40 min the standard sets.
- [ ] **One measured call on Ollama first.** Whether `prompt_eval_count` includes a cached
  prefix is not in its docs; whether `think: false` holds for this model; what the runner's
  process is called on 0.33.3 (found from source, not seen). Then the Ollama half.

## Driving what is built (each needs the served model; minutes)

- [ ] **A number for the local harness beside the page's.** `ml-stack-claude <flash-next>
  -- --print "say hello"` and `ml-stack-agent "read README.md and say in two sentences what
  this is" --model <flash-next> --allow Read` both ran for real on 2026-09-05 (the agent: 5
  turns in 382 s, 79k tokens read with 185k from the cache, 1.2k written, a right answer).
  Untried: the stream-idle watchdog (five minutes of silence aborts --
  `CLAUDE_STREAM_IDLE_TIMEOUT_MS`), and what `Usage` reports against the server's own
  `/metrics`. Then a small task set measured the bench's way.

## Store integrity

- [ ] **Two ladybug faults are worked around here and stay here.** Adam, 2026-09-04: no
  upstreaming to public repositories. 0.18.x: a single `DETACH DELETE` in a ~10k-node
  store blanks other nodes' string columns (reproduction: `tests/test_graph_store_scale.py`).
  0.20.2: the cached-physical-plan fast path re-executes a parameterized MERGE against a
  table rewritten since and segfaults, and the text index returns a node once per version
  written (reproduction: the store's `_written` docstring; two lines). ml-stack is on
  0.20.2 with the per-write guard and the pin `>=0.19,<0.21`; the probes gate any bump,
  which is the whole of what is left to do about them.

## Measuring across the fleet

- [ ] **Run it for real across two machines.** Everything is tested against fakes and
  loopback; nothing has crossed a real network or a real Windows box. One visit:
  `irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 |
  iex` (the app) or the `--headless` mode, then from here `ml-stack-fleet status`, a
  `ml-stack-fleet plan --users 3 --context 16384 --apply`, and a `sweep --fleet --serve
  gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf --sample 6`. Expect bugs; the daemon's log and
  `ml-stack-doctor` are the first two places to look. After that the Windows box follows
  releases or main on its own (`fleet status` shows COMMIT/UPDATES).
- [ ] **Two machines with the same name share one speed record and one run history.**
  Discovery is safe: each daemon mints a random token when it starts advertising and peers
  are keyed on that, so two machines both called `Mac` are two peers and both get work.
  Nothing else uses that token. `fleet/rates.py` keys measured speed on `(peer name, kind)`,
  so two machines of one name average into a single number and placement scores them
  identically; `fleet/bench.py` stamps gathered runs with `_name_of(peer)`, so their
  measurements land in the store indistinguishable. Nothing checks for the collision at
  join. The default name is the hostname, and this machine already records `"host": "Mac"`.
  Fixing it means a stable per-machine identity (the current token is per process, so a
  restarted machine is already a new peer to anything that remembers) and keying rates and
  the run host on that, leaving the name as a label. Adam's call on what a person sees:
  refuse a duplicate name at join, accept it and disambiguate in the listing, or accept it
  silently and fix only the keying.

- [ ] **A router across the fleet.** `ml-stack-fleet plan --apply` serves the placement;
  nothing yet sends a new session to a free seat on the best model. The daemon's `/infer`
  proxies by model name on one machine; the router picks the machine.

## What the window has not been driven through

- [ ] **What WKWebView renders the same as Chromium, so far.** Light-DOM custom elements,
  `[hidden]`, the sheet's backdrop blur, the context `input[type=range]`, the styled
  checkbox and radio rows, a card taller than the window with its own scrollbar, and the
  system sans stack. Nothing needed a resize to appear and no font fell back. Not yet
  driven in the window: the Chat screen with a model behind it, the Models screen's
  download, and the Fit screen's tables beyond confirming they draw.
- [ ] **`serve_forever` preferring `Settings.name` over the hostname has no test.** It was
  driven by hand -- the wizard renames the machine, the daemon restarts, `/health` answers
  with the new name -- but every test that boots a daemon builds `make_handler` directly, so
  nothing covers the resolution order. A subprocess boot with a settings file would.
- [ ] **The first-run library list showed PyTorch as not installed on a machine that could
  import it.** `/ui/libraries` decides `installed`, and the tick came out clear where torch
  was importable from the daemon's own interpreter. Worth finding out what it checks before
  someone downloads 200 MB they already have.

## From beehavior

`~/Documents/repos/beehavior` is the other repository that grew these shapes, and its own
`ML_STACK_MIGRATION.md` lists what it should now delete rather than keep. Two pieces came
here on 2026-09-05 (`ml-stack-suite`, `ml-stack-serve limits|reclaim`); three are still
worth taking, in this order:

- [ ] **A runtime redactor** (`tooling/compliance/{text_sanitizer,llm_sanitization}.py`,
  ~380 lines): a Presidio sanitizer built once per configuration and applied to prompts,
  model output and the turns a page stores. `ml_stack.redact` audits a repository at commit
  time and has nothing for the ask path, which is where a graph page sends real text.
- [ ] **The copyright carve-out gate** (`tooling/graph/ingestion/copyright_carveout.py`,
  183 lines): refuses a verbatim provenance snippet taken from a source that embeds
  third-party material under a permission granted to somebody else. Every ingested node
  keeps a snippet, so this is the same problem here.
- [ ] **A terminal command centre** (`tooling/daemon/tui.py`, 477 lines, Textual): the
  facts the fleet app's cluster view already shows, in a terminal. Worth it only if a
  headless machine wants one.

## The tensor toolkit

- [ ] **Six of the toolkit's functions have no caller here.** `graph.tensors` (the mapping
  graph as arrays) is called by `graph.smooth`, and `topology.knn_edges` by
  `graph.places.geocode`. `batch_graphs`, `resolvent_sweep`, `decompose_to_dags`,
  `mst_edges`, `morton_codes` and `spatial_window_edges` are tested library API that nothing
  in this repository calls. Each has an obvious home when the work arrives -- `batch_graphs`
  when a graph model is trained on many graphs at once, `resolvent_sweep` for influence down
  a DAG (a `reports_to` tree, a `part_of` hierarchy), `mst_edges`/`morton_codes`/
  `spatial_window_edges` for a cheaper `--near` on a graph too big for the full distance
  matrix `knn_edges` builds. The last one is the near-term task: `geocode --near K` is
  O(n^2) in placed entries, which is nothing at 300 and a problem at 30,000.

## Speech

- [ ] **The beacon does not say which machines can hear.** `refresh(b)` inside
  `fleet/daemon.py::serve_forever` fills `b.device` with `serving` and `models`;
  `"speech": [names that probe ok]` beside them would let `ml-stack-fleet status` and the
  cluster view show which peer to send a recording to, and would make
  `Peer.find_one(require="speech")` work. `ml_stack.speech.service.providers()` already
  returns exactly that list. Left out because another branch was editing `serve_forever`.
- [ ] **Nothing streams.** `StreamingASR` in `speech/protocols.py` is a protocol no
  provider implements, and `POST /speech/transcribe` takes one whole file. A push-to-talk
  button wants partial text while the person is still speaking: a provider that yields
  `Transcript`s off a chunk iterator, and a route that streams them the way `/ask/stream`
  does.
- [ ] **No microphone anywhere in the interface.** The chat screen has no record button, so
  the only way to reach any of this is the command line. The route and `Peer.transcribe`
  are the far end; what is missing is a control that records in the browser and posts the
  audio.

## The window

- [ ] **The Tauri window is proven on macOS only, and three things are unfinished.** `app/`
  is a Tauri 2 project (`app/src-tauri`, tauri 2.11.5, tauri-plugin-shell 2.3.6,
  tauri-plugin-window-state 2.4.1, CLI 2.11.4 pinned in `app/package.json`). It bundles no
  HTML: the window opens on `http://127.0.0.1:8770/ui/`, the page `ml_stack.ui.assemble`
  builds, and `window.mlStackNative()` in `close-sheet.html` returns null in a browser so
  the same page still works headless. The daemon is a sidecar, the PyInstaller binary
  `packaging/ml-stack.spec` builds; `packaging/build.py --bundle` freezes it, copies it to
  `app/src-tauri/binaries/ml-stack-headless-<target triple>` and runs the bundler.
  `--no-window` stops before that, for a machine with no Rust. An app command reached from
  the loopback page needs three things or it is refused by the ACL: the capability's
  `remote.urls`, `build.rs` declaring the command names, and the name in the capability's
  permissions.
  - **Windows and Linux have never been built.** The release workflow installs Rust, node
    and (on Linux) webkit2gtk and runs the same build; the NSIS setup and the AppImage the
    bundler makes there are what `install.ps1` and `install.sh` now expect, and neither has
    run. The Linux AppImage is installed as `~/.local/bin/ml-stack`.
  - **`hub.room` pulls the whole graph library in.** `room()` imports `serve.limits`, which
    runs `ml_stack/serve/__init__.py`, which imports `serve/shape.py`, which imports
    `graph.asking` and so `graph/__init__.py` and `graph/topology.py`, which needs numpy.
    The beacon calls `room()` every ten seconds. numpy is bundled into the frozen daemon to
    get past it (11.7 MB to 15.5 MB); the fix is that a memory number should not reach the
    tools layer.
  - **Adam's call: what a person downloading it gets on macOS.** There is no Apple
    developer certificate and there is not going to be one, so `packaging/build.py`
    ad-hoc signs the app, which is enough to open it on the machine that built it. A file
    fetched by `curl` carries no `com.apple.quarantine`, so the install script's path opens
    without complaint; a `.dmg` or a zip a browser downloaded is quarantined and refused
    with "the developer cannot be verified". Nothing publishes a `.dmg` today. Either it
    stays that way and the release page offers only the install command, or a `.dmg` is
    published and the README tells a first-time reader to open it from the right-click
    menu once.
  - **Adam's call: which updater, and what happens to the rule that nothing updates while
    a job runs.** `ml_stack.fleet.updates` is what runs now, unchanged: it asks GitHub
    once a day, checks the download against the digest GitHub publishes, swaps
    `ml-stack.app` into place and restarts, and `in_the_way` holds it back while a job is
    running, a benchmark is measuring or a model is loaded. It keeps working because the
    release zip still holds `ml-stack.app`. Tauri has its own updater, with its own
    keypair (`tauri signer generate`, the private key and its password in
    `TAURI_SIGNING_PRIVATE_KEY` and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` in the release
    workflow's secrets, the public key in `tauri.conf.json`; lose the private key and no
    published app can be updated again). It is not turned on, and it knows nothing about a
    running job, so turning it on means writing that gate again on the Rust side.

## The interface

- [ ] **Nobody has drawn 3,000 nodes in the page.** The page ships the whole graph as one
  JSON blob and lays it out in the browser; the biggest graph it has held is a few hundred
  nodes. `ml-stack-world make --size 5000` gives one to try, and `most_messages` already
  trims the quotes; nothing trims the drawing.

## The tool loop

- [ ] **`graph/ask.py:_converse` is 597 lines.** It takes an `Asking` now, so its signature
  is fourteen parameters, but the body still holds seven closures (`note`, `under`,
  `read_back`, `step`, `_searched`, `dispatch`, `settle`) and calls five schema-rewriting
  variants (`_rich`, `_tight`, `_batched`, `_singled`, `_few`) that each copy every schema
  to add a sentence. What to do with it: lift the schema variants into one `Asking`-driven
  pass that rewrites a set once instead of five times over; lift the closures that only
  read `graph` and `known` to module-level functions taking them; and split the loop body
  -- one turn: build the offer, call the model, read the reply, run the tools -- from the
  bookkeeping around it (`Answer.steps`, `Spent.part`, the emitted events). Nothing in it
  may change the system prompt or a tool description:
  `tests/test_asking_is_the_same_asking.py` hashes both for every way the bench measures,
  and `graph/cache.py:fingerprint` puts those bytes in the answer-cache key that 290 kept
  runs were measured against.

## The commands

- [ ] **Twenty commands still build their own parser and keep their work in the handler.**
  `ml_stack.command` holds the shared options and a `Group` that collects subcommands and
  answers as `main`; `ml_stack.cli.reference` holds one line per command and the README's
  table, and `tests/test_cli_reference.py` fails when the two disagree. On it so far:
  `ml-stack-serve` (work in `serve/ops.py`), `ml-stack-world` (`world/ops.py`) and
  `ml-stack-jobs`. The pattern, per command: make a `<package>/ops.py` whose functions take
  typed arguments and return values, raising rather than printing (`serve.ops.Refused`
  carries the lines); leave the handler to parse and print; declare each subcommand with
  `@COMMANDS.command(name, help=..., options=[option("port", ...), flag("--own", ...)])` in
  the order the old parser added them, so `--help` is unchanged; set `main = COMMANDS.run`
  and delete the `def main`. Diff every subcommand's `--help` and the command's own output
  against `main` before and after -- that is what catches a moved flag.
  Worth doing next in this order: `ml-stack-bench` (1,886 lines, two `main`s, and the last
  screen-scrape in `mcp.bench_show` and `do.bench_cli` is waiting on it), `ml-stack-setup`
  and `ml-stack-doctor` (one module, two entry points), `ml-stack-ingest`,
  `ml-stack-models`, `ml-stack-store`, `ml-stack-speech`, `ml-stack-train-run`,
  `ml-stack-train-tools`, `ml-stack-do`, `ml-stack-claude`, `ml-stack-agent`,
  `ml-stack-graph`, `ml-stack-audit`, `ml-stack-suite`, `ml-stack-fleet`,
  `ml-stack-peers`, `ml-stack-traind`. A command whose row in the README's table changes
  is a change to `ml_stack.cli.reference` and `scripts/reference --write`.

## Where state lives

- [ ] **`ML_STACK_HOME` does not move the llama.cpp builds.** `ml_stack.home` reads the
  environment when it is called, so every home and record file follows the variable -- but
  `serve/binary.py` still binds `MANAGED_ROOT`, `MANAGED_CURRENT` and `MANAGED_NAMED` at
  import, and `serve/build.py` derives `ROOT`, `SRC_DIR`, `BUILDS_DIR`, `CURRENT_LINK`,
  `NAMED_DIR` and `NAMED_SRC_DIR` from them, so a process that sets the variable after
  import still finds the builds under the old root. `hub.HUB_CACHE` is the same shape for
  `$HF_HOME`. Making them functions is about 40 call sites in `build.py` and eight
  `monkeypatch.setattr` in `test_serve_build.py`, `test_hub.py` and `test_doctor.py`.
- [ ] **`limits.json` and `idle.json` are under the cache root, not the state root.** What
  this machine is set to allow, and how long each server has been idle, are both state a
  person would lose by clearing caches -- the same reason `servers.json` moved to
  `~/.ml-stack`. Moving them needs the same read-the-old-path-once step `manager.lease_file`
  does, and `MLSTACK_LIMITS_FILE` keeps working either way.

## Layers

- [ ] **Seventeen imports still cross the layers `tests/test_layers.py` sets out.** The
  layers are core, model, graph, machine, tools; a package may import downwards, and
  sideways only when the other package does not import it back. `KNOWN` in that file lists
  every edge that breaks it, and the test fails both on a new violation and on a `KNOWN`
  entry that is no longer one, so the set only shrinks. Four are two-way cycles inside one
  layer: `serve` <-> `fleet` (3 files one way, 7 the other), `serve` <-> `setup` (2 and 1),
  `fleet` <-> `setup` (1 and 1), and `sources` <-> `world` (5 and 1, going when
  `world.Message` moves down). Nine reach up a layer: `setup`, `fleet`, `ingest` and
  `serve` into `bench` (1, 2, 2 and 1 files); `gguf`, `hub`, `graph` and `ingest` into
  `serve` (1, 1, 3 and 1); and `graph.data` into `train.backend` for a device handle, where
  the ops that sit under both belong below `graph` rather than in the tools layer. The
  `graph` -> `serve` three are function-local reaches: `profile_for` in
  `graph/asking.py`, `serve` in `graph/requests.py`, and `Run`, `Shape`, `seat` and `held`
  in `graph/serve.py`. The first goes when the profile store moves below `graph`; the other
  two are a page and a request handler leasing a server, which is what the machine layer is
  for.

## Verifying

```bash
python3 -m pytest tests -q -n 4 > /tmp/out.txt; echo $?   # never pipe into tail; -n 4 while a bench runs
ml-stack-setup                                             # the machine
ml-stack-bench status                                      # measuring, serving, what the job kept
ml-stack-serve profile                                     # every model's measured shape
```
