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
`ask`, `ingest` or `chat` -- read by `ml-stack-serve up --for WORKLOAD` (profile by default;
`--no-profile` serves the model bare), `sweep`, `extract` and `converse`).

Settled: Flash-Next answers (80% F1 at 27 s/q, 100 questions) and extracts (96% node / 76%
relation F1); draft length 4 for both; one slot for extraction; `single` +8 pts on E4B at
ten questions, unconfirmed.

Fixes first, then measurements that need the card, then what does not exist yet.
Inside the fixes, most blocking first.

Nothing here is blocked by a kept measurement, a hash or a pin: `CLAUDE.md` says what that
means and what it costs. An entry that reads "we cannot change that, it would invalidate
the benchmarks" is an entry someone should rewrite as the change plus the re-measurement.

## Fixes

What is broken, unproven, or claims more than it does. Nothing here is a new
capability; every line is something that already exists not being what it says.

### Getting it onto a machine that is not this one

- [ ] **CI has been red on `main` since 2026-09-04, through four pushes.** `gh run list
  --branch main` shows `ci` failing on 09-04, 09-05, 09-06 (twice) and 09-09; the one
  failure is
  `tests/test_serve.py::TestTheStartedProcess::test_a_server_that_exits_on_its_own_is_reaped
  -- DID NOT RAISE ChildProcessError`, with 4,257 passing beside it. Head of `main` is an
  attempted fix with no run against it. Either the reaper misses a process that exits
  between the poll and the wait, or the test races it; find out which, because the reaper
  is what stops a dead server holding a port.
- [ ] **CI never runs on the development branch, and the release publishes whatever
  built.** `.github/workflows/ci.yml` triggers on `main` only, so `0.2dev` -- 55 commits
  ahead -- has never been tested by anything but a person's own machine, and the
  release-please pull request's checks sit `action_required`. `release.yml`'s `publish`
  job is `if: always()`, so a failed bundle still publishes. The workflow should trigger on the
  development branch, the release pull request's checks should be approved, and `publish`
  should need `bundle`.
- [ ] **`pip install ml-stack` is in the README, `docs/install.md` and every release page,
  and it 404s.** `https://pypi.org/pypi/ml-stack/json` returns 404; no workflow publishes
  to PyPI -- `release.yml` attaches wheels to a GitHub release and stops. Register the name
  and add a trusted-publishing job, or take the line out of all three places. A first
  reader following the README hits this in the first minute.
- [ ] **A zero-extras install dies on `import numpy`, which contradicts the no-dependencies
  claim.** `pyproject.toml` says `dependencies = []`, but numpy is a module-level import in
  `graph/topology.py` and `vision/geometry.py`, and `graph/answers.py` reaches the second. On a
  bare interpreter `ml-stack-setup`, `ml-stack-doctor`, `ml-stack-serve`, `ml-stack-ingest`
  and `fleet join` all end in `ModuleNotFoundError: numpy`. `install.sh` only survives it
  because its `EXTRAS` pull numpy in through matplotlib. Either numpy is a dependency or
  the two imports are deferred behind a message naming the extra.
- [ ] **`ml-stack-doctor` checks a developer's checkout, and both installers run it as the
  last screen a new machine sees.** `setup.py` hardcodes `~/Documents/repos/ml-stack` and
  `~/ai_ceo`; on a fresh machine every finding is bad and the remedy offered is
  `pip install -e` a directory that does not exist. It never checks the things that would
  actually be wrong there: that the daemon answers, that the machine joined, that a GGUF is
  on disk, that a store opens, that 8770 and 8771 are free. Make doctor about the installed
  machine and put the checkout, hook and worktree checks behind `--checkouts`.
- [ ] **Neither installer has ever been executed, and CI runs one OS and one Python.**
  `tests/test_packaging_install.py` parses `install.sh` with `sh -n`, parses `install.ps1`
  with pwsh, and greps both; nothing runs either. `ci.yml` is `ubuntu-latest` on 3.12 while
  the classifiers claim 3.11, 3.12 and 3.13, and `docs/install.md` says outright that
  everything Windows-specific was written against a faked `platform.system()` on a Mac.
  `install.sh` also gates the unsloth fork build on `CHOSEN_BUILD`, which nothing assigns,
  and hardcodes `~/.ml-stack/traind` where `ML_STACK_HOME` should decide. A macOS job, a
  Windows job and a 3.11/3.13 matrix; then delete or wire `CHOSEN_BUILD`.

### Not losing what it read
- [ ] **Two ladybug faults are worked around here and stay here.** Adam, 2026-09-04: no
  upstreaming to public repositories. 0.18.x: a single `DETACH DELETE` in a ~10k-node
  store blanks other nodes' string columns (reproduction: `tests/test_graph_store_scale.py`).
  0.20.2: the cached-physical-plan fast path re-executes a parameterized MERGE against a
  table rewritten since and segfaults, and the text index returns a node once per version
  written (reproduction: the store's `_written` docstring; two lines). ml-stack is on
  0.20.2 with the per-write guard and the pin `>=0.19,<0.21`; the probes gate any bump,
  which is the whole of what is left to do about them.


- [ ] **Nothing detects a stale write-ahead log, and the recovery is written down nowhere.**
  `GraphStore.__init__` opens the database without looking for `<store>.wal`; a log left by
  a killed writer segfaults the engine on open, which no Python `except` can catch, so
  `ml-stack-store check`, `ml-stack-doctor` and `fold` all die the same way with no
  message. The recovery -- `ml-stack-ingest fold --rebuild` from the `.reads.json` files --
  appears once in `docs/ingest.md` as a note about deduplication. Warn at open when a log is
  present, name the command, and write the recovery down where someone looking for it will
  be. This has cost two rebuilds already.
- [ ] **A store that is half embedded reports success with no denominator.**
  `graph/vectors.py` logs `vectors: N of M` only when it is given a log, and
  `ingest/run.py` passes none, then prints `embedded N node(s)`. An embedding server that
  dies at node 3,000 of 9,700 prints `embedded 3000 node(s)` and every question over the
  rest of the store is answered by words alone, silently. Pass the log through, and have
  `ml-stack-store check` report vector coverage.
- [ ] **The store engine is unpinned inside its range and CI resolves it fresh.**
  `pyproject.toml` has `ladybug>=0.19,<0.21` while the two workarounds in `graph/store.py`
  are specific to 0.20's plan cache and its version-per-write; the pin's comment claims
  0.20.2 returned nothing from a fresh store's scans on Linux, which is the version
  installed here and contradicts what this file records under store integrity. Pin
  `ladybug==0.20.2`, and fix or delete the comment. The probes that would catch a bad
  version are `slow`, so a default test run skips them.

### Text it was never licensed to keep
- [ ] **A runtime redactor** (`tooling/compliance/{text_sanitizer,llm_sanitization}.py`,
  ~380 lines): a Presidio sanitizer built once per configuration and applied to prompts,
  model output and the turns a page stores. `ml_stack.redact` audits a repository at commit
  time and has nothing for the ask path, which is where a graph page sends real text.
- [ ] **The copyright carve-out gate** (`tooling/graph/ingestion/copyright_carveout.py`,
  183 lines): refuses a verbatim provenance snippet taken from a source that embeds
  third-party material under a permission granted to somebody else. Every ingested node
  keeps a snippet, so this is the same problem here.


- [ ] **Nothing refuses a snippet from a source that may not be redistributed.** There is no
  licence field on a source and no check anywhere in `ingest/` or `graph/`: `ingest/judge.py`
  stores the whole re-read span as `quote` and `verbatim`, uncapped, and the extractor asks
  for the book's own words by design. The caps that exist are on length, not on permission.
  For a corpus of standards a subscription grants one reader -- CSA, ASME -- this is the
  failure that matters, and the app driving this library had to invent `attrs.licensed` and
  a note about the reader's own subscription because the library offers nothing. A `licence`
  on the source record, and a refusal in `fold` and in `judge`'s quote path.
- [ ] **Nothing redacts at runtime.** `redact.Redactor` and `redact.names_in` have no caller
  anywhere in `src/`; the only non-test use is `world/check.py`, which imports the
  commit-time recogniser, and `ml-stack-audit` runs over `git ls-files` -- repository
  source, not the text a person feeds it. Prompts, answers and stored turns pass through
  unfiltered. A redaction pass in `ingest/fold.py` before the write and in
  `graph/serve.py`'s remember path, on by default, off by a flag.
- [ ] **Every question and answer is written into whatever store is being served.**
  `AskRoutes.answered` calls `remember()`, which opens the same `GraphStore` the corpus is
  in and writes the raw question and the raw answer as nodes; `--store` is documented as
  "a GraphStore path conversations are kept in", so pointing it at a corpus is the
  documented use. The demo app did exactly that, which is why restarting its server did not
  clear a conversation. Conversations belong in a store of their own, with a flag for
  anyone who wants them in the corpus.
- [ ] **`<store>.<slug>.reads.json` keeps every extraction in plaintext beside the store,
  for good.** The record carries `extracted` -- every concept with its source-worded
  definition -- and `raw`, the model's whole reply when it failed. `_keep_reads` only ever
  merges more in; nothing rotates, expires or deletes. `ml-stack-ingest forget`, and the
  licence and redaction gates above run before the write, not only before the store.
- [ ] **The name gate runs only in a git hook someone has to install, and degrades to
  nothing without an extra.** `scripts/install-hooks.sh` is by hand, `ci.yml` has no
  `no-real-names` step, and `redact/hook.py` returns `None` on `ImportError`, so on a
  machine without the `[privacy]` extra the property this repository claims -- a name it has
  never seen is refused -- is quietly absent. A CI job over the diff, and a non-zero exit
  when presidio is missing rather than a pass.

### Numbers that are quoted
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


- [ ] **No benchmark result belongs in a README.** Adam, 2026-09-10: "we shouldn't have
  benchmark results in a readme". This repository's README is already clean; the rule is
  written into `CLAUDE.md` under what a user reads, and what it means for the rest is
  below: a measurement lives in a document that names its date, its command, its store and
  its model, and prose points at that document rather than restating the figure.
- [ ] **`docs/report-2026-09-02.md` was never regenerated after the pairing bug was fixed,
  and two shipped serving defaults were derived from it.** The report holds 136 speedup
  figures and was last written on 09-02; the fix to how a drafted run is paired with its
  baseline landed 09-05. One quoted 4.69x is a row that also changed `-ub 2048` and the way
  of asking. `data/profiles.json` sets `spec_draft_max 2` for gemma-4 E2B and E4B from two
  of these rows. Re-run `ml-stack-bench report` over `runs.ladybug`, commit the regenerated
  file, and re-derive the profiles with `report --profile`.
- [ ] **`bench/report.py` prints acceptance for runs that carried no draft head, and then
  recommends serving them.** `docs/report-2026-09-02.md` has a `draft:none` row with 80%
  acceptance and a 2.74x speedup, followed by a line telling the reader to serve it; a run
  with no head cannot have an acceptance. The column is falling back to the label rather
  than reading the recorded head. Read the head, and drop the rows that have none.
- [ ] **Two documentation claims are not true as written.** `README.md` and
  `docs/FEATURES.md` say every feature has a check in `docs/verify_release.py`: there are
  83 bullets across 23 sections against 57 checks across 19 areas, and 16 sections have no
  check area at all. `docs/architectures/qwen4exp.md` and `gpt-oss.md` restate the suspect
  pairings as flat prose with no qualifier, and `docs/bench.md`, `docs/graph.md` and
  `docs/ingest.md` state measurements with no date, command or store. Soften the first to
  what is true; give every loose number its command and store or delete it.

### The vocabulary

Words this library coined that a reader has to learn before the code means anything. Adam,
2026-09-09, on `measured shape`: "it tells you nothing"; "look at the other jargon and see
if it's AI-ese". `lease` stays -- it says what it does for server talk. Counts are uses
across `src/`.

- [ ] **`held` as a variable name, 665 uses left in `src/`.** The public names are done --
  `serve.serving.servers()`, `hub.on_disk()`, `jobs.recorded()`, `converse(highlighted=)`,
  `AskRoutes.asker(highlighted=)`, `Ask.highlighted`, the `highlighted` field the page
  posts, and `bench.save(server=)` -- and so are `ingest/fold.py`, `graph/tidy.py`,
  `graph/serve.py`, the five modules `graph/ask.py` became, `bench/keep.py`,
  `bench/extract.py`, `bench/run.py`,
  `bench/serve.py`, `bench/speed.py` and `jobs.py`. What is left is per-file:
  `ingest/sources.py` (36), `bench/measure.py` (32), `serve/manager.py` (25),
  `bench/show.py` (23), `world/simulate.py` (21), `ingest/imports.py` (20),
  `graph/rebuild.py` (9), `graph/store.py` (7), `serve/cli.py` (15), then a long tail. Rename each to what it
  holds; renaming them all to one other word is the same problem again. Watch for the
  name already in that scope: three of these renames collided with an existing local and
  only the tests caught it.

### The interface

- [ ] **A graph of a few thousand nodes is drawn whole, and the pane holds about that
  many.** Measured 2026-09-10 headless at 1400x900 on synthetic graphs at 2.7 edges a node:
  at 3,000 nodes the marks settle 10 px apart -- two mark diameters -- over 61% of the pane,
  and at 5,000, 7 px apart over 63%. Scaling the charge, the repulsion cutoff, the pull to
  the centre and the collide radius with the live node count was tried across exponents 0.5
  to 3, with degree-scaled repulsion and with stronger and weaker centring, and left every
  one of those numbers the same or worse, because the layout grows in world units and the
  fit zooms back out (the measurements are in the body of "the link layer's ink falls with
  the number of links drawn in it", 2026-09-10). What would give a bigger graph room is
  drawing fewer of it -- a cap on marks with the rest reached by search or by opening a
  neighbourhood, the way `LABEL_CAP` caps names -- rather than another constant. Still
  unverified against a real corpus with cluster structure: 3,000 nodes in thirty planted
  communities, 85% of each node's edges inside its own, settled into one blob under every
  setting tried, and a citation or hierarchy graph may not.

### Files that hold more than one job

`deep-files` refuses a Python file over 900 lines and `deep-components` an HTML, JavaScript
or CSS file over 500. No component is over any more; twelve Python files are.
`claude-edit-guard` refuses a write that lengthens one of them, so each can only get
shorter from here, but nothing shortens them except somebody splitting them.
`scripts/budgets --show deep-files` lists them. `bench/run.py` (2,019) has its own entry
below, under the commands.

Each is a file to read before it is a file to split: the answer is a module per job with a
name, not a line count met by moving code sideways.

- [ ] **`serve/fit.py` (1,390)** -- parsing the llama.cpp load log (`Segment`, `Measured`,
  `parse_load_log`), the `Fit` record and its file, the rendering (`render`, `_block`,
  `_block_md`), the GGUF tensor table (`Tensor`, `tensors_of`, `render_tensors`) and the
  matplotlib plot.
- [ ] **`bench/measure.py` (1,279)** -- the questions (`read_questions`, `sample`), one
  question through the client with its bill (`Counting`, `_ask_once`) and its trace, a set
  of them (`measure`, `concurrent`), and what the server costs (`footprint`, `busy`,
  `slot_count`).
- [ ] **`world/organisation.py` (1,233)** -- an invented organisation as a graph: five
  kinds of group, the `_Build` that assembles one, the people and their personas.
- [ ] **`hub.py` (1,215)** -- `Found` and the search, `Head` and the draft head, `Chosen`
  and the pick, plus a `ctypes` memory probe that belongs nowhere near any of them.
- [ ] **`world/simulate.py` (1,209)** -- `_Relations`, the day clock and its conversation
  picking, `_Counting`, and `ModelWriter`, which writes the messages through a model.
- [ ] **`bench/extract.py` (1,095)** -- `MessageRow` and the truth behind a message, the
  resolution of a name to a node, and `_Extracting`, which runs the model.
- [ ] **`bench/report.py` (1,081)** -- the gathering of kept runs, and `Doc` with the
  sections it renders.
- [ ] **`serve/cli.py` (1,062)** -- twenty subcommands, each parsing its own arguments; the
  work is already in `serve/ops.py`, so this is parsers and printing. The one a newcomer
  meets first, and the file in the commands entry below.
- [ ] **`bench/show.py` (1,019)** -- the table, the questions behind a score, the rates and
  the frontier, the plot, and the `drafts` summary: five outputs over one reader.
- [ ] **`serve/build.py` (999)** -- finding a toolchain, the cmake invocation, the cache of
  what was built, and choosing the binary to run.
- [ ] **`train/tools.py` (953)** -- reading the worked examples out of tool schemas,
  inventing arguments and paraphrases, the synthesiser, turning kept bench traces into
  rows, and the command. Its own section banners name the seams.
- [ ] **`world/simulate.py`, `world/organisation.py`, `bench/extract.py` and
  `bench/measure.py` are four of the twelve and all four are read by the extraction
  bench**, so splitting them is one reader's job rather than four.

`fleet/models.py` (930) went that way, into `fleet/models.py` (the machine's model files),
`fleet/catalogue.py` (what the hub offers and what fits here) and `fleet/weights.py` (which
file in a repository is the model); the five helpers that crossed a module boundary lost
their leading underscore and every importer moved rather than being re-exported.

### The commands

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
  and delete the `def main`. `ml-stack-surface capture --out DIR --src SRC` writes every
  subcommand's `--help` and the output of everything safe to run; take one against
  `main`'s `src/` and one against the branch, and `ml-stack-surface diff BEFORE AFTER`
  names what moved.
  Worth doing next in this order: `ml-stack-bench` (2,019 lines, two `main`s, and the last
  screen-scrape in `mcp.bench_show` and `do.bench_cli` is waiting on it), `ml-stack-setup`
  and `ml-stack-doctor` (one module, two entry points), `ml-stack-ingest`,
  `ml-stack-models`, `ml-stack-store`, `ml-stack-speech`, `ml-stack-train-run`,
  `ml-stack-train-tools`, `ml-stack-do`, `ml-stack-claude`, `ml-stack-agent`,
  `ml-stack-graph`, `ml-stack-audit`, `ml-stack-suite`, `ml-stack-fleet`,
  `ml-stack-peers`, `ml-stack-traind`. A command whose row in the README's table changes
  is a change to `ml_stack.cli.reference` and `scripts/reference --write`.

### Layers

- [ ] **Fourteen imports still cross the layers `tests/test_layers.py` sets out.** The
  layers are core, model, graph, machine, tools; a package may import downwards, and
  sideways only when the other package does not import it back. `KNOWN` in that file lists
  every edge that breaks it, and the test fails both on a new violation and on a `KNOWN`
  entry that is no longer one, so the set only shrinks. Three are two-way cycles inside one
  layer: `serve` <-> `fleet` (4 files one way, 7 the other), `serve` <-> `setup` (2 and 1)
  and `fleet` <-> `setup` (1 and 1). Eight reach up a layer: `setup`, `fleet`, `ingest` and
  `serve` into `bench` (1, 3, 2 and 4 files); and `gguf`, `hub`, `graph` and `ingest` into
  `serve` (1, 1, 3 and 2). The `graph` -> `serve` three are function-local reaches:
  `profile_for` in `graph/asking.py`, `serve` in `graph/requests.py`, and `Config`,
  `Serving`, `slot` and `held` in `graph/serve.py`. The first goes when the profile store
  moves below `graph`; the other two are a page and a request handler leasing a server,
  which is what the machine layer is for.

### Finding a model

- [ ] **Two callers still decide for themselves what an `hf:` reference means.**
  `hub.located` is the one finder now, and it answers `None` for an `hf:` reference
  because the bench and `ml-stack-serve up` hand that string to llama.cpp for it to
  download. `fleet/sizing.py:_at_hand` wants the opposite -- the file an already-fetched
  reference points at -- so it strips the reference to a filename and asks `located` for
  that; `serve/profile.py:_heads` skips `located` for anything starting `hf:`. Folding
  either into `located` means deciding whether a fetched reference resolves to its local
  path, which changes what a sweep serves. That is a decision to make and then re-measure,
  not a reason to leave three answers in the tree: pick one meaning, make `located` the
  only place that holds it, and record which kept runs stop being comparable. Whoever takes
  it should check `bench.serve.served` and `serve/preflight.py` first.

### The shared fakes

- [ ] **Eighteen tests monkeypatch `serve()` with a `fake_serve` of their own.** Each takes
  `**lease` and records the dict, which is the shape that lets a keyword `ServerSpec` does
  not have through unnoticed; `ml_stack.testing.fakes.FakeServe` builds a real `ServerSpec`
  instead and puts it in `leased`. They are in `tests/test_ingest.py` (4),
  `tests/test_graph_bench.py` (5), `tests/test_bench_extract.py`,
  `tests/test_bench_extract_n_max.py`, `tests/test_bench_selfcheck.py`,
  `tests/test_claude_launcher.py` (2), `tests/test_graph_bench_speed.py`,
  `tests/test_harness.py`, `tests/test_ingest_server_gone.py` and
  `tests/test_serve_shape.py`. Each needs its assertions moved off the recorded dict and
  onto the spec.

- [ ] **Three tests define a `FakeClient` of their own** -- `tests/test_do.py:533`,
  `tests/test_entities_edits.py:150`, `tests/test_vision.py:172`. The shared one carries
  `Client.__init__`'s signature and `tests/test_testing_fakes.py` diffs it against the real
  one every run, so a keyword the real `Client` would refuse is refused there too.

### Measuring across the fleet
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
  identically; `fleet/sweeps.py` stamps gathered runs with `_name_of(peer)`, so their
  measurements land in the store indistinguishable. Nothing checks for the collision at
  join. The default name is the hostname, and this machine already records `"host": "Mac"`.
  Fixing it means a stable per-machine identity (the current token is per process, so a
  restarted machine is already a new peer to anything that remembers) and keying rates and
  the run host on that, leaving the name as a label. Adam's call on what a person sees:
  refuse a duplicate name at join, accept it and disambiguate in the listing, or accept it
  silently and fix only the keying.

### The window

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
  - **The frozen daemon bundles numpy because `ml_stack.serve` reaches the graph.**
    `packaging/ml-stack.spec` names `ml_stack.serve` as a hidden import, for the reclaim
    watcher and the hosting routes; `ml_stack/serve/__init__.py` imports `serve/serving.py`,
    which imports `graph.asking` and so `graph/__init__.py` and `graph/topology.py`, which
    needs numpy. It costs 11.7 MB to 15.5 MB of bundle. It goes when `graph.asking` leaves
    `serve/__init__.py`'s import chain, or when the spec names the `serve` modules the
    daemon reaches instead of the package.
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

### What the window has not been driven through

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
- [ ] **`ml-stack-walk` has never walked a screen with a model behind it.** Every walk so
  far was against a daemon serving nothing and an `ml-stack-graph serve` with no `--model`,
  so Chat reads "No model is running yet", the graph's question box is only read, and the
  review queue is skipped because `#review-box` stays hidden with no change requests
  waiting. Unwalked: sending a message on Chat, a download on Models, `--ask` through
  `/ask/stream`, and a review queue with something in it. `ml-stack-serve up <model>`
  first, then `ml-stack-walk fleet chat models` and `ml-stack-walk graph ask review --ask
  "..."`.

### The code itself

- [ ] **`tests/test_graph_ask.py` (2,383 lines) tests five modules under the name of one
  that is gone.** `graph/ask.py` is now `graph/prompts.py`, `graph/looking.py`,
  `graph/replies.py`, `graph/answers.py` and `graph/conversation.py`, and every test still
  sits in one file, so a reader looking for what covers `looking.py` has to search for it.
  Split it the way the module was, one file per module, moving whole test classes.
- [ ] **Four callers defer the modules `graph/ask.py` became, and it is not clear any of
  them has to.** `bench/measure.py:asking` defers `graph.conversation`, `graph.looking`
  and `graph.search`; `ingest/ask.py:ask`, `graph/serve.py:asker` and
  `world/simulate.py:ModelWriter.__call__` defer their own. Only `world.simulate` has a
  cycle to avoid, while `graph.page` reaches into `world`. The other three are paying for
  import time -- `graph` pulls numpy -- so measure what module scope costs `ml-stack-bench
  --help` before deciding, and hoist the ones that cost nothing.
- [ ] **A daemon probes for speech providers once, when it starts.** `serve_forever`
  runs `speech.service.working()` in a thread of its own and the beacon carries what it
  found, so a provider installed afterwards -- ffmpeg, a whisper model, torch -- is not
  advertised until that daemon restarts. The probe costs what importing faster-whisper
  and transformers costs: 2.0s and ~230 MB resident, measured 2026-09-10, paid by every
  daemon on a machine that has them. A frozen daemon has neither, so it probes in
  milliseconds and advertises `tts` and `vad`. Both go away if the probe runs somewhere
  that is not the daemon's own process.

## Measurements

Each needs the GPU and Adam's call. Estimate before it runs, smoke it before it is
paid for, and one thing on the card at a time.

### The store

- [ ] **The store holds APBiology and Biology2e chapter 2, sound; seven textbook PDFs are
  unread.** `~/.ml-stack/sources.ladybug` on ladybug 0.20.2: ~9,700 concepts with
  definitions, page provenance and the run that read each, after the judged pass (362
  merges, 1,489 inverse pairs, 622 conflicts judged with 282 edges dropped, 186 definitions,
  113 suspects), `ml-stack-store check` clean. The reads files beside the store are the
  truth (`ml-stack-ingest fold` rebuilds). One of the seven, Additive Manufacturing
  Essentials, was already tried and every one of its 80 units failed with
  `ServerUnreachable` -- nothing was serving on 8080 at the time -- so it needs
  `ml-stack-ingest retry --out ~/.ml-stack/sources.ladybug` before a `--resume` reads it for
  real; the other six have never been attempted. Whether they are worth it -- about three
  days of GPU at 86 s a unit, one slot -- is Adam's call; the command is `ml-stack-ingest
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

### Per-request draft depth

`docs/llama-cpp-per-request-speculative.md` is the patch written up as a pull request would
put it. `patches/llama.cpp/0001-speculative-per-request.patch` is the patch;
`ml-stack-serve build --from source` applies it and names the build directory after its
digest.

- [ ] **The draft-depth timings taken on 2026-09-05 were measured before a lease could be
  refused, so a server may have shared the card with them.** One 138s reading against a
  108-112s baseline is what that looks like. The runs carry no `beside` record -- it did
  not exist yet -- so nothing can say after the fact whether they were alone. Re-measure
  the depth sweep on a quiet card and compare.
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
- [ ] **Only `speculative.n_max` is per-request; the other six fields upstream guarded read
  their values once, when the implementations are built at server start.** Making `p_min`,
  `n_min` or `type` per-request means giving every implementation per-sequence parameters,
  which is a much larger change than this one. `Client` refuses those names rather than
  sending something the server would drop.

### What a draft head costs, and where the draft stops paying

`patches/llama.cpp/0002-speculative-timings.patch` splits generation into `draft_ms` and
`verify_ms` with `verify_n` passes, per request. `ml_stack.client.counters` reads the four
`llamacpp:spec_decode_*` counters off `/metrics` and differences them across a block of
work, labelled with a workload and a sampling regime; `Ledger` totals them per label and
pools them. Every build under `~/.ml-stack/llama.cpp/builds/` carries the counters, so this
works against any client, lm-eval included. Servers are now launched with `--metrics`.

- [ ] **The per-workload by sampling matrix on Flash-Next has not been run.** It needs the
  card and about 50G that another agent's server held all afternoon. Each workload
  (`ask`, `ingest`, `chat`) at greedy and at temperature 1.0, using the harnesses that
  exist: `ml-stack-ingest` over a chapter, `ml-stack-bench drafts` or `sweep` for graph
  answering, `ml-stack-world simulate --mix` or `ml-stack-bench extract --world` for prose,
  `ml-stack-bench standard` for GSM8K, HumanEval, MMLU-Pro and IFEval. Extraction and graph
  answering are greedy by profile, so their temperature arm describes a shape nobody serves
  and is there for the curve; prose and chat run hot for real. Wall clock is not comparable
  across the temperature arms, because the model writes a different number of tokens each
  time -- read acceptance and acceptance-by-position, which are per-token ratios, and fix
  the generated length before quoting seconds.
- [ ] **Only the ingest captures counters; the other harnesses do not.** `ml-stack-bench`
  (`standard`, `drafts`, `sweep`), `ml-stack-draft`, `ml-stack-world simulate` and
  `converse` each need a `counting()` around the block they already run, labelled with the
  workload and the sampling they used. No new harness -- the capture goes around what is
  there. `ml-stack-draft` wants it most: its `tok/pass` column is empty on any build
  without `0002-speculative-timings.patch`, and the counters answer the same question off
  `/metrics` on every build.
- [ ] **`ml-stack-draft` has not been run against Flash-Next.** It was driven end to end on
  gemma-4 E2B and E4B (2 questions an arm, `--anyway` on a busy machine): the arms serve,
  the table prints, the run is kept, the paste line is right. The command is
  `ml-stack-draft Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`, on a quiet machine,
  with no other flags. On E4B the head's own cache at q4_0 moved throughput by 0.1 tok/s
  against f16, which is why `--draft-kv` is a flag and not a default arm; whether that
  holds on Flash-Next is one `--draft-kv q4_0` away.
- [ ] **The pre-load estimate does not charge a draft head's KV cache.**
  `serve.preflight.draft_kv_estimate_bytes` computes it from the head's own GGUF (honouring
  `nextn_predict_layers`, without which an MTP head's header reads as its target's 49
  layers where llama.cpp builds one), and `fit` charges it from the load log. `Preflight`
  itself does not add it: its signature is nine parameters, over the edit guard's limit, so
  it cannot be edited until its six reader seams become a dataclass. That refactor reaches
  `bench/selfcheck.py` and the preflight tests.
- [ ] **The ingest's counter table has not been driven through `ml-stack-ingest` itself.**
  `_counted` and `_counter_lines` in `ml_stack/ingest/run.py` are exercised by the client
  path and by unit tests, not by a real read of a chapter.
- [ ] **`ml-stack-serve status` has not printed its drafting line for a server actually
  serving a head.** The head file, the speculation type and the depth come from the
  server's own command line, and the acceptance from `/metrics` through
  `ml_stack.client.counters`; both are covered against a real socket in
  `tests/test_serve_cli.py::TestStatusDrafting`, and the without-`--metrics` branch was
  read off a live server. `ml-stack-serve up <model> --draft auto --for ingest`, then
  `ml-stack-serve status --port <port>`, closes it in a minute.
- [ ] **Whether draft depth should adapt during generation is unanswered and was not
  built.** A fixed sweep at 4, 8, 12 and 16 on two biology sections tripled tokens per
  verification pass while generation time stayed flat, which says the per-pass cost rose to
  cancel the gain. Acceptance by position is the cheaper way to answer it: it shows where in
  the draft acceptance falls off, on one run, instead of a sweep per depth. Build adaptation
  only if that curve shows a knee worth chasing.

### Flash-Next, two builds: llama.cpp (unsloth GGUF Q4_K_XL, with and without the draft head) against Ollama (MLX, nvfp4)

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
  # 1. llama.cpp + draft head, against the page's server on 8080 (its profile's serving)
  ml-stack-bench sweep --on flash=http://127.0.0.1:8080 --plain-only --short
  ml-stack-bench speed --on flash=http://127.0.0.1:8080
  # 2. llama.cpp without the head, served by the bench with the same serving minus -md
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
  `--parallel` to the most streams asked (4) and the per-slot context to the largest prompt
  plus the generation, so set `--context` for the 4-stream cells. `ttft_s` is a streamed
  first token on llama.cpp (`ttft_from: stream`) and the server's prompt clock on Ollama
  (`prompt_ms`, marked `*`). Estimate before each: ~45 min the hundred-question graph
  bench, ~10 min speed, ~40 min the standard sets.
- [ ] **One measured call on Ollama first.** Whether `prompt_eval_count` includes a cached
  prefix is not in its docs; whether `think: false` holds for this model; what the runner's
  process is called on 0.33.3 (found from source, not seen). Then the Ollama half.

### Driving what is built

- [ ] **A number for the local harness beside the page's.** `ml-stack-claude <flash-next>
  -- --print "say hello"` and `ml-stack-agent "read README.md and say in two sentences what
  this is" --model <flash-next> --allow Read` both ran for real on 2026-09-05 (the agent: 5
  turns in 382 s, 79k tokens read with 185k from the cache, 1.2k written, a right answer).
  Untried: the stream-idle watchdog (five minutes of silence aborts --
  `CLAUDE_STREAM_IDLE_TIMEOUT_MS`), and what `Usage` reports against the server's own
  `/metrics`. Then a small task set measured the bench's way.

### Queued
- [ ] **Nothing has measured any model for `ingest` or `chat`.** The record now holds a
  shape per workload and the five shipped records are all `ask`, so `ml-stack-serve up
  --for ingest` serves the graph-asking shape (profile by default) and says so. What is missing is a
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
  where every other shape reported 70%. Prefill ~2,600 tok/s in every shape. So one slot
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

## New

Capabilities that do not exist yet.

### A router across the fleet

- [ ] **A router across the fleet.** `ml-stack-fleet plan --apply` serves the placement;
  nothing yet sends a new session to a free slot on the best model. The daemon's `/infer`
  proxies by model name on one machine; the router picks the machine.

### Speech
- [ ] **Nothing streams.** `StreamingASR` in `speech/protocols.py` is a protocol no
  provider implements, and `POST /speech/transcribe` takes one whole file. A push-to-talk
  button wants partial text while the person is still speaking: a provider that yields
  `Transcript`s off a chunk iterator, and a route that streams them the way `/ask/stream`
  does.
- [ ] **No microphone anywhere in the interface.** The chat screen has no record button, so
  the only way to reach any of this is the command line. The route and `Peer.transcribe`
  are the far end; what is missing is a control that records in the browser and posts the
  audio.

### The tensor toolkit

- [ ] **`batch_graphs`, `resolvent_sweep` and `decompose_to_dags` have no caller outside
  `tests/test_graph.py`.** `graph.tensors` (the mapping graph as arrays) is called by
  `graph.smooth`, and `topology.knn_edges` by `graph.places.geocode`.
  `topology.build_topology` already wires `mst_edges`, `morton_codes` and
  `spatial_window_edges` together with `knn_edges`, but `build_topology` itself is called
  only by that same test file -- nothing in the app calls it. `batch_graphs`,
  `resolvent_sweep` and `decompose_to_dags` each have an obvious home when the work arrives
  -- `batch_graphs` when a graph model is trained on many graphs at once, `resolvent_sweep`
  and `decompose_to_dags` for influence down a DAG (a `reports_to` tree, a `part_of`
  hierarchy). The near-term task is `geocode --near K`, which is O(n^2) in placed entries
  (nothing at 300, a problem at 30,000) because it calls `knn_edges` alone, which builds the
  full distance matrix `pairwise_distances` does; wiring in `build_topology` would not by
  itself fix that, because `build_topology`'s own `mst_edges` step also calls
  `pairwise_distances` -- only `morton_codes`/`spatial_window_edges` avoid it. A cheaper
  `--near` needs `geocode` to call `spatial_window_edges` on its own, skipping the kNN and
  MST parts of `build_topology` entirely.

### A terminal command centre

`~/Documents/repos/beehavior` is the other repository that grew these shapes, and its own
`ML_STACK_MIGRATION.md` lists what it should now delete rather than keep. Two pieces came
here on 2026-09-05 (`ml-stack-suite`, `ml-stack-serve limits|reclaim`); three are still
worth taking, in this order:
- [ ] **A terminal command centre** (`tooling/daemon/tui.py`, 477 lines, Textual): the
  facts the fleet app's cluster view already shows, in a terminal. Worth it only if a
  headless machine wants one.

## Verifying

```bash
python3 -m pytest tests -q -n 4 > /tmp/out.txt; echo $?   # never pipe into tail; -n 4 while a bench runs
ml-stack-setup                                             # the machine
ml-stack-bench status                                      # measuring, serving, what the job kept
ml-stack-serve profile                                     # every model's serving
```
