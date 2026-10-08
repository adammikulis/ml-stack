# Handoff

## Pending recovery work

- Board delivery latency exceeds the existing budgets: measured single-recipient DM 479 ms and Board 363 ms against 50/150 ms. Reduce graph opening and integrity-query costs; preserve integrity, privacy and latency budgets. Evidence: `/private/tmp/ml-stack-board-observer-handoff.md`.
- Preserve `/private/tmp/ml-stack-qwen-worker-project/published-source` while pending task scopes and the stopped worker reference it. Add an authenticated offline source-rebinding operation before removing this source anchor; preserve pending task states and grants.
- Triage the remaining failures from the October 6 macOS background full red-team run at `920cf217`: 130 failed, 30 errors, 13,729 passed, 364 skipped. Re-run the remaining Board latency, serving, training, gym and fixture failures after scoped setup, pairing and installer repairs. Evidence: `/private/tmp/dev-final-background-full.log`, `/private/tmp/dev-full-failures.txt` and `/private/tmp/ml-stack-full-lifecycle-project-diagnosis.md`. Linux testing remains owner-paused.
- Verify an authenticated connection and a Board message/reply with the other physical device after separate Dev profile activation. Local discovery and browser checks do not establish this roundtrip.

## Open from the claude session (2026-10-08)

- **Broker branches 2-5 of the one-thing-computing rule are not started** (branch 1, the device field, is landed). `broker-purpose-not-exclusive`: `Broker._decide` in `src/ml_stack/serve/broker.py` still serialises same-purpose asks behind a held server (`busy`, `_first_for_purpose`); a different model under one purpose must get its own resident server when memory admits. Then `gate-compute-lease-per-device` (`serve/gate.py`), `workspace-allocation-device`, and `rule-docs-and-bench-profiles` (measure a CPU model beside a GPU one for memory-bandwidth cost; per-platform setting, default allowed). Rule: `AGENTS.md`, "One thing computing on the GPU at a time".
- **Old GPU-rule wording remains** in `HANDOFF.md` (the serving section), `docs/experiments/iq-vs-kquant-metal.md:53`, `src/ml_stack/ingest/serving.py:87`, `src/ml_stack/bench/queue.py:21` and `tests/workspace_screen_corpus.py:35`; each states "one model on the GPU" and needs rewording or a decision.
- **Branch hygiene rule (f) has no mechanical check**: a bound on `scripts/test` fixed cost needs an opt-in timing harness (AGENTS.md, "Branch hygiene and landing").
- **Hooks are not verified end to end on the restored runtime.** SessionStart, SubagentStart/Stop and the nudge push (message text shown in the TUI) have only run from source; the nudge board fix needs a Board host restart. The installed `ml-stack-workspace` was down for a time during another agent's runtime deploy.
- **`authority_machine` removal (`4ee616e7`) is unverified on two real machines**: discovery, certificate pinning and the host-address path are covered by fakes and in-process TLS only. The Projects and Tasks pages were not driven in a browser (browser tests fail on this machine). `board_host` is self-declared and tied to a device only by the pinned certificate.
- **Owner has not reviewed the `AGENTS.md` rules changes** (authority registry, GPU rule, "The agents push; the owner does not").
- **Not built, discussed:** attested instructions (one-time hash on UserPromptSubmit, taint tracking for delegated gates) and decision-model screening of board messages at ingest.

- **The agent sandbox is staged, not installed or measured** (`docs/agent-sandbox.md`, `scripts/agent-sandbox`). The owner runs the fifteen acceptance steps; every sandboxed result in the compatibility matrix is predicted. Open design gaps: the workspace CLI writes `~/.ml-stack/workspace` from the agent's own process, so a sandboxed agent that may announce may also edit board files (needs the CLI to go through the daemon with `workspace/` read-only to agents); `testslots_rpc` binds an ephemeral loopback port that Linux and WSL cannot allow-list (needs a configurable fixed port); Codex's workspace-write cannot deny `.git/config` and `.git/hooks` in the shared git directory.

## Autostart units (prepare and install)

- **`runtime-ensure` has not run unattended against a real checkout.** The unit runs `ml-stack runtime ensure` as no agent (no `ML_STACK_WORKSPACE_AGENT` in its environment) from the `ml-stack` launcher in the `--launchers` directory; it needs a first `ensure --checkout --launchers` run by a person or agent, and hourly runs on a checkout with local edits or a held commit have not been observed.
- **No unit has run under a real service manager here.** The `launchctl bootstrap`/`bootout`, `systemctl --user`, `schtasks /XML` commands and copy-truncate log rotation under launchd are covered by a recorder and real files, not by a loaded unit. Run the reboot checklist in `docs/install.md` on a Mac and on the WSL device; the Windows task XML has not been loaded on Windows. systemd units have no watchdog because the daemon sends no `sd_notify`.
- **`autostart.install` (UI toggle, join, `--persist`) still writes units from a running process.** The Settings and join flows call it without a manifest; moving them to prepare plus a person's `install` is the rest of the owner's decision that agents never install units.

## Active work (2026-10-06)

- **Speed up checks without reducing coverage.** `scripts/gates/_floors.py` now reuses the gate's
  full tree fingerprint for its collection cache. An uncached `scripts/budgets` run still took
  95.6 seconds; further profiling and safe optimization remain.



## Pending setup and coverage

- [ ] **The release signing key does not exist yet.** Run `scripts/release-key create --write` at a
  terminal and commit `src/ml_stack/fleet/signing.py`. `packaging/install.sh` and `install.ps1`
  still install an unsigned download.

- [ ] **No real OS keystore prompt has been driven by hand.** Tests use fakes and the real-keystore
  guard stays on; what a person sees on a first Keychain prompt is as documented in `docs/keystore.md`
  and unobserved.
- [ ] **The macOS desktop-session heuristic is a guess.** `keystore` and the dialog decide whether a
  person could answer a prompt from a terminal or `DISPLAY`/`WAYLAND_DISPLAY`-style signals; on macOS
  that is an approximation not checked on a real login, SSH or launchd session.
- [ ] **`ml-stack-memory rekey` changes the file's salt.** A copy of a store made before a rekey
  stays sealed under the old salt and key.
- [ ] **Reputation hooks exist only in processes that call `reputation.install()`.** A process that
  does not call it records nothing about the sources it deals with.
- [ ] **The IQ-on-Metal refusal rests on thin evidence.** One model, one machine, nine and ten
  questions (`docs/serving.md`); the pre-registered re-measurement is pending.
- [ ] **Grant ledger, connector credential broker, a hard outward-send rule and the "what can it do
  now" listing are not built** (`docs/assistant-security.md`, "Order of work").
- [ ] **Requests inbox: sources not raised through it yet** (`docs/requests.md`). Fleet join requests
  (`fleet/onboard/requests.py`, `fleet/onboard/cli.py` accept and decline), `ml-stack-security
  approve-host` (`net/cli.py`, refusals in `net/policy.py`), the reputation notice
  (`reputation/notice.py`, its own dialog), keystore unlock, hold windows and rule suggestions keep
  their own prompts; the kinds exist in `ml_stack.requests.model.KINDS`.
- [ ] **Requests page in the shell.** The element (`ui/assets/ml-requests.js`) and
  `inbox.route.RequestsApp.dispatch` are not mounted in the workspace shell yet, and the page has
  only had DOM-level tests (a probe of `plain()` under node), no browser run.
- [ ] **Always allow answered in the UI saves no rule.** The call runs once; a rule is saved only from
  the terminal's second prompt.
- [ ] **A background process that cannot read the keystore keeps no requests on disk.** Sentinel's
  dialog keeps its own in memory for that process; the page does not see them.
## Fixes

What is broken, unproven, or claims more than it does. Nothing here is a new
capability; every line is something that already exists not being what it says.

### The agent loop

- [ ] **Taint tracking has been run against one served model only.** `docs/taint.md` has Qwen3-4B
  (20 web attempts per arm, 18 planted-text runs per guard). A larger model, more repetitions
  and the red-team's direct-injection arms would say how the usability cost and the proven
  versus unproven split move.
- [ ] **Taint events have no subscriber.** `taint.subscribe(fn)` receives a `TaintEvent` for each
  refused or questioned call; the sentinel branch's bus (`docs/sentinel.md`) is the intended
  subscriber (`taint.subscribe(bus.emit)`) and is not written yet.
- [ ] **No tool wrapper routes untrusted reads through `taint.extract`.** A tool whose answer has a
  typed shape (a model listing, a version, a status) could return only the validated fields, and
  then not contaminate; `taint.quarantined` does it for a callable, and nothing does it for an MCP
  `ToolSource`. The free-text case (summarise this page) has no schema and stays fenced data.
- [ ] **Pasted documents in the person's own turn are the person's text.** `Ledger.admit` takes a
  `Level.UNTRUSTED` label for them, but no entry point (the CLI, a chat front end) labels a paste.
- [ ] **MCP tasks are not driven.** The mcp 2.2 client has no tasks API, so a long-running tool
  call cannot be polled or cancelled through `McpTools`, and task progress is not an agent
  event. Progress for an ordinary call reaches `McpTools.on_progress` only.
- [ ] **No tool-calling base-versus-tuned comparison in one command.** `ml-stack-train-tools eval`
  scores one served model; comparing a base and a tuned model means running it against each
  server and diffing the JSON.

### The local agent (`docs/local-agent.md`)
- Its approvals go to `localtools.ask_a_person`, which answers no; replace its body with a raised
  request when the Requests inbox merges, so an agent in a role that asks can be answered.
- No context trimming: a task is held inside the 32768-token context by its caps. A task with large
  tool results needs a trim that drops whole oldest turns at a fixed boundary.
- The guard's model-based screen (`native_screen`) is not attached to the local agent's chat.
- `--orders-from` defaults to `claude`: whoever registers that id is obeyed.
- The general local agent has no worktree of its own; Coding mode works in the selected project
  directly, so edits need a separate checkout when isolation is required.

### Sentinel wiring (`docs/sentinel.md`, "What is armed by default")

- [x] **A bare `Agent`, the Broker (`_launch`, `lease`, `start`), the Broker and fleet daemons
  (scan loop, decoys, authenticator watch) use sentinel by default**, with a logged opt-out that
  needs a reason. Tests in `tests/test_sentinel_wiring_*.py` fail when the wiring is removed.
- [x] **Model output, compaction summaries, `scrub_env` at the spawn helpers, `mcp_allowed` at
  connect, behavioural canaries in the daemons' scan loop, aggregated guard events
  (`GuardLogHandler`, `broker_listener`, `sandbox_listener`, unmanaged listeners) and a loopback
  decoy endpoint are wired.** Tests: `tests/test_sentinel_canaries_scheduled.py`,
  `test_sentinel_score_wiring.py`, `test_sentinel_screens_wired.py`,
  `test_sentinel_decoy_endpoint.py`, each checked by breaking the wire.
- [ ] **Still not wired:** the scan loop, canaries and decoy listener live in the daemons only (a
  library user who calls `serve.serve()` or runs an `Agent` in a script gets start-time checks
  and the per-session score, no timer and no decoy endpoint); nothing quarantines an MCP server
  on its own; `exe_mismatches` is not scheduled; `taint.subscribe` has no sentinel subscriber;
  an environment passed as `None` (inherit) is not scrubbed; the canary baseline is the model's
  own first answers, and the false-alarm corpus for the decoy endpoint is 48 scripted runs.
- [ ] **A model ml-stack did not pull is trusted at its first start** (pinned then, source `first-use`, logged).
  A model it pulled (`gguf`, `safetensors`, through `net.download` / `net.accept`) is pinned at pull with
  `source=pull`, and a signed manifest accepted from a peer must agree with the file at load (docs/sentinel.md,
  "Supply chain"). Not armed: manifests match by file name only; a list is accepted only after a peer-first pull or the
  onboarding `fetch` (no command takes one from a file, no vendor digest list is read); the serial high-water mark is a
  sealed file the same user could reset; archives (llama.cpp builds, the Python bundle) are not pinned.
- [ ] **The npm and cargo audit gates and the SBOM step have only run on recorded reports and locally**
  (`tests/test_supply_chain_audit.py`); the first scheduled `audit.yml` run on GitHub is the real test (the cargo-audit
  version in the workflow is pinned by hand), and the SBOM omits npm packages.
- [ ] **A file edited in place with size and mtime restored is caught only by the next deep scan
  round**, not at start (the start skips the hash when path, size, mtime and inode match the last
  full verification).

### Serving: one broker, one request at a time per pool
- [ ] **Run `tests/test_serve_real_llama.py` on a card with nothing else on it** (`pytest
  --slow tests/test_serve_real_llama.py`). It leases a real llama-server through the
  broker, sends two requests from separate processes and checks the line of requests and
  the stop on release; it skips while any other llama-server runs, so it has not run
  against a real server yet.
- [ ] **The machine's broker does not deliver `on_event` or `say` to its caller.** Over
  its socket `RemoteBroker.start` answers once, and the caller is told only that the
  server is ready. A load's progress (`loading`, `restoring`, a memory wait) belongs
  streamed back as lines.
- [ ] **One `gpu` pool.** A machine with several CUDA devices queues all their requests
  in one line; `ServerSpec` has no device to put a server in a pool of its own.
- [ ] **`adopt_unmanaged = ask` has no prompt.** `ServerManager.confirm` is the hook; no
  command sets it, so `ask` adopts nothing from the CLI. vLLM, SGLang and MLX servers
  started by hand are not found at all, only `llama-server`.

### What still reaches the internet around the pipeline

`docs/internet.md` is the list; each of these is a path in it.

- [ ] **`ddgs` makes its own requests.** `web.ddgs_engine` hands the query to the library, which
  opens its own connections to the engines, so the address check, the host policy and the size
  cap do not apply to the search call itself (its results are cleaned and labelled). Replace it
  with an engine written against `net.default().open`, or make SearXNG the default.
- [ ] **pip installs unpinned dependencies.** `fleet/environment.py:Environment.pip` and
  `fleet/updates.py:pip_install` run `pip install`, which talks to the index with pip's own
  transport and checks no hashes. Download wheels through the pipeline into staging and install
  from there with `--no-index --require-hashes`, or state the index and the packages and pin
  them.
- [ ] **`ml-stack-bench standard` lets lm-eval fetch datasets.** `datasets` downloads through
  `huggingface_hub`'s transport. Fetch the datasets a task names through `hub.snapshot` and run
  the harness with `HF_HUB_OFFLINE=1`.
- [ ] **The browser resolves names itself.** `net/browserguard.py` checks the address of every
  request, then Chromium resolves the name again to connect, so a name whose answer changes in
  between reaches a private address. Pin with `--host-resolver-rules` from the checked answer,
  or send the browser through a local proxy that connects to the pinned address.
- [ ] **Windows Defender and Linux ClamAV are untested here.** The Defender command line and
  exit codes are tested against a stand-in script; ClamAV was exercised on macOS only. Run
  `tests/test_net_scan.py` on a Windows machine with Defender and on Linux with `clamav`
  installed, and keep the result in `docs/internet.md`.
- [ ] **The hash-reputation lookup has met no real service.** It is tested against a local
  server shaped like VirusTotal's answer, and nothing constructs `HashLookup` outside tests (there is no
  `ML_STACK_NET_HASH_LOOKUP` switch). Wire it behind an opt-in, run it once with a key
  (`ml-stack-credentials set VIRUSTOTAL_API_KEY`, then approve `www.virustotal.com`) and compare the fields.
- [ ] **A `/metrics` address in the fleet view is fetched from this machine.** `fleet/ui.py:_scraped`
  opens whatever a signed-in person typed, loopback and LAN included. Limit it to loopback and
  the fleet's peers, or send it through `net`.
- [ ] **Branch tracking installs what it pulls.** `fleet/updates.py:track_once` runs
  `pip install -e .` after a fast-forward of a branch a person chose to follow. Show the diff
  of the packaging files to the person, or install only from a tag with a digest.

### Getting it onto a machine that is not this one

- [ ] **`ml-stack` needs a similarity waiver from PyPI before anything can be uploaded.**
  PyPI's check ignores `-`, `_` and `.`, so `ml-stack` reads as `mlstack` -- 1.0b3,
  "Machine Learning toolkit and algorithms library", last uploaded 2018-09-14 -- and a
  pending publisher for it is refused. `ml-stack` itself is unregistered, so this is a
  request to waive the similarity check at `github.com/pypi/support`, not a PEP 541
  takeover of `mlstack`. The name stays `ml-stack` (the owner, 2026-09-18): `llm-stack` is
  refused the same way by `llmstack`, which unlike `mlstack` is live and in this field,
  and every other free single word is free because it is obscure. When the waiver lands,
  add the pending publisher -- the owner's GitHub account, repository `ml-stack`, workflow
  `release.yml`, no environment. Until then it uploads nothing and the docs install from
  git.
- [ ] **Set the `PYPI_ENABLED` repository variable to `true` once the waiver above lands
  and the pending publisher is registered.** `release.yml`'s `pypi` job stays off
  (`vars.PYPI_ENABLED == 'true'`) so every release before the waiver does not fail on a
  publisher PyPI has not registered yet; flipping the variable is the only step left to
  turn uploads on.

### Not losing what it read
- [ ] **Two ladybug faults are worked around here and stay here** (the owner, 2026-09-04: no
  upstreaming to public repositories). `CypherStore._run` prepares every statement that
  carries values afresh, and `access.read_lock` keeps a writer out while a read runs.
  `tests/test_graph_engine_contract.py` has one test for each that goes red when a ladybug
  release no longer needs it; drop the workaround in the same commit as that test.

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
- [ ] **The licence and redaction gates have to run before `_keep_reads`, not only before
  the store.** `<store>.<slug>.reads.json` is written as each unit comes back, with
  `extracted` (every concept with its source-worded definition) and `raw` (the model's whole
  reply when it failed) in plaintext. A refusal in `fold` or `judge` alone leaves both on
  disk until someone runs `ml-stack-ingest forget`.

### The vocabulary

Words this library coined that a reader has to learn before the code means anything. the owner,
2026-09-09, on `measured shape`: "it tells you nothing"; "look at the other jargon and see
if it's AI-ese". `lease` stays -- it says what it does for server talk. Counts are uses
across `src/`.

- [ ] **`held` as a variable name, 665 uses left in `src/`.** The public names are done --
  `serve.serving.servers()`, `hub.on_disk()`, `jobs.recorded()`, `converse(highlighted=)`,
  `AskRoutes.asker(highlighted=)`, `Ask.highlighted`, the `highlighted` field the page
  posts, and `bench.save(server=)` -- and so are `ingest/fold.py`, `graph/tidy.py`,
  `graph/serve.py`, the five modules `graph/ask.py` became, `bench/keep.py`,
  `bench/extract.py`, `bench/run.py` and the modules it became,
  `bench/serve.py`, `bench/show.py`, `bench/speed.py` and `jobs.py`. What is left is
  per-file:
  `ingest/sources.py` (36), `bench/measure.py` (32), `serve/manager.py` (25),
  `world/simulate.py` (21), `ingest/imports.py` (20), `serve/cli.py` (15),
  `graph/rebuild.py` (9), `graph/store.py` (7), then a long tail. Rename each to what it
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

### The commands

- [ ] **Nineteen commands still build their own parser and keep their work in the handler.**
  `ml_stack.command` holds the shared options and a `Group` that collects subcommands and
  answers as `main`; `ml_stack.cli.reference` holds one line per command and the README's
  table, and `tests/test_cli_reference.py` fails when the two disagree. On it so far:
  `ml-stack-serve` (work in `serve/ops.py`), `ml-stack-world` (`world/ops.py`),
  `ml-stack-jobs` and `ml-stack-bench` (`bench/ops.py`, with its flags in
  `bench/options.py`). The pattern, per command: make a `<package>/ops.py` whose functions take
  typed arguments and return values, raising rather than printing (`serve.ops.Refused`
  carries the lines); leave the handler to parse and print; declare each subcommand with
  `@COMMANDS.command(name, help=..., options=[option("port", ...), flag("--own", ...)])` in
  the order the old parser added them, so `--help` is unchanged; set `main = COMMANDS.run`
  and delete the `def main`. `ml-stack-surface capture --out DIR --src SRC` writes every
  subcommand's `--help` and the output of everything safe to run; take one against
  `main`'s `src/` and one against the branch, and `ml-stack-surface diff BEFORE AFTER`
  names what moved.
  Worth doing next in this order: `ml-stack-setup`
  and `ml-stack-doctor` (one module, two entry points), `ml-stack-ingest`,
  `ml-stack-models`, `ml-stack-store`, `ml-stack-speech`, `ml-stack-train-run`,
  `ml-stack-train-tools`, `ml-stack-chat`, `ml-stack-claude`, `ml-stack-agent`,
  `ml-stack-graph`, `ml-stack-audit`, `ml-stack-suite`, `ml-stack-fleet`,
  `ml-stack-peers`, `ml-stack-traind`. A command whose row in the README's table changes
  is a change to `ml_stack.cli.reference` and `scripts/reference --write`.

- [ ] **`do.bench_cli` runs four bench subcommands in-process and hands back what they
  printed.** `BENCH_SUBS` in `do.py` names `compare`, `animate`, `standard` and `speed`;
  `standard` and `speed` detach and return a log and a pid, but `compare` and `animate`
  go through `mcp._captured(lambda: _main([sub, *args]))`, so a model reads a fixed-width
  progress listing instead of a value. Each has a function already: `compare` is
  `bench.comparison.assemble` and `write`, which return the document and where it went;
  `animate` returns the path of the file it rendered. Both tools take `args: list[str]` and
  are documented as following the command line, so giving them typed arguments changes what
  an agent driving them writes.

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

### Red-teaming

`python -m ml_stack.redteam` and `docs/redteam.md` exist; what is missing from them:

- [ ] **Re-run `docs/redteam/baseline-2026-10-02.json` after each of agent, decide, hardening and
  model-discovery lands.** `python -m ml_stack.redteam run --against docs/redteam/baseline-2026-10-02.json`
  needs an installed GGUF and `pip install -e ".[redteam]"` in its own virtualenv. The kept baseline
  was measured on `0.2dev` merged with `agent/port-pcbe` and `agent/hardening` (see the baseline's
  header); the `loop` and `compaction` scenarios import `ml_stack.agent`, so
  `.github/workflows/redteam.yml` runs only `extraction,chat,fleet` until it is on the
  development branch -- add `loop,compaction` to its `--scenarios` then.
- [ ] **The guard benchmark is not written.** The brief's precision/recall of tool-call guards on
  injected against benign calls needs a `guard` scenario: labelled tool-call contexts built from
  `styles.json`, the pages and the PyRIT converters, each guard asked `before_tool_call`, recall and
  false-positive rate reported per guard. `scenarios/loop.py:PolicyGuard` is the reference guard to
  start from. `ml_stack.decide.guard.ToolCallGuard` uses its own `ml_stack.interventions` (`Call`,
  `Context`, `Verdict`), not `ml_stack.agent.interventions`; one of the two has to go before the
  benchmark can take both.
- [ ] **Multi-turn attacks and a judge are not wired.** PyRIT's `CrescendoAttack`, `PAIRAttack` and
  `RedTeamingAttack` need an adversarial chat target and a scorer that is not a canary; the local
  model could play both (`pyrit_bridge.ResponderTarget` wraps any `Responder`), with its scores
  reported as noisy. Every scorer today is objective evidence.
- [ ] **The findings in `docs/redteam/findings.md` marked open have an owner and no fix.** Each
  names the file, a reproducing `python -m ml_stack.redteam run --scenarios ...` line and the
  test that should go red when it is fixed.
- [ ] **Check the `redteam` extra on Python 3.11, 3.12 and 3.14 by installing it.** PyRIT 1.1.0
  declares `>=3.10,<3.15` and `uv pip compile` resolves it for all three; only 3.13 was installed
  and run.

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

- [ ] **`hub.find`, `hub.files`, `hub.fetch`, `hub.card` and the draft lookups still go
  through `huggingface_hub`.** `hub.remote` (listing, search) and `hub.pull` do the same over
  plain HTTP with resume, checksums and progress. Move the first group onto the second so
  `ml-stack-models find|files|fetch` work without the `hub` extra and put files in the
  store `discover()` reads first, and so `hub.fetch` stops being a second download path.
- [ ] **Folders for LM Studio, GPT4All, Jan, ModelScope and KaggleHub, and every Linux and
  Windows folder in `hub.places`, follow each tool's documentation.** Only the Hugging Face
  cache, llama.cpp's cache and Ollama were read off installs, all on macOS. Run
  `ml-stack-models where` on a machine with each tool and set `verified` in `hub/places.py`.
- [ ] **The memory estimate is measured only with flash attention on, on Apple silicon.**
  Flash attention off, a draft model, partial offload, CUDA, ROCm and Windows are formulas
  with no load log behind them, and `hub.probe`'s `nvidia-smi` and `rocm-smi` readers have
  only been run on sample output. Each needs a machine with that hardware and the table in
  `docs/model-discovery.md` extended.
- [ ] **`suggest_model` ranks by parameters and quantisation, not by a score.** The model
  ranking in `docs/model-ranking.md` and `data/profiles.json` are per workload and per model
  family; feed them in so "recommended for this machine" follows a measurement.
- [ ] **Pulls read `HF_TOKEN` and the login file by hand and fetch with `ml_stack.http`.**
  When `ml_stack.credentials` and the guarded fetch land (`agent/hardening`), read the token
  through the first and route the redirect target through the second.

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
  `packaging/install.ps1` from the release repository (the app) or the `--headless` mode, then from here `ml-stack-fleet status`, a
  `ml-stack-fleet plan --users 3 --context 16384 --apply`, and a `sweep --fleet --serve
  gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf --sample 6`. Expect bugs; the daemon's log and
  `ml-stack-doctor` are the first two places to look. After that the Windows box follows
  releases or main on its own (`fleet status` shows COMMIT/UPDATES).

### The window

- [ ] **The Tauri window is proven on macOS only.** `app/`
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
  - **Nobody has opened the Windows or Linux window.** The bundles themselves build:
    `gh run view 32881712199` (v0.1.7, 2026-08-25) shows `bundle (ubuntu-latest,
    ml-stack-linux-x86_64)` and `bundle (windows-latest, ml-stack-windows-x86_64)` both
    succeeded, against a matrix identical to today's, and the release workflow's own
    `it starts and serves the interface` and `the screens answer` steps ran there. What
    has not happened is a person installing the NSIS setup or the AppImage on a real
    machine of that kind and using it -- which is what `install.ps1` and `install.sh`
    expect to exist. The Linux AppImage is installed as `~/.local/bin/ml-stack`.
  - **`release.yml`'s `it starts and serves the interface` step curls a file that does
    not exist.** `curl -sf ... http://127.0.0.1:8899/ui/static/app.js` 404s against a
    freshly built headless binary: the page has no separate `app.js` (`shell.html`'s
    `__SCRIPTS__` is filled with each component's own inline `<script>`, and
    `ml_stack/fleet/web/` holds only `shell.html`, `style.css` and `components/`), and
    nothing in the tree writes one. Either the step is checking for a file this UI
    architecture never produces and should check something that does exist (`style.css`,
    or a real screen), or `app.js` is meant to exist and something upstream of packaging
    stopped writing it.

### What the window has not been driven through

- [ ] **What WKWebView renders the same as Chromium, so far.** Light-DOM custom elements,
  `[hidden]`, the sheet's backdrop blur, the context `input[type=range]`, the styled
  checkbox and radio rows, a card taller than the window with its own scrollbar, and the
  system sans stack. Nothing needed a resize to appear and no font fell back. Not yet
  driven in the window: the Chat screen with a model behind it, the Models screen's
  download, and the Fit screen's tables beyond confirming they draw.
- [ ] **`ml-stack-walk` has never walked a screen with a model behind it.** Every walk so
  far was against a daemon serving nothing and an `ml-stack-graph serve` with no `--model`,
  so Chat reads "No model is running yet", the graph's question box is only read, and the
  review queue is skipped because `#review-box` stays hidden with no change requests
  waiting. Unwalked: sending a message on Chat, a download on Models, `--ask` through
  `/ask/stream`, and a review queue with something in it. `ml-stack-serve up <model>`
  first, then `ml-stack-walk fleet chat models` and `ml-stack-walk graph ask review --ask
  "..."`.

### The code itself

- [ ] **The hard 900-line file gate does not look at `tests/` at all.**
  `scripts/gates/deep_files.py` sets `ROOTS = ("src/ml_stack",)`, so thirteen test files
  are over the limit and invisible to the gate that refuses a source file at 901 lines:
  `test_graph_bench.py` 4,828, `test_graph_ask.py` 2,383, `test_serve_fit.py` 1,574,
  `test_ingest.py` 1,549, `test_graph_page.py` 1,403, `test_client.py` 1,340,
  `test_serve.py` 1,102, `test_serve_cli.py` 1,077, `test_fleet_ui.py` 1,023,
  `test_fleet_daemon.py` 1,002, `test_fleet_models.py` 969, `test_hub.py` 963,
  `test_fleet_join.py` 936. Widen `ROOTS` to `tests/` and split all thirteen; the gate
  takes no budget line, so this lands as one branch per file, not as a recorded number.
  Held until 0.2.0 was cut (the owner, 2026-09-18) on sequencing alone -- a 4,828-line file is
  not a one-branch job -- and nothing else.
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

### The test suite

- [ ] **The red-team runs do not use the verdict cache yet.** `ml_stack.testing.verdicts` has the
  cache (key: model-file hash, attack id, guard or prompt version, source of the modules the
  attack touches; a failing verdict is never served), `sample` (seeded tenth plus every class
  touching a changed file) and `Limits` (few tokens, thinking off, `cache_prompt`, the canary as a
  stop string). `ml_stack.redteam` lives on `agent/redteam` and is not on this branch, so nothing
  calls them. After that branch lands: wrap each scenario's `execute` in `verdicts.run`, add
  `--no-cache` and `--quick` (the sample) to the red-team command and keep the full sweep as the
  nightly and release command, declare each scenario's `surfaces`, and send `Limits.body(...)` with
  every request. Then measure per-attack and total wall time with and without the cache against a
  real model through the broker, serially, and check the cached verdicts equal an uncached full
  run on a sample; none of that has been measured, only the scripted Bash-guard layer (20
  attacks: 1.3 s uncached, 0.00 s on a warm cache).
- [ ] **Tests that assert a wall-clock bound fail on a loaded machine.** `test_serve.py` (a
  negative-cache lease under 0.5 s), `test_fleet_join.py` (`peer_pause` with no cluster under
  1 s), `test_client.py` (a dead process polled under 2 s), `test_fleet_bench.py` (a preparing
  handle under 2 s, an install slower than 4 s), `test_bench_selfcheck.py` (under 10 s),
  `test_fleet_work.py` (under 10 s), `test_ingest.py` (a 300-unit fold under 20 s). Each wants the
  clock handed in, or a count of polls or calls in place of seconds.
- [ ] **`scripts/test fast` has not been timed on a quiet machine.** The target is under 30 s at
  `-n 4`; it was measured only beside other suites (load 90 to 270), where it took 143 CPU
  seconds. Time it alone, and re-run `scripts/test heavy --junit j.xml` if it is over.
- [ ] **`scripts/test quick` reruns everything when the installed packages change.** pytest-testmon
  records the package list in `.testmondata`, so any `pip install` in the shared interpreter
  makes the next `quick` a full run, and each new worktree pays one full run to record its map.
  Seeding a new worktree's map from the last one, or ignoring the packages that change often
  (`testmon_ignore_dependencies`), would stop that.
- [ ] **`scripts/verify-quick` has covered seven modules, not fifteen, and `quick` reruns about
  855 tests whatever changed.** It mutated `serve/ports`, `serve/escalation`, `client/tokens`,
  `client/cache`, `harness`, `claude` and `hub/naming`; the one miss (`claude.main`, reached only
  through `tests/test_cli_help.py`'s computed import) is covered by `affected.dynamic_tests` and
  caught on the re-run. Not yet covered: `fleet`, `http`, `ui`, `train`, `units` and the
  credential code in `fleet/session`. The 855 tests are a constant set testmon selects on top of
  the change (831 to 855 in every row, load 190 to 240); what makes them always affected is not
  found, and it is most of `quick`'s run time.
- [ ] **Two slow tests fail under load and pass alone.** `test_fleet_daemon.py::
  test_the_default_is_still_one_job_at_a_time` and `test_fleet_discovery.py::
  test_peers_ls_reports_the_running_daemon` (a fixture error) failed in a `--slow -n 2` run of
  the fleet files beside other suites (load 40 to 100) and passed alone. Both wait on a real
  daemon with a fixed deadline; they want a deadline that scales or a readiness probe. The rest of
  the slow tier has not been run under `_no_public_network`, which refuses a connection or name
  lookup beyond the LAN in every test.

### The agent workspace

`ml_stack.workspace` (`docs/workspace.md`) is a library, a command and MCP tools over a state
directory. What it does not do yet:
- [ ] **A loopback daemon** answering `macauth` signed requests, for an agent that runs as
  another user or in a sandbox that cannot share the directory. Today the boundary is the
  account.
- [ ] **A repair command** for a damaged log: a broken chain refuses new rows until someone
  moves the file aside by hand.
- [ ] **Claims keyed to a process start time**, so a reused pid does not keep a dead owner's
  claim alive until its TTL.
- [ ] **Scratch size enforced as files are written**: the limit is checked when a folder is
  made and when `scratch-ls` runs, not on each write.
- [ ] **Embedding search over notes** through the graph store; keyword search through
  `graph.search.lexical` is what exists.
- [ ] **Embedding search over files**: `file search` ranks words with `graph.search.lexical`
  fused per word; vectors are not used.
- [ ] **A request to the person for a held file**: a held file waits in the quarantine
  (`quarantine-ls`); nothing raises a `requests` item for it.
- [ ] **The verified or claimed model on a file's `posted_by` edge**, once model identity is
  on the integration branch; the edge carries `agent-claimed` only.
- [ ] **The Board page lists a file's download on its message** but has no per-board file list.
- [ ] **Files outlive their message only for the person**: `gc` shreds the content of a
  file once its messages are pruned; a longer-lived file store needs its own retention.

## Measurements

Each needs the GPU and the owner's call. Estimate before it runs, smoke it before it is
paid for, and one thing on the card at a time.

### Served shapes with no paired baseline

`docs/report-2026-09-23.md` pairs a drafted run only with the undrafted run whose identity
is the same minus the head. Three served shapes in `src/ml_stack/data/profiles.json` have
no such pair, so what their head buys is unmeasured; each is read again with
`ml-stack-bench report --profile` once its baseline is kept.

- [ ] **Whether the gemma-4 E2B and E4B heads are worth serving at all.** Both records serve
  `mtp-gemma-4-E{2,4}B-it.gguf` at `spec_draft_max 2` with a q8_0 cache, set from the
  hundred-question rows `gemma-4-E2B-it-plain-kv-q8_0` (1.5 s/q, 30% F1) and
  `gemma-4-E4B-it-plain-kv-q8_0` (3.1 s/q, 40%). No undrafted run shares those rows' identity.
  The paired sweeps over fewer questions put length 2 at 0.99-1.29x and length 4 at
  0.85-1.30x, the fastest inside the noise, so depth 2 is kept and nothing yet says the head
  beats none. `ml-stack-bench drafts gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf --draft ''
  --draft mtp-gemma-4-E4B-it.gguf --n-max 2 --n-max 4 --serve-kv q8_0`, then the same for
  E2B; at the recorded s/q a hundred questions is ~5 and ~3 minutes an arm plus a load.
- [ ] **The gemma-4 26B-A4B record serves a head nothing has paired.** Its record is new,
  from `gemma-4-26B-A4-plain-rb4096` (34 questions, 37% F1, 25.4 s/q): the MTP head and
  `--reasoning-budget 4096`, against 43% F1 at 39.5 s/q plain with no head and unlimited
  thinking, a fall `held_up` reads as inside the noise at 34 questions. The report has no
  `draft:none` run at that identity. `ml-stack-bench drafts gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf
  --draft '' --draft mtp-gemma-4-26B-A4B-it.gguf --reasoning-budget 4096`, ~15-25 minutes
  an arm at 34 questions.
- [ ] **Flash-Next's `-ub 2048` and `--spec-draft-p-min 0.5` rest on one nine-question run
  each whose record carries neither flag.** The `ub2048` and `pmin05` rows share the recorded
  identity of the plain row (25.5 and 23.6 s/q against 29.2-29.4), so only their labels say
  what differed, and `extra_args` holds both because `report --profile` carries them from
  the older record. Runs now record `spec_p_min`; a sampled sweep with and without each flag
  on the profile's serving (`--sample 20`, ~10 minutes an arm) settles both.

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
  days of GPU at 86 s a unit, one slot -- is the owner's call; the command is `ml-stack-ingest
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

### Draft trees for llama.cpp

`patches/llama.cpp/0003-speculative-tree.patch` verifies a tree of drafted nodes in one pass
-- ancestor-masked attention, per-path DeltaNet state, a commit that keeps the accepted path
-- and `ml-stack-serve up --spec-tree W` asks for it. `docs/llama-cpp-tree-speculative.md`
says what it does and what it refuses.

- [ ] **The bench cannot ask for a tree.** `ServerSpec.spec_tree` and `up --spec-tree` carry
  it, but `ml-stack-bench speed/sweep/drafts` build their serving from `bench/serve.py` and
  `bench/profiles.py`, which have no field for it, so a tree arm has to be served by hand
  with `up` and measured with `--on`. Give `Serving` a tree width the way it carries
  `draft_n_max`, and the drafts table a tree arm.
- [ ] **Trees serve one slot.** The server refuses `--spec-tree` beside `--parallel > 1`: a
  tree batch is one sequence, and the KV cache keeps one tree per sequence. Several slots
  need the drafted trees of each slot in one batch, which is a batch-level tree rather than a
  context-level one.
- [ ] **A tree pass costs 2.6x a chain pass of the same width, and the recurrent layers are
  why.** Measured 2026-09-17 on Qwen3.8-27B Q4_K_M, f16 KV, one slot, 3 prompts x 128 tokens
  greedy, `llama-server` at commit 3466812 + the three patches: no draft 19.3 tok/s (51.6 ms a
  pass); chain at 4 drafted 12.9 tok/s, 2.81 tokens a pass, 198 ms; chain at 12 drafted 11.5
  tok/s, 3.09 tokens a pass, 228 ms; tree W=2/6 nodes 5.6 tok/s, 3.01 tokens a pass, 516 ms;
  tree W=3/12 nodes 5.4 tok/s, 3.48 tokens a pass, 603 ms. The tree keeps more tokens a pass
  than the chain of the same width, and loses several times over on the pass itself. Graph
  rebuilding is not the cause: letting a repeated tree shape reuse its graph left the pass at
  644 ms (shapes rarely repeat), so that narrowing was taken back out. What is left is
  `build_recurrent_attn`'s tree branch, which materialises a full DeltaNet state per node and
  runs `max depth + 1` rounds of gather-and-update over all of them -- tens of GB of state
  traffic a layer a pass. The fix is the shape the MLX engine already uses
  (`ml_stack/spec/deltanet.py:walk`): walk each node's ancestors carrying the `p` vector rather
  than the state matrix, as one op, so a tree pass costs what a chain pass of the same width
  costs. Until then a tree is slower than a chain on this model and the numbers above are the
  baseline to beat.
- [ ] **Width and node budget have not been swept.** 3 branches and 12 nodes was the first
  shape tried. What the curve looks like against depth, and against the chain at the same
  verification cost, is a `ml-stack-draft`-shaped question once the bench can ask for a tree.
- [ ] **The ngram tree drafter has not been measured.** It builds a trie of the continuations
  that followed the last n-gram; only the MTP tree drafter has been driven end to end.

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

the owner, 2026-09-03: which is better, faster, or both; a LinkedIn graphic; "if ollama can't
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

### Tree speculative decoding on MLX

- [ ] **The full tree-decoding table has not been run.** Smoke numbers only, on a busy
  card. One model on the card at a time, the embedding and page servers down, AC power:
  ```
  # llama.cpp, Flash-Next UD-Q4_K_XL (the GGUF in the Hub cache): no head, then its MTP head (unsloth fork build)
  ml-stack-bench speed --serve Qwen3.8-Flash-Next-UD-Q4_K_XL --serve-label flash-gguf --no-draft --prompts 512,4096 --streams 1 --generate 512
  ml-stack-bench speed --serve Qwen3.8-Flash-Next-UD-Q4_K_XL --serve-label flash-gguf-mtp --serve-draft embedded --prompts 512,4096 --streams 1 --generate 512
  # the 27B at 4 bits on MLX: one node a pass, then each drafter
  ml-stack-bench speed --serve mlx:lmstudio-community/Qwen3.8-27B-MLX-4bit --serve-label q27-mlx-none --no-profile --prompts 512,4096 --streams 1 --generate 512
  ml-stack-bench speed --serve mlx:lmstudio-community/Qwen3.8-27B-MLX-4bit --serve-draft ngram --serve-label q27-tree-ngram --no-profile --prompts 512,4096 --streams 1 --generate 512
  ml-stack-bench speed --serve mlx:lmstudio-community/Qwen3.8-27B-MLX-4bit --serve-draft mlx-community/Qwen3.8-27B-MTP-bf16 --serve-label q27-tree-mtp --no-profile --prompts 512,4096 --streams 1 --generate 512
  ml-stack-bench speed --serve mlx:lmstudio-community/Qwen3.8-27B-MLX-4bit --serve-draft JonasLoos/Qwen3.8-27B-DFlash2-b32 --serve-label q27-tree-dflash --no-profile --prompts 512,4096 --streams 1 --generate 512
  # the same arms on real work (graph questions with tool calls), accepted tokens per pass and drafting share
  ml-stack-draft mlx:lmstudio-community/Qwen3.8-27B-MLX-4bit --draft JonasLoos/Qwen3.8-27B-DFlash2-b32 --depth 8 --depth 16 --depth 32
  ml-stack-draft mlx:lmstudio-community/Qwen3.8-27B-MLX-4bit --draft mlx-community/Qwen3.8-27B-MTP-bf16 --depth 16 --depth 32
  # qwen-spec itself against the port, same prompts: its server is not a lease, so it is measured with --on
  uvx --from git+https://github.com/JonasLoos/qwen-spec qwen-spec-server --port 8098 --greedy
  ml-stack-bench speed --on qwen-spec=http://127.0.0.1:8098 --prompts 512,4096 --streams 1 --generate 512
  ml-stack-bench show   # tok/s, draft acceptance; tok/pass is completion_tokens / verify_n in each row's timings
  ```
  Plain mlx-lm with no verifier in the way is `mlx_lm.generate --model lmstudio-community/Qwen3.8-27B-MLX-4bit --max-tokens 512 -p ...`;
  it is not a lease, so its tok/s is read off its own report. Resident peak is the bench's
  memory record for a leased arm; upstream's is not tracked.
- [ ] **The Flash-Next lossless witness is unfinished, and so are its mutation checks.**
  `ml-stack-bench tree lossless` passed for the `ngram` drafter on the chat, code and math
  prompts (`docs/tree-flash-next-2026-09-17.md`); the `mtp` and `dflash` drafters and the
  2,300-token prompt that decodes past the indexer budget have not been through it. The
  three paths the witness is there to hold -- the PLE commit in `spec/qwen4.py:tree_ple`,
  the indexer selection in `_sparse_mask`, and the hyper-connection inject in
  `Qwen4ExpLayout.block` -- have not been mutation-checked against it, so what it catches
  is unknown. Both want a machine with 80G free: with 25-37G of other agents' servers
  resident, the 74G load swapped and one 2,300-token prompt took over twenty minutes.
- [ ] **The Flash-Next speed table has not been run.** The arms and the command are
  ```
  ml-stack-bench tree speed --samples 3 --tokens-list 128 512 \
      --drafter ngram --drafter mtp=<MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf> \
      --drafter dflash=PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash
  ```
  which times plain MLX and each drafter in one process and then llama.cpp with and
  without the shared MTP head through the broker. Every sample records the bytes held
  beside it, so a row taken under contention says so, but the run refuses a machine that
  is not quiet unless `--anyway`.
- [ ] **The mapped n-gram table makes a long prefill disk-bound.** A single-token step
  reads sixteen rows and costs nothing measurable; a 2,300-token prefill reads about
  37,000 scattered rows out of the 37G table (measured 129-285 MB/s, GPU 70-97%).
  `mlx_vlm.models.qwen4_exp.ple_storage` offers `cache_rows` and an interleaved row store
  (`materialize_interleaved_ple_store`); neither has been tried, and what a long prefill
  costs with the table resident instead is unmeasured.
- [ ] **On an M4 the verify pass costs 4.1x one token at 16 nodes.** Measured on 27B 4-bit:
  the dense MLP's quantized matmul grows from 22 ms to 91 ms between 1 and 16 nodes, the
  DeltaNet mixer from 12 to 46, attention from 4 to 13. A small-M 4-bit matmul kernel that
  reads each weight group once was slower than stock at every row count (0.64 against
  0.41 ms at 8 rows), so the cost is compute, not memory. Untried: fusing gate/up, q/k/v
  and qkv/z projections into one matmul each, which cuts kernel launches per pass by
  about a third.
- [ ] **The tree pass flips one-ulp bf16 ties against sequential decoding.** On 27B, greedy,
  one code prompt: token 22 had a 0.125 logit margin and the verifier's own single-node pass
  took the other token; every layer differs from mlx-lm's decode step by bf16 rounding
  from layer 0 (float32 conv and DeltaNet in the parallel form). Greedy tree decoding is
  token-identical to mlx-lm on the float32 test model. Whether matching mlx-lm's dtypes
  exactly (bf16 conv) removes the flip on the 27B is unmeasured.
- [ ] **Nothing tests that the engine decodes on the thread that loaded it.** Loading on
  the main thread and decoding on the worker failed on the 27B with its DFlash drafter
  ("There is no Stream(gpu, 1) in current thread"), and the tiny float32 model does not
  reproduce it, so moving the load back to the main thread leaves every test green.

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

the owner, 2026-09-04: "we're never going to have that many users, so flash-next is the way to
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
  agents' suites were running, so the wall clock is unreliable either way (the owner). Next:
  `show --trace flashprefix-plain`, find what invites the second `show` (the show nudge now
  offers every tool), then a quiet `--sample 20` of both.
- [ ] **The fine-tuned tool caller.** `ml-stack-train-tools from-bench` over the traced runs
  (traces are on by default at ≤20 questions; the hundred-question runs before that carry
  none -- rerun Flash-Next's hundred with `--trace` for ~5,000 turns), then
  `ml-stack-train-run --recipe tool-calls --size e4b --lora --export-gguf --yes` (~18 h
  here; the owner's go-ahead), then the measure in `docs/research/tool-caller-finetune.md`.
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

### Decision models

- [ ] **Trained deciders are not in model discovery.** `ml_stack.decide.registry` records them
  by name under the state root; `hub.discover` (model-discovery branch) should list a directory
  holding `decider.json` as a model of format `decider`.
- [ ] **The pointer backend takes the GPU without a lease.** `PointerDecider` loads onto MPS or
  CUDA directly, so it can run beside a served model; it should ask `ml_stack.serve` for a lease
  and release it when `/decide` has been idle.
- [ ] **The pointer backend is slow on Apple silicon at guard-prompt length.** 200 to 450 ms
  median for 300 to 600 tokens: transformers runs the Gated DeltaNet layers in a PyTorch
  reference loop. A Metal chunk kernel, or one forward over the shared state for the three
  guard questions, is the fix; neither exists here.
- [ ] **No per-process memory figures.** The decision-models document has latency, accuracy,
  Brier and ECE but only the pointer model's device memory (3.7 GB); measure resident memory
  of each llama-server and the embedding server with the `ps` of the pid `ml-stack-serve up`
  prints.
- [ ] **The guard benchmark has one author.** 412 cases written and checked by one person;
  a second reviewer, and cases from a project other than the one the templates came from, are
  what would make an accuracy figure mean more than "on these templates". `make-cases` output
  is rule-labelled and has never been read in bulk.
- [ ] **A Qwen3.5-4B Q4_K_M server returns non-finite logits** on llama.cpp build 10816 (the
  first token is `@` with null log-probabilities). Check against a newer build.
- [ ] **MLX does not train a decider.** `ml-stack-train-decider` is PyTorch only.

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

### Security (ml-stack issue #18)

What the 2026-10 hardening pass left open; `docs/security.md` has the model and the findings.
- [ ] **Task grants remain tied to their original cluster after Dev rekey.**
  Automatic Dev renewal preserves the agent capability and Board history, but task eligibility
  and remote task authorization compare the complete project grant, including `cluster_id`.
  Compare stable project authority without changing task specs or hashes; retain negative
  authorization coverage for different projects and authorities.
- [ ] **A daemon's certificate is renewed only when it starts.** `tls.identity` makes a new
  one when under 30 days of the 90 are left, but a daemon that runs longer than 60 days
  keeps serving the old one until it restarts. Renewing in place means rebuilding the
  `SSLContext` the server holds and re-announcing the new certificate; peers re-pin from the
  next beacon.
- [ ] **File and model downloads and the `/infer` stream are signed and not sealed by the
  application.** They travel inside the daemon's TLS; with `ML_STACK_FLEET_TLS=off` they are
  readable on the segment. Sealing them needs a framed stream of sealed chunks.
- [ ] **A machine that joined from a recovery file holds no hash of the passphrase**, so it cannot
  take other machines in by passphrase and the web interface cannot sign it in by passphrase.
- [ ] **`web.py`, `scrape/` and `ingest/run.py` still fetch through `http.check` and urllib.**
  `ml_stack.httpguard.fetch` pins the checked address for the connection and checks every
  redirect; the page reader and the browser (a redirect or a sub-request inside Playwright
  reaches a private address unchecked) should use it, or a `page.route` that applies
  `httpguard.resolve`. Another branch owns `web.py`.
- [ ] **`hub/` calls `huggingface_hub` without `token=`.** `credentials.get("HF_TOKEN")` is the
  resolver; `listing.py`, `cards.py`, `drafts.py`, `spec/engine.py` and `mlx_tree.py` should
  pass it so `ML_STACK_CREDENTIALS_FILE` and the keychain reach a download.
- [ ] **37 `except ...: pass` sites report as S110 and S112** (device probes, memory
  readings, the accelerator reports in `train/accelerator.py`). Each should name the
  exception it expects and log the rest at debug.
- [ ] **`fleet/api.py` `do_GET` and `_route_post` are 99 and 115 statements.** Splitting each
  route into a method is what clears PLR0915 there.
- [ ] **`scripts/test-on-linux` was not run for this pass,** and CI's `pip install` lines are
  unpinned and unhashed (there is no lockfile to audit with `pip-audit` or `osv-scanner`).
- [ ] **Windows job objects.** A server survives a killed host only until the watchdog
  notices; on Windows the watchdog is all there is, and it was not run there.

### Integration (0.3.0)

`docs/INTEGRATION-REPORT.md` has the merge order, the commands and the numbers.
- [ ] **Two integration decisions to check by a person.** (1) `redteam/tools.py` reads a page with
  the plain extractor on purpose, so the hidden-text attacks still reach the guards behind the
  reader; the reader the agent really uses (`web.read`) strips them first, and
  `test_redteam_toolbox.py` pins that. A red-team arm that runs the toolbox through `web.read`
  would measure the pipeline's stripping instead. (2) `ML_STACK_DECIDE_URL` is loopback-only unless the host is named in
  `ML_STACK_FETCH_ALLOW_HOSTS` (decided by the lead: a decider sees every call it judges), so the
  net-scan exemption for it is a guarantee (`docs/security.md`, "What is exempt from the net scan";
  `tests/test_decide_hosts.py`).
- [ ] **The `spake2` package was not checked against `fleet/onboard/pake.py` on the wheel the
  `fleet-onboard` extra installs;** the pairing tests ran with whatever `spake2` this machine has.
- [ ] **Nine slow-tier tests fail here and eight of them on `agent/hardening` alone:**
  `test_fleet_bench.py::test_a_frozen_peer_installs_the_bench_after_accepting_the_job`,
  `test_fleet_chat.py::...test_asking_for_a_model_answers_before_it_has_arrived`,
  `test_fleet_daemon.py::test_serve_forever_prefers_the_settings_name_over_the_hostname`,
  `test_walk.py::test_saying_something_in_the_fleet_chat_waits_for_the_whole_reply`, the two
  offline-install tests in `test_packaging_install_runs.py` (they need built wheels and
  `ladybug`), two graph-page browser waits and a TLS handshake timeout in
  `test_fleet_bind.py`. Each needs a cause, not a retry.
- [ ] **The `--redteam` tier runs nowhere by default** and is what found a red-team policy guard
  that denied every call. Put it on a schedule or in the default tier.
- [ ] **Python 3.11, `scripts/test-on-linux`, Windows and Linux were not run** for 0.3.0 (the 3.11
  build segfaults on this machine, Docker's daemon was down). `vermin` says 3.11 is the minimum
  of `src`, `scripts` and `tests`.
- [ ] **The isolation guard watches files, not directories.** `~/.ml-stack/gate/gpu` changes
  when anything leases a real model; a test that wrote only a directory there would not fail the
  run.
- [ ] **Record `tests-collected` on a machine with every extra installed.** `scripts/budgets
  --update` refuses without `nemoguardrails`.
- [ ] **Sandbox: Linux is argv-only.** `bubblewrap.arguments` is tested as a list; no Linux host
  has run it. Landlock and seccomp are not written.
- [ ] **Sandbox: container backend.** `sandbox/container.py` raises `NotImplementedError`; Apple
  `container` was installed but its service was not started, so latency and mounts are unmeasured.
- [ ] **Sandbox: host allow-lists.** Network is deny, loopback or named ports. Host names need the
  `sandbox-runtime` proxy as an optional backend.
- [ ] **Sandbox: confined model server is opt-in** (`ML_STACK_SANDBOX_SERVE=1`). Make it the default
  after speculative heads, mmproj and multi-shard models have run inside it.
- [ ] **Sandbox: Claude Code's Bash sandbox** is configured in `harness.confined_bash()` and
  checked as options only; no model-driven run has exercised it.

## Pending verification

- [ ] **The destructive-action classifier's model layer has not been measured on a real decider**
  (`docs/destructive-actions.md`); only the stub logprob server has exercised it.
- [ ] **Destructive verdicts are not written to the activity log.** They reach `Run.events` and
  `DestructiveRail.on_verdict` only.
- [ ] **An approved plan is not classified when it is approved.** A destructive or unsure call asks
  even when a step names it.
- [ ] **The Board is not in the MCP tools or the chat tools.** Agents reach it through
  `ml-stack-workspace board ...` only; an MCP `workspace_board_*` set needs the same membership checks
  and a red-team pass.

- [ ] **Board subscriptions deliver a board's older messages to a new member.** A subscription made
  after others posted delivers every unread row after the inbox cursor, bounded by `--limit` only.
- [ ] **The page's live feed is a long poll on one wake pipe name per person (`<id>.web`).** Two open tabs
  share one pipe, so one of them can wake up to 2 s late; `board.view` is recorded for thread lists only.
- [ ] **A board post wakes every member's follower pipes, and a waiting `wait` rescans on each.** The cost is
  fine at 40 agents (p99 38 ms); a board with hundreds of members needs the signal limited to subscribers
  and open followers.
- [ ] **No board search, edit or delete, and no per-board retention.** Board messages age out with the bus
  (`retention_s`); `boards.jsonl` is never pruned.
