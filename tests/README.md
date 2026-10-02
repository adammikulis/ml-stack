# The test suite

Nothing here serves a model, touches a GPU, reaches the Hub, or reads anything under
`~/.ml-stack`, `~/.cache` or a real home directory — see *Nothing reads the machine* below for
how that is enforced rather than remembered. Every person, company, place and model file is
invented.

## Running them

One command per tier, `scripts/test <tier>`; `-n N` sets the workers (default 4) and any other
argument goes to pytest.

| tier | what runs | when |
| --- | --- | --- |
| `quick` | the tests the change reaches (below) | after every edit |
| `fast` | every test not marked `slow` or `heavy`, and not the four that import mlx while collecting | before a commit |
| `full` | every test not marked `slow` | what bare `pytest` runs; before reporting |
| `slow` | only the tests marked `slow` | after touching packaging, the page or the fleet |
| `all` | everything, `--slow` included | what CI runs; before a merge |

```sh
scripts/test quick --explain      # which file selected which test file
scripts/test fast -n 2            # while a bench has the GPU
pytest -n 0                       # one process, in file order, when a failure needs a clean order
pytest --durations=0 --durations-min=1.8    # what is costing the wall clock
```

`-n 4` is the one to use while a measurement is running: a full `-n auto` run competes with
the bench for cores and both get slower, and a bench's wall clock is the thing being measured.

### How `quick` chooses

It diffs the working tree against the merge-base with `0.2dev` (`--base` to change it), then
takes the union of two selections:

- **pytest-testmon** records, per test, the functions it executed, in `.testmondata` (ignored
  by git, one per checkout). The first `quick` in a checkout runs the whole full tier to
  record it; after that testmon reruns the tests that executed a changed function, and the
  ones that failed last time.
- **The import graph** (`scripts/affected.py`): every test file within two imports of a
  changed module, counting imports inside functions and a module named in a string
  (`-m ml_stack.x`, `import_module`). It exists for what testmon cannot see, such as code that
  only a child process runs. `--explain` prints each file and what selected it.

It runs the full tier, and says why, when the change touches `pyproject.toml`,
`tests/conftest.py`, `budgets.json`, a file that is not source, a test or prose, a deleted
module, or when pytest-testmon is not installed (`pip install -e '.[test]'`).

`quick` adds the cheap tree-wide checks (`test_layers`, `test_wiring`, the conftest and isolation
guards) to every selection and leaves out the slow ones (`test_budgets`, `test_gates_*`,
`test_no_data_files`): the pre-commit hook runs the budgets, and `full` and `all` run all
of them. `tests/test_quick_select.py` pins the selection rules.

### `heavy`

`tests/heavy-modules.txt` lists the modules that cost the most; `conftest.py` marks them
`heavy`. `scripts/test heavy --junit j.xml` rewrites the list from a `--junitxml` run.

## What is slow, and why

`@pytest.mark.slow` means the cost is a browser, a subprocess, a wheel build or a real network
timeout — something over about two seconds *every* time, rather than a one-off import or model
load that the first test in a worker happens to be charged for. Dropping them takes the wall
clock from about 145 s to about 109 s on four workers.

| where | what it costs |
| --- | --- |
| `test_graph_page.py` (whole module) | launches headless Chromium and drives a real page |
| `test_fleet_discovery.py` (whole module) | real UDP on a real interface and a real daemon subprocess |
| `test_fleet_join.py`, `test_fleet_daemon.py`, `test_fleet_serving.py` | health and beacon deadlines waited out for real |
| `test_no_real_names.py` (the two wrapper tests) | runs the commit hook through `sh`; the in-process `check()` tests are fast and stay in |
| `test_packaging.py`, `test_fleet_environment.py` | builds a wheel |
| `test_bench_selfcheck.py` (four of them) | runs the whole self-check path |

Two costs are *not* marked, because marking them would move the cost rather than remove it:

- `test_graph_thread.py::long_thread` — a module-scoped fixture that builds a two-hundred-turn
  conversation once (~5 s) and is then read by five tests. Marking those five would drop real
  recall coverage from the fast subset to save five seconds, once.
- The first `presidio` test in a worker (`test_no_real_names.py`, `test_redact.py`) pays for
  loading the analyser. Marking it slow just charges the next test instead.

## Nothing reads the machine

`conftest.py` has one autouse, suite-wide fixture, `_no_machine_state`. Every test gets it, so
a test that forgets cannot reach any of these:

- `ML_STACK_HOME` points the whole state root at an empty directory under the test's own
  `tmp_path`, which moves the runs store, the fit and profile records, the job files and
  every other home at once. The runs store is where an evening of measuring lives; a test
  that read it would pass or fail on what the laptop had been doing.
- `bench.progress.serving_lines` and `bench.progress.results_since` — what is serving on this machine
  right now, and what the last job kept — answer empty.
- Every variable that moves one corner back out of the state root
  (`MLSTACK_BENCH_HOME`, `MLSTACK_INGEST_HOME`, `MLSTACK_FIT_FILE`, `MLSTACK_PROFILES_FILE`
  and the rest) is *deleted*, along with `MLSTACK_BENCH_CEILING`, `MLSTACK_BENCH_TRACE`,
  `MLSTACK_LLAMA_BUILD`, `MLSTACK_SEARCH` and `MLSTACK_TRAIN_CEILING`, so a shell that
  exports one cannot change a result.

A test that means to exercise one of these overrides it with its own `monkeypatch.setenv`
or `monkeypatch.setattr`, which runs after the autouse fixture and is undone with it.

What still reads the real machine, on purpose:

- `test_gguf.py` compares the shipped `source_dirs()` against `Path.home() / ".unsloth"`, which
  is the value under test — it asserts what the default *is*, and never opens the path.
- `test_web.py`'s one live search is marked `live_net` and skipped unless `ML_STACK_LIVE_NET=1`.
- `test_fleet_install.py` asserts the `HF_HOME` an installer *writes* into a plist or unit
  file. It composes a path from a home directory; it does not read one.
- `test_serve_build.py` runs a real (tiny, hand-written) executable and real `strings` against
  fake dylibs, all inside `tmp_path`; the packaging tests build real wheels there. Neither
  compiles anything or reaches the network.

## Live services

No test reaches a paid or quota-limited API (Anthropic, OpenAI, any cloud model) or a public
endpoint on its own. A test that has to is marked `live_api` or `live_net`, and conftest skips it
unless `ML_STACK_LIVE_API=1` or `ML_STACK_LIVE_NET=1` is set. A key or login in the environment
switches nothing on, and `conftest.py` deletes the credential variables (`live.CREDENTIALS`) from
every test's environment. An autouse fixture refuses every other test a connection or a name
lookup beyond loopback, the LAN and link-local addresses; the test fails and names the call site.
`test_no_live_calls.py` also fails on an unmarked import of a paid SDK, a read of a credential
variable, or a spawn of the `claude` command anywhere under `tests/`. Local models served through
the broker are not remote services and are untouched by all of this.

## Attack runs against a served model

`ml_stack.testing.verdicts` is what a model-backed red-team, canary or guard-eval run uses to
avoid repeating itself. Attacks still go to the model one at a time through the broker.

- **`run(attacks, execute, Subject(model_hash, guard_version))`** serves a *passing* verdict from
  `$ML_STACK_CACHE/verdicts/attacks/` while its key is unchanged: the hash of the model file
  (`model_hash`, read once per file version), the attack id, the guard or prompt version, and the
  bytes of every module or file in `Attack.surfaces`. Editing a guard, a prompt, a surface or
  swapping the model changes the key, so the attack runs again and a regression shows. A failing
  verdict is never served; `use_cache=False` (the runner's `--no-cache`) runs everything and
  records nothing.
- **`sample(attacks, fraction=0.1, seed=..., changed=...)`** is the quick mode: a seeded tenth plus
  every attack in a class (`Attack.klass`) with an attack that touches a changed file. The full
  sweep is the nightly and release command.
- **`Limits(max_tokens, stop, thinking=False, cache_prompt=True).body(base)`** is the request a run
  sends: few tokens, thinking off, the prompt prefix kept warm so only the attack text is
  processed, and the canary as a stop string so generation ends at the objective.

`test_verdicts.py` runs the real Bash guard as the attack target: it breaks a copy of the guard
and checks the cached run reports the regression.

## The shared fakes, in `conftest.py`

Import them like the tests already do: `from conftest import write_gguf`.

| name | what it is |
| --- | --- |
| `server` (fixture) | a real `http.server` on a free port with a caller-supplied handler; closes the socket as well as stopping the thread |
| `json_reply` | the `(status, body)` pair such a handler answers with |
| `threaded_server` | a context manager for a handler *class* on a free port — the eight lines fourteen modules had each written |
| `write_gguf` | a real, minimal GGUF v3 header in `tmp_path`; refuses a metadata type it cannot write rather than stringifying it |
| `LLAMA_SERVER_HELP`, `fake_binary` | llama-server's `--help`, and an executable that answers it |
| `fake_process`, `fake_memory` | one row of `psutil.process_iter`, and what `virtual_memory()` answers |
| `a_row`, `scored_rows` | one measured bench question, and *n* of them with *h* hits over *s* seconds — so a run's F1 is `hits / questions` exactly |
| `fit_files` (fixture) | points both halves of the fit source of truth (`package_file` and `$MLSTACK_FIT_FILE`) at `tmp_path`, fills them, and optionally fixes `hub.room` |
| `on_a_fresh_loop` | awaits a coroutine on an event loop in a thread of its own, so `asyncio.run` cannot trip over a loop a neighbouring test left running |

## The rules

- A test builds its own fixtures in `tmp_path`. It never reads `data/`, `~/.ml-stack`,
  `~/.cache` or anything scraped.
- Every name is invented. Reproduce a real value's *shape* when that is what revealed a bug,
  never its content. `tests/known-fixtures.txt` lists the invented names already in use.
- A test that would pass against a broken implementation is a bug. Two shapes to watch for:
  a loop whose only assertion is inside it (add a non-emptiness guard first — an empty
  sequence passes anything), and a `try: ... except Exception: pass` around the call under
  test (assert the seam was reached instead).
- Name a test after the behaviour it pins, as a sentence.
