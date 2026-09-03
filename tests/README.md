# The test suite

About three thousand tests in a little over two minutes on every core. Nothing here serves a
model, touches a GPU, reaches the Hub, or reads anything under `~/.ml-stack`, `~/.cache` or a
real home directory — see *Nothing reads the machine* below for how that is enforced rather
than remembered. Every person, company, place and model file is invented.

## Running them

```sh
pytest                      # everything, on every core (`-n auto` is in pyproject.toml)
pytest -n 4                 # while a bench has the GPU: four workers, not sixteen
pytest -m "not slow"        # the fast subset -- no browser, no subprocess, no network wait
pytest -n 0                 # one process, in file order, when a failure needs a clean order
pytest -n 0 -p no:randomly  # the same, with any ordering plugin disabled
pytest --durations=0 --durations-min=1.8    # what is costing the wall clock
```

`-n 4` is the one to use while a measurement is running: a full `-n auto` run competes with
the bench for cores and both get slower, and a bench's wall clock is the thing being measured.

The default run has no `-m` filter, so `pytest` alone still runs every test including the slow
ones. `-m "not slow"` is a convenience for the inner loop, not the suite of record.

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

- `bench.HOME` (and `bench.extract.HOME`, which binds it at import) and `mcp.MCP_HOME` are
  pointed at an empty directory under the test's own `tmp_path`. The runs store is where an
  evening of measuring lives; a test that read it would pass or fail on what the laptop had
  been doing.
- `bench.run.serving_lines` and `bench.run.results_since` — what is serving on this machine
  right now, and what the last job kept — answer empty.
- `MLSTACK_BENCH_HOME`, `MLSTACK_INGEST_HOME`, `MLSTACK_FIT_FILE` and `MLSTACK_PROFILES_FILE`
  are set into `tmp_path`; `MLSTACK_BENCH_CEILING`, `MLSTACK_BENCH_TRACE`, `MLSTACK_LLAMA_BUILD`,
  `MLSTACK_SEARCH`, `MLSTACK_TRAIN_CEILING` and `MLSTACK_WEB_PROFILE` are *deleted*, so a shell
  that exports one cannot change a result.

A test that means to exercise one of these overrides it with its own `monkeypatch.setattr`,
which runs after the autouse fixture and is undone with it.

What still reads the real machine, on purpose:

- `test_gguf.py` compares the shipped `SOURCE_DIRS` against `Path.home() / ".unsloth"`, which
  is the value under test — it asserts what the default *is*, and never opens the path.
- `test_web.py`'s one live search is skipped unless `MLSTACK_NET` is set on purpose.
- `test_fleet_install.py` asserts the `HF_HOME` an installer *writes* into a plist or unit
  file. It composes a path from a home directory; it does not read one.
- `test_serve_build.py` runs a real (tiny, hand-written) executable and real `strings` against
  fake dylibs, all inside `tmp_path`; the packaging tests build real wheels there. Neither
  compiles anything or reaches the network.

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
