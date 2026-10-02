# Integration report: `integration/0.3.0`

One branch, cut from `0.2dev` (5c1353a), carrying the finished agent branches. 203 commits ahead of
`0.2dev`; 451 files changed. Run on one Apple-silicon Mac (Python 3.13.5, 16 cores, load 20 to 190
from other agents' runs), 2026-10-02.

## What merged, in order

| Branch | Head | Merge | Conflicts and how they were resolved |
|---|---|---|---|
| `agent/taint` (carries native-guard, guardrails, decide, port-pcbe, serve-admission, redteam) | f0fdb68 | 8cb12fa | none |
| `agent/hardening` (carries audit) | c11835b | bf52522 | `http.py`: both import sets kept, so the request-queue hooks (`gate`) and the address policy (`httpguard`, `macauth`) both apply. `serve/manager.py`: `ServerManager` keeps the broker routing and takes `stop_on_exit`, the exit guard and the orphan sweep before the per-port lock. `fleet/api.py`, `fleet/daemon.py`: `/decide` reads the body the signed-request guard has already read, and the daemon is built with the decider and the TLS listener. `cli/reference.py`, `tests/test_layers.py`, `budgets.json`: union. |
| `agent/model-discovery` | 67a2363 | bb4ee2e | `serve/manager.py`: `reusing_installed` runs in `lease` and in the broker-side start. `http.py`: `head_once` beside the queue hooks. `budgets.json`. |
| `agent/sentinel` | 30a973b | 3d66175 | `tests/test_layers.py`, `budgets.json`: union. |
| `agent/sentinel-integ` (tests and notes only) | 047e52f | 310dc24 | `tests/test_layers.py`, `budgets.json`. Its tests were written against the pre-native-guard `Guard` class and needed the new adapters (below). |
| `agent/web-research` | da887fc | 31a7d94 | `http.py`: `open_stream(guard=...)` and the queue hooks; `tests/test_layers.py`, `tests/test_wiring.py`. |
| `agent/ui-primitives` | 2dadf36 | 9dc1e36 | `budgets.json` only. |
| `agent/py312` (the Python 3.11 floor) | 17bf2ad | 4942588 | `docs/packages.md`. |
| `agent/release-ready` | 6768564 | 00c790e | three workflow files: comment-only conflicts, the version-annotated side kept. |
| `fix/test-speed` | b8fb60e | 1a3051d | `tests/conftest.py` (heavy and live-API markers kept with the redteam deselection), `pyproject.toml` (markers), `budgets.json`. |

`agent/redteam-integ`, `agent/audit`, `agent/decide`, `agent/guardrails`, `agent/port-pcbe`,
`agent/redteam` and `fix/serve-admission-control` have no commit that is not already in the merged
branches.

### Not merged

- **`agent/lan-onboarding`** (9a54ca5, 5978f85, 859bbe9) and **`agent/internet-pipeline`**
  (88 commits ahead of `0.2dev`). The task said to skip branches still in progress; a later
  message from the lead said both were finished and to merge them. The permission classifier
  refused each merge (`git merge --no-ff agent/lan-onboarding`: "Modify Shared Resources";
  `agent/internet-pipeline`: "Interfere With Workloads"). Nothing in the repository was changed
  for either. Work that depends on them is not done: the `http-servers` fold of
  `fleet/onboard/web.py`, device-only pairing as the default, the SPAKE2 library check against
  the branch's `spake.py`, the net-pipeline no-bypass scan and its network seal across the merged
  suite, the ClamAV skip, and the internet-pipeline HANDOFF leftovers. The owner decides whether
  to run the two merges (`git merge --no-ff agent/lan-onboarding`, then
  `agent/internet-pipeline`) in this worktree.
- SPAKE2, from what is knowable without the branch: `spake2` 0.9 on PyPI is a pure-Python wheel
  (`py3-none-any`) with the SPAKE2 protocol over Ed25519, used by magic-wormhole; the
  `cryptography` package has no SPAKE2 or generic group arithmetic to build it from. Whether it
  is a drop-in for `fleet/onboard/spake.py` was not checked and stays an open risk.

## Integration bugs found and fixed

Each was found by a test run on the merged tree, not by reading diffs.

1. **Sentinel read a guard that no longer exists.** `sentinel-integ`'s tests call
   `ml_stack.guard.Guard.default()`; native-guard replaced it with `guard.start(...)` and
   `interventions.Run`. `sentinel.adapters` gained `screening(run)` (the `verdict` for
   `Sentinel.screen`) and `GuardLogHandler` reads a `Run`'s warnings. The documents that quote
   attacks (`guardrails.md`, `redteam.md`, `sentinel.md`, `security.md`) are left out of the
   "ordinary pages are not held" corpus, and its bound is a share of the paragraphs.
2. **`/decide` read the request body twice** once hardening's signed-request guard had read it.
   It now takes the body the guard read. The decider tests build their tokens with
   `load_or_create_token` because a plain bearer string is no longer a token.
3. **The daemon ignored its configured decide backend** and used `auto` for any request that
   named none; under load `auto` fell through to loading the pointer model's weights. It uses
   `Config.backend` first.
4. **`taint.labels.Labelled` used a Python 3.12 type parameter list**, which the 3.11 floor from
   `agent/py312` does not parse. `scripts/notices.py` had a backslash inside an f-string.
5. **Admission calls `sysctl`** (`hub.machine_room`) before a start, which broke three tests that
   replace `subprocess.Popen` to prove nothing starts. They set the machine's room. This failed on
   `agent/taint` alone.
6. **The red-team fleet tests used a plain bearer token and a module-scoped daemon**, so the
   ninety wrong-token attacks locked the address out before the test that sends the right token.
7. **The red-team `PolicyGuard` read a `Call` as a dict**, so every call raised, the hook denied
   it, and the "policy stops it" arms passed for the wrong reason. It reads the fields. Only the
   `--redteam` tier runs these tests, so the default suite never showed it.
8. **`scripts/test-on-linux` did not install the `mcp` extra CI installs**; the Windows lock test
   imported `ml_stack` after faking the platform (hardening's `__init__` imports
   `importlib.metadata`); a docs row named a flag its command lacks; the CI `licenses` job had no
   pip cache; three workflows installed the `serve` extra that no longer exists.
9. **The embedding import probe** required `serve` and `client` to import with the standard
   library alone; `psutil` is now a core dependency, so it allows `psutil` and `packaging`.
10. **Three copies of the 0.80 / 0.95 memory thresholds** (`ui/assets/verdict.json`,
    `serve/admission.py`, `serve/estimate.py`). Admission and the estimator now read
    `ml_stack.ui.verdict`'s, and a test compares all three.
11. **The real-llama test picked the first shard of a sharded 90 GiB model as "the smallest GGUF"**
    and admission refused it. It skips shards.
12. **Four more GGUF readers that trusted the file** (the discovery path among them): see below.
13. **Test isolation.** The suite's "real state root was written" guard stayed green in the three
    full runs. A probe on `home.home()` over a full run saw no test resolve the real root after the
    session fixture moved it. `~/.ml-stack/gate/gpu` was already there (14:16 and 14:34, before
    this work) and its mtime moved at 17:18 during `scripts/guard-eval --native`, which leases a
    real model through the real broker: real use, not a test. The guard looks at files, not
    directories, so a directory-only change under the real root would not fail it.

### One bounded reader for model-file headers

The model-discovery path read GGUF headers with its own unbounded walker (`hub/header.py`), as did
`hub/cards.py` and `serve/tensors.py`; hardening had bounded only `serve/preflight.py` and the
safetensors size in `serve/mlx_tree.py`. A 2^60-byte string raised `MemoryError` and a 2^50-item
or nested array looped for as long as it liked. `ml_stack.hub.modelfile` is now the only reader:
`scan_gguf` and `safetensors_header` check every count and length against the file size, and hold a
header to a pair cap, an item budget, a nesting depth, a dimension cap and a deadline. The
`gguf` package's reader is only handed a file `scan_gguf` has accepted. `tests/test_hub_modelfile.py`
(32 tests) feeds every caller a huge string length, huge array counts, nested arrays, truncated
files and lying pair, tensor and dimension counts, for GGUF and safetensors. Mutation check: of ten
single-line breaks of the reader, seven fail a test; three survive because a second check in the
same reader refuses the same input (string length and array room are also caught by `take`/
`skip`; the dimension cap by `take`; an unknown array kind by the value reader).

## Tests

All commands from the worktree, through the machine's shared test-slot queue.

| Run | Command | Result |
|---|---|---|
| Default tier, final tree | `python scripts/test full` | `6948 passed, 5 skipped, 7 warnings in 325.76s (0:05:25)` |
| Default tier, before the fixes | `pytest tests -n 2 -q` | `5 failed, 6867 passed, 6 skipped, 5 warnings in 687.48s (0:11:27)` |
| Default tier, after test-speed merge | `python scripts/test full` | `3 failed, 6945 passed, 5 skipped, 9 warnings in 430.37s` (two table tests and the decide backend; fixed) |
| Slow tier, model and server modules, serial | `pytest --slow -n 0 tests/test_serve_real_llama.py tests/test_sentinel_real_model.py tests/test_serve_three_callers.py tests/test_serve_broker.py tests/test_serve_orphans.py tests/test_serve_exit_guard.py tests/test_serve.py tests/test_guard_native.py tests/test_spec_serve.py tests/test_fleet_serving.py tests/test_sentinel_crash.py tests/test_isolation_guard.py` | `1 failed, 210 passed, 1 skipped in 303.55s`; the failure was bug 11. `ML_STACK_TEST_GGUF=<gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf> pytest --slow -n 0 tests/test_serve_real_llama.py`: `1 passed in 23.65s` (a real llama-server leased through the broker, queued, stopped; `servers.json` empty afterwards) |
| Slow tier, the rest | `pytest --slow -m slow -n 2` over the other modules | `9 failed, 219 passed, 5 skipped, 1 warning in 1271.63s (0:21:11)`. Run alone on `agent/hardening`, 8 of the same 9 tests fail; on this tree 7 of 9 failed on a rerun (one TLS handshake timeout and one browser click timeout are load). Causes: no built wheels or `ladybug` for the offline-install tests, a managed environment the bench job cannot install into, a browser wait. They fail without the integration. |
| Red-team suite with PyRIT | throwaway venv, `pip install ".[redteam]"`; `pytest --redteam -n 0 tests/test_redteam_*.py` (11 modules) | `85 passed, 5 warnings in 57.85s` (1 failed before bug 7) |
| Sentinel integration | `pytest tests/test_sentinel_integration.py` | 6 passed (in the full tier) |
| Canary | `python -m ml_stack.testing.canary` | 17 of 18 attacks succeed with the rails off, 0 of 18 by default; benign tasks completed 2/2 |
| Guard eval on a leased small model | `scripts/guard-eval --native` (Qwen3-4B-Instruct-2507 Q4_K_M, leased through the broker, stopped afterwards) | native judge: eval 16/16 caught, 0/19 benign flagged; fresh 15/17, 0/31; redteam 24/24; adaptive 7/8; hard 0/16 flagged. Built-in markers alone: redteam 16/24, adaptive 5/8 |
| Budgets | `scripts/budgets` | every delta 0. Against `0.2dev`: broad-excepts 172 to 167, local-imports 453 to 451, long-docstrings 83 to 81, ruff-blind-except 19 to 17, ruff-other 463 to 386, ruff-security 85 to 64. Nothing rose. `tests-collected` stays 5991: `scripts/budgets --update` refuses to record it here because `nemoguardrails` is absent |
| Lint and types | the `ruff-*` and `pyright-errors` rows of `scripts/budgets` (ruff 0.13.3, pyright 1.1.411; `typeCheckingMode` is `off` with a few error rules) | pyright 1 (unchanged), ruff rows above |
| Dependency audit | `pip install --dry-run --report` of `.[store,hub,web,plot,graph,scrape,vision,pdf,gguf,train,test,mcp,claude,privacy,decide,guardrails,guard-model,fleet-tls,arrays]` in a throwaway venv, then `pip-audit -r pins --no-deps` | 136 pinned packages, "No known vulnerabilities found". `scripts/notices.py --check` reports `socksio 1.0.0: UNKNOWN` licence in this environment. |
| Import smoke | bare venvs with `packaging` and `psutil`; every `ml_stack` module imported | Python 3.12.8 and 3.13.5: 470 modules import; the rest need numpy, mlx, manimpango, metal-smi or pytest and say so |
| Python 3.11 | `ruff check --target-version py311 --select E9`; `vermin --target=3.11-` over `src`, `scripts` and `tests` | clean; minimum Python 3.11. **No 3.11 interpreter ran**: `pyenv install 3.11.14` fails here (the freshly built interpreter segfaults running `setup.py` on macOS 26.6, twice) |
| `scripts/test-on-linux` | not run | the Docker CLI is installed but its daemon is not running (`failed to connect to the docker API at unix://~/.docker/run/docker.sock`); Docker Desktop was not started |

### Mutation spot-check of the seams between branches

A line was broken, the named tests run, the file restored.

| Seam | Break | Result |
|---|---|---|
| Broker routing | `ServerManager.lease` calls the private start | caught: `test_serve_no_bypass.py::test_every_public_entry_of_the_manager_reaches_the_broker` |
| Address policy on the model download path | the redirect target is not checked in `hub/transfer.py` | **survived** (the pull tests patched `http.check` away); added two tests; then caught by `test_hub_pull.py::test_a_redirect_to_a_host_the_address_policy_refuses_is_never_followed` |
| Taint rail on the `Agent` default | taint dropped from the default rails | **survived**; added two tests; then caught by `test_agent_interventions.py::test_an_agent_given_no_interventions_carries_the_taint_rail` |
| Sentinel rail hook | `screening` never reports a denial | caught: `test_sentinel_integration.py::test_real_guard_verdicts_become_security_events_and_enforce_holds_the_content` |
| Sentinel peer hook | a quarantined peer is not refused | caught: `test_sentinel_integration.py::test_a_replayed_request_raises_events_and_the_peer_is_blocked` |
| Guard fails closed | a check that cannot run proceeds | caught: `test_decide_guard.py::test_a_decider_that_fails_asks_a_person_by_default_and_denies_when_told_to` |
| Signed requests | the daemon skips `_guard` | caught: `test_fleet_daemon.py::TestSpeechOverTheNetwork::test_the_route_needs_the_credential` |
| Request queue | `http` skips its turn | caught: `test_gate.py::test_requests_from_separate_processes_to_two_servers_never_overlap` |
| No-bypass static scan | a module that starts `llama-server` outside the manager | caught: `test_serve_no_bypass.py` names the module |
| Bounded header reader | ten single-line breaks | seven caught, three redundant (above) |

## Not verified

- Python 3.11 and 3.14 were not run; `scripts/test-on-linux`, Windows and Linux were not run.
- The slow tier ran once, before `fix/test-speed` merged, and not again; the default tier ran on
  the final tree.
- No fleet of more than one machine; TLS between two real machines; a daemon left running for
  60 days (certificate renewal is at start only).
- `gate/gpu` isolation is shown by a probe and a guard that watch files, not by a directory
  watcher.
- The `quick` and `fast` tiers of `scripts/test` were not timed.
- Both unmerged branches (above).

## Open items carried from the branches' `HANDOFF.md`

See `HANDOFF.md` for the full entries. In short: taint tracking measured on one model only and
its events have no subscriber; the machine broker does not deliver `on_event` or `say` to its
caller and has one `gpu` pool; `adopt_unmanaged = ask` has no prompt; the redteam baseline is to
be re-run after these merges, the guard benchmark is unwritten and multi-turn attacks are not
wired; `hub.find`, `hub.files`, `hub.fetch`, `hub.card` and pulls still use `huggingface_hub`
or read `HF_TOKEN` by hand; memory estimates are measured only with flash attention on Apple
silicon; the red-team runs do not use the verdict cache yet; tests that assert a wall-clock bound
fail on a loaded machine; trained deciders are not in model discovery and the pointer backend
takes the GPU without a lease; a daemon's certificate is renewed only at start, replies are not
signed, a cluster joined before protocol 2 cannot onboard by passphrase; `web.py`, `scrape/` and
`ingest/run.py` still fetch through `http.check`; 37 `except ...: pass` sites; Windows job
objects.

## Decisions for the owner

1. Run the two remaining merges (`agent/lan-onboarding`, `agent/internet-pipeline`), then the
   work listed under "Not merged". Device-only pairing as the default, `cryptography` staying
   optional (`fleet-tls`) with a clear message when it is missing, and a vetted SPAKE2 library
   are the lead's instructions for the first.
2. Whether `docs/INTEGRATION-REPORT.md` stays after the release.
3. Whether a fresh `tests-collected` is recorded from a machine that has every extra installed.
4. Whether the red-team tier (`--redteam`) joins the default tier or a scheduled job; it is what
   found bug 7, and it runs nowhere by default.
5. Whether the slow-tier failures that also fail on `agent/hardening` alone are fixed before the
   release or listed (they need built wheels, `ladybug` and a managed environment).
