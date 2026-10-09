# Environment: what each job needs, and what happens when it is missing

`scripts/preflight [purpose ...]` says whether this machine can do a job and prints the one
command that closes the gap (exit 1 with the fix, exit 0 and one line when ready). `scripts/test`
runs it first (`dev`, and `gate` for `scripts/test gate`); the CI `gates` job runs
`scripts/preflight gate`. `scripts/cold-start-check` does the same from nothing: it clones the
committed tree, builds a fresh venv and runs the steps, so it finds what a machine that has never
seen the repo would hit. Run it before a release.

The Python packages a purpose needs are not a list kept in one place. They are read from the
imports at module level that the files the purpose runs reach (`scripts/envcheck.py`), so a
conftest that starts importing a package makes `dev` ask for it at once, and
`tests/test_envcheck.py` fails until the `test` extra in `pyproject.toml` and the CI `gates` job's
install line provide it.

| purpose | what it runs | needs | one-line fix |
|---|---|---|---|
| `runtime` | the library | the `dependencies` in pyproject, Python >= 3.12 | `pip install .` |
| `dev` | `scripts/test`, pytest | runtime + the `test` extra (pytest, xdist, testmon, numpy, keyring, ...) + git | `pip install '.[test]'` |
| `gate` | `scripts/budgets`, `redteam_coverage.py`, `reference`, `scripts/test gate` | the imports those reach + ruff and pyright at `scripts/gates/pinned.txt` + git | `pip install -r scripts/gates/pinned.txt` |
| `commit` | the pre-commit name check | the `privacy` extra and the spaCy model `en_core_web_sm` | `pip install '.[privacy]' && python -m spacy download en_core_web_sm` |
| `build` | `packaging/build.py`, the app | `build`, `hatchling`, cargo at the `rust-version` in `app/Cargo.toml` | `rustup update stable` |
| `base` (not in `--all`) | any branch work | the branch contains the development branch's tip as last fetched | `git fetch origin && git merge origin/<dev>` |

Not needed anywhere above: node, docker, git-lfs, a GPU. node is used by `pi` and the browser UI
tests and those skip with their own message.

## Failures found by the adversarial pass (2026-10-09)

| # | failure | cause | evidence | fix |
|---|---|---|---|---|
| 1 | `pip install '.[test]'` then pytest: `ModuleNotFoundError: numpy` at conftest | `tests/conftest.py` imports `ml_stack.testing`, which imports numpy; the `test` extra did not list it (CI hid it by installing numpy by hand) | fresh 3.12 venv with `.[test]` | `test` extra includes `arrays`; `test_the_test_extra_installs_everything_the_suite_imports_at_collection` |
| 2 | `scripts/test` in a bare install: raw traceback `No module named 'pytest'` | the runner imports pytest-dependent modules before checking anything | bare `pip install .` clone | `scripts/test` runs `envcheck` first and exits with the install line; `test_scripts_test_stops_with_the_fix_before_importing_what_is_missing` |
| 3 | CI `gates` job failed twice | conftest gained imports (psutil, keyring, cryptography) the job's install line did not name | CI history | `test_the_ci_gates_job_installs_everything_the_gate_tests_import` derives the imports and compares; the job also runs `scripts/preflight gate` |
| 4 | `scripts/budgets` passed with 5 metrics uncounted and printed "ruff 3.13.5" | a pyenv shim exits 127 and its message names the Python versions holding the tool; the version regex read `3.13.5` out of it, and uncounted metrics did not fail the run | clone outside `$HOME` (pyenv then has no `.python-version`) | `_pins.tool` accepts only a command that runs, `_pins.version` requires exit 0, `scripts/budgets` exits 1 with `pip install -r scripts/gates/pinned.txt`; `tests/test_environment_gates.py` |
| 5 | the Tauri crate could not `cargo check` on a clone | `tauri.conf.json` names a sidecar file that only `packaging/build.py` creates, and `tauri_build` panics without it | `cargo check --locked -p ml-stack-app` on a clone | `app/src-tauri/build.rs` writes an empty git-ignored placeholder for check and debug builds and panics with the build command for release |
| 6 | a daemon kept running old code after `runtime ensure` (the board daemon, for hours) | ensure selects what the next start runs and signals nothing | incident | `runtime status` lists processes whose command line is ours, that began before the selected runtime was verified and that do not run from it, and says to stop them through their own control; `--json` carries `older_processes`; `tests/test_runtime_stale.py` |
| 7 | packaging tests used a stale `dist/` wheel | freshness was `mtime(src/*.py)` against the wheel: a checkout of an older branch, a changed `pyproject.toml` or `packaging/` file was not seen; `test_runtime_wheel` took any wheel | reading the tests | `tests/wheel_support.fresh_wheel` stamps the wheel with a digest of its inputs and rebuilds on any change; `tests/test_wheel_support.py` |
| 8 | `git commit` in a bare install: refused by pre-commit | the name check needs presidio and a spaCy model | clone commit step | already failed fast with the fix; now also `preflight commit` and in `--all` |
| 9 | `scripts/redteam_coverage.py --check` and `scripts/test gate` fail on a clean checkout of the tip | a merged change added a spawn (`platform.py:private_dir`) without regenerating `docs/redteam/coverage.json`; partial surfaces rose 197 to 198 | every clone at the tip | not fixed by this branch: it needs a coverage decision (cover the spawn, or classify it n/a in `docs/redteam/coverage-map.toml`) and `--write` |
| 10 | branches cut from an old tip test code that has moved | nothing compared the branch with the development branch | n/a | `scripts/preflight base` |

Checked and found sound: the test-reuse key already includes the interpreter, the installed
versions of the packages a file imports, the plugin versions, the environment variables the file
names and its whole import closure, and bars any file that spawns a process; the pyright
incremental cache is keyed on the pyright version and the interpreter; the `gates` job's install
line covers conftest today; the red-team, command-reference and notices checks run in a bare
`pip install .`.

## Not covered

- No network: `scripts/cold-start-check --find-links DIR` installs from a wheelhouse; the steps
  after the install need no network except `cargo check` without a populated registry.
- Python 3.14 is not installed under pyenv here; `--python 3.14` skips with that message.
- Windows: the checks use `sys.executable -m pip` and `shutil.which`, but the cold-start script
  assumes POSIX paths (`venv/bin`).
