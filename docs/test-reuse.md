# Test result reuse and the async runner

`scripts/test` counts a passing test file from an earlier run when that run executed the identical
file against identical code under identical settings, and runs suites as background jobs. This note
defines what counts as identical, what is never reused, how entries are protected, and where the
protection stops.

## When reuse applies

Reuse applies to a run in tier `fast`, `full`, `slow` or `all` that names test files
(`scripts/test all tests/test_x.py ...`). It does not apply to a node selector (`file::test`), a
directory, `quick`, `record`, `gate`, or `--no-reuse`. A tier run with no file selector and the
`gate` pytest step always execute and write an entry for every file that passes, so the
coordinator's combined-tree gates and background suites refresh the store and are never satisfied
by it. A named-file run reuses an entry only when its pytest flags equal the flags of the run that
wrote it (`all` writes `--slow`, `fast` writes its marker expression).

A reused file satisfies a worker's affected-test requirement only when the report says
`reused from <run id>` for that file, with the tree hash and age the runner printed.

## The key

Each file has a lookup key and a manifest. The lookup key is a sha256 over:

1. the bytes of the test file, every applicable `conftest.py`, `pyproject.toml` (and `pytest.ini`,
   `tox.ini`, `setup.cfg`) and `tests/heavy-modules.txt`;
2. the Python version and implementation, platform and machine;
3. `name==version` of every installed distribution that registers a pytest plugin;
4. the pytest arguments that can change a result (markers, `-k`, `-m`, `--slow`, options), with file
   selectors, `-n`, `-q`, `--junitxml` and runner plugins removed;
5. the values of the environment variables named in the test file or its conftests, plus a fixed list
   (`PATH` without temporary entries and with the checkout root masked, `CI`, `TZ`, `LANG`, `LC_ALL`,
   `CLAUDECODE`, `ML_STACK_NOTIFY`, `ML_STACK_LIVE_API`, `ML_STACK_LIVE_NET`, `DEV_TEST_SLOTS`);
6. the installed `ml-stack` distribution and how it was installed.

The manifest is what the run depended on: the first-party files in the test file's static import
closure (imports at any depth, including inside functions and packages' `__init__`), every file the
test's collection or tests opened under the checkout, every module they imported, the names in each
directory they listed, the `name==version` of the third-party packages those files import, the
values of the environment variables those files name, and a digest of every file under `src/`,
`scripts/` and the non-test `tests/` helpers when the test file or a helper spawns a process
(`subprocess`, `Popen`, `os.system`, `os.exec*`, `multiprocessing`, `pexpect`), because a child's
reads are not seen. The pytest plugin in `scripts/testreuse_plugin.py` records the opened files and
listings with an audit hook and attributes them to the test file whose collection or test was
running; modules imported at collection are attributed from `sys.modules`.

A hit needs a stored entry for the lookup key whose manifest still matches the checkout byte for
byte. Nothing in either is a path, so two worktrees with identical content hit the same entry.

Known gaps: a module another test file had already imported and that is reached only through a
run-time `importlib` call is not attributed; a file read by a child process of a test that spawns
nothing the regex recognises is not seen; a network or clock dependency has no file to record.
The canary below is the check on these. pytest-testmon's database is not read: it is one SQLite file per checkout keyed by function
checksums, so it cannot be compared across worktrees.

## Never reused

A file is executed and no passing entry is written for it when any of the following holds:

- the test file carries marker `heavy`, `live_api`, `live_net`, `redteam`, `gpu` or `model`, or is
  listed in `tests/heavy-modules.txt`, or names a model or GPU lease (`Lease`, `llama-server`,
  `ml-stack-serve`), or the marker appears on any of its items at run time;
- the run was red. A failure is written to the chain as a `fail` entry for attribution; it is never
  a hit, because a hit needs a `pass` entry;
- its tests wrote under the checkout other than to cache directories, wrote outside the temporary
  root, or opened anything under the real `~/.ml-stack`, `~/.ssh`, `~/.gnupg`, `~/.config`,
  `~/.cache/huggingface` or `Library/Keychains` (the suite's own isolation guard in
  `tests/conftest.py` reads real state to check it is untouched; those reads are ignored);
- the run emitted a warning that mentions an isolation violation;
- any test in it was skipped or xfailed, because a skip depends on conditions the key lacks;
- the source tree changed during the run.

## Integrity

Only the runner process writes entries, after it executed pytest itself; no command records an
entry. An entry holds the schema, the lookup key, the manifest and its digest, the exact command, the
junit file's sha256 and counts, the tree hash, the runner's pid and start time, the executing
workspace agent (id, label, parent, taken from the authenticated session and not from arguments), a
timestamp and the previous chain hash. `entries/<id>.json` is written through a rename; the id is
the first twenty hex digits of the entry's own sha256. `chain.jsonl` appends
`{seq, kind, lookup, id, entry_sha256, prev, row_sha256}` under a file lock, like the activity log;
`latest/<lookup>` names the newest pass.

A read parses the entry against its schema, recomputes its hash, finds its chain row and checks the
chain from the first row, then recomputes the manifest from the current tree. Any failure is a miss.

What an agent with filesystem access can still do: the store is ordinary files owned by the same
operating system user as the agent. That user can rewrite an entry and the whole chain
consistently, delete the chain, edit the runner or the plugin, or set `DEV_TEST_REUSE_DIR` to a store
it made. The chain detects corruption, a partial write and an edited entry; it does not stop a
deliberate forger. A forged hit is caught only by the canary on its sample and by the tier and gate
runs, which execute. The store sits under a per-project directory (a hash of the project key), and
board verification reads only the asking agent's project directory.

## Single flight

Before executing, the runner claims each missing key with an exclusive create of
`inflight/<key>.json` holding pid, process start time and agent. A second request for a key whose
owner is alive (pid and start time match, zombies excluded) runs its other files first, then waits
for the claim to clear, polling, up to fifteen minutes, and uses the entry if one passes. If the
owner died or finished red, the waiter executes the file and the board thread says so. A claim from
a dead owner is removed by whoever finds it.

## Reporting and the canary

Each file prints `ran` or `reused from <run id> (tree <hash>, <age>, by <agent>)` with its key, and
the summary counts both. The exit code is pytest's status for what ran; a run of reused files only
exits 0. `--no-reuse` forces execution. Each hit is re-executed with probability `CANARY_RATE`
(0.05; `DEV_TEST_REUSE_CANARY` overrides it). A cached pass that fails fresh prints
`CANARY MISMATCH`, fails the run, writes an `incident` chain row and a disabling record for the key
(reuse stays off while the manifest equals the one the cached pass recorded), and posts one
`#announcements` line.

## Async jobs

`scripts/test submit <tier> [paths]` records a job under the project's store, starts a detached
runner that goes through the same broker admission as a foreground run (no extra permit class and
no second queue), and prints the job id. `status [JOB]`, `wait JOB [--timeout S]`, `result JOB`
(status, summary, failing node ids, kept junit paths) and `cancel JOB`. A job is owned by its
submitting agent; only that agent cancels it. A job survives the shell, keeps its output in `log`,
and is pruned three days after it finishes. `scripts/test <tier>` is submit plus follow; Ctrl-C
cancels. One whole-tier job (a tier with no test path) may be live at a time; a second submit is
refused and names the owner.

## Workspace integration

- A job, lease and entry carry the authenticated workspace agent (id, label, parent). Leases carry
  it as `agent`; the board's testslots view shows `agent/label (lease label)` and `registered`,
  which says whether the workspace has that identity.
- Entries, in-flight claims and jobs sit under a per-project directory. `task-checkpoint`/`task-submit`
  verification and board threads are project-scoped by the board's own membership.
- Notices go out under the agent's `<agent>/test-runner` delegate, because a message never reaches
  its sender's inbox. A run that takes a key opens a thread on the project board and replies when it
  ends; a job opens a thread when it starts and replies when it ends. A finishing job also sends one
  direct message to the submitter and to the watchers of the task it was attached to (`--task`),
  except those already subscribed to the thread. Waiters follow the owner's thread and are told when
  the run failed.
- Following uses the board's subscriptions: `scripts/test subscribe JOB`, `subscribe --key PREFIX`
  (a reuse key prefix from the report) subscribe to the thread; `subscribe --task ID` is the task
  board's `task-subscribe`.
- `task-checkpoint` takes `test_entry` and `task-submit` takes `test_entries`: runner entry ids. The
  board verifies each in the asker's project store and records its file, key, tree, junit hash,
  outcome, counts and executing agent with the checkpoint or proposal.
- Board text is data. The thread number in a claim record is used only to subscribe, which the board
  checks against membership; nothing read from the board enters a key, a hit or a verification.
  Notice failures are printed and never fail a run.

## Failure modes

- An unrecorded input (see the known gaps) serves a stale pass until the canary, a tier run or a
  gate run catches it.
- Store missing or unreadable: every key misses and the runner executes.
- Two runners finish one key: each writes an entry; `latest` names the later pass.
- Runner killed mid-run: its claim is removed by the next finder; no entry is written.
- Job runner killed: `status` reports the job failed.
- Chain compaction and entry pruning are not implemented; the store grows by one entry per passing
  file run.
