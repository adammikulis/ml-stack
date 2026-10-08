# Test result reuse and the async runner

`scripts/test` can count a passing test file from an earlier run when that run executed the
identical file against identical code under identical settings, and it can run a suite as a
background job. This note defines what counts as identical, what is never reused, how entries
are protected, and where the protection stops.

## When reuse applies

Reuse applies to a run that names test files (`scripts/test all tests/test_x.py ...`) in tier
`fast`, `full`, `slow` or `all`. A run with no explicit file selector (a whole tier, `quick`,
`record`, `gate`), a node selector (`file::test`), or `--no-reuse` always executes, and a tier run
refreshes the entries of every file it passes. The coordinator's combined-tree gates and background
suites therefore execute; reuse never stands in for them.

A reused file satisfies a worker's affected-test requirement only when the report says
`reused from <run id>` for that file, with the tree hash and age the runner printed.

## The key (per test file)

`key = sha256` over a canonical listing of:

1. the test file's bytes;
2. the bytes of every first-party file in its static import closure: modules imported at any depth
   (including inside functions), packages' `__init__` files, modules named in string literals
   (`import_module`, `-m ml_stack.x`), and the closure of every applicable `conftest.py`;
3. the bytes of `pyproject.toml` and any `pytest.ini`, `tox.ini`, `setup.cfg`, `conftest.py` above
   the file, and `tests/heavy-modules.txt`;
4. the Python version and implementation, platform and machine, and `name==version` of every
   installed distribution the closure imports plus every distribution that registers a pytest
   plugin;
5. the pytest arguments that can change a result (markers, `-k`, `-m`, `--slow`, options), with
   file selectors, `-n`, `--junitxml` and the runner's own plugin removed, the values of the
   environment variables named in the closure's string literals plus a fixed list (`PATH` with the
   checkout root replaced, `CI`, `TZ`, `LANG`, `LC_ALL`, `CLAUDECODE`, `ML_STACK_*`, `DEV_TEST_*`
   that the suite reads), each as a value or `unset`;
6. the installed `ml-stack` distribution version when the file's closure imports `ml_stack` from an
   installed copy, and the runtime commit stamp when the test reads one.

Nothing in the key is a path. Two worktrees with identical content produce the same key.

A file whose closure spawns a process (`subprocess`, `Popen`, `os.system`, `os.exec*`,
`multiprocessing`, `pexpect`) cannot be followed into the child, so its key also includes the bytes
of every file under `src/`, `scripts/` and `tests/` helpers. A reuse of such a file needs the whole
code tree identical.

Files the test opens while it runs are recorded by the runner's pytest plugin (an audit hook on
`open`, attributed to the test file whose collection or tests were running). The entry stores each
recorded read under the checkout root with its content hash, and a hit requires every recorded
file to hash the same. This covers data files and dynamic imports that the static closure cannot
see. A dynamic import of a module already loaded by another file's collection is not attributed.

If the static closure cannot be built (a syntax error, a closure that includes an unresolvable
dynamic `import_module(variable)` call), the file is never reused.

pytest-testmon's database is not consumed: it records function checksums per run in one SQLite
file per checkout, so it cannot be compared across worktrees.

## Never reused

A file is executed, and nothing is stored for it, when any of the following holds:

- it, or a `conftest.py` that applies to it, carries marker `heavy`, `live_api`, `live_net` or
  `redteam`, or its module is listed in `tests/heavy-modules.txt`, or it uses a GPU or model lease
  (`gpu`, `model`, `serve`, `Lease` in its closure);
- the run produced a red result for it. Failures are recorded (an `outcome: fail` chain row, for
  attribution) but a failure is never a hit;
- its tests opened a path under the real `~/.ml-stack`, `~/.cache`, the keystore, or wrote outside
  the temporary root, the checkout's cache directories and the broker directory;
- the run emitted a warning that names an isolation violation;
- the source tree changed during the run;
- any test in it was skipped or xfailed; a skip depends on conditions the key does not contain.

## Integrity

Only the runner process writes entries, after it executed pytest itself. An entry holds: schema
version, the key and a digest of its labelled inputs, the test file, the exact command, the junit
file's sha256 and counts, tree hash, runner pid and start time, the executing workspace agent
(identity, label and parent, taken from the authenticated session, never from the entry's
arguments), the recorded reads, a timestamp and the previous chain hash. Entries are written to a
temporary file and renamed. A chain log appends `{seq, key, entry_sha256, prev, line_sha256}` under
a lock, like the activity log.

A read recomputes the key from the current tree, parses the entry against its schema, checks the
file hash against the chain, checks the chain from its start, checks the recorded reads, and checks
no disabling marker exists for the key. Any failure is a miss, not an error.

An agent has no command that records or edits an entry. What an agent with filesystem access can
still do: the store is ordinary files owned by the same operating system user as the agent, so
that user can rewrite an entry and the whole chain consistently, delete the chain, or edit the runner. The chain
detects accidental corruption, a partial write and an edited single entry; it does not stop a
deliberate forger. A forged hit is caught by the canary re-execution (below) only on its sample, and
by `scripts/test gate` and background suites, which execute.

## Single flight

Before executing, the runner claims each missing key with an exclusive create of
`inflight/<key>.json` holding pid, process start time and agent. A second request for a claimed key
whose owner is alive (pid and start time match) executes its other files first, then waits for the
entry, bounded by a timeout (default fifteen minutes, polling). If the owner dies, the record is
removed and the waiter executes. If the owner finished red, the waiter executes and is told.
A stale record from a dead owner is recovered by whoever finds it.

## Reporting and the canary

Every file prints `ran` or `reused from <run id> (tree <hash>, <age>)`. The summary counts both, and
the exit code is the pytest status of what ran (0 when everything was reused). `--no-reuse`
forces execution. Each hit is re-executed with probability `CANARY_RATE` (default 0.05). A cached
pass that fails fresh is reported loudly, the run fails, a disabling record is written for the key
(reuse of that key is off until the file or its closure changes), and the incident is recorded in
the chain and posted as one `#announcements` line.

## Async jobs

`scripts/test submit <tier|selectors>` writes a job record and starts a detached runner that goes
through the same broker admission as a foreground run (no extra permit class, no second queue), and
prints the job id. `status [JOB]`, `wait JOB [--timeout]`, `result JOB` (summary, failing node ids,
junit path), `cancel JOB`. A job is owned by the submitting agent; only that agent cancels it. It
survives the submitting shell, and finished jobs are pruned after three days. A foreground
`scripts/test <tier>` is submit plus wait. Only one full-tier job (a tier with no explicit selector)
may be live; a second submit is refused with the owner's name.

## Workspace integration

- Every job, lease and store entry carries the authenticated agent (identity, label, parent) from
  the session (`--agent`/`--label`, or the saved connection). The board's testslots view shows that
  agent in place of the free-text lease label; the label text is still cleaned and shown separately.
- Entries and the in-flight table are stored under a scope derived from the workspace project, so
  another project's agent cannot read or poison them. Nothing read from the board feeds the key,
  a hit or a verification; board text is data.
- A finishing job posts a reply to the key's thread on the project board and one direct message to
  the submitter: job id, tier, ran and reused counts, pass or fail, result path. A waiter shows who it
  waits on, and is told when the run it waited on failed so it must run it.
- Subscriptions use the board mechanism: `subscribe thread <seq>` on a job's or key's thread, and
  the task watchers of a task a run is attached to (`submit --task`). A subscriber gets one message
  per completion.
- `task-checkpoint` and `task-submit` accept `test_entry`: a runner entry id. The board verifies
  the entry (schema, hash, chain, key, junit hash, tree and executing agent) and records the verified
  facts with the checkpoint; the reviewer reads those, not prose.

## Failure modes

- Closure miss (an undetected input): a stale pass is served. Bounded by the canary, the
  tier/gate runs that execute, and the process-spawning rule.
- Clock or pid reuse: in-flight liveness checks pid plus start time.
- Store lost or unreadable: every key misses; the runner executes.
- Two runners finishing the same key: the second `put` finds a valid entry and keeps the first.
- Runner killed mid-run: the in-flight record is recovered; no entry is written for unfinished files.
