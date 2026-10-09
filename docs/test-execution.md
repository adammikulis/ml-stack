# Shared test execution

Run tests with `python scripts/test fast`, `full`, `slow`, `all`, or `quick`.
Explicit test paths are checked before broker admission using pytest’s argument parser.
A missing file or directory, or a node selector that collects no tests, returns
exit status 4, including in `quick`; an empty affected-test selection remains valid.
`-n N` limits the worker pool; scoped runs default to an automatic pool of what the broker grants (`-n 0`); `-n 1` runs sequentially. `-n 0` requests
an automatic pool up to broker capacity. Explicit worker ceilings remain effective.
Every checkout shares the same admission broker.

Agents run affected tests for their own changes. The main agent coordinates shared structural
and security gates once per consolidated integration batch before publication, handles full
end-to-end checks, and schedules background full suites. Workers do not repeat those checks.
The main agent owns `quick`, whose cold-map recording and fallback can start full runs; workers
use explicit affected selectors.

`scripts/test <tier>` runs as a background job that the command follows: `submit` prints the job id
and returns, `status`, `wait`, `result` and `cancel` act on it, and Ctrl-C on a followed run
cancels its job. Jobs use the admission below and are owned by the submitting workspace agent,
which `scripts/testslots.py status` shows beside each lease. A run that names test files reuses
earlier passes of unchanged files and says so per file ([test-reuse.md](test-reuse.md)).

CPU capacity defaults to the logical CPU count minus one, with a minimum of
one. `DEV_TEST_RESERVED_CORES` changes that reservation. This reserves capacity;
it does not pin execution to a particular efficiency core. `DEV_TEST_BUDGET`
sets an explicit owner override. Normal load uses the full configured capacity;
severe load above `DEV_TEST_LOAD_HIGH` times the CPU count reduces the default
capacity. Existing active permits finish before new work enters.

The supervisor admits pytest startup, then each worker's configuration and collection, and each
test's complete setup, call and teardown. Idle workers hold no CPU permits. A
long final test therefore leaves the remaining capacity available to other
suites. A worker pool retains enough workers to use freed capacity after another
suite finishes. Multi-threaded model and benchmark test files also acquire a heavy
lane before their CPU permit. The `slow` marker alone does not require a lane.
Thread-library defaults are one thread per test process; explicitly configured
thread settings remain effective.

Lease records are published as complete JSON on an already locked inode under
the scheduler mutex. Published updates retain that inode and run under the same
mutex. Readers skip malformed or incomplete records and retry on the next
admission pass. A queued lease restores a missing or replaced record while
retaining its original FIFO position and a live file lock.

Normal command waits report a changed queue state at most once per second,
or repeat unchanged status after twenty seconds. Short waits stay quiet.
Per-test RPC waits do not print into pytest's progress output; use the shared
status command to inspect their queue.

`python scripts/testslots.py status` reports active permits and waiting work.
A queued minimum cannot exceed the configured capacity. Small requests can fill
a temporarily unusable gap twice before an older larger request reserves the
gap. Wait limits raise an error; they never disable admission. A command started
inside a live test permit fails promptly instead of queuing behind its parent.
Run an independent suite after the parent releases its permit.

`python scripts/testslots.py pytest --want auto --min 1 --label NAME -- COMMAND`
supervises a pytest command that loads `-p testslots_pytest` and uses
`DEV_TEST_WORKERS` as its worker ceiling. `{workers}` in a command argument is
replaced by that ceiling. `DEV_TEST_SLOTS_DIR` identifies the shared host broker.
The supervisor supplies `DEV_TEST_PYTEST_ENDPOINT`, `DEV_TEST_PYTEST_TOKEN`, and
`DEV_TEST_REMOTE_BROKER`; propagate those into a supervised container. The token
is private to that run and its admission endpoint closes when the run ends.

`--container` exposes the authenticated host endpoint to Docker through
`host.docker.internal`. Docker Desktop file locks do not coordinate with macOS
host locks, so container workers request host permits through this endpoint.
Native supervisors bind to loopback. The Linux runner installs dependencies
under one setup permit, releases it, then starts supervised pytest. Serial
execution through `scripts/test` uses `-n 1`. A directly supervised pytest command can use `-n 0`
to run without xdist workers.

On macOS, `scripts/test --confine` (or `DEV_TEST_CONFINE=1`) runs pytest under the Seatbelt
confinement kernel; without it the run follows the ordinary path and the isolation guards in
`tests/conftest.py`. The kernel constructs private startup home, state, cache and temporary
roots before collection and launches pytest through the maintained Seatbelt backend, which
denies every write outside the private scratch root. Source reads name tracked files
individually; dependency prefixes are read-only. The child receives only its authenticated
private Unix admission and descriptor-transfer endpoints. Fixtures that need more require
explicit pre-launch endpoint reservations, and the kernel accepts only reviewed gate files and
the nodes listed in `scripts/test_kernel_selectors.py`; other selectors fail before collection
rather than receiving loopback access. The human-floor terminal fixture uses a supervisor-owned
PTY bank through bounded authenticated operations on that endpoint. The kernel does not walk the
real state root: a canary file outside the scratch root and the profile's write denial carry
the isolation proof.

The supervisor pins stdout and stderr descriptors before the confined launch. Regular output
files and JUnit destinations must be owned, have one hard link, and stay outside protected
state by physical path and directory identity. Child output forwarding and JUnit promotion each
permit at most 16 MiB. Sockets, named FIFOs and unidentified device destinations are refused;
output writes have a five-second deadline per forwarded block. A failed preparation removes the
control and holder directories it created.

`scripts/test all --confine --artifact-output` creates fresh private console, JUnit and
evidence files. Confined output and supervisor diagnostics stay in that namespace; the caller
receives its location before collection. Private shadow Git metadata disables executable
configuration and filters. Tree hashing there uses raw source bytes, tracked deletions,
executable modes, symlinks and recorded gitlinks; repository `.git/info/exclude` and global
ignore configuration are not imported, and unsupported split or sparse indexes fail closed.

# The real state root is write-denied

Without `--confine`, `scripts/test` (and `scripts/testslots run`) still keep every test process from
writing under the real `~/.ml-stack` (`scripts/testwritedeny.py`), so a test that reaches for it gets
`EPERM` at the line that did it instead of failing the run afterwards ("real state root changed
during the run", which also names other agents' live writers the suite did not cause). On macOS the
pytest tree runs under `sandbox-exec` with an allow-default profile that denies `file-write*` under the
account's state root and the one the launching environment names; on Linux under `bwrap` with the root
bound read-only, but only when a probe on a scratch directory shows both that it denies and that a
nested `bwrap` still starts. Windows and a host without either tool run unwrapped, and the
after-the-fact check alone applies. `DEV_TEST_WRITE_DENY=0` turns it off for one run.

A process that is already sandboxed cannot apply another `sandbox-exec` profile (measured: the call
fails with `sandbox_apply: Operation not permitted` unless the two profiles are identical). The test
modules that start `sandbox-exec` themselves are listed in `tests/seatbelt-modules.txt`, marked
`seatbelt` by `tests/conftest.py`, and a run that selects them is split in two: everything else under
the denial, then those modules beside it without, with the JUnit files merged. A module that fails with
that message under the denial belongs on the list (`tests/test_test_write_deny.py` fails when one that
names the Seatbelt backend is missing). A full `fast` run measured this way (macOS, 2026-10-09) showed
no other test writing the real root.

A test that tries the forbidden write leaves a "touched" note in the reuse store, so
`tests/test_real_state_write_denied.py` is never stored as reusable; it costs two seconds a run.

# Holder channel resources on macOS

Reviewed holder protocol nodes use three offline-prepared immutable fixture prefixes. The
supervisor verifies the reviewed manifest digest, complete file/link inventory and source
stamp, then copies the manifest into its private control directory before collection. One
fresh, private, short invocation namespace supplies TMPDIR and permits Unix bind, inbound and
outbound operations only inside that namespace. TCP stays denied.

## Scheduling: measured duration, shortest work first

Every run writes a duration history. The `testdurations` pytest plugin records the CPU and wall
seconds of each passing test, merges them under a file lock into `test-history/durations.json` in the
repository's common Git directory (shared by every worktree; `scripts/test` passes the path in
`DEV_TEST_HISTORY`), keeps an exponentially weighted value per test with a sample count (a sample is
clipped to three times the current value once three exist, so one loaded run barely moves it) and drops
tests whose file or definition is gone. A test's estimate is its CPU seconds, and at least a quarter of
its wall seconds. `scripts/test heavy` rewrites `tests/heavy-modules.txt` from this history.
Only measured durations decide a run's length: the `slow` and `heavy` labels (the marker and
`tests/heavy-modules.txt`) never enter the estimate or the class and must not bar a test from any
tier, cache or queue.

Before admission `scripts/test` estimates the run: the recorded seconds of the selected files and nodes,
summed per file without collecting, plus 30 s for each file with no record, divided by the workers the
broker would grant. Named files whose pass the test-reuse store would serve (a read-only lookup, no
claim) cost nothing; a reused file never runs, so the plugin records no duration for it, and a
selection of directories or nodes is estimated in full. It prints the estimate (`test: estimated 38 s`, or `estimated 42 min from 3,120
recorded tests`). An estimate of `DEV_TEST_BACKGROUND_S` (default 180) or more classes the run
`background`, below it `interactive`, whatever the tier name. With fewer than 50 recorded tests the
command shape decides: tiers `full`, `slow`, `record`, `all` or `fast` with no file or node selector,
and `--redteam` are background. `--background` forces the class (`scripts/land` passes it to its full
run); `quick` and `gate` are interactive. A background run prints one line saying why, its pytest
processes start at `nice` +10 (no change where `os.nice` is absent), and `testslots.py status` shows each
lease's class.

The estimate travels with the lease (`DEV_TEST_ESTIMATE_S`). Queued requests are granted shortest
estimated work first: the order key is the estimate minus the seconds waited, so a longer run arriving
at about the same time as a shorter one goes behind it and a long wait outweighs a large estimate. A
request that has waited `DEV_TEST_BACKGROUND_WAIT_S` (default 600) is granted ahead of the ordering and
of the cap. While an interactive run is queued or active, background runs together hold at most half
the budget (rounded up, at least 1). The cap applies to new grants; running workers are never
preempted, but each test takes a fresh one-worker lease, so a background run shrinks to the cap at its
next test. The cap holds inside normal hours, weekdays 08:00-21:00 machine-local, and is lifted outside
them. `DEV_TEST_NORMAL_HOURS="HH:MM-HH:MM"` changes the window; `off` keeps the cap on at all hours.
Nothing is held or delayed to a time of day. While a lease written by older code (`version` below 2)
is live, every lease keeps arrival order and no cap applies.

Each `scripts/test` run keeps a run record in the slots directory (`*.run`: class, estimate, workers
wanted and granted, enqueue and start time). `testqueue.forecast(runs, budget, now)` (module
`scripts/testqueue.py`) returns for each run its position among queued runs under the real ordering,
the count and estimated work ahead, and an estimated start time from the remaining estimate of the
running runs plus the work ahead; `testslots.py status` prints these per queued run.

## Landing a batch

`scripts/land` (modules `scripts/land_*.py`) lands several ready branches with one verification.
Every test and gate runs through `scripts/test`, so the broker admits them like any other run.

- `plan [--cover] [BRANCH...]` lists each branch's unique patches (patch-id, `git cherry`
  semantics), drops branches another one contains, orders the rest so each merges cleanly onto the
  plan so far (`git merge-tree --write-tree`, no checkout), and reports files, commit and behind
  counts, predicted conflicts and dirty worktrees. It warns past 50 commits or 20 behind.
- `run [--dry-run] [--conflicts=eject|stop] [-n N] BRANCH...` merges with `--no-ff` and rerere in a
  sibling `<repo>-land-<stamp>` worktree on `land/<stamp>`, keeping both sides of `HANDOFF.md`.
  A documentation-only diff runs `scripts/budgets` and `git diff --check`. A code diff runs
  `scripts/test gate` and `scripts/test all` on the selectors `scripts/affected.py` computes from the
  combined diff. A diff touching `tests/conftest.py`, `scripts/test*`, `scripts/testslots*`,
  `pyproject.toml` or `packaging/`, an unmapped path, or more than 120 files or 4000 changed lines
  also starts one background `scripts/test full`; a second is refused while a land full run or a
  broker entry labelled `full` is live.
- A pass is recorded with `record_run` under the tree hash; the same tree and check later print
  `reused from <id>`. A failing check is first run on a clean worktree of the base: failing there
  too is reported as `baseline`. Otherwise the merge commits are bisected with only the failing
  selectors, the branch is ejected, the integration branch is rebuilt without it, and the failed
  checks run again. An ejection prints branch, check, evidence, owner (the tip's `Agent-Label`
  trailer, else its worktree name) and commit count.
- `finish [--apply]` is a dry run unless `--apply`. It fast-forwards the target only when the
  primary checkout is clean and on the target, otherwise prints the command for the lead. It never
  pushes. Then it removes each landed source branch's worktree and branch (never forced), keeping
  dirty trees, trees holding the current directory and branches with unique patches, and the
  integration tree while its background full run is alive.

The last line of every command is a JSON summary.

Pending: `scripts/test` passing a `full:`/`land:` label so `other_full_run` in
`scripts/land_check.py` can tell a full run from any other pytest coordinator.
