# Shared test execution

Run tests with `python scripts/test fast`, `full`, `slow`, `all`, or `quick`.
Explicit test paths are checked before broker admission using pytest’s argument parser.
A missing file or directory, or a node selector that collects no tests, returns
exit status 4, including in `quick`; an empty affected-test selection remains valid.
`-n N` limits the worker pool; scoped runs default to one worker. Explicit `-n 0` requests
an automatic pool up to broker capacity. Explicit worker ceilings remain effective.
Every checkout shares the same admission broker.

Agents run affected tests for their own changes. The main agent coordinates shared structural
and security gates once per consolidated integration batch before publication, handles full
end-to-end checks, and schedules background full suites. Workers do not repeat those checks.
The main agent owns `quick`, whose cold-map recording and fallback can start full runs; workers
use explicit affected selectors.

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
