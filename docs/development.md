# Working on ml-stack

## `contracts/` is data, not code

`contracts/` holds JSON describing things a runtime and a non-Python host both need to
agree on: the RAM→model tier ladder, the sampler surface, GBNF grammars. It contains no
code, so a native or scripting host can read it directly.

There is exactly one copy on disk. The wheel pulls it in at build time,
so there is no synced duplicate in the source tree to drift.

Resolution order at runtime: `$ML_STACK_CONTRACTS` → the copy inside the installed wheel → a
`contracts/` found by walking up from the source file. The walk-up is last on purpose: if a
wheel is installed *and* a repo happens to be an ancestor, the wheel's own data should win,
because that is what its version was tested against.

## Hooks

`python scripts/install-hooks.py` installs the Git hooks using that Python interpreter,
including on Windows, and preserves hooks managed by someone else. Run it from the primary
checkout after landing the reviewed branch. The POSIX `scripts/install-hooks.sh` remains
available for existing shell setups. Installed hooks run the checks in the invoking checkout:
`no-real-names` refuses a commit whose staged files carry a person's name, `commit-msg`
refuses one whose message does. A third hook there is for Claude Code rather than git:
`scripts/hooks/claude-bash-guard` is a PreToolUse hook on Bash that refuses the shells
which keep getting written instead of ml-stack commands -- a hand-written `pgrep` waiter,
`nohup`, `llama-server` started directly, `find`-ing for GGUFs, `hf download`, curl probes
at the model, killing llama by name, `SKIP_NAME_CHECK=1` -- and names the command to run
instead. Wire it into a project's `.claude/settings.json` (the docstring shows the JSON);
`MLSTACK_GUARD=off` disables it for a session. Both hooks are tested:
`tests/test_no_real_names.py` and `tests/test_bash_guard.py`.

Set `git config --local pull.ff only` for editor sync: a divergent pull refuses before
creating a merge. Reconcile divergence in an integration worktree, then land with a reviewed
fast-forward. The pre-push hook refuses dirty checkouts and unfinished Git operations for
people and agents.

Coding tasks own their branch and worktree through assignment, claim, review, integration
and cleanup ([task lifecycle](tasks.md)). A reviewed coding proposal is accepted; completion
requires the landed commit and verified removal of its worktrees and merged branches.

The worktree rule is in `src/ml_stack/worktreerules.py` and reaches every harness. Claude Code
runs `claude-edit-guard` (Write, Edit, MultiEdit, NotebookEdit) and `claude-bash-guard` before
a tool call; both refuse a write inside the primary checkout (the first entry of `git worktree
list`), a git command that changes its tree (`add`, `commit`, `checkout`, `switch`, `reset`,
`restore`, `stash`, `rebase`, `cherry-pick`, `am`, `apply`, and any `merge` that is not
`--ff-only <branch>`) and a `pip install -e` of a tree that is not the primary checkout.
`claude-subagent-start` (SubagentStart) puts the rule and the development branch in each
subagent's context, and `.claude/settings.json` sets `worktree.baseRef` to `head` so worktrees
branch from the development branch. `ml_stack.harnesshook pre` applies the same refusal to Codex
and local-model sessions the launchers start (add a `[[hooks.PreToolUse]]` entry running it to a
Codex config the launchers did not write), and `scripts/hooks/primary-only` refuses a commit by an
agent in the primary checkout or on the development branch. Tests: `tests/test_worktree_rules.py`.

What `no-real-names` takes for a name, and what stands a name-shaped pair down, is data:
`contracts/name-shapes.json` holds the place prefixes and suffixes (a gazetteer's
"North Carolina", "Colorado River"), the job-title endings (a role catalogue's
"Software Engineer"), the `no-real-names: shapes off` file marker and the data-file
suffixes it implies, the RFC 2606 reserved domains, the `noreply` mailboxes, the uuid
and name patterns, and the file suffixes never read -- each section with a `why` saying
what it is for and when it was learned. A refusal names the rule (`patterns: nameish;
nothing stood it down`), and `NAMES_WHY=1` (or `python -m ml_stack.redact.hook --why`)
prints, for every pair a rule cleared, which section and which word did it
(`'North Carolina' cleared by place_first: north`), so the next exception is a word added
to a known section rather than a code change. `NAMES_SHAPES=path.json` reads another
rules file instead of the shipped one.

## Python versions

The library imports and its suite runs on 3.12 to 3.14; CI runs each, and 3.15 as an
experiment that may fail. The app builds its own environment on 3.13 (`fleet.environment.PYTHON`,
the installers). Code in `src/` and `tests/` uses nothing newer than 3.12:

- generics use PEP 695 (`class C[T]`, `def f[T]`, `type X[T] = ...`), not `Generic` and `TypeVar` (ruff UP046, UP047)
- `os.waitid` is missing on macOS before 3.13

`vermin -t=3.12- --no-tips src tests scripts` lists what a change added.

## Testing

```
python -m pytest tests/ -q
```

Run tests through `scripts/test`, which queues for workers in the machine-wide budget:

| tier | what it runs | about |
|---|---|---|
| `scripts/test quick` | the tests a change reaches (import graph, then testmon's map) | under a minute for one module |
| `scripts/test gate` | the structural checks a merge is gated on: budgets, red-team coverage, command reference, layers, the human-floor tests | a minute |
| `scripts/test fast` | everything not marked `slow` or `heavy` | |
| `scripts/test full` | everything not marked `slow`; writes its wall and CPU time for the ratchet | 3 minutes on an idle 16-core machine |
| `scripts/test all` | slow included: what CI runs | |
| `scripts/test ratchet` | compares the last `full` run with `tests/full-tier-time.json` (10 % tolerance) | |

`quick` needs a per-checkout testmon map (`.testmondata`, not committed). The first `quick` in a
checkout runs the test files the import graph reaches and starts `scripts/test record` detached
(three workers, a lock file, a log in `.testmondata.log`); the map is written to a scratch file and
moved into place when the run ends, so the second `quick` is fast and never sees half a map. A
change to something no module maps (`pyproject.toml`, `tests/conftest.py`, ...) still runs the full
tier.

`tests/full-tier-time.json` is the full tier's wall time (at a worker count) and the sum of every
test's own time. A later run more than 10 % over either fails `scripts/test ratchet`; wall time is
compared only at the same or a higher worker count. `scripts/test ratchet --update` records a
lower time and refuses a higher one (`--allow-increase` is the owner's, refused for an agent). Where
the time went, and the causes already fixed: `docs/experiments/test-durations.md`.

Rules that keep the suite fast and not flaky:

* A test never nests a slot lease. A test that runs `scripts/test-on-linux` or
  `testslots.py run` inside a full run waits for slots the run holds (it deadlocked a whole run
  for seven minutes); give the child `DEV_TEST_WORKERS` so it skips the queue.
* A test that checks a failure path does not wait out the production timeout: pass a small one.
* Servers a test starts shut down in milliseconds (conftest polls every 20 ms); do not add sleeps
  to wait for them.
* The real-state-root guard (`tests/conftest.py`, `LIVE_PATHS`) fails a run when a file under
  `~/.ml-stack` changed. Other agents and runs append to a few logs while yours is going, so only
  `workspace/`, `harness/`, the activity log and the sentinel's event and anchor logs are tolerated.
  The keystore, every `.key` file, manifests, canaries, honey, requests and credentials stay guarded
  (`tests/test_live_paths.py`). The guard cannot tell which process wrote a tolerated log; a test
  that escapes into one of those paths is not caught.
* A test never depends on a neighbour having run first (a module fixture it forgot to request, a
  process-wide observer): run it alone before you push.

Nothing here mocks the transport. Every client test runs a real `http.server` on a real
socket, because the failures these modules exist to prevent are transport-shaped: a server
that answers `/health` while still loading, one that ignores a `Range` header, one that
returns 500 on a concurrent request. A mocked `urlopen` reproduces none of them.

The fleet tests go further: they boot real `ml-stack-traind` subprocesses and speak real
UDP on a real interface, on randomised ports so a run never answers -- or gets answered
by -- a daemon you actually have running on your LAN. A forged beacon, a replayed reply,
a multicast group a router quietly drops: a fake socket reproduces none of those either.

What *is* faked -- the model's answers, a `serve()` that would load 87G, a preflight that
would read it -- is faked once, in `ml_stack.testing.fakes`, with the real signature.
`FakeClient` is built exactly as `Client` is built, `fake_serve` / `FakeServe` take what
`serve()` takes, `FakePreflight` returns a real `Report`, and `ScriptedModel` replays tool
calls through `Client.chat`'s signature. None takes a `**kwargs` the real one lacks: a fake
that accepts every keyword lets a test pass on a keyword the real thing refuses, which is
how a `--also tight` flag once reached `Client.__init__` in a benchmark and took the load
down with it. `tests/test_testing_fakes.py` diffs every fake's signature against the real
one (`mirrors`, `drift`), so a change to the real one fails the suite until the fake follows.

