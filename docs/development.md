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

`scripts/install-hooks.sh` links the git hooks in `scripts/hooks/` into `.git/hooks`:
`no-real-names` refuses a commit whose staged files carry a person's name, `commit-msg`
refuses one whose message does. A third hook there is for Claude Code rather than git:
`scripts/hooks/claude-bash-guard` is a PreToolUse hook on Bash that refuses the shells
which keep getting written instead of ml-stack commands -- a hand-written `pgrep` waiter,
`nohup`, `llama-server` started directly, `find`-ing for GGUFs, `hf download`, curl probes
at the model, killing llama by name, `SKIP_NAME_CHECK=1` -- and names the command to run
instead. Wire it into a project's `.claude/settings.json` (the docstring shows the JSON);
`MLSTACK_GUARD=off` disables it for a session. Both hooks are tested:
`tests/test_no_real_names.py` and `tests/test_bash_guard.py`.

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

## Testing

```
python -m pytest tests/ -q
```

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

