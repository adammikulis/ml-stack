# Claude Code instructions

[AGENTS.md](AGENTS.md) is the complete repository policy. Read it first; everything here applies
in addition to it and only covers Claude Code.

## Do what was said, when it was said

An instruction from the owner is carried out as stated, now, in the order and on the branch named. It
is never postponed, reordered behind other work, or widened. A request to land a branch lands that branch and nothing else: no new
features, renames or fixes in that branch or turn; anything else found becomes a separate item after it
lands. If something blocks the instruction, say so in one line at once and do every part that is not
blocked. Briefs to workers state exactly what to land and forbid additions.

## Names

The lead is its own session name (for example `claude-6e1a2f`). `claude-code` is the harness, which can also run Qwen. A lead joining
passes its own model id (`--model <id>`; `claude-sonnet-5-5` unless it knows otherwise).

## Subagent model order

Use Haiku 5.5 (`claude-haiku-5-5`) for read-only search, summarising, narrow mechanical edits,
and lowering a budget ratchet when the brief names the metric and the test selector that checks it.
Haiku also writes a small standalone tool or script with its own tests and wires it in with additive
integration edits to existing files (an index or docs row, an import, a registry line, a regenerated
file): edits that only add references to its new code and change no existing behaviour. A trial on
2026-10-08 (a commit-history chart script) passed review: it worked, its tests failed when the code
was broken, it broke no repo rule and it took a correction cleanly. The lead reviews every such diff
to existing files before it lands.
Haiku now has a thinking effort setting, and on the owner's OSWorld 2.1 cost chart (docs/model-benchmarks.md)
Haiku at `xhigh` (about 68%) beats Sonnet at low and medium for under half the cost, and at `max` (about 72%) it
matches Sonnet at high (about 73%) for about 45% of it. Only Sonnet at xhigh or max (about 81% and 84%) goes
higher. So run Haiku for well-specified code (a clear brief, tests it writes for its own code, additive
integration edits): `xhigh` by default, `max` when `xhigh` fell short or the task is harder, passing the
effort to the Agent tool. Use Sonnet, at the effort the task needs, when the work needs more than
Haiku's ceiling or when judgment decides the result: design, conflict-heavy merges, cross-module
refactors, anything near security. The chart is a computer-use benchmark, not our code: when a Haiku
run at `xhigh` or `max` fails review, move that kind of task to Sonnet and note it here.
Haiku never deletes or weakens a test or an assertion, edits authorization, grant, claim, guard,
hook or red-team code, resolves a semantic merge conflict, changes anything outside its worktree
(an install, an interpreter, a shared service), or decides that a branch is ready to land.
Use Sonnet 5.5 (`claude-sonnet-5-5`) for any other work that writes or reviews code. Use Opus 5.5
(`claude-opus-5-5`) only after Sonnet has failed on the task. Fable is used only when the owner
asks. The evidence is in docs/model-benchmarks.md. Select within that order using the capability
rules in AGENTS.md, "The main session and its agents".

## Subagents and the Agent tool

The Agent tool's worktree isolation provides the worktree half of a subagent brief; the brief
still names the branch. The SubagentStart hook registers the subagent as its own identity under a
board-assigned name and delivers it in the brief; SubagentStop announces done as the subagent and
retires it. A subagent prompt carries the workspace line from AGENTS.md and needs no invite or token. The
`.claude/agents/branch-worker.md` agent is the one-branch worker: it reads AGENTS.md and this
file, announces, works in its worktree, commits named files and reports. The harness confines it
to that fresh tree, so `.claude/agents/branch-finisher.md` (no isolation) is the one for a branch
that already has a claimed sibling worktree.

A worker that has delivered its report is stopped at once with TaskStop, and the same goes for a
worker that sends a report again. The lead does not wait for it to exit and does not read a
repeat report as new work. The branch and worktree stay; only the agent ends.

## The heartbeat

A lead keeps a recurring heartbeat prompt, at most 45 minutes apart, for the whole life of its session
(the prompt cache goes cold at 60), and recreates it at each session start and every seven days. It is a
standard part of driving this project and is never deleted when the queue empties: each firing does real
work or ends in one line. The prompt and the rules are in docs/heartbeat.md.

## Contact comes first

The `SessionStart` hook registers the lead, announces it and puts the inbox in its context. The
lead's first action each session is to read that inbox and answer what is in it; when the hook
output is absent, it runs `poolhouse-workspace inbox` before any other work.
Between tasks it runs `poolhouse-workspace inbox` again. The lead is its own session name
(for example `claude-6e1a2f`), run with no `--agent`.

## Hooks and settings

`.claude/settings.json` wires `scripts/hooks/claude-bash-guard` and `scripts/hooks/claude-edit-guard`
before tools run, and `claude-session-start`, `claude-subagent-start` and `claude-subagent-stop`
around sessions and subagents. `POOLHOUSE_GUARD=off` turns both guards off.

- The Bash guard refuses `git add -A`, `.` and `-u`, `git commit -a`, `llama-server` by hand, `poolhouse-serve up` flags that skip the lease,
  a push to `main` (a promotion is a pull request from a `promote/<date>` snapshot, merged only
  under the owner's live `release-main` authorization; `POOLHOUSE_PUSH_MAIN=yes` opens nothing), and a force push,
  a remote ref deletion, `--all` or `--tags` pushes, past any `NAME=value` written in front of the command.
- The edit guard refuses, at the moment it is written, a function whose body already exists
  elsewhere, a raw HTTP call, a docstring over twelve lines, a signature over eight parameters,
  and a write that takes a file over its line limit or makes an already-over file longer. It
  reads the two limits from the checkers.
- Claude Code sets `CLAUDECODE` for every command it runs. `--allow-increase` on the budgets and
  the `pre-push` hook's agent restrictions key on it; a terminal sets nothing.
- `POOLHOUSE_WINDOW_POSITION` is set in `.claude/settings.json`.

## Commit attribution

End git commit messages with the attribution trailer the session's system reminder gives, and
end pull request descriptions with the line it gives. The commit message rules in AGENTS.md
apply to the subject and body above the trailer.

## Pushing

The landing runner pushes the development branch after each batch it lands (AGENTS.md, "A worker lands
its own branch through the queue"); I push only what I land by hand, immediately, and the owner never
pushes it. I never report a commit as "unpushed, the owner's to push" and never ask whether to push it.

## Landing is the worker's

A worker lands its own branch: fetch, rebase its own linear commits onto `origin/<dev>` (merge it when
the branch holds merges or is shared), run the affected tests, then
`poolhouse-workspace land-request BRANCH SHA --test ...`. The runner (`scripts/land up`) batches, gates,
fast-forwards, pushes only the development branch and cleans up; a request that fails is ejected alone and
the rest land. I step in only for cross-branch conflicts, security review and release, and my heartbeat
runs `scripts/land up` when `digest --status` says the runner is `NOT RUNNING`. A worker never pushes, and
never reports landed without a gate result for its exact SHA.

## Browsers

Never drive the owner's own browser through the `claude-in-chrome` tools to test this project's
pages: that window is on their primary display and every click takes their screen. Drive your
own Chromium through playwright, or run headless and read screenshots (AGENTS.md, "Driving a
browser").

## poolhouse-claude

`poolhouse-claude` runs Claude Code against a local model under a broker lease. Drive it in a
scratch directory, never in a checkout being edited (AGENTS.md, "Driving a model on this
machine").
