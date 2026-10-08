# Claude Code instructions

[AGENTS.md](AGENTS.md) is the complete repository policy. Read it first; everything here applies
in addition to it and only covers Claude Code.

## Names

The lead agent is `claude`. `claude-code` is the harness, which can also run Qwen. A lead joining
passes its own model id (`--model <id>`; `claude-sonnet-5-5` unless it knows otherwise).

## Subagent model order

Use Haiku 5.5 (`claude-haiku-5-5`) for read-only search, summarising, narrow mechanical edits,
and lowering a budget ratchet when the brief names the metric and the test selector that checks it.
Haiku never deletes or weakens a test or an assertion, edits authorization, grant, claim, guard,
hook or red-team code, resolves a semantic merge conflict, changes anything outside its worktree
(an install, an interpreter, a shared service), or decides that a branch is ready to land.
Use Sonnet 5.5 (`claude-sonnet-5-5`) for any other work that writes or reviews code. Use Opus 5.5
(`claude-opus-5-5`) only after Sonnet has failed on the task. Fable is used only when the owner
asks. The evidence is in docs/model-benchmarks.md. Select within that order using the capability
rules in AGENTS.md, "The main session and its agents".

## Subagents and the Agent tool

The Agent tool's worktree isolation provides the worktree half of a subagent brief; the brief
still names the branch. The `SubagentStart` and `SubagentStop` hooks join and release a
subagent's workspace access under the lead's identity, so a subagent prompt carries the
`--agent <lead name> --label LABEL` line from AGENTS.md and needs no invite or token. The
`.claude/agents/branch-worker.md` agent is the one-branch worker: it reads AGENTS.md and this
file, announces, works in its worktree, commits named files and reports. The harness confines it
to that fresh tree, so `.claude/agents/branch-finisher.md` (no isolation) is the one for a branch
that already has a claimed sibling worktree.

## Contact comes first

The `SessionStart` hook registers the lead, announces it and puts the inbox in its context. The
lead's first action each session is to read that inbox and answer what is in it; when the hook
output is absent, it runs `ml-stack-workspace inbox --agent claude` before any other work.
Between tasks it runs `ml-stack-workspace inbox` again (it is joined as `claude`).

## Hooks and settings

`.claude/settings.json` wires `scripts/hooks/claude-bash-guard` and `scripts/hooks/claude-edit-guard`
before tools run, and `claude-session-start`, `claude-subagent-start` and `claude-subagent-stop`
around sessions and subagents. `MLSTACK_GUARD=off` turns both guards off.

- The Bash guard refuses `git add -A`, `.` and `-u`, `git commit -a`, `llama-server` by hand, `ml-stack-serve up` flags that skip the lease,
  a push to `main` without `ML_STACK_PUSH_MAIN=yes`, and a force push, a remote ref deletion,
  `--all` or `--tags` pushes, past any `NAME=value` written in front of the command.
- The edit guard refuses, at the moment it is written, a function whose body already exists
  elsewhere, a raw HTTP call, a docstring over twelve lines, a signature over eight parameters,
  and a write that takes a file over its line limit or makes an already-over file longer. It
  reads the two limits from the checkers.
- Claude Code sets `CLAUDECODE` for every command it runs. `--allow-increase` on the budgets and
  the `pre-push` hook's agent restrictions key on it; a terminal sets nothing.
- `ML_STACK_WINDOW_POSITION` is set in `.claude/settings.json`.

## Commit attribution

End git commit messages with the attribution trailer the session's system reminder gives, and
end pull request descriptions with the line it gives. The commit message rules in AGENTS.md
apply to the subject and body above the trailer.

## Pushing

I push `0.2dev` myself after every landing (AGENTS.md, "The agents push; the owner does not"). I
never report a commit as "unpushed, the owner's to push" and never ask whether to push it.

## Browsers

Never drive the owner's own browser through the `claude-in-chrome` tools to test this project's
pages: that window is on their primary display and every click takes their screen. Drive your
own Chromium through playwright, or run headless and read screenshots (AGENTS.md, "Driving a
browser").

## ml-stack-claude

`ml-stack-claude` runs Claude Code against a local model under a broker lease. Drive it in a
scratch directory, never in a checkout being edited (AGENTS.md, "Driving a model on this
machine").
