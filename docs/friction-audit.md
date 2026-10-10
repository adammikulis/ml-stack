# Friction audit of the hooks, guards and gates

Audited 2026-10-09 from the checkout at the head of the Poolhouse rename branch, against the owner's order:
a hook that forces the owner to be present for a standard operation goes. Every refusal in
`scripts/hooks/*`, `.claude/settings.json`, `src/poolhouse/worktreerules.py` and the commit-msg / pre-commit /
pre-push chain is listed. "Who can resolve today" is who can make the refused step succeed with what exists now.

Classes: **DELETE** (the check adds no safety for this operation), **AGENT-RESOLVABLE** (a standing, scoped,
audited, revocable grant or an independent second-agent review replaces the human-only step; AGENTS.md
graduated autonomy), **KEEP-FLOOR** (no push or merge to `main` without live `release-main` authorization, an
agent is never a human, keystore and secrets, one GPU job at a time, forgery of the person record).

Status: **done** is committed with failing-first tests; **patch** is in `docs/hooks-protected.patch` because the
edit guard keeps agents out of that file; **next** needs infrastructure that does not exist yet and is filed below.

| # | Hook | Refusal | Who can resolve today | Class | Change |
|---|------|---------|-----------------------|-------|--------|
| 1 | no-data-files | a renamed file under a data directory, or with a data extension, counted as newly added | nobody; the owner bypasses with `--no-verify` | DELETE (rename only) | done: a rename keeps any rule its old path already broke; a rule it newly breaks, a growth over the size limit or a move into a data directory still fails |
| 2 | no-data-files | a merge's files taken from the other parent | nobody | DELETE (merge only) | done: staged mode checks only files that differ from both parents, as the name check does |
| 3 | no-data-files | a new data file, weights, a file over 1 MiB, a copy of a data file | the owner moves it out of the tree | KEEP-FLOOR | none: the repository is public |
| 4 | budgets | every site in a renamed .py file counted as new | nobody (`SKIP_BUDGETS=1` by hand) | DELETE (rename only) | done: the new path is counted against the old path's content; an added site, a new file, a copy and a move in from a non-.py file still fail |
| 5 | budgets | a merge | none needed | no refusal | none: merges were already skipped |
| 6 | budgets-only-fall | a merge carrying the other parent's budget numbers | owner terminal (`POOLHOUSE_BUDGET_RISE=yes`); an agent is refused outright | DELETE (merge only) | done: a number may equal either parent's; a rise past both still fails |
| 7 | budgets-only-fall | a raised or dropped budget number | owner terminal | KEEP-FLOOR | none: a budget only falls is the ratchet |
| 8 | weakened-assertions | a renamed guard, authority or red-team test, or a renamed test function in it | any agent that types `Reviewed-by-second:` | DELETE (rename only) | done: the diff pairs old and new paths; a hunk that swaps a test name for another with the body untouched is a rename; a dropped assertion, a removed test, a new skip, and a guard test renamed to an unguarded name still fail |
| 9 | weakened-assertions / commit-msg | a real weakening needs a `Reviewed-by-second:` line | any agent, by typing the line, so nothing verifies a second reviewer exists | AGENT-RESOLVABLE | next (item A below): the line must name a distinct board identity whose review of that exact diff is on the board |
| 10 | no-real-names / commit-msg | a pure rename re-reads every line of every moved file for names | nobody | DELETE (rename only) | done: `added_lines` pairs a renamed file with its old path, so only the lines the rename changed are read; a merge was already limited to what differs from both parents |
| 11 | no-real-names / commit-msg | a name or contact detail on an added line; `SKIP_NAME_CHECK=1` | the owner, at a terminal; the Bash guard refuses an agent that sets it | KEEP-FLOOR | none: a real community on a public repository |
| 12 | pre-push | an agent pushing an existing `promote/*` branch (a new one already passed) | the owner | DELETE | patch: `docs/hooks-protected.patch` allows a fast-forward of a `promote/*` branch; a rewrite, an unknown remote commit, `main` and any work branch stay refused |
| 13 | pre-push | an agent pushing anything but the development branch and `promote/*` | none needed: land it, then push the development branch | already agent-resolvable | none |
| 14 | pre-push / Bash guard | `main`, a force, a deletion, `--all`, `--mirror`, `--tags` | `main`: live `release-main` authorization; the rest: the owner | KEEP-FLOOR | none |
| 15 | primary-only / worktreerules | an agent commit on `main`, or on the development branch from a linked worktree | none needed: commit on the task branch, land with `merge --ff-only` | already agent-resolvable | none |
| 16 | worktreerules (Bash guard) | `checkout`, `reset`, `rebase`, `apply`, `restore`, `stash`, `cherry-pick` and a non-fast-forward `merge` in the primary checkout | none needed: do it in a worktree; `switch <branch>` and `merge --ff-only <branch>` are allowed there | already agent-resolvable | none |
| 17 | worktreerules (Bash guard) | `cd` or `git -C` into a sibling worktree | not refused (the guard follows `cd` and `-C`) | no refusal | none |
| 18 | worktreerules (Bash guard) | `git worktree add` inside an existing checkout | none needed: put it beside the checkout | already agent-resolvable | none |
| 19 | Bash guard | `git add -A`, `.`, `-u`, `commit -a` | none needed: name the files | already agent-resolvable | none |
| 20 | Bash guard | a bare `git stash`, `stash pop`, an untagged `stash push` | none needed: `stash push -u -m <tag>`, then `stash apply <sha>` | already agent-resolvable | none |
| 21 | Bash guard | `nohup`, `llama-server` by hand, `poolhouse-serve up` flags that skip the lease, `pkill` of the server, hand-written waiters, curl probes, GGUF hunts | none needed: the named command | KEEP-FLOOR (one GPU job at a time) / already agent-resolvable | none |
| 22 | Bash guard | `poolhouse-workspace mint` and `delegate` | the owner, at a terminal | KEEP-FLOOR | none: an agent never creates identities |
| 23 | Bash guard / edit guard | reading or writing the person record, its key, `claude-user-prompt`, `person-consume` (except `propose`), `person_*.py` | nobody; the harness hooks write it when the person types | KEEP-FLOOR | none: forgery |
| 24 | Bash guard | a subagent's workspace command without `--agent` | none needed: add `--agent <name>` | already agent-resolvable | none |
| 25 | edit guard | an edit to `claude-bash-guard`, `claude-edit-guard`, `pre-push`, `rules_loader.py` or `.claude/settings.json` for a reviewed change | the owner, at a terminal (as with the patch above) | AGENT-RESOLVABLE | next (item B below): a scoped, audited, revocable `guards.edit` grant plus a second-agent review of the exact diff |
| 26 | edit guard | an edit on `main` | none needed: use a development branch | KEEP-FLOOR | none |
| 27 | edit guard | a duplicate function body, a raw HTTP call, a docstring over 12 lines, a signature over 8 parameters, a file over its limit | none needed: refactor | already agent-resolvable | none |
| 28 | `POOLHOUSE_GUARD=off` | switches off the soft rules for a person's own session | the owner | KEEP-FLOOR | none: the hard rules stay in force |
| 29 | `.claude/settings.json` hooks | wiring only; none refuses by itself | n/a | no refusal | none |
| 30 | post-commit, post-merge, runtime-refresh, session and subagent hooks, `tree_watch` | warn or announce; none refuses | n/a | no refusal | none |
| 31 | release / promote rules | merging a promotion pull request into `main` | live `release-main` authorization | KEEP-FLOOR | none |
| 32 | generated files | none of the hooks refuses a regenerated file | n/a | no refusal | none |

## Counts

32 rows: 7 DELETE (rows 1, 2, 4, 6, 8, 10, 12), 6 of them committed with tests and row 12 as the patch;
2 AGENT-RESOLVABLE that need infrastructure (rows 9 and 25); 8 already agent-resolvable by the named form, no
change (rows 13, 15, 16, 18, 19, 20, 24, 27); 10 KEEP-FLOOR (rows 3, 7, 11, 14, 21, 22, 23, 26, 28, 31);
5 that refuse nothing (rows 5, 17, 29, 30, 32).

## What the owner applies once

`git apply docs/hooks-protected.patch` (it changes only `scripts/hooks/pre-push`, which the edit guard keeps from
agents). Everything else is committed. After applying it, `tests/test_pre_push_promote_update.py` uses the
real hook.

## Next items (infrastructure that does not exist yet)

- **A. A verified second reviewer.** `poolhouse-workspace review sign SHA --agent NAME` writes a board record
  (reviewer, the commit's tree hash, the diff's hash, a time); the `Reviewed-by-second:` line names that
  reviewer; `weakened-assertions` checks the record exists, is by a different identity than the commit's author
  and covers this diff. Without the record the line is a self-typed claim that nothing checks. This is the
  independent review that rows 9 and 25 both depend on.
- **B. A `guards.edit` grant.** A gate in `poolhouse.authority` (state `person` by default) that a lead agent can
  delegate, scoped to named files, with an expiry, logged in the authority audit log and revocable with
  `authority set person guards.edit`. `claude-edit-guard` consults it for `claude-bash-guard`,
  `claude-edit-guard`, `pre-push`, `rules_loader.py` and `.claude/settings.json`, and still refuses the
  person-record files (`person_*.py`, `claude-user-prompt`, `person-consume`), which stay KEEP-FLOOR. It needs
  item A, so the grant is only used with a second agent's signed review of the diff.

## Observation, not changed

`SKIP_BUDGETS=1` is not refused by the Bash guard (unlike `SKIP_NAME_CHECK`), so an agent can switch the budget
check off for one commit. That is a hole in the opposite direction from friction, and is reported, not touched.
