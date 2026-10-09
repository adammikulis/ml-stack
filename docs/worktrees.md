# Worktrees: owners, orphans and the one way to remove one

Every worktree of a repository has an **owner** and a **state**, kept in one registry file in the
shared git directory (`ml-stack-trees.json`, next to the repository's other git state; it is not
versioned). Code: `src/ml_stack/trees.py` (registry, states, `close`, `sweep`),
`src/ml_stack/trees_notice.py` (thresholds, who is told, caps), `scripts/worktrees` (the commands),
`scripts/hooks/tree_watch.py` and `scripts/hooks/post-commit` (the hooks' side) and the gate
`scripts/gates/orphan_trees.py`.

## How a tree gets an owner

- **Harness isolation / SubagentStart**: the tree the subagent starts in is registered to the
  subagent's board name.
- **`scripts/land`**: its integration worktree is registered to `land`.
- **By hand (`git worktree add`)**: no hook is needed. `worktrees scan` (run by SessionStart, the
  attention hook, the post-commit hook, `worktrees` and the gate) registers any tree it has not
  seen as owner `unknown`. A commit made in the tree by an agent claims it for
  `ML_STACK_WORKSPACE_AGENT`.
- `worktrees claim TREE OWNER [PURPOSE] [--pid N] [--scratch]` registers one explicitly. A
  `--scratch` tree (a baseline or throwaway checkout) never counts and is never nagged about;
  `sweep` removes it once its owner stops, if it is clean.

## States

| state | meaning |
| --- | --- |
| active | the owner is alive: its process (when a pid is recorded) is running, else it showed a sign of life within `ttl_h`; an `unknown` owner is assumed alive for `claim_h` after the tree appeared |
| waiting | the owner stopped within the grace and the tree holds work: "finished, waiting to land". Visible, never an orphan, never counted by the gate |
| orphan | the owner stopped more than `grace_h` ago and the tree is not landed, bundled or abandoned |
| landed | the owner is not active, every commit is in the development branch (by ancestry or by patch, so a rebased and cherry-picked tree counts) and nothing is dirty |
| scratch | a `--scratch` tree |

A clean tree with no commits looks the same as a landed one. Only the owner's liveness tells a
fresh tree of a running worker from a finished one, so **a tree whose owner is active is never
removed by `sweep` and `close` refuses it** (the owner itself may close its own tree). The
development branch is the branch the primary checkout is on. The primary checkout is never a row.

## Seeing trees

Waiting and orphan trees show with no delay in `ml-stack-workspace digest --status`, the lead's
attention hook and SessionStart context, the heartbeat, and `scripts/worktrees`. The report exits 1
for an orphan (not for a waiting tree) or a tree over a hygiene limit. Each line names the tree,
branch, unlanded commits, dirty files, owner, how long ago the owner stopped and the actions.

```
scripts/worktrees                                   # report
scripts/worktrees close TREE --landed               # work is all in the development branch, clean
scripts/worktrees close TREE --bundle               # unlanded commits go to a verified git bundle first
scripts/worktrees close TREE --abandon "reason"     # reason recorded; the dirty diff is saved first
scripts/worktrees sweep                             # close every landed tree whose owner is not active
```

`close` is the only removal path for agents and the lead. `--landed` refuses unless the claim is
true; `--bundle` refuses dirty files (commit them or abandon) and verifies the bundle holds the tip
before removing anything; `--abandon` needs a reason. Bundles and dirty diffs go to
`ml-stack-bundles/` in the git directory. A tree removed some other way (`git worktree remove`,
`rm -rf`) is not an error: the next scan drops it and records "removed-outside-the-tool" in the
registry history.

## Notices

Per tree and condition: commits ahead of the development branch at or over `max_ahead`; more than
`max_behind` behind; unlanded commits older than `max_age_h`; "finished, waiting to land"; and
"ORPHAN" past the grace. A condition is told when it first appears and again only when its value
changed and `repeat_min` has passed; an unchanged one goes once more to the coordinator when the
owner stayed silent, then not again. The board gets the note; the owner is told while alive; the
coordinator when the owner has stopped, is unknown, or was silent. Each recipient (and the board)
gets at most `cap_per_hour` messages an hour: the last is a roll-up line, the rest are held and
show in `scripts/worktrees`. Delivery runs in a detached process so a slow board never holds up a
hook. The coordinator is the session that last started as lead (`worktrees lead NAME`, or
`ML_STACK_TREES_LEAD`).

## The gate

`orphan-trees` (hard, no allowance, no `--allow`) counts orphans: owners stopped more than
`grace_h` ago. A tree that has just finished is waiting, and cannot fail an unrelated landing. The
finding names the tree and the exact `close` command that fits it.

## Friction budget

Measured with 25 registered trees on a loaded machine (load average about 24), see the audit in the
commit message. The registry caches each tree's counts against the development tip and asks git for
`status` only for trees whose owner is not active, so a warm report costs one `git worktree list`.
The post-commit hook is a shell prefilter: it exits in a primary checkout, below `max_ahead`, and
for 30 minutes after it last ran, with no Python. Every hook path is bounded by an 8 second alarm,
waits at most 2 seconds for the registry lock and fails open with one warning line: a locked,
missing or garbled registry, a slow git or an unreachable board never blocks a commit, push or
tool call. A person using plain git in the primary checkout sees nothing; in a linked tree they
get one stderr line when the commit count crosses the threshold.

## Policy

Defaults: `grace_h` 2, `claim_h` 1, `ttl_h` 12, `max_ahead` 10, `max_behind` 20, `max_age_h` 4,
`repeat_min` 30, `cap_per_hour` 6. Override per repository in the registry's `policy` object, or
per pool or session with the variable named in `trees.ENVIRONMENT` (for example
`ML_STACK_TREES_GRACE_H=4`); the environment wins. The post-commit prefilter reads only
`ML_STACK_TREES_MAX_AHEAD` and `ML_STACK_TREES_REPEAT_MIN` from the environment.
