# Worktrees: owners, orphans and the one way to remove one

Every worktree of a repository has an **owner** and a **state**, kept in one registry file in the
shared git directory (`ml-stack-trees.json`, next to the repository's other git state; it is not
versioned). Code: `src/ml_stack/workspace/trees.py` (registry, states, `close`, `sweep`),
`trees_notice.py` (thresholds and who is told), `scripts/worktrees` (the commands),
`scripts/hooks/tree_watch.py` (the hooks' side) and the gate `scripts/gates/orphan_trees.py`.

## How a tree gets an owner

- **Harness isolation / SubagentStart**: the tree the subagent starts in is registered to the
  subagent's board name.
- **`scripts/land`**: its integration worktree is registered to `land`.
- **By hand (`git worktree add`)**: no hook is needed. `worktrees scan` (run by SessionStart, the
  attention hook, the post-commit hook, `worktrees` and the gate) registers any tree it has not
  seen as owner `unknown`. The post-commit hook (installed by `scripts/install-hooks.sh`) claims
  the tree for `ML_STACK_WORKSPACE_AGENT` on the first commit made in it.
- `worktrees claim TREE OWNER [PURPOSE] [--pid N]` registers one explicitly.

## States

| state | meaning |
| --- | --- |
| active | the owner is alive: its process (when a pid is recorded) is running, else it has shown a sign of life within `ttl_h`; an `unknown` owner is assumed alive for `claim_h` after the tree appeared |
| landed | the owner is not active, every commit is in the development branch (by ancestry or patch) and nothing is dirty |
| orphan | the owner is not active and the tree is not landed, bundled or abandoned |
| bundled / abandoned | decided with `close`; the tree is removed in the same step |

A clean tree with no commits looks the same as a landed one. Only the owner's liveness tells a
fresh tree of a running worker from a finished one, so **a tree whose owner is active is never
removed by `sweep` and `close` refuses it** (the owner itself may close its own tree). The
development branch is the branch the primary checkout is on.

## Seeing orphans

An orphan is shown at once, with no grace: `ml-stack-workspace digest --status`, the lead's
attention hook and SessionStart context, the heartbeat, and `scripts/worktrees` (exit 1). Each line
names the tree, branch, unlanded commits, dirty files, owner, how long ago the owner stopped and
the three actions.

```
scripts/worktrees                                   # report; exit 1 on a flagged tree or an orphan
scripts/worktrees close TREE --landed               # work is all in the development branch, clean
scripts/worktrees close TREE --bundle               # unlanded commits go to a verified git bundle first
scripts/worktrees close TREE --abandon "reason"     # reason recorded; the dirty diff is saved first
scripts/worktrees sweep                             # close every landed tree whose owner is not active
```

`close` is the only removal path for agents and the lead. `--landed` refuses unless the claim is
true; `--bundle` refuses dirty files (commit them or abandon) and verifies the bundle holds the tip
before removing anything; `--abandon` needs a reason. Bundles and dirty diffs go to
`ml-stack-bundles/` in the git directory. Never `git worktree remove` or `rm -rf` a tree.

## Subagent stop

SubagentStop marks the stopping agent's trees finished. A tree with unlanded commits or dirty files
is posted to the board and sent to the lead at once and is an orphan from that moment.

## Notices

Per tree, at most once per `repeat_min` per condition: commits ahead of the development branch at
or over `max_ahead`; more than `max_behind` behind; unlanded commits older than `max_age_h`; and an
orphan. Run from SubagentStart/Stop, SessionStart, the attention hook, the heartbeat and the
post-commit hook. The board gets the note. The owner is told by direct message while alive; the
coordinator is told too when the owner has stopped, is unknown, or was already told this same
condition and has not fixed it. The coordinator is the session that last started as lead
(`worktrees lead NAME`, or `ML_STACK_TREES_LEAD`). Conditions also show in `scripts/worktrees` and
`digest --status`.

## The gate

`orphan-trees` (hard, no allowance, no `--allow`) counts orphans whose owner stopped more than
`grace_h` ago. Grace is for the gate only; visibility has none.

## Policy

Defaults: `grace_h` 2, `claim_h` 1, `ttl_h` 12, `max_ahead` 10, `max_behind` 20, `max_age_h` 4,
`repeat_min` 30. Override per repository in the registry's `policy` object, or per pool or session
with `ML_STACK_TREES_<NAME>` (for example `ML_STACK_TREES_GRACE_H=4`); the environment wins.
