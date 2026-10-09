# Worktree lifecycle recovery

Coding sessions use their authenticated agent identity. Connect and device-session setup may
run automatically for the agent's own session through supported tooling. Agents never initialize
a human workspace, read an owner credential file, or mint a person's credentials. Explicit
invalid credentials remain errors; a selected remote board never falls back to a local board.

## Recovery workflow

1. Acquire authenticated branch, worktree and file-area claims before mutation. Record the task,
   owner, process identity, source commit and expiry. Use task-scoped handoffs for an existing
   reservation and verify expired or dead-worker recovery.
2. Create isolated worktrees beside the primary checkout. A failed checkout releases its
   reservation. Keep independent workers on separate branches and paths.
3. Preserve uncommitted changes and unique commits before integrating. Review independently and
   run affected tests and structural/security gates. Integrate through a gated fast-forward to
   `0.2dev`; synchronize only the development branch with its upstream.
4. Before cleanup, inspect unmerged commits, tracked/untracked changes, ignored state and live
   worker ownership. Preserve unique data and never remove a tree used by an active worker.
5. Remove a landed, clean worktree from outside that tree; delete its merged local branch and
   prune registrations. Verify `git worktree list` no longer contains the path before completion.
6. When Windows or Cloud Files refuses cleanup, record the exact path and retryable state.
   Recovery verifies absolute administrative paths remain within the selected repository and
   avoids following reparse-point targets. Keep completion pending while registration remains.

## Coverage

Affected Windows checks use the maintained test broker. Linux testing remains paused until the
owner explicitly resumes it. Full suites run in the background after reviewed batches; workers
run affected checks and report exact commands, commits, results and platform gaps.
