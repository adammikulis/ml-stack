---
name: worktree-cleanup
description: Reconcile a repository's extra Git worktrees, land useful changes, account for obsolete work, and prune completed checkouts. Use when the user requests worktree cleanup or wants only the primary checkout remaining.
---

# Worktree cleanup

The goal is one retained primary checkout and no extra worktrees. Account for every checkout's contents before removal; do the integration and cleanup rather than stopping at an inventory.

Read the repository's agent instructions and identify its primary checkout and development branch. Follow its ownership, review, integration, testing and publication rules, including any platform testing pause. Use a maintained cleanup/lifecycle helper when one exists so removal also records the required completion proof.

## Account for every worktree

Inspect `git worktree list --porcelain`, including locked, detached and prunable entries. A registration marked prunable is not proof that its files or commits are disposable. Resolve platform-specific paths before treating a checkout as absent.

For every extra checkout, inspect:

- Its branch or detached commit, unique commits relative to development, and patch equivalence (`git cherry` helps detect already-landed work).
- Tracked changes, staged changes, all untracked files and ignored files. Distinguish rebuildable caches from unique data or artifacts.
- Live writers, task assignments, claims, locks and pending reviews. Coordinate an authenticated handoff or wait for active work to reach a safe boundary; do not seize another worker's checkout.

Classify the actual contents, not the branch name, age or a passing test. Record a brief disposition for each checkout: useful work to land, already landed or equivalent, obsolete with a concrete reason, or blocked on a named condition.

## Land useful work

Review useful commits and unfinished changes, including interactions with current development. Claim named files before completing unfinished work. Commit changes by named files on the checkout's branch so they can be reviewed and integrated.

Use independent review and the repository's required affected checks. Reconcile conflicts in an isolated sibling checkout. Fetch before integration and publication, preserve remote advancement, and use normal development-branch pushes only when authorized. Do not promote protected branches, publish tags or force-push as part of cleanup.

Keep the original Git commit history and ancestry when landing useful work. Integrate the existing commits with normal merges or fast-forwards; do not replace a branch with an archive or snapshot import, squash away its useful commits, or recreate its patch while discarding its original history. Disposable local project or Board history is separate from Git history.

Do not cherry-pick useful fragments and silently lose the rest. Account for every unique commit and file, and verify that the useful result is present in the final development tree.

## Remove completed and obsolete checkouts

Run removal from outside the target checkout. Immediately before removal, recheck unique commits, staged and unstaged changes, untracked and ignored contents, and live ownership. Preserve any unique work that has not been deliberately accounted for.

Already-landed or equivalent commits need no second implementation. For obsolete unique commits, record why they are unnecessary; retain their branch or an explicit recovery reference when needed to avoid losing unmerged history. A retained branch does not require retaining its worktree. Do not turn uncertain work into a deletion just to reach zero.

Use the repository's maintained cleanup command when available; it should verify the landing and record cleanup before deleting the checkout and its merged branch. Otherwise use `git worktree remove` without force, delete merged branches with `git branch -d`, and inspect a prune dry run before pruning stale registrations. Do not use forced deletion to bypass preservation checks or remove an active lock.

Apply the same checks to any integration or investigation checkout created during this cleanup. Preserve the primary checkout.

## Finish at zero extra worktrees

Fetch and verify the final development/upstream relationship when publication is in scope. Re-run `git worktree list --porcelain` and confirm that only the primary checkout remains and that removed paths and registrations are gone. Check that useful work is landed and required lifecycle receipts are recorded.

Report the landed commit, the disposition of each extra checkout, any deliberately retained recovery branches, and the verified remaining worktree count. If active work, unique data or missing authority prevents removal, name the exact path, responsible worker and required next action. Keep the task incomplete until those conditions are resolved; do not report cleanup complete with extra worktrees left.
