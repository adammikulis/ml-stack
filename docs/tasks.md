# Tasks and independent outcomes

Open **Tasks** in the main UI to inspect structured work. Search or filter by status, select
an item, and inspect its account, physical device, authenticated worker, bound model,
resource lease, checkpoints, artifact hashes and independent reviews. The Board remains
available for discussion; a discussion message does not acquire a task or resource lease.

A person creates a task with a title and explicit acceptance criteria. The coordinator
stores it in `coordination.db`, with task, device, worker, allocation, lease, checkpoint,
proposal, artifact and review nodes linked by actual graph edges. A stable workspace node
identifies this coordinator independently of its directory. A source key deduplicates an
identical task; reusing it for a different specification is refused.

Workers claim through the authenticated `TaskBoard.claim` service with a scheduler-issued
allocation ID. The service checks registered authority, accepted dependencies, required
capabilities and the actual live broker allocation before atomically acquiring the task.
Execution uses the scheduler's prepared worktree and immutable baseline, never a path from
the task description. Models and harnesses are task provenance; balances belong to the
authenticated physical-device base account.

A heartbeat renews the task lease. Worker checkpoints, proposed checks and claimed usage
remain claims. Submission records artifact hashes and the actual allocated model/runtime;
it does not complete the task. An independent person or authorized reviewer records passed
checks covering every acceptance criterion before an outcome is accepted. Coding tasks remain
accepted until their reviewed commit lands and their worktrees and branches are removed; only
then do they become completed. Reviewers
need existing project permissions, cannot review their own work, and nonhuman reviewers
cannot grade another worker from their own physical-device account. Rejected work and
infrastructure blockage are separate outcomes.

Each project runs in one of two enforcement modes. In `open` mode, the default, a task with no
assignees is claimable by any registered non-human worker in its project, assignees and reviewers
need only be registered in the task's project, and any registered project member other than the
worker may review. In `strict` mode an unassigned task is open only when a person created it,
assignees and reviewers must belong to the task's project, and a reviewer must be a person, a
designated reviewer or the creator of the worker's parent. `poolhouse-workspace enforcement`
shows the mode (`show`, readable by anyone) and a lead identity changes it: `check` lists the
tasks that strict mode would strand, `promote` switches to strict and `demote` to open in one
step, and `set open|strict` does the same by name. A helper identity cannot change it. The mode
is stored per project key in `enforcement.db`, outside the project record that tasks carry, and
every change writes an `enforcement.set` audit entry with the identity, time, previous and new
mode. A change applies to new claims, assignments and reviews; leases, submitted work and
accepted reviews continue under the rules they started with. `whoami` reports the mode of the
caller's project.

Expired working leases require **Recover expired lease**. Blocked work requires
**Authorize resume**, with a reason describing the changed condition. These actions preserve
checkpoint history and do not grant resources; the next worker claim still validates a live
allocation. Recovery consumes the configured retry budget. Supported limits currently cover
model choice, wall time and retries. Memory, context and token limits require maintained
broker/runtime enforcement before they can be represented as guarantees.

Independent review automatically records the verified outcome in the encrypted work ledger.
Accepted work earns completion credits once, with independently evidenced quality bonuses
when recorded by the reviewer. If the ledger is unavailable, the review remains saved and
the UI reports credit recording as pending. **Record verified outcome** retries the same
review idempotently; it cannot mint a second award or substitute worker checks. Quality is
shown as unrated until independently assessed. Reliability uses accepted/rejected outcomes;
infrastructure blockage is excluded. **Runs are free**: balances and reputation do not
restrict access or spend credits.

The person-facing API is `GET /ui/tasks`, `GET /ui/tasks?id=task:…`, and same-origin
`POST /ui/tasks`. POST actions are create (`spec`), review (`id`, `decision`), recovery or
resume (`id`, `reason`), and credit retry (`id`). Fleet authenticates the browser; the
workspace person identity stays server-side. Browser requests cannot supply worker tokens,
claim resource allocations, or grant project permissions.

Agents use the maintained workspace CLI (`tasks`, `task ID`, `task-claim ID ALLOCATION`,
`task-heartbeat ID`, `task-checkpoint ID JSON`, `task-submit ID JSON`, `task-review ID JSON`,
and `task-credit ID`). JSON payloads can be read from stdin with `-`. The corresponding
`workspace_task*` MCP tools use the launcher’s existing authenticated identity; they cannot
create tasks, enroll identities or grant project permissions. Claims require a live trusted
allocation. Review uses the same independent-review and idempotent outcome adapter as the
person UI; a locked credit store leaves the saved review intact and reports pending credit.

Reviewed native work can land through `task_integration.integrate(workspace, token, task_id)`.
The same authenticated operation is available as `poolhouse-workspace task-integrate ID`
and the `workspace_task_integrate` tool. It needs existing development-branch claim authority;
an accepted review does not grant publication rights. A foreign live development claim
blocks integration before any candidate or source-claim changes.
The helper requires the exact committed task branch, tracked artifact hashes and committed
full patch that the independent review accepted. It merges unchanged source into its own
candidate worktree, runs the maintained quick, structural and serving-security gates,
then fast-forwards the checked-out development branch. Local integration is the default;
publication requires an explicitly authorized `publish=True` request.
Before preparing a candidate and before landing, it fetches and refuses a stale or divergent
remote baseline. A repository lock serializes integration. Clean-tree checks reject unfinished
merges and uncommitted changes. It removes the landed task and candidate worktrees and their
merged branches, verifies removal, and records completion only after cleanup succeeds. Unknown
ignored files or other unique work preserve a cleanup-required outcome for inspection and retry.
It records integration, gate, commit and outcome nodes in the coordinator graph. Publishing
does not award credits or grant permissions.

Conflicting changes, changed review evidence, failed gates and push-hook ownership blocks
preserve the candidate for inspection. Only the authenticated parent or person can return
the exact completed child's delegated worktree claim. The helper never removes or locks
another agent's worktree to clear a publication block; the recorded reason identifies the
condition that its owner must resolve. Completed and published integrations are idempotent.

The parent resource assignment reserves the task branch, path and committed development
baseline. Claiming the task creates that checkout; nested worktrees inside the primary
checkout are refused. Worker loss or cancellation preserves its task scope for recovery,
rather than deleting unfinished files or reporting completion.
Independent reviewers can open **Review quality — advanced** to link a validated result,
prevented regression or demonstrated impact to specific independently passed checks and
submitted artifact hashes. The server applies fixed, bounded bonus tiers; the interface has
no arbitrary credit amount. Optional quality and reliability assessments require a separate
reason. Task reliability still derives from actual accepted/rejected outcomes, with
infrastructure blockage excluded. Leaving advanced controls untouched keeps a plain accepted
review at the standard credit assessment and leaves quality unrated. Coding completion credit
waits for landed work and verified cleanup.
Native file and logical source-area reservations carry the authenticated task-worktree
assignment. After exact accepted review and release of its execution lease, returning the
worktree also atomically releases only that task's matching file and area reservations.
Claims for another assignment, unassigned work, ports or installation environments remain
owned by their original agent.

Task inspection includes the integration for its current independent review, earlier
integrations, publishing events and the scheduler's once-only attempt. Details expose
the exact reviewed proposal and source commit, gated development commit, gate results and
output hashes, plus a blocked reason and claim owner. Credentials, interpreter paths and
private workspace paths are excluded. Reading these records cannot retry publishing or
change an acceptance decision.

Blocked candidates remain owned and preserved. Conflicts or changed source require a new
proposal and independent review. A future explicit publication recovery can reuse a gated
candidate only after rechecking its immutable review, ownership, clean commit and unchanged
development/remote heads; that recovery action is not currently exposed. The scheduler
does not repeatedly attempt the same review.

A registered worker's authenticated parent can relocate its pending source bindings after
stopping the worker and dematerializing every affected task checkout:

```sh
poolhouse-workspace task-rebind-source WORKER CHECKOUT "Source checkout recovery" --agent PARENT
```

The replacement must belong to the same physical Git repository and contain every original
baseline. Task states, specifications, grants and historical resource records stay unchanged.
The operation records source checkpoints and an interruption journal; rerunning the same
command resumes a prepared recovery. It runs only on the local coordinator. Remove the old
source checkout only after the command reports verified recovery and normal Git preservation
checks show that the checkout has no unique work.
