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
checks covering every acceptance criterion before an outcome becomes completed. Reviewers
need existing project permissions, cannot review their own work, and nonhuman reviewers
cannot grade another worker from their own physical-device account. Rejected work and
infrastructure blockage are separate outcomes.

Expired working leases require **Recover expired lease**. Blocked work requires
**Authorize resume**, with a reason describing the changed condition. These actions preserve
checkpoint history and do not grant resources; the next worker claim still validates a live
allocation. Recovery consumes the configured retry budget. Supported limits currently cover
model choice, wall time and retries. Memory, context and token limits require maintained
broker/runtime enforcement before they can be represented as guarantees.

Independent review automatically records the canonical outcome in the encrypted work ledger.
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
