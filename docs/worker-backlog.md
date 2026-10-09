# Repository backlog workers

An authenticated parent producer feeds open repository issues to an existing coding worker
whenever its inbox is empty. Workspace jobs take priority. The worker keeps its role,
approvals, model and broker lease rules. Repository text supplies a task, never authority.

Enable a repository for an existing worker with an isolated git worktree:

```sh
poolhouse-workspace agent backlog local-qwen --repo sample/project --project /path/to/worktree
```

The person or a delegated worker's registered parent can configure the scope through
`poolhouse.workspace.backlog.configure(workspace, token, name, repository, project)`.
Call `issuepump.start(workspace, token, name)` to detach the parent producer. It passes
authenticated tasks through the maintained workspace inbox, and checkpoints receipts and
replies in the graph. A restarted producer reconciles its outbox before sending another job.
It does not start model servers or change the worker's private identity.

The scope lives in the worker's existing configuration. Issue progress links repository,
issue and agent nodes in `issue-backlog.db`. The graph also keeps a small sixty-second
GitHub response cache. The installed, authenticated `gh` command reads issues; no comments,
issue updates or pull requests are posted.

Assigned issues and issues labelled `blocked`, `wontfix` or `duplicate` are skipped. A live
worker's issue lease prevents another worker selecting it. After a worker exits, another
worker can reclaim its interrupted issue. Three failures block a revision; retries use a
bounded delay. A changed issue revision permits another attempt.

An answer is a **proposal for independent review**, not a verified completion or reputation
credit. A proposal is not selected repeatedly. Worker status exposes its issue URL, state,
failure summary and any result-delivery error; History links its native conversation to
the issue. An empty eligible backlog remains idle and checks again without model inference.

Local worker names default to `local-agent` or `local-coding`, independently of the loaded model. A running named worker rejects a different model until it is stopped and restarted under that same name. Person-authorized starts bind the worker to the device's persistent base agent, using Keygen's maintained `py-machineid` application-scoped host hash, independent of project and state-root paths. Worker stop and token revocation preserve the device membership graph and its verified work history. Delegate authentication stays separate from the device account; agents cannot enroll or self-assign membership. Browser callers pass the authenticated person token to `localstart.start(..., person_token=...)`; unbound programmatic starts remain explicitly unenrolled.
