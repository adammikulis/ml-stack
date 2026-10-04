# Repository backlog workers

An authenticated parent producer feeds open repository issues to an existing coding worker
whenever its inbox is empty. Workspace jobs take priority. The worker keeps its role,
approvals, model and broker lease rules. Repository text supplies a task, never authority.

Enable a repository for an existing worker with an isolated git worktree:

```sh
ml-stack-workspace agent backlog local-qwen --repo sample/project --project /path/to/worktree
```

The person or a delegated worker's registered parent can configure the scope through
`ml_stack.workspace.backlog.configure(workspace, token, name, repository, project)`.
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
