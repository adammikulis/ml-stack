# Project task workers

Coding workers consume structured TaskBoard assignments. Board chat messages discuss work;
they do not start jobs. An authenticated registered parent runs
`ml-stack-workspace agent schedule WORKER_NAME --agent PARENT`, and the maintained coding
launcher runs `ml_stack.workspace.localcoding` for the saved worker identity.

The scheduler renews the existing delegated seat within the parent's authority and configured
TTL. It selects queued tasks whose dependencies are independently accepted and whose model
and capability requirements match the actual broker grant and configured worker. Blocked tasks
require authorized resumption; the scheduler does not retry them automatically.
Person-created tasks use the same worker eligibility rules as TaskBoard claims. An explicit
project must match a current person-set worker or inherited parent grant. Empty project metadata
uses the trusted configured worker source; it does not grant access to a path in task text.

Each task gets a claimed branch and sibling git worktree beneath the saved source worktree's
parent directory. Its immutable baseline is the source HEAD. Uncommitted changes in other task
worktrees remain separate. The coordination graph stores the owner, task, worker, source path,
execution path, branch, baseline, device account and managed resource allocation. Native tools
receive the trusted allocation's execution path; task descriptions cannot select another path.

The worker reloads its private token before each task transition, heartbeats its task lease,
and cancels its owned native process when stopped, its wall limit expires or heartbeat fails.
Checkpoints record successful native tool returns, the repository commit and runtime/interpreter
provenance. The native worker commits its changes before answering. Uncommitted changes block
submission and remain available for review. The runner commits a report and binary patch covering
the task baseline through the native commit, including deleted files. Its proposal records the
final clean commit and hashed artifacts for independent review. A final answer alone does not
award reputation or accept a task.

Coding workers currently use the Claude harness. Its maintained coding profile caps native
turns with `--max-turns`; output tokens per response use `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, and
effort stays within the saved worker ceiling. These are not a total task token budget. The
TaskBoard accepts model, wall time and retry limits; memory, context, total-token and custom
step requirements remain unsupported and are rejected by its schema. Memory admission and
server context remain the existing managed broker's responsibility. Linux tests remain on hold.

The native controls use the maintained [Claude CLI reference](https://code.claude.com/docs/en/cli-reference)
and [environment-variable reference](https://code.claude.com/docs/en/env-vars).

Stop ends only the saved worker process and its verified broker holder. It retains the
private identity, token, preferences and device account. Start reuses that identity
when changing models; identity revocation is a separate operation.

Repository issue intake creates project tasks with source-revision keys,
acceptance criteria and coding limits. Board chat replies cannot complete those tasks.
Blocked intake stays blocked until an authorized review changes its condition; issue
edits and elapsed time do not retry a blocked task. The scheduler selects eligible
queued tasks and assigns an isolated worktree before native execution.

Parent scheduler and repository intake poll independently accepted worker reviews through
the maintained development integration helper. A graph attempt binds the exact review
hash before publication starts. Published, blocked and interrupted attempts are not
repeated automatically; the owner inspects a blocked or interrupted candidate before
requesting further integration. Proposal submission alone never starts publication.

An obsolete repository issue can be excluded by the person or registered worker parent:
`ml-stack-workspace agent supersede-issue WORKER --issue NUMBER --reason "Current owner decision" --agent PARENT`.
The graph records the authenticated decision and its issue relationship. Subsequent
issue edits do not remove that decision; nothing is posted to GitHub.

The person or actual registered parent can explicitly recover a blocked legacy issue
projection with `agent resume-issue WORKER --issue NUMBER --reason "Changed condition" --agent PARENT`.
Recovery preserves the failed attempt and reason as linked graph nodes and permits
one new intake attempt. This does not resume project tasks or change their retry
budget; blocked tasks use the existing task queue recovery path.
