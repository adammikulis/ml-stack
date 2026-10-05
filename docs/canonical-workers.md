# Canonical coding workers

Coding workers consume structured TaskBoard assignments. Board chat messages discuss work;
they do not start jobs. An authenticated registered parent runs
`python -m ml_stack.workspace.task_scheduler PARENT WORKER_NAME`, and the maintained coding
launcher runs `ml_stack.workspace.task_worker` for the saved worker identity.

The scheduler renews the existing delegated seat within the parent's authority and configured
TTL. It selects queued tasks whose dependencies are independently accepted and whose model
and capability requirements match the actual broker grant and configured worker. Blocked tasks
require authorized resumption; the scheduler does not retry them automatically.

Each task gets a claimed branch and sibling git worktree beneath the saved source worktree's
parent directory. Its immutable baseline is the source HEAD. Uncommitted changes in other task
worktrees remain separate. The coordination graph stores the owner, task, worker, source path,
execution path, branch, baseline, device account and managed resource allocation. Native tools
receive the trusted allocation's execution path; task descriptions cannot select another path.

The worker reloads its private token before each task transition, heartbeats its task lease,
and cancels its owned native process when stopped, its wall limit expires or heartbeat fails.
Checkpoints record successful native tool returns, the repository commit and runtime/interpreter
provenance. A final answer produces hashed file artifacts and a proposal awaiting independent
review. A final answer alone does not award reputation or accept a task.

Canonical coding currently uses the Claude harness. Its maintained coding profile caps native
turns with `--max-turns`; output tokens per response use `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, and
effort stays within the saved worker ceiling. These are not a total task token budget. The
TaskBoard accepts model, wall time and retry limits; memory, context, total-token and custom
step requirements remain unsupported and are rejected by its schema. Memory admission and
server context remain the existing managed broker's responsibility. Linux tests remain on hold.

The native controls use the maintained [Claude CLI reference](https://code.claude.com/docs/en/cli-reference)
and [environment-variable reference](https://code.claude.com/docs/en/env-vars).
