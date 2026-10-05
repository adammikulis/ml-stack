# A local model as a workspace agent

One command starts a downloaded model, joins it to the workspace and runs its loop detached:

    ml-stack-workspace agent start [--model auto|ID] [--name NAME] [--role ROLE] [--project PATH]
                                   [--effort off|low|medium|high|auto] [--max-effort LEVEL]
                                   [--orders-from NAMES] [--no-wait]
    ml-stack-workspace agent list
    ml-stack-workspace agent stop NAME

`start` and `stop` are for a person at a terminal (a process an agent started, or one with no terminal,
is refused). The Agents panel does the same from the browser: `ml-stack-workspace board-serve` prints
a link, `/agents?session=...`, that sets the browser session cookie the start and stop routes require.

## What `start` does

1. **Model.** `auto` is the best already-downloaded mixture-of-experts Qwen model ranked for agent
   work on this machine; Flash-Next is never chosen. With none downloaded it prints the one command
   that fetches one (`ml-stack-models fetch REF`, or the `find` that locates it) and downloads nothing.
   A model rated red for this machine's memory is refused in one line with the smaller downloaded
   choice to pass as `--model`.
2. **Identity.** The agent is named `local-` and the model's short name (or `--name`) and gets the
   standard agent role. Its token is minted directly, as `setup` does, written to
   `tokens/NAME` (0600) and never printed. Its project board is placed as for a joined agent.
3. **Loop.** `python -m ml_stack.workspace.localloop NAME` runs detached (`ml_stack.jobs.detach`), its
   pid and start time are recorded, and its log is `local-agents/NAME.log`. The same name again
   reports the running agent and changes nothing.
4. **Model lease.** The loop leases the model from the broker (memory admission and the queue are the
   broker's) on one slot using the maintained harness profile: measured settings, the requested
   context, q8 KV cache, an installed matching MTP head, and the maintained patched chat template.
   Installed Hub references select the cached snapshot path without downloading. A compatible
   managed server used by Chat or Coding is reused across ports; different head, template, cache,
   or context requirements remain distinct.
   A refusal ends the agent with `failed` and the broker's one-line reason.

`stop` stops the owned process and releases its lease. Its saved identity, token, settings and
device account remain available for the next start.

## Taking and giving orders

The loop waits on the agent's inbox through the workspace's wake pipe. A message is acted on only when
it is a clear `task` or `question` (not held in quarantine, not to `*`; on a board it must `@name`
the agent) from a live identity that is the person (`human`), a `lead`, an agent named with
`--orders-from` (default `claude-code`; any identity registered under a named id is trusted, so name
only ids you control), or a delegate of one of those. Everything else is counted as information and
never reaches the model. The task text goes to the model fenced as data. The result is sent back on the
thread: `answer` when the model called `done`, `status` when it stopped.

Under its role the agent has ml-stack's own tools (`ml-stack-chat`'s) plus `workspace_roster`,
`workspace_thread`, `workspace_send` (kind `task`, `question` or `status`, up to five a task, no `task`
after reading an agent it does not obey; not offered in the role that only reads) and `set_effort`.
Anything the role says needs a person goes to `localtools.ask_a_person`, which answers no: nobody can
answer in a detached process. That function is the one seam the Requests inbox replaces. The agent
cannot change roles, rules, quarantine, approvals, its ceiling or its caps; it holds no credential.

Caps per task: 12 tool-calling rounds, 30 tool calls, 24 model calls, 600 s, and the kill switch
(the stop file, checked before every model call). Workspace rate limits apply to what it sends.

## Effort and the prompt cache

Effort is compute only: `off` (no thinking, 2048 tokens a reply), `low`, `medium`, `high` (16384).
The default is `off`. The model may raise or lower its own effort with `set_effort(level, reason)` up
to `--max-effort` (default `medium`); above the ceiling is refused with the ceiling in the message.
The change applies from the next task, so a task's prompt is never rewritten; it is recorded in the
workspace audit log. `auto` picks per task from a rule table (status or lookup: off; plan, diagnose,
review, design: medium; otherwise low), never above the ceiling.

Within a task the prompt only grows: the system message and tool list are fixed bytes, new turns are
appended, and thinking, sampling and token ceiling stay constant. Requests go to slot 0 with
`cache_prompt`, and the server runs with `cache_idle_slots`, so the prefix survives between tasks.
No context trimming is done; the task caps keep a task inside the 32768-token context.
`tests/test_workspace_local_agent.py::test_each_turn_extends_the_last_prompt_byte_for_byte_and_tasks_share_their_prefix`
checks this against a fake server.

## Profiles

`--profile chat` (default) serves 32K with the caps above. `--profile coding` is 256K (`--ctx 256k`
accepts k and K) with Qwen3.8-27B (Q4_K_XL first; `ml-stack-serve memory` rates it 27.2 GiB at 256K,
q8_0 cache, MTP head shared) and caps of 60 rounds, 150 calls, 120 model calls and an hour. A coding
agent runs on the Codex harness through `ml_stack.coding.launch_coding_agent(model, role, project,
harness='codex')`; until that lands `start` prints the one command to run (`localharness.stub_command`).
Flash-Next is used only when named with `--model`. Before starting, the memory estimator checks the
context; when it does not fit, `start` says the longest context that does and prints the person-only
`ml-stack-serve memory --for ... --apply`. Past 85% of the context the chat loop drops whole oldest turns
down to 50%, leaving one fixed marker; nothing kept is edited, so the cached prefix survives.


Coding agents retain their registered workspace identity and process authorized inbox tasks
serially through the maintained native harness. The parent worker reads task data and posts
threaded results; the model does not need workspace shell access to finish or report a job.
Each task has a graph conversation, native session, bounded runtime and cancellable process.
The task role and native sandbox remain in force. An idle worker waits without running inference.

The local-agent routes accept POST `/agents/pause` and `/agents/resume` with `{"name":"NAME"}`
under the same person-session guard as start/stop. Pause takes effect before the next task; an
active task finishes first. The listing exposes `identity`, `paused`, `state`, `tasks` and the
latest message. Stopping a delegated worker revokes its child token, preserving its parent.
`agent.task` activity links the workspace message, graph conversation, project and native session;
task completion is separate from independent verification and reputation credit.

Coding start accepts `harness` (`codex` or `claude`), also available as CLI `--harness`.
The installed Claude Agent SDK's bundled executable is reused when `claude` is absent from PATH.
The person-authorized start registers one worker identity before launching its inbox loop;
native tasks reuse that identity rather than minting a new agent for each job.
