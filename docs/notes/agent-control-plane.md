# Driving agents entirely from the ml-stack UI (planned, not built)

Owner goal, 2026-10-03: a person should be able to drive agents entirely through the ml-stack UI,
without opening an editor.

## Pieces that exist or are in flight
Workspace identity and messaging (connect/join, labels, delegation), the agent Board, roles and
Always/Never rules, the destructive-action classifier, hold windows and undo (planned), the
Requests inbox (planned, docs/notes/requests-inbox.md), the activity log, the Broker for model
leases, the test worker queue, the sentinel and memory.

## Missing
1. **Agent runner:** start, supervise, resume and stop agents, each in its own worktree with its
   workspace identity and role wired in. Adapters: our `ml-stack-chat`; Claude through its Agent
   SDK; Codex through its app-server protocol.
2. **Live transcript** per agent: tool calls and results as they happen, with a stop button.
3. **Review view:** per-agent diffs, branch and worktree status, gate results from a ledger, and a
   merge action that follows the existing rules (full gate on the merged tree, human decides).
4. **Task board:** work items with owner, state and acceptance command.
5. **Agents' own approval prompts routed into the Requests inbox** (Claude's permission-prompt
   hook, Codex's approval modes), so approvals are in one list.

## Security rules for the control surface
Same local-server protections as the others (Host/Origin, browser session, no tokens in pages,
plain-text rendering of everything agents write); human-only actions answered only by a person; each
launched agent in a sandboxed worktree with a scoped identity and role, never the person's full
rights by default; every launch, stop and answer is an activity-log record; a runaway agent can be
stopped from the UI even if its process is wedged.

## Order
1. Finish: workspace quickstart, activity log, swarm scale, classifier.
2. Requests inbox, then the Board, in the UI.
3. Agent runner with one adapter (our chat agent) and an Agents panel: start, stop, transcript.
4. Claude and Codex adapters, then the review and task views.

## First runner milestone: start a local model from the UI and have it contribute (owner priority)
From the UI: pick a downloaded model (fit meter, a recommended default: the best MoE ranked for
agents, thinking off, a warning on quants known to be slow on Metal), a role (reader, operator,
runner), a project and a task or "pick up work from the board", then Start. ml-stack leases the model
through the Broker, creates a fresh worktree and branch, connects the agent to the workspace with its
own identity, and launches it there with the board brief. The adapter reuses `ml-stack-claude MODEL`
(Claude Code on a locally served model, in the settings it scored best with) so the local model has
real coding tools; `ml-stack-chat` has none and only operates ml-stack. Its approval prompts route to
the Requests inbox; its transcript streams into an Agents panel with a stop button; its work stays in
its worktree and branch, tests go through the shared queue, and merging needs the gates and the person.
Expectations: small local models did not finish a board end to end in the live runs, so scope the first
tasks (small fixes, tests, docs, triage, review); one big model at a time because it holds most memory.
