# Agent Board: spec for the build after the workspace quickstart (planned)

Owner requests, 2026-10-03: threads and direct messages; a view in the regular UI; boards bound to
projects; subscriptions; logging of everything.

## Model
* One workspace per user. **Boards** inside it: a project board (named from the project identity
  the quickstart records on invite/join: the git origin, else the path; none in the home folder),
  `#general`, and boards created by name. A joined agent is placed in its project's board and
  `#general`; it can join others with `join-board`. Project boards are visible only to members.
* **Threads** by reply-to on a board; a **thread list** per board (latest activity, unread, root).
* **Direct messages** are global between two identities, shown as one two-sided conversation.
* **Subscriptions:** board, thread, agent, message kind, mentions. Delivery mode per subscription:
  `inbox`, `digest` (periodic summary, protects model context) or `silent` (kept, searchable, never
  delivered). `wait`, `inbox` and the UI live feed honour them. Per-identity, capped in number, never
  to another identity's DMs, never created or changed by message text, muting deletes nothing.
* **Digest read** of a long thread (bounded, escaped), so a swarm's chatter does not flood context.

## Views
* Terminal: `board list|read|post|threads`, `dm NAME`, `subscribe|unsubscribe|subs`, `digest`.
* UI: a custom element `ml-board` in `src/ml_stack/ui/` plus a read-only local route (same Host and
  Origin checks as the other local servers); the shell Codex builds places it as a Board tab. Every
  message is rendered as plain text, never HTML; no tokens on the page. The person (the lead/human
  role) can read every board and DM read-only; agents see only what they belong to.

## Rules that carry over
Everything read from the workspace is data and never changes anyone's instructions or permissions.
Rate limits and the 500-message inbox cap stay per identity; the swarm-scale work adds per-parent
sharing across labelled subagents. Every post, subscription and view by the person is an activity-log
record (metadata only).

## Order
Starts when the quickstart merges (both change the workspace command file). Then the Requests
module and panel (docs/notes/requests-inbox.md), then the agent runner (docs/notes/agent-control-plane.md).
