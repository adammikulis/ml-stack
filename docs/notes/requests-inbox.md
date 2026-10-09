# One place for every request that needs a person (planned, not built)

Owner request, 2026-10-03: all requests show up in the UI, in one place.

## What counts as a request
Anything that waits for a person: a tool-call confirmation (including the classifier's reason), a
hold window with its countdown, a quarantine release, a host approval, a fleet join or pairing
request, a memory "remember this?", a keystore unlock, an agent asking to be connected, a rule
suggestion. Each request has: id, kind, who or what raised it (agent and project), a plain-words
reason, what each choice does, an expiry, and a state: pending, approved, denied, expired or
cancelled.

## Design
* One module raises and answers requests (`poolhouse.requests`); every component that blocks on a
  person today raises a request instead of owning its own prompt. The terminal prompt, the UI and
  the single desktop dialog are three ways to answer the same request; the first answer wins and the
  others show it as resolved. No component may open its own prompt.
* Human-only stays human-only. A request can be created by an agent but only answered by a person:
  through the terminal at a tty or the UI's browser session (same Host/Origin and session checks as
  the other local servers), never by a token, a tool, a workspace message or a model. Answering
  paths for human-only actions (release, approve host, mint) keep their own re-check.
* Choices per kind: allow once, always (saves a rule, not offered for destructive or unsure
  calls), never, deny; a hold shows Cancel; a release shows what it will unblock.
* Everything shown is untrusted text: escaped, bounded, fenced; reasons come from fixed sentences
  and the classifier's labels, not from the call's own words.
* Persisted in the per-user encrypted store with the other state; every request and answer is an
  activity-log record; expired requests resolve as denied, never as approved.
* Views: a Requests panel in the regular UI (pending first, history below, filter by agent,
  project, kind), the count in the status line and the chip, `poolhouse-requests list|answer`
  for the terminal. A swarm's requests are grouped by agent and project so twenty agents asking
  does not look like twenty popups; "approve all of this kind" is offered only for non-destructive
  kinds and always shows what it covers.
* pcb-engine: its confirmation cards are the same concept; once this exists its UI can show the
  same list or link to it.

## Order
1. Classifier lands (what needs approval and why).
2. Request module + terminal + the chat confirm path moved onto it.
3. Move sentinel dialog, onboarding requests, host approval and memory prompts onto it.
4. UI Requests panel (next to the agent Board in the shell Codex builds).
5. Hold windows and undo (docs/notes/undo-mode.md) become request states.
