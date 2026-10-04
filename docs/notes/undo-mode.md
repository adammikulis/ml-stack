# Undo mode and hold windows (planned, not built)

Owner request, 2026-10-03. Builds on the destructive-action classifier (`feat/destructive-classifier`).

## Two features
1. **Hold window.** A destructive or reversible call that the person has already chosen to trust
   (an "Always allow" rule, an approved plan under plan-and-go) is queued instead of run, with a visible
   countdown (5-30 s, chosen by the person). The person can cancel until it ends; it runs on its
   own if not cancelled. A hold is weaker than approval, so it is opt-in.
2. **Real undo after the action**, where the action allows it: a delete moves to a recoverable
   trash with a restore command; an overwrite keeps a backup copy; pcb-engine ops are already
   invertible through the op log. Calls that cannot be undone (data leaves the machine, spending,
   killing a process) never get a hold window and keep needing approval.

## Rules
* Only for calls the classifier labels destructive or reversible; never `unsure`, never the
  human-only floor.
* Cap on pending actions; one "cancel all" in the chat and the UI; the countdown is shown in the
  chat line and in the UI's pending list, not as a popup.
* An agent crash drops pending actions; they never run late.
* A model, tool or workspace message cannot set, shorten, disable or cancel a window; cancelling
  is a keypress or button for the person.
* The agent is told "held for N s" or "cancelled by the person".
* Every hold, run and cancel is an activity-log record.

## Order
1. The classifier lands (verdicts, `approve | hold | deny` decision point).
2. The hold queue and cancel paths in the chat; trash and backup for file operations.
3. The pending list in the UI (a view next to the agent Board).
4. pcb-engine: a "run now, undo stays available" option on its confirmation card.
