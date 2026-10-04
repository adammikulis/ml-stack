# The self-improving loop: the goal, and the rails it has to run on

The owner's goal: a loop in which a local model (Qwen, a mixture-of-experts model, thinking off by
default) working through ml-stack improves ml-stack, and the next round starts from the better
stack. Nothing here is built as one piece yet; this note is the contract that each piece must
meet, so that the loop can be started later without removing a safety rail to do it.

## The loop

1. **Pick a target** from a ranked list the person approved (an issue with a re-runnable
   acceptance check; a benchmark that regressed; a budget that can fall).
2. **Work in its own worktree and branch** (`ml-stack-workspace agent start`, role
   `approve-first` or `plan-and-go`), announcing on the workspace.
3. **Change code and tests**, run only the tier that matches (`quick`, the files' own tests).
4. **Measure against a fixed judge**: the gates, the budgets ratchet, the red-team coverage
   ratchet, JevBench for deciders, and the speed/memory benchmarks. The judge is outside the
   agent's write access.
5. **Report the result as evidence**, never as a claim: the command and its output, the
   before and after numbers, and the mutation check (break the change, see the test fail).
6. **A person lands it.** The merge is a request in the Requests inbox with the diff summary,
   the judge's numbers and what is touched. Approve, send back, or reject.
7. **Record what was learned** in project memory (encrypted graph, project scope) so the next
   round does not repeat a dead end; failed attempts are recorded as failures with the reason.

## Rails that must hold before the loop runs unattended

* **The judge cannot be edited by the thing being judged.** The gates, budgets, coverage map,
  red-team tests, scoring code, benchmark sets and this contract are write-protected from the
  loop's worktrees (a path list the loop's tools refuse; changes to them are human-only, like a
  role or a saved rule).
* **The human-only floor is untouched**: no change to `sentinel/`, `keystore.py`, `person.py`,
  `roles`/`rules`, `requests/`, the destructive-action classifier or the human-only tables lands
  without a person reading the diff; the loop may propose, never merge, those.
* **Tighten-only for safety code**: a change that loosens a guard, raises a budget or removes a
  test is refused by the judge, not argued about.
* **Hold-out evaluation**: improvements are scored on items the loop never saw (the deciders'
  public set versus a private one), so it cannot overfit to the judge.
* **Bounded**: steps, tool calls, wall-clock and memory per round are capped by the person;
  one round at a time; a kill switch (`agent stop`) that always works; a daily budget.
* **No ambient authority**: the loop holds no credential, never touches the keystore, never
  pushes, tags or publishes. Network goes through the net pipeline allow-list.
* **Reputation and tripwires**: a round that edits a protected path, asks for a human-only
  action, or raises its own caps is stopped and logged, and the agent's reputation drops.
* **Everything is logged**: each round's prompt prefix is stable (cache-friendly), its tool calls
  go to the activity log (bodies excluded), and the result is in the hash-chained event log.

## Pieces that exist, and what is missing

Exist: worktrees and the workspace board; roles and rules; the destructive-action classifier;
the Requests inbox; the activity log; reputation; the red-team coverage ratchet and budgets;
decision models and JevBench; project memory.

Missing, in order: the local-model agent loop (`feat/runner`); the write-protected path list for
loop worktrees; a "round" runner that ties target, worktree, judge and the merge request
together; the hold-out evaluation sets; the daily budget; hold windows/undo
(`docs/notes/undo-mode.md`).
