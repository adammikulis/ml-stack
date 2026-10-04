# The self-improving loop: the goal, and the rails it has to run on

The owner's goal: a loop in which a local model (Qwen, a mixture-of-experts model, thinking off by
default) working through ml-stack improves ml-stack, and the next round starts from the better
stack. Nothing here is built as one piece yet; this note is the contract that each piece must
meet, so that the loop can be started later without removing a safety rail to do it.

## The loop

1. **Pick its own target, no approval needed.** The person approved the *kind* of work once, not
   each target. It takes the highest of: an open issue with a re-runnable acceptance check; a
   failing or skipped test; a documented gap in the docs/notes and HANDOFF lists; a benchmark that
   regressed; a budget that can fall (long functions, local imports, ruff sites); an uncovered or
   partial red-team surface; a surviving mutation (`scripts/gates/survivors.txt`). When that list is
   empty it falls back to behaviour-preserving refactors and bug fixes (what the mutation and
   complexity tools point at), still inside the rails below. It claims the target on the workspace
   so two rounds never take the same one.
2. **Work in its own worktree and branch** (`ml-stack-workspace agent start`, role
   `approve-first` or `plan-and-go`), announcing on the workspace.
3. **Change code and tests**, run only the tier that matches (`quick`, the files' own tests).
4. **Measure against a fixed judge**: the gates, the budgets ratchet, the red-team coverage
   ratchet, JevBench for deciders, and the speed/memory benchmarks. The judge is outside the
   agent's write access.
5. **Report the result as evidence**, never as a claim: the command and its output, the
   before and after numbers, and the mutation check (break the change, see the test fail).
6. **Landing has two tiers, so nobody has to be around.** A change that touches no protected path,
   leaves every ratchet equal or lower, passes the judge and its own mutation check merges by itself
   into a staging branch (`loop/integration`), never into `0.2dev`, never pushed. A change that
   touches a protected path, or that the judge cannot fully measure, is not merged: it waits as a
   request in the Requests inbox with the diff summary and the judge's numbers. The person later
   reviews the staging branch as one batch and moves the dev branch, or sends rounds back; the
   loop keeps working meanwhile and never blocks on an answer.
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
