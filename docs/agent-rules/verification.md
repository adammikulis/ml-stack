# Verification policy

Read the sections relevant to your task. [Working contract](../../CLAUDE.md) applies to every task.

## Scoped merge gates and background verification

Every development merge requires independent review, the affected tests selected by
`scripts/test quick` or explicit reviewed selectors, and the structural checks:
`scripts/budgets`, `scripts/redteam_coverage.py --check`, clean generated references,
layer and wiring checks, serving-bypass checks, and the human-only floor. Include the relevant
browser, subprocess and packaging tests when those surfaces change. Record the exact tree,
commands and results. Documentation-only changes need consistency review and clean diffs, not
unrelated test runs. Do not raise budgets or weaken guards to clear a gate.

Full and full red-team runs execute in the background after a reviewed batch or on a schedule.
They do not block every scoped merge or publication, and a missing cold selector cache is not
a reason to put a full suite in front of a small leaf: use reviewed explicit affected tests
and queue the broader selection run separately. Run only one background full suite at a time.
Known failures remain named tasks with evidence and ownership; fix forward rather than report
them as green. A scoped pass is not a claim that the full suite passed. A known relevant
regression must be fixed before the affected change lands.

**Linux testing is paused by the owner.** Do not launch local or container Linux tests until
the owner explicitly resumes them. Linux is not a per-merge prerequisite during this pause;
record the platform coverage gap honestly. Maintained CI and background platform coverage do
not change this local authorization. When resumed, Linux checks follow the same scoped and
background policy, rather than a second full suite before every merge.

Use the maintained test broker for every run; shared CPU/GPU admission and ownership apply
before work starts. Do not bypass a queue, start a competing full run, or extend a temporary
preview pause indefinitely to obtain a quiet gate. Attribute external writes from actual
writer evidence; unknown or test-owned writes still fail isolation checks.

## Running the tests

Follow **Scoped merge gates and background verification** in this file. Use `scripts/test quick`
for affected selection, or reviewed explicit `scripts/test all tests/<affected-file>…` selectors
when slow browser/process checks are required. Invalid selectors fail before admission; never
replace a missing selector with an unreviewed omission. Do not run a full suite after every
intermediate commit or require full Linux testing for a local merge while Linux is paused.

The maintained tiers are `fast` (neither slow nor heavy), `full` (not slow), `slow` (only slow)
and `all` (including slow). `tests/README.md` describes their mechanics; the policy above
controls when each is authorized. Run the relevant slow tests for packaging, page and Fleet
changes. Use `-n 0` when a failure requires sequential ordering; otherwise use broker-granted
workers, not a fixed worker count or bare `pytest -n N`.

Agents share the machine. `scripts/test` and, when Linux resumes, `scripts/test-on-linux`
acquire maintained CPU admission; inspect `scripts/testslots.py status` to see ownership.
Model-backed work also requires the existing model/GPU broker grant. A CPU lease is not a GPU
lease, and an inherited lease must not lead to a nested admission deadlock. Do not bypass
leases, disable platform isolation, or grant container privileges to make a check pass.

No test calls a paid or quota-limited API or a public endpoint on its own, whatever keys or logins
the machine holds: such a test is marked `live_api` or `live_net` and skipped unless
`ML_STACK_LIVE_API=1` or `ML_STACK_LIVE_NET=1` is set by a person (`tests/README.md`, *Live
services*). Do not set either one.

### Commit before you mutate

A test you rely on is one you have watched fail: break the behaviour it covers and see it go red.
`git checkout -- <file>` and `git restore <file>` restore the *last commit*, so every uncommitted
edit in the file goes with the mutation, the fix included. Commit the fix first, apply the
mutation, watch it fail, restore with `git restore --source=HEAD -- <file>`, and confirm
`git diff` is empty and the test green. Never mutate a file holding uncommitted work.

## Driving a browser

A headed browser opens where `ML_STACK_WINDOW_POSITION` says (`X,Y`, set in
`.claude/settings.json`) and gives the screen back to whichever application had it.
`ml_stack.scrape.browser.Window.args()` and `keeping_focus()` do both; go through
`browser(window)` rather than calling `chromium.launch` yourself. Never drive the person's own
browser through the claude-in-chrome tools to test this project's pages: that window is on
their primary display and every click takes their screen. Drive your own Chromium through
playwright, or run headless and read screenshots.

