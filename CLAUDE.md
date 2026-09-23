# Rules for working in this repo

## Comments and docstrings

Write what the code does. Never why it was written that way.

**Banned:** rationale, war stories, "deliberately", "on purpose", "the reason is", "this is what
X's lesson looks like", explanations of what would happen if the code were different, arguments
against changing it, anything that reads as a message to a future reader about a decision.

**Allowed:**
- A one-line docstring on a public function saying what it returns.
- A short docstring on a module saying what is in it.
- A comment only where the mechanics are genuinely non-obvious — an API that behaves
  unexpectedly, a magic constant, a workaround for a specific bug. One line.

Stale prose is worse than none: it is confidently wrong and nobody notices. If a decision needs
recording, it goes in a commit message, not in the file. Default to fewer words; if a docstring
is longer than the function, delete most of it.

## Commit messages

Start the subject with `feat:`, `fix:`, or `chore:`. release-please reads them: `feat` bumps the
minor, `fix` the patch, `chore` neither. A subject with no prefix is a change that never reaches
a release.

After the prefix, the subject says what changed. Nothing else. **Banned:** "actually", "real",
"finally", "now works", scare quotes, before/after contrasts, anything that editorialises about
the previous state or sounds pleased with itself. Write: `feat: native window instead of a
browser tab`. `chore: install CPU torch in CI`. `fix: remove the tier system`.

The body is for detail a reader would want later: what was wrong, what the fix is, what it costs.
Plain sentences. No war stories, no rhetorical questions, no lines that argue with a future
reader.

## Nothing is cemented until 1.0

This library is a work in progress. No prompt, schema, serving default, pin, name, signature or
file layout is frozen, and none is worth keeping in a shape that is wrong.

Never keep bad code to keep something green -- not a test, not a hash, not a budget, not a kept
benchmark, not a caller that would otherwise have to change. If the right shape breaks one of
those, change the shape and then fix what broke: update the fixture in the same commit so cause
and effect are one diff, say which kept runs stop being comparable, and write what needs
re-measuring where the next person will read it. **Banned:** a duplicate kept so an old path
still works, a wrapper preserving an old name, a branch for a caller nobody has, a module
boundary drawn around a hash, a number left in a document because re-measuring is inconvenient,
"we can't change that, it would invalidate the benchmarks".

**There are no users but the owner**, on any platform, unless he says otherwise. A requirement
that serves somebody hypothetical is not a requirement: a fallback to an older interpreter,
per-platform advice in an error path, a `--force` escape hatch, a migration for state nobody
holds, a softened refusal. When a decision narrows what is supported, write the code as though
the narrow thing is what this runs on, and delete whatever existed only to straddle. This is a
**stop, not a judgement call**: if you are about to write something worse to serve someone who
is not there, ask first rather than deciding it and reporting after. What is refused is the
*accidental* version. `tests/test_asking_is_the_same_asking.py` and
`graph/cache.py:fingerprint` catch bytes moving when nobody meant them to: they are detectors,
not vetoes. A red you can explain is a change; a red you cannot is a bug.

## The gates

**A budget is a debt, not a permission.** Every number in `budgets.json` is a count of violations
nobody has fixed yet. The file exists so the numbers can be driven down and so a new violation is
refused. There is no acceptable violation and no shape this repository has decided to live with.

**Nothing is ever grandfathered.** A violation that was here before your branch is your work the
moment you touch the file it lives in, and everybody's work the rest of the time. The files over
the size limit are not a baseline; they are files to split. **Banned as a reason to leave
something alone:** "pre-existing", "not introduced by this change", "already over budget",
"grandfathered", "out of scope for this branch", "the budget allows it", "it was already like
that". None of those says whether the code is right.

Leave every number you touched lower than you found it, and say by how much. A number may never
rise on its own, and no agent may raise one at all: `--allow-increase` is refused whenever
`CLAUDECODE` is set -- Claude Code sets it for every command it runs and a terminal sets nothing
-- so raising a number is the owner's, at his own terminal, like the push it resembles.

**And say what you found.** A tolerated violation you noticed and did not fix goes in the message
where you found it, not into a budget file for the owner to discover. Reporting a tolerance as a
good state ("holding at nineteen") is worse than not mentioning it.

Six checks refuse a change rather than describing what it should have been. Run them before you
ask whether the suite passes. `budgets.json` holds the highest count each shape in
`scripts/gates/` is allowed. `scripts/budgets` prints metric, budget, actual and delta, a total
under the table -- what the budgets add up to, what the tree holds, the distance between them
-- and every site that is over. `tests/test_budgets.py` fails when a number rises, and also
when it falls without being recorded, so a branch that lowers one runs `scripts/budgets
--update` and commits the file; `--update` refuses to raise a number. `scripts/budgets --show
METRIC` lists the sites, and `SKIP_BUDGETS=1` skips the pre-commit check.

Two entries are not ceilings. `floors-only-rise` holds `tests-collected`: it may not *fall*,
because a module that stops being collected leaves the suite still saying passed.
`scripts/budgets` returns 1 when the count is under the floor or any module failed to collect,
and `--update` refuses to record a count taken while one did. The count is only comparable where
every extra is installed, so on a machine missing one it says so rather than reporting a low
number. `module-skips` counts the other way in: a skip *outside* a test function takes a whole
file out of collection, which the floor cannot see coming. `ratchet` is the date and total the
debt is measured from, `RATCHET` at the top of `scripts/budgets` is the share it should fall by
each month, and the scoreboard prints what is due and whether the tree is on track. **It refuses
nothing** -- a repo-wide debt must never block an unrelated branch, which is the pressure that
gets gates gamed.

`scripts/hooks/budgets-only-fall` closes the other door: a staged `budgets.json` whose numbers
rose is refused whatever wrote it, and a metric dropped from the file counts as a rise, because
the next `--update` puts it back at whatever the tree holds. An agent is refused outright;
`ML_STACK_BUDGET_RISE=yes` is for the owner's own commit. The pre-commit chain is opt-in, so
`ci.yml` runs the same check with `--against` the pull request's base.

`tests/test_layers.py` sets out core, model, machine, graph, tools, and reads every import
including the ones inside functions. A package imports downwards, and sideways only when the
other does not import it back. `KNOWN` lists what still crosses; it only shrinks.
`tests/test_wiring.py` requires every package to be imported by something, back a console
script, or be named in `STANDALONE` with a reason. Code nothing calls is a gap to close, never
a reason to delete.

The size gates carry no number. `deep-files` (900 lines of Python) and `deep-components` (500
lines of the HTML, JavaScript and CSS a page is assembled from) set `HARD = True`, take no line
in `budgets.json`, and fail on a single finding. A file over the limit is a file to split,
never an allowance to record. `scripts/gates/duplicates.py` hashes normalised function bodies
and reports the pairs. `tests/test_gates_duplicates.py` names pairs that must still be found,
so a normalisation that quietly tightens is caught.

`scripts/mutate` samples functions out of `src/ml_stack`, changes one thing each -- a flipped
comparison, a swapped `and`, a negated or dropped branch, a constant return, an emptied body --
and runs the test files that name the module, in a copy of the tree made from `git ls-files`. It
is seeded by the commit, so a given commit samples the same functions every time. A mutation the
tests keep green is a survivor: those tests do not read what the function answers.
`scripts/gates/survivors.txt` holds the survivors already found, `mutation-survivors` counts the
rows whose function is still in the tree, and `scripts/mutate --verify` re-runs every row and
says which the tests now catch. A mutation that cannot change what anyone observes is recorded as
`equivalent` with the reason.

`mutation-survivors: 0` means no recorded survivor is left, not that the tests catch everything:
the run is a sample, so `scripts/mutate` writes a `campaign` row saying what it sampled and how
many functions it did not mutate, and `scripts/budgets` prints the same caveat under the table. A
deeper `--mutations` finds more mutations of a function already sampled, so a function with a row
is not one that is done. A module no test file names is measured by nothing:
`scripts/mutate --unmeasured` lists them, and each is a test file to write. `from
ml_stack.sources import rows` does not name `ml_stack.sources.rows`, so a well-tested module can
sit in that list.

`pyproject.toml` selects ruff's rules and pyright's checks, and `scripts/gates/` budgets both:
`ruff-blind-except`, `ruff-bugbear`, `ruff-security`, `ruff-other`, `pyright-errors`. Neither
tool is a dependency, so a checker that cannot find its tool prints why and its metric is left
out rather than counted as zero.

`scripts/hooks/pre-push` lets an agent push the development branch and nothing else, and lets the
owner's own push through: Claude Code sets `CLAUDECODE` for every command it runs and a terminal
sets nothing. The development branch is the one the primary checkout is on, and `main` is never
it. A push to `main` publishes every commit on it at once, so it is the owner's, and an agent
makes it only when asked, on a command that says so: `ML_STACK_PUSH_MAIN=yes git push origin
main`. That opener opens `main` and nothing else -- never a force, a deletion, `--all` or
`--tags`, and never another branch. The Bash guard refuses all of those before the command runs,
past any `NAME=value` written in front of it.

`scripts/hooks/claude-edit-guard` refuses, at the moment it is written, a function whose body
already exists elsewhere, a raw HTTP call, a docstring over twelve lines, a signature over eight
parameters, and a write that takes a file over its line limit or makes an already-over file
longer. It reads the two limits from the checkers. `.claude/settings.json` wires it and the Bash
guard; `MLSTACK_GUARD=off` turns both off. `scripts/install-hooks.sh` installs the pre-commit
chain.

## Anything a user reads

Release notes, the README, the interface, error messages. Write for someone seeing it for the
first time: they did not see the previous version, so telling them it is fixed only raises a
question they did not have. Describe what the thing does, not what it no longer does wrong, not
what changed, not how long it took — "trains across every machine on your network", never
"training now works". Before/after belongs in a commit message. No benchmark result goes in a
README. A measurement lives in a document that names its date, the command that produced it,
the store it read and the model it ran on, so a reader can repeat it; the README points at that
document and quotes no figure. A number without those four is not a measurement, it is a claim.

## HANDOFF.md

It lists what is still pending. Nothing else.

When something is done, **delete its entry**. Do not strike it through, do not mark it `[x]`,
do not move it to a "completed" section, do not leave a line saying it was finished. The same
goes for anything that turned out to be wrong: delete it, rather than adding a note that an
earlier entry was mistaken. A reader opens this file to find out what is left; anything already
dealt with is noise they read past. If a finished piece leaves something behind — a limit, a
gap, a follow-up — write that as its own pending entry, in its own words, not as a postscript
to the item that is going away. An empty HANDOFF.md is a good state: delete the file rather
than leave headings with nothing under them.

## What belongs here, and what belongs to the app that drives it

Anything true of any graph, model or scrape is this library's, with a test and a command; a line
that names one community, its vocabulary, or where its data lives belongs to the app (`~/ai_ceo`
is the first). An app holds only wrappers and one-line switches -- a script that calls one of
ours with its own arguments, an environment variable that flips one of our parameters, a lambda
that says where its graph keeps its pointers, its copy and kinds handed to our page. When an app
needs more than that, the missing piece is a command or a parameter here.

## The main session and its agents

The main session plans, writes the briefs, lands branches, and does what an agent cannot: a
decision that needs the whole conversation, a conflict between two agents' work, a check on a
claim before it is relayed. Everything else -- reading a subsystem, writing the code and its
tests, running the suite, merging its own branch -- goes to a subagent, one per branch, in its
own worktree. It does a piece itself only when handing it off would cost more: a one-line edit, a
change that needs what only this conversation knows, a thing an agent has failed at twice.

**Pick the model the task needs** -- not a rank to stay under and not a default. Haiku only to
explore: a search, a read, a lookup across files, never an edit, because a change it writes is
a change someone has to redo. Every read-only search is Haiku's unless the search itself needs
judgement a larger model has to supply -- say so in the brief when you go above it. Sonnet for
ordinary code with tests. Opus where the agent has to decide *what* the right change is, not
just make it: a module boundary, a failure that needs diagnosing, a measurement whose meaning
is in question. Fable sparingly, only for work whose difficulty is the thinking rather than the
typing and that Opus has fallen short on; it is the exception, never the default for work that
merely feels important. Getting it wrong costs in both directions: too small and the work comes
back wrong or comes back twice, too large and a rename that needed no judgement was paid for at
the rate of one that did. When a task turns out harder than the brief assumed, that is what a
second, better-modelled agent is for -- not a reason to send everything up front.

## Worktrees

Every agent works in its own worktree on its own branch — the main session as much as any
subagent it spawns; "I am the one driving" is not an exemption. Nobody edits the primary
checkout, and no two agents share a branch. Branch from the development branch the primary
checkout is on (`git branch --show-current` there says which it is), never from `main`.

```
git worktree add -b <branch> ../ml-stack-<branch> "$(git -C ../ml-stack branch --show-current)"
```

`main` is the release branch: a commit that arrives there is a commit queued to publish. Work
lands on the development branch, and promoting that to `main` is the owner's. Whoever made a
branch finishes it. Fetch, merge into the development branch, push it, then take the worktree
and the branch away:

```
git fetch origin
git merge --ff-only <branch>
git push origin "$(git branch --show-current)"
git worktree remove ../ml-stack-<branch>
git branch -d <branch>
git worktree prune
```

The development branch is pushed after every merge that lands on it, so the remote is never
behind what has landed; a work branch is not pushed. Whoever merges, prunes: a subagent that
lands its own branch removes its own worktree and branch, and when the main session merges it
does so in the same step. A merge is not finished until `git worktree list` shows only trees
with live work in them. Before removing a tree, check that it holds nothing unique -- unmerged
commits (`git cherry <dev-branch> <branch>`), uncommitted changes, or ignored state that is not
a rebuildable cache -- and never remove one while an agent is still working in it; if one is,
leave it and say so.

- The primary checkout is what *runs* — the editable install, a detached ingest, the page. It
  changes only by landing a branch: tests green on the branch, a fast-forward or rebase merge
  into the development branch it is on. Never an edit, never a `git add`, never a bare
  `git commit` there, and never a merge into `main`.
- A brief to a subagent names the worktree rule and gives it a branch (the Agent tool's worktree
  isolation does the first half). A subagent told to commit nothing still commits on its own
  branch by named files before it reports — staged-and-uncommitted is the state that leaks — and
  reports the branch, the commits, and the suite result on that branch.
- The main session lands each branch it asked for, or the agent does, but one of them does, the
  same day. A branch nobody lands is work nobody has.

A new worktree has no `dist/`, and one test builds a real environment out of it: run
`python packaging/build.py` there before trusting a full test run.

## Running the tests

The suite is ~3,800 tests and about six minutes on a quiet machine, longer when several agents
are running it at once. Run it **once, immediately before merging**, on Linux:

    scripts/test-on-linux tests/ -q --slow

A green macOS run does not clear a branch. CI is Linux and the difference is not visible from the
code: a graph read with no `ORDER BY` comes back in insertion order here and in another order
there, and a killed process is reapable a couple of milliseconds later there.

`scripts/test-on-linux` copies the worktree into a container, installs what `ci.yml` installs,
and hands pytest whatever arguments follow, so `scripts/test-on-linux tests/test_serve.py -q` is
the targeted form and `--single` adds `-n 0`. The first run builds a venv in a docker volume,
after which the container costs a couple of seconds; `--help` covers the docker credential
helper. Run the macOS suite as well; the app ships for both.

While you are working, run the tests that touch what you changed -- seconds, not minutes:

    PYTHONPATH=src python3 -m pytest tests/test_<what_you_touched>.py -q -n 4

The tests marked `slow` — a browser, a subprocess, a wheel build, a network timeout — are left
out unless you ask for them with `--slow`. CI runs with `--slow`, so a change only they catch
still fails there; run `--slow` yourself before merging anything that touches packaging, the
page or the fleet. `-n 0` runs them in one process when a failure needs a clean order; one
Linux CI entry runs the whole suite that way, so a test that only passes beside its neighbours
is caught there. Re-running the whole suite after every intermediate commit buys nothing: the
branch has not landed, and it is rebased onto a moved development branch before it does.

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

## One thing on the GPU at a time

Never put two pieces of work on the GPU at once -- not a question beside a reading, not two
benchmark rows, not a smoke test while a long run is going. Serve one slot and let the second
request wait.

Two at once is more than twice as slow, and it takes the meaning out of every number either one
produces: a row measured under load cannot be compared with a row measured alone, and neither can
be trusted afterwards. Measured 2026-09-09, one machine, Qwen3.8-Flash-Next: a one-line reply
asked on a second slot while an extraction ran took 81s, against a second or two alone, and the
extraction was slowed too. So: `--parallel 1` unless something genuinely needs concurrent
conversations, and a program that reads and answers over the same model does both through the
same server, one after the other. `ml-stack-serve status` says how many slots a server has; check
it before starting a run that will take hours.

## Driving a model on this machine

Never point `ml-stack-claude`, `ml-stack-agent` or `ml-stack-do` at a checkout you are editing.
An agent with file access edits the files it finds, and a small model will happily rewrite
`CLAUDE.md` because it was asked to say hello. Drive them in a scratch directory.

Never `git add -A` when anything else may be writing to the tree — another agent, a running
ingest, a model you just drove. Add the files you changed, by name. (2026-09-04: a 0.8B model
driven in the primary checkout deleted two paragraphs of this file and changed a heading;
`git add -A` swept it into an unrelated commit.)

## Saying that something works

Drive it the way a person does before you say it works: open the interface, click through the
screen, type into the box, press the button, read what comes back.

**A request is not a person.** `curl` against a route proves the route answers. It does not prove
there is a button that reaches it, that the button is on a screen anyone can find, that the reply
renders, or that the next screen follows. Every bug that has shipped here has been on the side of
the line `curl` does not cross.

**A green suite is not a person either.** The tests were written against the same understanding
that wrote the code, so they agree with it by construction: they catch a change that breaks
something, not something that was never right. If you have not driven it, say what you did
instead, in the same breath as the claim: "the route answers, I have not opened the screen".
Never let "it works" stand for "the parts I checked did not fail". This applies hardest to
anything a person only does once — first run, setup, an uninstall — the paths with no second
chance to notice.

## Reporting a problem

Fix it. Then say what you fixed. A problem you found and did not fix is only worth raising if you
are **actually blocked**: you need a decision only the owner can make, you need hardware or an
account you do not have, or fixing it would go outside what was asked. Say which of those it is,
in one line.

**Say it short.** A few lines, the result or the question first, findings as a list. Prose that
has to be mined for its content is work handed back. **If it is a decision, ask it** -- through
the ask/answer tool, in the same message, as one question with the two or three options and
what each costs, not a paragraph describing that a decision exists. "That's your call", "I'd
want your view" and "let me know how you want to proceed" are deferring dressed as deference.
If you cannot write it as one question with options, it is not a decision but a judgement that
is yours: make it and say what you chose.

**Nothing you could change is a blocker.** Not existing code, not code you did not write, not a
function that returns the wrong thing on one platform, not a missing branch, not a test that was
never written -- and not something this repository does not have yet: a dependency nobody has
taken, a tool that is not installed, a setting nothing wires, a helper nobody wrote. Those are
the work: add it and write the straightforward code, because downloading costs nothing. Never
leave something unfixed on purpose, and never write worse code to avoid adding something --
string-matching a file a parser would read, a hand-rolled version compare, a shape copied because
importing the real one would mean a new name in `pyproject.toml`. A test-only dependency is not
bound by `dependencies = []`, a promise about what a *user* installs; put it in the `test` extra
and in the line CI installs. Upgrade on the same terms: a package below its pin is the
environment being wrong, not a version to code around, so upgrade it, run the suite, and say what
moved. `ml-stack-doctor` reports what is below its pin and `ml_stack.installed` holds the check.

Watch for the passive voice that turns a bug into weather: "the field is simply absent", "psutil
isn't available there", "that platform doesn't expose it". Every one of those is a sentence about
something you could have changed. If it is genuinely impossible, say why in terms of the thing
that makes it impossible, not what currently happens. The bar for mentioning a problem at all is
the bar for a commit: it changes what someone would do next.

**A measurement that names a cause we control is a task, not a finding.** "Precision was low
because the model selected everything it read" names a prompt, a flag or a setting, so the
sentence is not finished until it says what was changed and what the re-measurement showed. Write
the fix, run the smoke, queue the sampled run, and report cause, change and number together. A
cause we cannot control (the weights, the hardware, an upstream PR) is reported as such, with
what would change it.

## Never a real person

No name, handle, email or phone number of a real person may appear anywhere in this repository:
not in source, not in a test, not in a fixture, not in a docstring, not in a commit message. Test
data is invented. If a real value revealed a bug, reproduce its *shape* — the casing, the
punctuation, a dot in a handle, a missing surname — never its content.

Long-dead public figures are not covered: a fixture may use a name like Alan Turing, Ada
Lovelace or Grace Hopper, listed in `tests/known-fixtures.txt`. A living person never.

The other exception is attribution a license requires: a copyright line in `NOTICE`, `LICENSE` or
a vendored file's own license header names its holder, because the license makes keeping it a
condition of using the code. The hook does not read those files. `scripts/hooks/` enforces it —
`no-real-names` on staged files, `commit-msg` on the message — and is worth installing:

    scripts/install-hooks.sh
    pip install -e '.[privacy]' && python -m spacy download en_core_web_sm

It refuses a person it has never seen, not merely a list of known names. Invented names go in
`tests/known-fixtures.txt`. Both hooks read `NAMES_GRAPH`, `NAMES_SCRAPE`, `NAMES_FIXTURES` and
`PYTHON` from the environment, so a machine holding a local database of names can wrap them with
an untracked `.git/hooks/` script that exports those and execs the tracked one — the installer
leaves such a wrapper alone.
