# Quality policy

Read the sections relevant to your task. [Working contract](../../CLAUDE.md) applies to every task.

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

## What belongs here, and what belongs to the app that drives it

Anything true of any graph, model or scrape is this library's, with a test and a command; a line
that names one community, its vocabulary, or where its data lives belongs to the app (`~/ai_ceo`
is the first). An app holds only wrappers and one-line switches -- a script that calls one of
ours with its own arguments, an environment variable that flips one of our parameters, a lambda
that says where its graph keeps its pointers, its copy and kinds handed to our page. When an app
needs more than that, the missing piece is a command or a parameter here.

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

