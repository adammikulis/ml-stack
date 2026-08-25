# Rules for working in this repo

## Comments and docstrings

Write what the code does. Never why it was written that way.

**Banned:** rationale, war stories, "deliberately", "on purpose", "the reason is",
"this is what X's lesson looks like", explanations of what would happen if the code were
different, arguments against changing it, anything that reads as a message to a future
reader about a decision.

**Allowed:**
- A one-line docstring on a public function saying what it returns.
- A short docstring on a module saying what is in it.
- A comment only where the mechanics are genuinely non-obvious — an API that behaves
  unexpectedly, a magic constant, a workaround for a specific bug. One line.

Prose goes stale, and stale prose is worse than none: it is confidently wrong and nobody
notices. If a decision needs recording, it goes in a commit message, not in the file.

Default to fewer words. If a docstring is longer than the function, delete most of it.

## Commit messages

Start the subject with `feat:`, `fix:`, or `chore:`. release-please reads them: `feat`
bumps the minor, `fix` the patch, `chore` neither. A subject with no prefix is a change
that never reaches a release.

After the prefix, the subject says what changed. Nothing else.

**Banned:** "actually", "real", "finally", "now works", scare quotes, before/after
contrasts, anything that editorialises about the previous state or sounds pleased with
itself. "A real app, and a CI that can actually run the tests" says the old one was fake
and the old CI was lazy; neither is a description.

Write: `feat: native window instead of a browser tab`. `chore: install CPU torch in CI`.
`feat: add Trainer`. `fix: remove the tier system`.

The body is for detail a reader would want later: what was wrong, what the fix is, what
it costs. Plain sentences. No war stories, no rhetorical questions, no lines that argue
with a future reader.

## Anything a user reads

Release notes, the README, the interface, error messages. Write for someone seeing it for
the first time. They did not see the previous version and do not know what was broken in
it — telling them it is fixed only raises a question they did not have.

Describe what the thing does. Not what it no longer does wrong, not what changed, not how
long it took. "Trains across every machine on your network", never "training now works".

Before/after belongs in a commit message, where the reader came looking for it.

## HANDOFF.md

It lists what is still pending. Nothing else.

When something is done, **delete its entry**. Do not strike it through, do not mark it
`[x]`, do not move it to a "completed" section, do not leave a line saying it was
finished. A reader opens this file to find out what is left; anything already dealt with
is noise they have to read past to get there.

The same goes for anything that turned out to be wrong: delete it. A note explaining that
an earlier entry was mistaken is another thing to read past.

If a finished piece leaves something behind — a limit, a gap, a follow-up — write that as
its own pending entry, in its own words. Do not write it as a postscript to the item that
is going away.

An empty HANDOFF.md is a good state. Delete the file rather than leaving headings with
nothing under them.

## Worktrees

Every agent works in its own worktree on its own branch. That means the main session as
much as any subagent it spawns — "I am the one driving" is not an exemption. Nobody edits
the primary checkout, and no two agents share a branch.

Make one before the first edit, branching from `main`:

```
git worktree add -b <branch> ../ml-stack-<branch> main
```

Whoever made it finishes it. Merge into `main`, then take the worktree and the branch
away:

```
git worktree remove ../ml-stack-<branch>
git branch -d <branch>
git worktree prune
```

If the branch was pushed, delete it on the remote too. A branch nobody is working on
still shows up in every list of branches, and the next person has to work out whether it
matters.

A subagent merges and prunes its own work. A session that spawns three agents gets three
merges done by three agents, not three branches handed back for it to sort out.

Do not remove a worktree you did not create — another agent may still be in it. Leave it
and say so.

A new worktree has no `dist/`, and one test builds a real environment out of it. Run
`python packaging/build.py` in the worktree before trusting a full test run there.

## Saying that something works

Drive it the way a person does before you say it works. Open the interface, click
through the screen, type into the box, press the button, read what comes back.

**A request is not a person.** `curl` against a route proves the route answers. It does
not prove there is a button that reaches it, that the button is on a screen anyone can
find, that the reply renders, or that the next screen follows. Every bug that has shipped
here has been on the side of the line `curl` does not cross.

**A green suite is not a person either.** The tests are written against the same
understanding that wrote the code, so they agree with it by construction. They catch a
change that breaks something. They do not catch something that was never right.

If you have not driven it, say what you did instead, in the same breath as the claim:
"the route answers, I have not opened the screen". Never let "it works" stand for
"the parts I checked did not fail".

This applies hardest to anything a person only does once — first run, setup, an
uninstall. Those are the paths with no second chance to notice.

## Reporting a problem

Fix it. Then say what you fixed.

A problem you found and did not fix is only worth raising if you are **actually blocked**:
you need a decision only the owner can make, you need hardware or an account you do not
have, or fixing it would go outside what was asked. Say which of those it is, in one line.

**Existing code is not a blocker.** Neither is code you did not write, a function that
returns the wrong thing on one platform, a missing branch, or a test that was never
written. Those are the work. Reporting them as findings, with the fix left undone, is
handing back a list instead of a result.

Watch for the passive voice that turns a bug into weather: "the field is simply absent",
"psutil isn't available there", "that platform doesn't expose it". Every one of those is
a sentence about something you could have changed. If it is genuinely impossible, say why
in terms of the thing that makes it impossible, not in terms of what currently happens.

The bar for mentioning a problem at all is the same as the bar for a commit: it changes
what someone would do next.
