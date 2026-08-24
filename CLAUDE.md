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

The subject says what changed. Nothing else.

**Banned:** "actually", "real", "finally", "now works", scare quotes, before/after
contrasts, anything that editorialises about the previous state or sounds pleased with
itself. "A real app, and a CI that can actually run the tests" says the old one was fake
and the old CI was lazy; neither is a description.

Write: `Native window instead of a browser tab`. `Install CPU torch in CI`.
`Add Trainer`. `Remove the tier system`.

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
