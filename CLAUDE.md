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
