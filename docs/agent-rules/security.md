# Security policy

Read the sections relevant to your task. [Working contract](../../CLAUDE.md) applies to every task.

## Tests never touch the person's keystore

No test reads, writes or prompts for an item in the real OS keystore (macOS Keychain). In process
the real backends refuse (`tests/conftest.py`); `tests/conftest.py` also sets
`ML_STACK_NO_REAL_KEYSTORE=1` for the whole run, and every process a test starts inherits it, so
the keystore reads as absent there. A test child that needs a working keystore sets
`PYTHON_KEYRING_BACKEND=onboard_support.FileKeyring` and `ML_STACK_TEST_KEYRING=<file>` (see
`tests/onboard_support.py`) and puts `tests` on its `PYTHONPATH`; a test that spawns a child with
an environment built from scratch must do the same. A test that stripped the agent markers
(`CLAUDECODE`, `ML_STACK_NONINTERACTIVE`) to look like a person at a screen is the most likely to
reach the keystore: give it the file keyring. Nothing in the repo pops more than one dialog; a
notice goes through `sentinel/heads_up.py` only, and `ML_STACK_NOTIFY=off` silences all of it.

## System settings are human-only

Changing a machine setting (the wired memory limit `iogpu.wired_limit_mb`, a boot-time daemon,
a guard or sentinel policy, a role, a saved rule, a quarantine release) is done by a person at
their own screen or terminal: never offered to a model, role, MCP or chat tool, workspace agent or
channel message. A privileged step goes through the operating system's own administrator prompt;
ml-stack never sees or stores the password, and never installs a passwordless `sudoers` rule.

## Never a real person

No name, handle, email or phone number of a real person may appear anywhere in this repository:
not in source, not in a test, not in a fixture, not in a docstring, not in a commit message. Test
data is invented. If a real value revealed a bug, reproduce its *shape* — the casing, the
punctuation, a dot in a handle, a missing surname — never its content.

A licence beats this rule. Where a licence requires a copyright holder to be named for code we
copy, port or redistribute (the `NOTICE` file and the licence texts that travel with such code),
write the name exactly as the licence requires; that is the only exception, it lives in those
files, and nothing else may carry the name. Never drop a required attribution to satisfy this rule.

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
