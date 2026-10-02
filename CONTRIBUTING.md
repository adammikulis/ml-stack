# Contributing to ml-stack

Issues and pull requests are welcome. This is a one-person, pre-1.0 project: expect slow reviews and changes to
names and layout. `CLAUDE.md` holds the working rules in full; the short version:

- Python 3.13. `python -m pip install -e ".[test]"` then `python -m pytest` (`--slow` adds the slow tests).
- Commit subjects start with `feat:`, `fix:` or `chore:`; release-please builds the changelog from them.
- Comments say what the code does, not why it was written that way.
- The gates in `scripts/gates/` refuse a new violation; run `scripts/budgets` before you push. A number in
  `budgets.json` may fall, never rise.
- Tests build their own fixtures with invented names and never read `~/.ml-stack`. No real names, emails, hostnames
  or home paths in files or commit messages (`scripts/hooks/no-real-names` checks this).
- A new dependency needs a licence compatible with Apache-2.0. `python scripts/notices.py` regenerates
  `THIRD_PARTY_NOTICES.md` and `--check` fails on a disallowed licence. GPL and AGPL packages stay optional extras that
  are never bundled.
- Do not copy code from a project whose licence is incompatible with Apache-2.0.

By contributing you agree that your contribution is licensed under Apache-2.0.

Security issues: see `SECURITY.md`; do not file them as public issues.
