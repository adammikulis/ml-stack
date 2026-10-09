# Brief: the gated auto-publisher for `0.2dev` (owner: Codex)

## Decision (owner, 2026-10-04)
"yes to auto-publish 0.2dev. start local then move to ci. scan gated pushes are fine." The owner
does not want to be the person who publishes; the controls below replace that step. This
authorises ONLY pushing the branch `0.2dev` to `origin` (public repo adammikulis/ml-stack) by the
publisher defined here. It does not authorise any agent to run `git push`, to push other refs,
to tag, to create a GitHub release, to upload a package, or to set a version number. The owner
sets the version and releases.

## What to build (phase 1: local publisher)
`ml-stack-publish` (one console script, new module `ml_stack/publish/`):
1. **Agents never hold credentials.** The push credential (a fine-grained GitHub token or deploy
   key scoped to this one repo, contents: write, no admin, no workflows/secrets) lives in the OS
   keystore under a dedicated purpose, read only by the publisher process. `ml-stack-publish setup`
   is person-only (terminal, no agent marker) and is where the owner provides it once; it never
   prints it. Agents call `ml-stack-publish request` (allowed for agents: it records "please
   publish commit X") and `status`; only the publisher pushes. Add the command names and the
   credential purpose to the human-only floor tests and the red-team coverage map.
2. **Refuses everything except a fast-forward of `0.2dev`.** Tags, other refs, `--force`, deleting
   refs, non-fast-forward, a commit that is not an ancestor-descendant of `origin/0.2dev`, a
   dirty or detached worktree: all refused with a one-line reason. The publisher builds the
   push argv itself from fixed pieces (no shell, no user text), pushes `refs/heads/0.2dev` only.
3. **Gates, all green on the exact commit (checked out clean in a throwaway worktree):**
   `scripts/test gate`; `scripts/test full` (documented load flakes allowed only by a listed
   allow-list with the failing test named, never a blanket retry); the red-team tier
   (`pytest --redteam -n 0 tests/test_redteam_*.py`); budgets, red-team coverage counts and
   `tests/full-tier-time.json` not risen; a secrets scan (reuse sentinel's scanners / pattern
   rules; add a high-entropy and key-shape check) and the private-terms/real-names audit
   (`scripts/notices.py` rules plus the owner's local denylist if present) over the diff since
   `origin/0.2dev` AND over added files in full; the licence-attribution check; no file over the
   size cap; no new binary; `git fsck`; commit trailer present; no version number string added
   (grep the diff for the forbidden strings the owner named in CLAUDE.md/memory).
4. **Protected paths need the owner.** Maintain `PROTECTED_PATHS` (sentinel/, keystore.py,
   person.py, requests/, inbox/, roles/rules/chatpolicy, destructive classifier, bash guard
   hooks, scripts/gates, the publisher itself, CODEOWNERS, pyproject.toml scripts and
   dependencies, .github/). If the diff touches any, the publisher does not push: it raises a
   Requests-inbox item (kind `publish_protected`, listing the paths and a diff summary) and
   waits; the owner answers allow/deny. Everything else publishes without a person.
5. **Second-model review.** A publish request carries a review record: another agent (not the
   author; Claude Code or Codex) re-ran the gate and read the diff, with its model identity.
   No review record, no push. (Reviews are data written to the workspace; the publisher
   verifies the reviewer identity is registered and is not the author.)
6. **Limits and kill switch.** At most N pushes per hour (default 6); `ml-stack-publish pause`
   and `resume` (person-only); a red result, a refused push or a gate flake outside the
   allow-list pauses the publisher and raises a Requests item; every attempt (commit, gates run,
   result, reviewer) goes to the activity log (no secrets) and the hash-chained event log.
7. **Dry run is the default in tests and `--dry-run` everywhere**: tests use a bare local repo as
   `origin` and fake scanners; never the real network, never the real Keychain
   (ML_STACK_NO_REAL_KEYSTORE; children use tests/onboard_support.FileKeyring); mutation-check each
   refusal and each gate once. Red-team rows for the new surfaces. Budgets only fall.
8. Docs: docs/publishing.md (what auto-publishes, what never does, how to pause, how to rotate
   the credential), `python3 scripts/reference --write`, no version number.

## Phase 2: CI
Add `.github/workflows/` (needs the owner's one-time review because `.github/` is protected):
a Linux job running the same gates on pushes to `0.2dev` and on PRs, required checks via branch
protection, and move the publish decision to "CI green on the exact commit". Keep the local
publisher as the pusher until CI is trusted; then the push comes from CI auto-merge. Includes
the Linux run (`scripts/test-on-linux`) that has not been done yet.

## Order and rules
Build on a branch from `0.2dev` (not the dev branch directly), announce with `ml-stack-workspace
announce`, keep messages short, tell the lead the branch and head. The lead merges it (the
publisher is a protected path, so the owner reads its diff once before it is first enabled).
Until the publisher exists and is enabled by the owner, nothing is pushed by any agent.
