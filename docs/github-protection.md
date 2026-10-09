# GitHub-side protection

The target configuration of the GitHub repository, why each setting is there, what it costs, and the command that
sets it.

- `scripts/github-protection-apply` applies the target. It is the owner's, run at his own terminal: it refuses an
  agent marker and a missing terminal, prints a diff and changes nothing unless given `--apply`, and saves the current
  settings before it changes any.
- `scripts/github-protection --check` compares the repository with the target and prints one line per difference,
  each pointing at a section below. It issues `gh api` GET requests only, and `tests/test_github_protection.py` fails
  when its source gains a write.
- The commands in the sections below are what the apply script runs, written out for anyone who wants to apply one
  setting by hand. `tests/test_github_protection_apply.py` fails when a JSON body here differs from the script's.

Shell variables used in the commands:

```sh
R=OWNER/REPO                       # the repository
BACKUP=~/github-protection-backup  # where the state before a manual change is kept
mkdir -p "$BACKUP"
```

## Threat model

An agent runs as the owner's operating-system user, so it can use the owner's `gh` login, git credential helper and
ssh keys. Every guard that lives in the checkout (`scripts/hooks/`, the Bash guard, `pre-push`) can be rewritten or
skipped by such a process. The one layer it cannot rewrite is the server, and only when the credentials it holds
carry no right to change the server's rules. What the red-team pass found:

1. The credentials agents use can edit rulesets and secrets, and the previous ruleset lists the administrator role
   and the owner as bypass actors that always bypass.
2. `release.yml` signed every artifact with `RELEASE_SIGNING_KEY` when any `v*` tag was pushed, whatever commit it
   named. A repository secret is readable by a workflow file on any branch a push can create.
3. The gate scripts ran from the pull request's own checkout, so a pull request could edit the gate that judges it.
   `release-dry-run.yml` gave pull-request runs write access and an OIDC token.
4. CODEOWNERS did not cover the hooks, the gates, `budgets.json`, signing, the guard or the person modules.
5. `gh pr merge` merges on the server with no `git push`, which no local push guard sees.
6. `RELEASE_KEY` is empty, so a signed update cannot verify (see [release.md](release.md)).

## Order of work

Nothing after step 1 constrains an agent that still holds the owner's credentials, so step 1 comes first.

1. [Give agents their own identity](#agent-identity) and start the harness with it.
2. `scripts/github-protection-apply --repo $R` prints what would change.
3. `scripts/github-protection-apply --repo $R --apply --identity-ready` creates the `release` environment, then stops
   because the environment has no signing key yet.
4. `scripts/release-key create --write` (or `rotate`) stores the key in the environment; commit
   `src/poolhouse/fleet/signing.py` ([release.md](release.md)).
5. Run the apply command from step 3 again. It removes repository secrets, replaces the rulesets, sets the Actions
   and security settings, and ends with `scripts/github-protection --check`, which must print `0 findings`.
6. Register the PyPI publisher with the `release` environment ([PyPI](#pypi)).
7. From the agent's environment: `scripts/github-protection --check --agent`.

What the owner does not know in advance and the first apply run shows: whether the repository is public or private
(environment reviewers and rulesets on a private repository need a paid plan; an unavailable setting is a `*.read`
or a failed step), the default branch name (the main ruleset uses `~DEFAULT_BRANCH`), and whether the repository
deletes merged branches on its own (the target sets `delete_branch_on_merge` to false).

## Agent identity

Agents authenticate as something other than the owner, and push everything with it, workflow files included.

- **GitHub App** (preferred): installed on this repository only, with Contents, Pull requests and Workflows read and
  write, and Metadata read. No Administration, no Secrets, no Environments. An installation token lasts an hour;
  whatever launches the harness mints a fresh one.
- **Fine-grained personal access token** on the owner's account, limited to this repository, with the same
  permissions. The user interface is the only place to make one; it can last up to a year.

Workflows write is acceptable because nothing a workflow pushed to a branch can read is worth taking: the signing key
is a secret of the `release` environment, which deploys from `main` only; no repository-level secret exists
(`scripts/github-protection` reports any); and the way onto `main` is a pull request with required checks and a
ruleset the identity cannot edit. The identity holds no right to edit rulesets, environments or secrets, so a ruleset
without bypass actors binds it fully.

The harness is started by the owner, at his terminal, with an environment that holds the agent token and nothing of
his own:

```sh
export GH_CONFIG_DIR="$HOME/.config/gh-agent"     # an empty directory: no stored login
export GH_TOKEN="<the agent token>"
unset GITHUB_TOKEN GH_ENTERPRISE_TOKEN
export GIT_CONFIG_GLOBAL="$HOME/.config/git/agent-config"
export GIT_CONFIG_NOSYSTEM=1
export SSH_AUTH_SOCK=
export GIT_SSH_COMMAND=/usr/bin/false
```

`$HOME/.config/git/agent-config` carries a credential helper that answers with the token, and the agent's commit
identity:

```ini
[credential "https://github.com"]
	helper =
	helper = !f() { test "$1" = get && echo username=x-access-token && echo "password=$GH_TOKEN"; }; f
[user]
	name = agent
	email = agent@users.noreply.github.com
```

The remote of every checkout an agent uses is the HTTPS URL (`git remote set-url origin https://github.com/$R.git`),
since an SSH remote would use the owner's key.

Limits that remain: a process of the same operating-system user can still read `~/.config/gh` and the keychain item
the owner's own `gh` and `git` use. A separate operating-system account for agents closes that; the environment above
removes the credentials from every command the harness runs by ordinary means, not from a process that goes looking.
`scripts/github-protection --check --agent`, run from the agent's environment, fails when the credentials in use are
the owner's account or can administer the repository.

Rollback: unset the variables; the harness falls back to the owner's `gh` login.

## Rulesets

One ruleset per ref family, none with a bypass actor. A ruleset with no bypass actor binds the owner too; the one
exception he needs is made by setting that ruleset's `enforcement` to `disabled` for the moment it takes and back to
`active`:

```sh
gh api -X PUT repos/$R/rulesets/$ID -f enforcement=disabled
gh api -X PUT repos/$R/rulesets/$ID -f enforcement=active
```

The apply script creates the three rulesets (or updates those of the same name), then removes the one named
`default`. Manually, keep a copy of the old one first:

```sh
gh api repos/$R/rulesets --jq '.[] | [.id, .name] | @tsv'
OLD=<id of the ruleset named default>
gh api repos/$R/rulesets/$OLD > "$BACKUP/ruleset-default.json"
```

Rollback for the section: `scripts/github-protection-apply --restore <backup file> --apply`, or delete the three by
id (the rollback line under each) and restore the old one with
`jq 'del(.id,.source,.source_type,.node_id,.created_at,.updated_at,._links,.current_user_can_bypass)' "$BACKUP/ruleset-default.json" | gh api -X POST repos/$R/rulesets --input -`.

### Main branch ruleset

Blocks deleting `main` and rewriting its history, and requires a pull request whose required checks pass. It requires
no approving review: the human gate on a promotion is described under [Who merges](#who-merges).

```sh
gh api -X POST repos/$R/rulesets --input - <<'JSON'
{
  "name": "main",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "pull_request", "parameters": {
      "required_approving_review_count": 0,
      "require_code_owner_review": false,
      "dismiss_stale_reviews_on_push": false,
      "require_last_push_approval": false,
      "required_review_thread_resolution": false,
      "allowed_merge_methods": ["merge"]
    }},
    {"type": "required_status_checks", "parameters": {
      "strict_required_status_checks_policy": false,
      "required_status_checks": [
        {"context": "gates"},
        {"context": "privacy"},
        {"context": "licenses"},
        {"context": "test (ubuntu-latest, 3.13, --slow)"}
      ]
    }}
  ]
}
JSON
```

The context names are the job names a pull request's check list shows (`gh pr checks N`); copy them from there if the
workflow names change. The check reads the `gates`, `privacy` and `licenses` jobs and any job whose name begins
`test (`.

- The merge method is a merge commit, since release-please reads the commit subjects of the merged branch; a squash
  would reduce a branch to one subject.
- Nobody pushes to `main`. A direct push of it, approved through `person-consume` or not, is refused by GitHub once
  this ruleset is on; a promotion is a pull request.
- A pull request opened by release-please gets no workflow runs from its own token, so its required checks never
  report. Closing and reopening it as a user starts them.

Optional: GitHub enforces the owner's approval. Run `scripts/github-protection-apply --repo $R --apply --owner-review`
(it sets `required_approving_review_count` to 1 and `require_code_owner_review`, `dismiss_stale_reviews_on_push` and
`require_last_push_approval` to true) and check with `--require-owner-review`. The cost: the owner cannot approve a
pull request he opened, so every pull request to `main` comes from the agent identity or release-please; each
promotion needs an approval click in GitHub besides the word in chat; and a push to the pull request's head after the
approval voids it.

Optional: signed commits. Add `{"type": "required_signatures"}` to `rules` and check with `--require-signatures`. A
merge commit made on the web is signed by GitHub, but a pull request holding unsigned commits cannot be merged into a
branch that requires signatures.

Rollback: `gh api -X DELETE repos/$R/rulesets/$(gh api repos/$R/rulesets --jq '.[]|select(.name=="main")|.id')`

### Development branch ruleset

`0.2dev` and every `*dev` branch: agents push to it directly, so it only loses its deletion and history rewrites.

```sh
gh api -X POST repos/$R/rulesets --input - <<'JSON'
{
  "name": "development",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {"ref_name": {"include": ["refs/heads/*dev"], "exclude": []}},
  "rules": [{"type": "deletion"}, {"type": "non_fast_forward"}]
}
JSON
```

Rollback: `gh api -X DELETE repos/$R/rulesets/$(gh api repos/$R/rulesets --jq '.[]|select(.name=="development")|.id')`

### Release tag ruleset

A `v*` tag cannot be moved, re-pointed or deleted.

```sh
gh api -X POST repos/$R/rulesets --input - <<'JSON'
{
  "name": "release-tags",
  "target": "tag",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {"ref_name": {"include": ["refs/tags/v*"], "exclude": []}},
  "rules": [{"type": "deletion"}, {"type": "non_fast_forward"}, {"type": "update"}]
}
JSON
```

Tag creation is not blocked by default, so any account with write access can push a `v*` tag. What a stray tag does:

- It starts `release.yml` from the workflow file at the tagged commit, not the one on `main`. The `build` job
  (`release-build.yml`, read-only, no secrets) runs unguarded and costs runner time.
- Its `pypi` and `publish` jobs name the `release` environment, which deploys from `main` only; a run on a tag ref is
  refused the environment, so it neither signs nor uploads. A tagged commit that deletes the `environment:` line from
  those jobs would still reach no secret, since the key is a secret of the environment and no repository secret
  exists.
- The tags release-please makes are created with `GITHUB_TOKEN` and start no tag workflow; the release runs from
  `release-please.yml` on `main`, where the `guard` job requires the tag to name the checked-out commit and that
  commit to be an ancestor of `main`.

To block creation too, add `{"type": "creation"}` to `rules` and check with `--strict-tags`. release-please can then
no longer tag, because no ruleset can exempt its token.

Rollback: `gh api -X DELETE repos/$R/rulesets/$(gh api repos/$R/rulesets --jq '.[]|select(.name=="release-tags")|.id')`

## The release environment

`RELEASE_SIGNING_KEY` is a secret of the `release` environment and of nothing else. The `pypi` and `publish` jobs of
`release.yml` run in it, and nothing else reads the key. The environment requires the owner's approval for every
deployment (the one place GitHub itself holds a human gate) and deploys only from `main`, so a workflow file on any
other branch or tag gets neither the key nor the PyPI token. The `pypi` job checks the key is set before it uploads,
so a missing secret ends the run before anything is published.

```sh
OWNER_ID=$(gh api user --jq .id)
gh api repos/$R/environments/release > "$BACKUP/environment-release.json" || true
gh api -X PUT repos/$R/environments/release --input - <<JSON
{
  "reviewers": [{"type": "User", "id": $OWNER_ID}],
  "can_admins_bypass": false,
  "deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}
}
JSON
gh api -X POST repos/$R/environments/release/deployment-branch-policies -f name=main -f type=branch
```

`scripts/release-key create` and `rotate` store the key as a secret of this environment, so the environment must exist
first. A key that is a repository secret is removed by the apply script once the environment holds one
(`gh secret delete RELEASE_SIGNING_KEY`); a secret's value cannot be read back, so it cannot be copied across.

Friction: a release asks for approval twice (the `pypi` job and the `publish` job each name the environment).

Rollback: `gh api -X DELETE repos/$R/environments/release` removes the environment and its secrets; restore the key
with `scripts/release-key rotate` after recreating it, or remove `environment: release` from the two jobs.

## Actions settings

```sh
for p in permissions/workflow permissions permissions/selected-actions permissions/fork-pr-contributor-approval; do
  gh api repos/$R/actions/$p > "$BACKUP/actions-$(echo $p | tr / -).json"
done

gh api -X PUT repos/$R/actions/permissions/workflow --input - <<'JSON'
{"default_workflow_permissions": "read", "can_approve_pull_request_reviews": true}
JSON

gh api -X PUT repos/$R/actions/permissions --input - <<'JSON'
{"enabled": true, "allowed_actions": "selected", "sha_pinning_required": true}
JSON

gh api -X PUT repos/$R/actions/permissions/selected-actions --input - <<'JSON'
{"github_owned_allowed": true, "verified_allowed": false, "patterns_allowed": [
  "Swatinem/rust-cache@*", "dtolnay/rust-toolchain@*", "googleapis/release-please-action@*",
  "ossf/scorecard-action@*", "pypa/gh-action-pypi-publish@*", "softprops/action-gh-release@*"
]}
JSON

gh api -X PUT repos/$R/actions/permissions/fork-pr-contributor-approval \
  -f approval_policy=all_external_contributors
```

- The default token is read-only; a job that writes asks for it in its own `permissions`.
- `can_approve_pull_request_reviews` stays `true` because it is the setting that lets release-please open its pull
  request.
- Only GitHub's own actions and the listed ones run, and every `uses:` is pinned to a commit
  (`scripts/github-protection` checks that in the workflow files).
- A workflow run from an outside collaborator's pull request waits for the owner.

Rollback: `PUT` the files saved in `$BACKUP` back to the same paths (`gh api -X PUT repos/$R/actions/permissions/workflow --input "$BACKUP/actions-permissions-workflow.json"`).

## Code security settings

```sh
gh api repos/$R > "$BACKUP/repository.json"
gh api -X PATCH repos/$R --input - <<'JSON'
{
  "allow_auto_merge": false,
  "delete_branch_on_merge": false,
  "security_and_analysis": {
    "secret_scanning": {"status": "enabled"},
    "secret_scanning_push_protection": {"status": "enabled"},
    "dependabot_security_updates": {"status": "enabled"}
  }
}
JSON
```

Push protection refuses a push that contains a recognised secret, which covers a token an agent pastes into a file.
`delete_branch_on_merge` is false so that merging the pull request from a snapshot of `0.2dev` into `main` cannot ask
GitHub to delete a long-lived branch. Private vulnerability reporting and Dependabot alerts are in Settings > Code
security.

Rollback: `scripts/github-protection-apply --restore <backup file> --apply`, or `PATCH` the values saved in
`$BACKUP/repository.json` (`allow_auto_merge`, `delete_branch_on_merge`, `security_and_analysis`) back.

## Who merges

`gh pr merge` merges on the server and needs no `git push`, so the push guards never see it. The main ruleset requires
a pull request and passing checks and no review, so the credential an agent holds can merge a pull request whose
checks are green. The human gate on a promotion to `main` is therefore ours, not GitHub's:

- The promotion is a pull request from a frozen snapshot branch of the development branch (`promote/<date>`, pushed
  by the agent; `scripts/hooks/pre-push` lets an agent create such a branch and nothing under `main`). The snapshot
  keeps the pull request's head still while the development branch moves on; a pull request opened from `0.2dev`
  itself would change under its own checks with every push.
- The agent merges it only while a live `release-main` person authorization exists for the pull request head's exact
  SHA ([person-delegation.md](person-delegation.md#release-main-on-a-pull-request)). The owner gives it by selecting
  the approval question (`scripts/hooks/person-consume propose`). The guard that refuses the merge command without
  it is not written yet; until it is, nothing but the agent's own rule stops a merge.
- What enforces that is our hooks and the agent credential's lack of administrator rights over the repository (it
  cannot edit the ruleset, the environment or the secrets). GitHub does not enforce it. The environment's reviewer is
  the one approval GitHub holds, at publication.

To have GitHub enforce an approval as well, use the `--owner-review` switch described under the main ruleset.

The first promotion after this configuration is applied runs under the gates `main` holds, which are older than those
on `0.2dev`: `scripts/budgets` from `main` scores a tree with budgets and checkers it has never seen. Dry-run it before
opening the pull request: check out `main`, merge the snapshot into a scratch branch, run `scripts/test gate`, and
settle what fails there before the pull request exists. Changes to a gate land before the numbers they move (the
checker code comes from the base commit and the data beside it from the pull request).

## CODEOWNERS

`.github/CODEOWNERS` begins with `* @owner`, so the owner already owns every path, and the lines for specific paths
change nothing GitHub does: whole-repository code-owner review is what the `--owner-review` switch turns on. The
specific lines record which paths decide what the gates accept: `.github/`, `.claude/`, `packaging/`,
`pyproject.toml`, `budgets.json`, `scripts/hooks/`, `scripts/gates/`, `scripts/budgets`, `scripts/land*`,
`scripts/test*`, `scripts/release-key`, `scripts/github-protection*`, `scripts/audit_gate.py`, `scripts/notices.py`,
`tests/known-fixtures.txt`, the three gate tests, `release-please-config.json`, `.release-please-manifest.json`,
`src/poolhouse/fleet/signing.py`, `src/poolhouse/fleet/updates.py`, `src/poolhouse/guard/`, `src/poolhouse/redact/`,
`src/poolhouse/person.py`, `src/poolhouse/worktreerules.py` and `src/poolhouse/workspace/person_*.py`.
`scripts/github-protection` holds the same list and reports a tracked path that falls under the `*` rule, and a listed
pattern that matches nothing. A pull request that lowers a number in `budgets.json` is reviewed like any other.

## Workflow rules

`scripts/github-protection` reads `.github/workflows/*.yml` and reports:

- an action not pinned to a 40-character commit;
- `secrets: inherit`, and a job that reads a secret outside an environment;
- a workflow-level write permission, and a job that runs on `pull_request` with a write permission other than
  `security-events`;
- `pull_request_target`;
- an OIDC token (`id-token: write`) in a job without an environment (the Scorecard workflow is the one exception);
- `release.yml` without the `merge-base --is-ancestor` check on the tagged commit, and `ci.yml` without the base
  commit's checkers.

What a pull request can and cannot change in the checks that judge it:

- A `pull_request` run uses the workflow files the pull request carries. A pull request that deletes the `trusted`
  steps from `ci.yml`, or the jobs themselves, defeats the base-commit mechanism, and so does one that edits a test
  the gate runs. Only a reviewer of the diff stops that; with `require_code_owner_review` on it is the owner, and
  until then it is whoever merges.
- `scripts/budgets`, the other `scripts/gates/*.py` and `scripts/hooks/` run from the base commit, as does the
  `budgets.json` only-falls comparison. `pinned.txt`, `survivors.txt` and `budgets.json` are the pull request's own.
- The gate tests (`tests/test_budgets.py`, `test_wiring.py`, `test_layers.py`), `scripts/notices.py`, the
  release-please files and the name detector (`src/poolhouse/redact/`, installed from the pull request's tree) are the
  pull request's own. The privacy job runs the base commit's hook script, but the detector it imports is the pull
  request's.
- `ci.yml` runs on every pull request with no path filter, since a required check that never starts leaves a
  documentation-only pull request unmergeable.

## PyPI

Register the publisher for project `poolhouse`: owner, repository, workflow `release.yml`, **environment `release`**.
Set the repository variable `PYPI_ENABLED` to `true` (`gh variable set PYPI_ENABLED --body true`). The token the
`pypi` job mints is accepted only for a run the environment approved on `main`.

PyPI matches the workflow named in the token's `workflow` claim. A run reached through `release-please.yml` calling
`release.yml` may carry the caller's filename; after the first release, read the claim in the PyPI publisher error
if the upload is refused and register the filename it names.

Rollback: remove the environment name from the publisher and delete `environment: release` from the `pypi` job.

## Verifying

```sh
scripts/github-protection --check                     # as the owner
scripts/github-protection --check --agent             # from the agent's environment
scripts/github-protection --check --local             # CODEOWNERS and workflows, no network
scripts/github-protection --check --require-owner-review --require-signatures --strict-tags   # the optional rules
```

Exit 0 is no finding, 1 is drift, 2 is no repository to read. A setting the credentials cannot read is a finding
(`*.read`), since it cannot be confirmed. Secrets scoped to an organisation are not visible to a repository token and
are not checked.
