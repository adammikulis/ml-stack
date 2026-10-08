# GitHub-side protection

The target configuration of the GitHub repository, why each setting is there, what it costs, and the command that
sets it. `scripts/github-protection --check` compares the repository with this document and prints one line per
difference, each pointing at a section below.

**Every command here changes GitHub settings and is the owner's, run at his own terminal.** None of them is in a
script, a hook or a workflow, and no agent runs them. `scripts/github-protection` only reads: it issues `gh api`
GET requests, and `tests/test_github_protection.py` fails when its source gains a write.

Shell variables used throughout:

```sh
R=OWNER/REPO                       # the repository
BACKUP=~/github-protection-backup  # where the state before each change is kept
mkdir -p "$BACKUP"
```

Each section saves the current state before changing it and ends with a rollback line.

## Threat model

An agent runs as the owner's operating-system user, so it can use the owner's `gh` login, git credential helper and
ssh keys. Every guard that lives in the checkout (`scripts/hooks/`, the Bash guard, `pre-push`) can be rewritten or
skipped by such a process. The one layer it cannot rewrite is the server, and only when the credentials it holds
carry no right to change the server's rules. What the red-team pass found:

1. The credentials agents use can edit rulesets and secrets, and the existing ruleset lists the administrator role
   and the owner as bypass actors that always bypass.
2. `release.yml` signs every artifact with `RELEASE_SIGNING_KEY` when any `v*` tag is pushed, whatever commit it names.
   A repository secret is readable by a workflow file on any branch a push can create.
3. The gate scripts run from the pull request's own checkout, so a pull request can edit the gate that judges it.
   `release-dry-run.yml` gave pull-request runs write access and an OIDC token.
4. CODEOWNERS did not cover the hooks, the gates, `budgets.json`, signing, the guard or the person modules.
5. `gh pr merge` merges on the server with no `git push`, which no local push guard sees.
6. `RELEASE_KEY` is empty, so a signed update cannot verify (see [release.md](release.md)).

## Order of work

Nothing after step 1 constrains an agent that still holds the owner's credentials, so step 1 comes first.

1. [Give agents their own identity](#agent-identity) and start the harness with it.
2. Save the current state: `gh api repos/$R/rulesets > "$BACKUP/rulesets.json"`.
3. [Create the `release` environment](#the-release-environment) and move the signing key into it.
4. Replace the existing ruleset with the [main](#main-branch-ruleset), [development](#development-branch-ruleset)
   and [release tag](#release-tag-ruleset) rulesets.
5. Set the [Actions](#actions-settings) and [code security](#code-security-settings) settings.
6. Register the PyPI publisher with the `release` environment ([PyPI](#pypi)).
7. Set `RELEASE_KEY` (`scripts/release-key create --write`, see [release.md](release.md)).
8. Run `scripts/github-protection --check` as the owner, then `scripts/github-protection --check --agent` from the
   agent's environment. Both print `0 findings`.

## Agent identity

Agents authenticate as something other than the owner.

- **GitHub App** (preferred): installed on this repository only, with Contents read and write, Pull requests read and
  write, Metadata read. No Workflows, no Administration, no Secrets, no Environments. An installation token lasts an
  hour; whatever launches the harness mints a fresh one.
- **Fine-grained personal access token** on the owner's account, limited to this repository, with the same three
  permissions. The user interface is the only place to make one; it can last up to a year.

Either holds no right to edit rulesets, environments, secrets or workflow files, so a push that changes
`.github/workflows/` is refused by GitHub itself, and a ruleset without bypass actors binds it fully.

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

One ruleset per ref family, none with a bypass actor. Find the existing ruleset and keep a copy:

```sh
gh api repos/$R/rulesets --jq '.[] | [.id, .name] | @tsv'
OLD=<id of the ruleset named default>
gh api repos/$R/rulesets/$OLD > "$BACKUP/ruleset-default.json"
```

Create the three below, then remove the old one: `gh api -X DELETE repos/$R/rulesets/$OLD`. Rollback for the
whole section: delete the three by id (the rollback line under each) and restore the old one with
`jq 'del(.id,.source,.source_type,.node_id,.created_at,.updated_at,._links,.current_user_can_bypass)' "$BACKUP/ruleset-default.json" | gh api -X POST repos/$R/rulesets --input -`.

A ruleset with no bypass actor binds the owner too. Anything that needs an exception is done by setting the
ruleset's `enforcement` to `disabled` for the moment it takes and back to `active`:

```sh
gh api -X PUT repos/$R/rulesets/$ID -f enforcement=disabled
gh api -X PUT repos/$R/rulesets/$ID -f enforcement=active
```

### Main branch ruleset

Blocks deleting `main` and rewriting its history, requires a pull request approved by the code owner after the last
push, and requires the checks to pass on the tip.

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
      "required_approving_review_count": 1,
      "require_code_owner_review": true,
      "dismiss_stale_reviews_on_push": true,
      "require_last_push_approval": true,
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

Why each rule:

- `pull_request` with `require_code_owner_review` makes the owner's approval the gate. CODEOWNERS names him for the
  paths that decide what the gates accept. `require_last_push_approval` means a push after the approval needs a new
  one, and the account that made the last push cannot approve it.
- The merge method is a merge commit, since release-please reads the commit subjects of the merged branch; a squash
  would reduce a branch to one subject.
- The checks run `scripts/budgets` and `budgets-only-fall` from the base commit (`ci.yml`), so a pull request does
  not edit its own gate.

Friction: the owner cannot push to `main` or merge his own pull request. Every pull request to `main` is opened by
the agent identity (or release-please) and approved and merged by the owner. A promotion from `0.2dev` is a pull
request from `0.2dev` to `main` opened by the agent. A pull request opened by release-please gets no workflow runs
from its own token; the owner closes and reopens it, which starts the checks under his account.

Optional, once the commits on the development branch are signed (`scripts/release-key agent`): add
`{"type": "required_signatures"}` to `rules` and run the check with `--require-signatures`. A merge commit made on
the web is signed by GitHub, but a pull request holding unsigned commits cannot be merged into a branch that requires
signatures.

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

Tag creation is not blocked by default. release-please tags the commit when its pull request is merged, with a
token no ruleset can exempt. A tag of any other commit starts `release.yml`, whose `guard` job fails
unless the tag names the checked-out commit and that commit is an ancestor of `main`; its `pypi` and `publish` jobs
need the guard and the `release` environment, which the owner approves and which only deploys from `main` and `v*`
tags. A stray tag therefore builds and publishes nothing.

To block creation too, add `{"type": "creation"}` to `rules` and run the check with `--strict-tags`. release-please
can then no longer tag, and the owner cuts each release himself: set the ruleset to `disabled`, run
`git tag -s vX.Y.Z <commit on main> && git push origin vX.Y.Z`, set it back to `active`.

Rollback: `gh api -X DELETE repos/$R/rulesets/$(gh api repos/$R/rulesets --jq '.[]|select(.name=="release-tags")|.id')`

## The release environment

`RELEASE_SIGNING_KEY` becomes a secret of the `release` environment instead of the repository. The `pypi` and `publish`
jobs of `release.yml` run in it, and nothing else reads the key. The environment requires the owner's approval for
every deployment and deploys only from `main` and `v*` tags, so a workflow file on any other branch gets neither the
key nor the PyPI token.

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
gh api -X POST repos/$R/environments/release/deployment-branch-policies -f 'name=v*' -f type=tag
```

Move the key. `scripts/release-key create` and `rotate` store it as an environment secret from now on. For a key that
already exists as a repository secret:

```sh
gh secret set RELEASE_SIGNING_KEY --env release < <private key file>
gh secret delete RELEASE_SIGNING_KEY
```

Friction: a release asks for approval twice (the `pypi` job and the `publish` job each name the environment). Approve
from the run page or with the notification.

Rollback: `gh api -X DELETE repos/$R/environments/release` removes the environment and its secrets; restore the key
with `scripts/release-key rotate` and remove `environment: release` from the two jobs.

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
  request. Its approval does not satisfy the code owner rule.
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
  "delete_branch_on_merge": true,
  "security_and_analysis": {
    "secret_scanning": {"status": "enabled"},
    "secret_scanning_push_protection": {"status": "enabled"},
    "dependabot_security_updates": {"status": "enabled"}
  }
}
JSON
```

Push protection refuses a push that contains a recognised secret, which covers a token an agent pastes into a file.
Private vulnerability reporting and Dependabot alerts are in Settings > Code security.

Rollback: the same body with `"disabled"` and `true` for `allow_auto_merge`.

## Who merges

`gh pr merge` merges on the server and needs no `git push`, so the push guards never see it. Two settings keep it the
owner's: the main ruleset requires an approval by the code owner, and the agent identity cannot supply one (the
account that opened or last pushed to a pull request cannot approve it, and an agent holds no code-owner account).
Auto-merge is off so a queued merge cannot complete after an approval. An agent with write access can press merge
after the owner has approved; the approval is what the owner gives, and `require_last_push_approval` voids it when
anything is pushed afterwards.

## CODEOWNERS

`.github/CODEOWNERS` names the owner for the paths that decide what the gates accept: `.github/`, `.claude/`,
`packaging/`, `pyproject.toml`, `budgets.json`, `scripts/hooks/`, `scripts/gates/`, `scripts/budgets`,
`scripts/land*`, `scripts/test*`, `scripts/release-key`, `scripts/github-protection`, `scripts/audit_gate.py`,
`tests/known-fixtures.txt`, `src/ml_stack/fleet/signing.py`, `src/ml_stack/guard/`, `src/ml_stack/person.py`,
`src/ml_stack/worktreerules.py` and `src/ml_stack/workspace/person_*.py`. `scripts/github-protection` holds the same
list and reports a tracked path that falls under the `*` rule, and a listed pattern that matches nothing. The rule has
effect once the main ruleset sets `require_code_owner_review`.

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

`ci.yml` runs on every pull request with no path filter, since a required check that never starts leaves a
documentation-only pull request unmergeable. A pull request still runs the workflow file it carries, so a change to
`.github/` is reviewed by the owner through CODEOWNERS before it is trusted.

## PyPI

Register the publisher for project `ml-stack`: owner, repository, workflow `release.yml`, **environment `release`**.
Set the repository variable `PYPI_ENABLED` to `true` (`gh variable set PYPI_ENABLED --body true`). The token the
`pypi` job mints is accepted only for a run that the environment approved on `main` or a `v*` tag.

Rollback: remove the environment name from the publisher and delete `environment: release` from the `pypi` job.

## Verifying

```sh
scripts/github-protection --check                     # as the owner
scripts/github-protection --check --agent             # from the agent's environment
scripts/github-protection --check --local             # CODEOWNERS and workflows, no network
scripts/github-protection --check --require-signatures --strict-tags   # the optional rules
```

Exit 0 is no finding, 1 is drift, 2 is no repository to read. A setting the credentials cannot read is a finding
(`*.read`), since it cannot be confirmed. Secrets scoped to an organisation are not visible to a repository token and
are not checked.
