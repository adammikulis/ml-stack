# Releasing and repository settings

## How a release is made

1. Work lands on the development branch (`0.2dev`). A push to `main` is the owner's.
2. A push to `main` runs `release-please.yml`, which opens or updates a release pull request from the
   `feat:` and `fix:` commit subjects. Merging that pull request tags `vX.Y.Z`.
3. `release.yml` calls `release-build.yml`, which builds the wheel, a CycloneDX SBOM (`sbom.cdx.json`) and the macOS,
   Windows and Linux bundles with read-only permissions. Its `guard` job requires the tag to name the checked-out
   commit and that commit to be an ancestor of `main`. Its `pypi` and `publish` jobs run in the `release` environment,
   which needs the owner's approval: `pypi` uploads when the repository variable `PYPI_ENABLED` is `true`, and
   `publish` signs the bundles and attaches everything to the GitHub release.
   `release-dry-run.yml` calls `release-build.yml` on pull requests, without publishing and without a write grant.
   The environment, the rulesets and the other repository settings are in [github-protection.md](github-protection.md).

## Checks before a release

- `python scripts/notices.py` regenerates `THIRD_PARTY_NOTICES.md`; `python scripts/notices.py --check` fails on a
  GPL, AGPL, LGPL or unknown licence in the packages a bundle ships. The same check runs in CI as the `licenses` job.
- `python -m build` then `twine check dist/*`; the wheel carries `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` under `dist-info/licenses`.
- `gitleaks git --log-opts="--all" .` reports no leaks; `python -m pytest`; `scripts/budgets`.
- `audit.yml` (weekly and on lockfile changes) fails on a known vulnerability in the Python, npm and Cargo dependencies
  unless `.github/pip-audit-allow.json` accepts it (reason, expiry; `scripts/audit_gate.py`).
- `python scripts/sbom.py --out sbom.cdx.json` (run in an environment with the extras installed) writes the CycloneDX SBOM
  that every release produces: `release.yml` runs it in a clean venv, checks it with `--check` and uploads it as `sbom`; the
  publish job attaches it to the GitHub release. It lists the packages `scripts/notices.py` lists, plus the Cargo.lock crates.

## PyPI trusted publishing

See `HANDOFF.md` for the name-similarity waiver PyPI has to grant first. Then, on PyPI, add a pending publisher for
project `ml-stack`: owner `adammikulis`, repository `ml-stack`, workflow `release.yml`, environment `release`. Set the
repository variable `PYPI_ENABLED` to `true`. No token is stored anywhere. See [github-protection.md](github-protection.md#pypi).

## Signing and verifying a download

The publish job signs every `ml-stack-*.zip` with `ssh-keygen -Y sign -n ml-stack-release`, using the Ed25519
private key in the secret `RELEASE_SIGNING_KEY` of the `release` environment, and attaches `<asset>.zip.sig` beside it. The job fails
when the secret is empty or no signature was written.

`signing.RELEASE_KEY` (`src/ml_stack/fleet/signing.py`) holds the matching public key as one `ssh-ed25519 AAAA...`
line. The updater (`fleet/updates.py`) downloads the asset and its `.sig`, and installs the asset only when the
signature was made by that key, in that namespace, over the SHA-256 or SHA-512 of the downloaded bytes. A missing
signature, a signature from another key or namespace, and an asset that differs from what was signed are each refused
and the download is deleted. The verifier is Ed25519 in plain Python; nothing else is installed.

`ml-stack-cluster join --track BRANCH` fetches the branch and runs `git verify-commit FETCH_HEAD` with `RELEASE_KEY` as
the only allowed signer; the checkout then fast-forwards to exactly that commit. A tip commit that is unsigned or
signed by another key is not pulled. Commits on the branch are signed with the same key, loaded into ssh-agent by `scripts/release-key agent`, which
prints the git settings to use:

```
git config gpg.format ssh
git config user.signingkey 'key::ssh-ed25519 AAAA...'
git config commit.gpgsign true
```

The key is made and kept by `scripts/release-key`, run by a person at a terminal (it refuses an agent and a process
with no terminal). It needs `ssh-keygen` and an authenticated `gh` in this repository.

- `scripts/release-key create [--write]` generates an Ed25519 key, sets the secret `RELEASE_SIGNING_KEY` of the
  `release` environment from stdin, stores the private key in the OS keystore (credential `ML_STACK_RELEASE_SIGNING_KEY`) and prints the
  public line. `--write` sets `RELEASE_KEY` in `src/ml_stack/fleet/signing.py`; commit that file. It refuses when a key
  is already stored.
- `scripts/release-key show-public` prints the stored key's public line.
- `scripts/release-key rotate [--write]` replaces the stored key, the repository secret and `RELEASE_KEY`. Releases
  signed before it verify only against the old public key.
- `scripts/release-key agent` loads the stored key into ssh-agent.

If `gh` fails, nothing is stored and `signing.py` is unchanged.

`RELEASE_KEY` must be set before a release is published: `scripts/release-key create --write` at the owner's terminal,
then commit `src/ml_stack/fleet/signing.py`. Without it the signed-update path fails closed. `updates.download_release`
deletes every downloaded asset with "no release key is set" and installs nothing, and `ml-stack-cluster join --track`
refuses every fetched commit with the same message, so no machine updates itself from a release or a tracked branch.
A release published before the key is set is signed with a key nothing pins, and its assets verify against nothing.

To check an asset by hand: `ssh-keygen -Y check-novalidate -n ml-stack-release -s <asset>.zip.sig < <asset>.zip`
confirms the signature is well formed, and `ssh-keygen -Y verify -f allowed_signers -I ml-stack-release -n
ml-stack-release -s <asset>.zip.sig < <asset>.zip` checks it against a file holding `ml-stack-release <public key>`.
Build provenance attestations (`actions/attest-build-provenance`) are not enabled; they need `attestations: write` on
the calling workflows in `release-please.yml`.

## GitHub settings

[github-protection.md](github-protection.md) holds the target configuration: the rulesets for `main`, the
development branches and `v*` tags, the `release` environment and its reviewer, the Actions defaults, secret scanning
with push protection, the credentials agents use, and a command and rollback for each. Only the repository owner
applies them. `scripts/github-protection --check` prints what differs.

- Settings > Code security: enable private vulnerability reporting (`SECURITY.md` points to it), Dependabot alerts and
  the dependency graph (the dependency-review workflow needs it).
- The README one-line installers fetch `packaging/install.sh` from `main`. Pin a tag in the URL when you document one.

## Standalone conversation smoke test

The frozen daemon includes the Ladybug graph engine, native bindings and the package metadata
that declares UI route extensions. Tensor, dataframe, world and benchmark dependencies remain
outside this profile. Plain conversation history opens without installing search extensions.

After building the standalone daemon, run its socket lifecycle check through the shared broker:

```sh
ML_STACK_FROZEN_BINARY=dist/bundle/ml-stack-headless python scripts/test slow -n 1 tests/test_packaging_conversations.py
```

Use the executable with the `.exe` suffix on Windows. The test uses isolated daemon and workspace
roots, checks the maintained Board extension discovered from bundled metadata, imports an existing
conversation and verifies edits and deletion across a restart. It starts no model server.

The Coding worker additionally exercises frozen multiprocessing, native-session resume,
cancellation, identity revocation and the frozen permission-hook dispatcher. Its test build
replaces only the model-serving boundary with `tests/frozen_coding_broker.py`; the maintained
harness launcher, role hooks and subprocess lifecycle still run. The test installs its own
fixture Codex executable in an isolated PATH and starts no model server.

Create a separate proof spec from `packaging/ml-stack.spec`, add
`runtime_hooks=[str(repository / "tests/frozen_coding_broker.py")]` to `Analysis`, and use the
absolute path of `packaging/launcher-headless.py` as its entry point. Build that spec with
PyInstaller in the standalone build environment, then run:

```sh
ML_STACK_FROZEN_CODING_BINARY=/path/to/proof/ml-stack-headless python scripts/test slow -n 1 tests/test_packaging_coding.py
```

Keep that fixture hook out of production bundles. Actual local-model acceptance is a separate,
exclusive brokered run with an installed harness and an isolated project.
