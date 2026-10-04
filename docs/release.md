# Releasing and repository settings

## How a release is made

1. Work lands on the development branch (`0.2dev`). A push to `main` is the owner's.
2. A push to `main` runs `release-please.yml`, which opens or updates a release pull request from the
   `feat:` and `fix:` commit subjects. Merging that pull request tags `vX.Y.Z`.
3. `release.yml` builds the wheel, a CycloneDX SBOM (`sbom.cdx.json`), the macOS, Windows and Linux bundles,
   uploads to PyPI when the repository variable `PYPI_ENABLED` is `true`, and attaches everything to the GitHub release.
   `release-dry-run.yml` builds the same on pull requests without publishing.

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
project `ml-stack`: owner `adammikulis`, repository `ml-stack`, workflow `release.yml`, no environment. Set the
repository variable `PYPI_ENABLED` to `true`. No token is stored anywhere.

## Signing and verifying a download

The publish job signs every `ml-stack-*.zip` with `ssh-keygen -Y sign -n ml-stack-release`, using the Ed25519
private key in the repository secret `RELEASE_SIGNING_KEY`, and attaches `<asset>.zip.sig` beside it. The job fails
when the secret is empty or no signature was written.

`signing.RELEASE_KEY` (`src/ml_stack/fleet/signing.py`) holds the matching public key as one `ssh-ed25519 AAAA...`
line. The updater (`fleet/updates.py`) downloads the asset and its `.sig`, and installs the asset only when the
signature was made by that key, in that namespace, over the SHA-256 or SHA-512 of the downloaded bytes. A missing
signature, a signature from another key or namespace, and an asset that differs from what was signed are each refused
and the download is deleted. The verifier is Ed25519 in plain Python; nothing else is installed.

`ml-stack-fleet join --track BRANCH` fetches the branch and runs `git verify-commit FETCH_HEAD` with `RELEASE_KEY` as
the only allowed signer; the checkout then fast-forwards to exactly that commit. A tip commit that is unsigned or
signed by another key is not pulled. Commits on the branch are signed with the same key:

```
git config gpg.format ssh && git config user.signingkey ~/.ssh/ml-stack-release && git config commit.gpgsign true
```

Creating the key:

```
ssh-keygen -t ed25519 -N '' -C ml-stack-release -f ml-stack-release
```

`ml-stack-release` is the private key, for the secret `RELEASE_SIGNING_KEY`; the contents of `ml-stack-release.pub` are
`RELEASE_KEY`. Until `RELEASE_KEY` is set, no release asset and no tracked commit verifies.

To check an asset by hand: `ssh-keygen -Y check-novalidate -n ml-stack-release -s <asset>.zip.sig < <asset>.zip`
confirms the signature is well formed, and `ssh-keygen -Y verify -f allowed_signers -I ml-stack-release -n
ml-stack-release -s <asset>.zip.sig < <asset>.zip` checks it against a file holding `ml-stack-release <public key>`.
Build provenance attestations (`actions/attest-build-provenance`) are not enabled; they need `attestations: write` on
the calling workflows in `release-please.yml`.

## Recommended GitHub settings

Only the repository owner can change these.

- Branch protection or a ruleset for `main`: pull request required, status checks `test`, `gates`, `privacy`, `licenses`
  and the CodeQL `analyze` jobs required, force pushes and deletion blocked. `0.2dev` needs no more than blocking
  deletion.
- Settings > Code security: enable private vulnerability reporting (`SECURITY.md` points to it), Dependabot alerts and
  security updates, secret scanning with push protection, and the dependency graph (the dependency-review workflow needs it).
- Settings > Actions > General: default `GITHUB_TOKEN` permission read-only; allow actions from GitHub and verified creators
  (every action here is pinned by commit SHA); require approval for first-time contributors; do not send write tokens to
  workflows from pull requests.
- Tag ruleset for `v*`: restrict creation to the owner; block deletion and force updates.
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
