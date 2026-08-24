# Handoff

Pending work. Read `CLAUDE.md` first — its rules are not suggestions, and three of them
were written because they were broken in the session that produced v0.1.4.

`main` is the default branch and holds everything. `v0.1.4` is published with 17 assets.
The suite is 908 passing, `docs/verify_release.py` is 40/40.

## In flight

**A `release-notes` branch, not merged.** `git worktree list` shows a second worktree at
`../ml-stack-notes` holding an uncommitted change to `.github/workflows/release.yml`.

`generate_release_notes: true` lists merged pull requests. Everything lands here as
direct commits, so v0.1.4's release body is one compare link and nothing else. The change
builds the body from `git log --no-merges` between the previous tag and this one. It was
run against the real `v0.1.3..v0.1.4` range and produced all 21 subjects plus a compare
link, but has never run inside the workflow.

The owner asked whether a library should do this instead. It is an open question:

- **git-cliff** groups by Conventional Commit type, one binary, no repo restructuring,
  and its `cliff.toml` can group on plain subjects instead of prefixes.
- **release-please** also bumps the version across all twelve `pyproject.toml` files and
  `packaging/ml-stack-app.spec`, which is thirteen files edited by hand at every release.
  That is the part worth having.
- Both want Conventional Commits. Nothing in `CLAUDE.md` forbids `feat:` / `fix:` — a
  previous session claimed it did, and was wrong. `feat: native window instead of a
  browser tab` satisfies the rule as written.

Decide between the `git log` version and a tool before the next release. Do not merge the
branch to avoid the decision.

## Parity: a coding agent cannot drive all of this

The owner asked for parity between what the interface can do and what can be driven
without it. There is not parity.

Twenty-one `/ui/*` routes; `ml-stack-peers` has `setup init key token ls pause resume
when busy`. Missing from the command line entirely: models, chat, conversations,
clusters, serving, updates, uninstall, libraries.

Most of the logic is already importable and does not need moving — `Models`,
`fleet.chat`, `Conversations`, `discovery.join`/`leave`/`memberships`, `uninstall.plan`/
`remove`, `updates.apply_if_newer`, `Environment.install`. A CLI over those is assembly,
not design.

**The one real gap is serving.** `UI.start_serving` and `UI.stop_serving` hold the only
code that leases a model server, passes the draft model and the context length, and
registers it. Nothing outside a live `UI` object can start a model. Move that into
`serving.py` as functions and have `UI` call them; then the API is complete.

## Never driven by a person

The suite and the release checks do not cover these.

- **Two machines.** A machine with the bundle and nothing else — no `llama-server`, no
  `ml-stack-serve` installed by hand, no venv, no model store — holding a conversation
  with a model a second machine is serving. This is the property the whole release is
  for. `verify_release.py` asserts it against two daemons on one box; nobody has done it
  across a network.
- **The interface on Windows or Linux.** CI proves the bundle starts and every screen
  answers on all three platforms. The wizard has only been clicked on macOS.
- **Run, on Windows or Linux.** The llama.cpp asset names resolve for all six platform
  and architecture pairs against a real release, and macOS arm64 is verified the whole
  way — downloaded, unpacked, executable, `--version` answers. No other platform has
  executed the binary.
- **Updating itself.** `relaunch` is tested with `Popen` intercepted, so it launches the
  right path and reports correctly. A copy replacing itself and coming back has never
  happened. v0.1.4 exists to update *from*, so this is testable now for the first time.

## Asked for, not built

1. **Fleet-wide dataset catalogue.** `POST /fetch` moves files peer-to-peer; nothing
   indexes what datasets each machine holds. Browse every dataset on every machine,
   deduplicated by content digest, so three copies of one path that do not match show as
   three that do not match.
2. **Two more recipes** — image classification, and fine-tuning from an existing model.
   `text-lm` and `classify-text` work; the shape is the same.
3. **Local-SGD.** One training run split across machines, averaging safetensors with
   numpy so a Mac and an AMD box contribute to one model.
4. **Semantic search over conversations.** Search is keyword. `embed()`, `cosine()` and
   `top_k()` in `ml-stack-client` are ready; the cost is underneath them. llama-server
   cannot serve embeddings and chat at once — `backend.py` drops `--jinja` when
   `--embeddings` is set — so it needs a second server, a second model download, and a
   vectors sidecar recording the model and dimension it was built with.
5. **Sharing conversations across machines.** They stay where they were held.
6. **`-ngl 99` is not a choice.** `ServerSpec` defaults `n_gpu_layers="auto"`, which is
   every layer, and the draft model gets `-ngld 99`. Pressing Run claims the whole GPU.
   The owner serves their GPU to something else; this should be a setting.

## Things that will bite you

- **The owner's GPU is serving other work.** Do not leave a `llama-server` running, and
  do not run `docs/verify_release.py` casually — it trains. Check `pgrep -fl llama-server`
  before finishing.
- **A fresh worktree has no `dist/`.** `test_fleet_environment.py` builds a real
  environment and finds the packages by walking up for a `dist/` holding wheels, which
  git ignores. Run `python packaging/build.py` there first, or link an existing one, or
  that one test fails for reasons unrelated to your work.
- **Lazy imports are invisible to PyInstaller.** `ml_stack.serve` is reached only through
  one, and is listed in `hidden` in both spec files. Anything else imported inside a
  function needs the same.
- **`Advertiser(group=...)` is a multicast address**, not a cluster name. Passing a name
  makes `inet_aton` refuse it and every advertiser fail to bind — a machine in two
  clusters announcing to neither. Clusters are told apart by the key their beacons are
  signed with.
- **Several daemons on one machine share the discovery port and only one answers.** A
  real fleet is unaffected. It is why the peer-copy check in `verify_release.py` is given
  an address rather than discovering one.
- **The full suite takes longer than the 120s Bash timeout.** Background it and poll.
- **`node --check web/app.js` catches syntax, not scope.** A helper defined inside one
  screen and used from another parses fine and throws at runtime. There is a test that
  extracts every address `app.js` calls and asks the daemon for each; there is nothing
  equivalent for identifiers.
- **The windowed app is one file outside macOS**, beside `ml-stack-headless`. It was a
  directory holding the executable with its runtime alongside, which `install.sh` skipped
  entirely — the test was `-f` against a directory — and `install.ps1` launched a path
  that did not exist. Keep them at the same level.

## Verifying

```
python -m pytest tests/ -q                    # 908, exceeds a 120s Bash timeout
PYTHONPATH=$(ls -d packages/*/src | tr '\n' ':') python docs/verify_release.py   # 40/40
node --check packages/ml-stack-fleet/src/ml_stack/fleet/web/app.js
python packaging/build.py --bundle
```

`ruff` reports about 45 findings; all but five are `E702` (`self._send(...); return`)
throughout `daemon.py`, which is house style. CI does not run it.

## Releasing

Bump thirteen files — twelve `packages/*/pyproject.toml` at line 7, plus
`packaging/ml-stack-app.spec` (`CFBundleShortVersionString`). Commit, `git tag vX.Y.Z`,
push the tag. `release.yml` builds twelve wheels and three bundles and publishes only on
a `refs/tags/` ref. `workflow_dispatch` on a branch builds all three platforms and
publishes nothing, which is how to test a packaging change without cutting a release.

Actions are free: the repository is public and every runner is a standard GitHub-hosted
one. Larger runners would be billed; there are none.
