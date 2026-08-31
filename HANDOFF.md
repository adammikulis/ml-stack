# Handoff

Pending work. Read `CLAUDE.md` first — its rules are not suggestions, and three of them
were written because they were broken in the session that produced v0.1.4.

`main` is the default branch and holds everything. `v0.1.5` is published with 17 assets.
The suite is 949 passing, `docs/verify_release.py` is 40/40.

## Parity: a coding agent cannot drive all of this

The owner asked for parity between what the interface can do and what can be driven
without it. There is not parity.

Twenty-one `/ui/*` routes; `ml-stack-peers` has `setup init key token ls pause resume
when busy`, and `ml-serve` has `status up down`. Missing from the command line entirely:
models, chat, conversations, clusters, updates, uninstall, libraries.

Every piece of the logic is already importable and does not need moving — `Models`,
`fleet.chat`, `Conversations`, `discovery.join`/`leave`/`memberships`, `uninstall.plan`/
`remove`, `updates.apply_if_newer`, `Environment.install`. A CLI over those is assembly,
not design.

## A model put up with `ml-serve` is not advertised to the other machines

`ml-serve up` leases through `ml_stack.serve` and records the port in the lease file.
`fleet.serving.Serving.register` is what puts a port in the beacon, and nothing calls it,
so a peer asking who is serving does not see it. Either `ml-serve up` registers when a
daemon is running on this machine, or the daemon learns to read the lease file.

## Never driven by a person

The suite and the release checks do not cover these.

- **Two machines.** A machine with the bundle and nothing else — no `llama-server`, no
  `ml-stack-serve` installed by hand, no venv, no model store — holding a conversation
  with a model a second machine is serving. This is the property the whole release is
  for. `verify_release.py` asserts it against two daemons on one box; nobody has done it
  across a network.
- **The interface on Windows or Linux.** CI proves the bundle starts and every screen
  answers on all three platforms. Every screen has been clicked only on macOS: the
  wizard, going back through it, and the question the close button asks.
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
6. **A machine holding only a draft model is never asked for it.** Drafts are left out
   of the listing a beacon carries, so `ensure_draft` asks the machines that hold the
   model itself, and falls back to the internet. A machine with the draft and not the
   model is invisible.
7. **`-ngl 99` is not a choice.** `ServerSpec` defaults `n_gpu_layers="auto"`, which is
   every layer, and the draft model gets `-ngld 99`. Pressing Run claims the whole GPU.
   The owner serves their GPU to something else; this should be a setting.

## PyPI

`ml-stack` is not published, and nothing in the repository tries to publish it. The
release builds the wheel and attaches it; that is where it stops.

PyPI refused twelve uploads and six retries with `429 Too many new projects created`,
an account-wide limit spent by the twelve names this repository used to have. The limit
clears on its own. The name `ml-stack` is free and `PYPI_API_TOKEN` is still a secret on
the repository -- delete it if it is not going to be used.

Publishing, when it is wanted, is `twine upload dist/*.whl` from a checkout, or the job
that was removed in this commit's parent.

## Stale

- **`docs/images/setup.jpg`** is the preferences step as it was before the wizard had a
  back button, and the step bar has one segment too many. Retaking it means a screenshot
  of the window, which needs screen-recording permission for whatever runs it.

## Things that will bite you

- **The owner's GPU is serving other work.** Do not leave a `llama-server` running, and
  do not run `docs/verify_release.py` casually — it trains. Check `pgrep -fl llama-server`
  before finishing.
- **A fresh worktree has no `dist/`.** `test_fleet_environment.py` builds a real
  environment and finds the packages by walking up for a `dist/` holding wheels, which
  git ignores. Without them its two real-environment tests skip. Run
  `python packaging/build.py` there, or link an existing `dist/`, to make them run.
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
  screen and used from another parses fine and throws at runtime, and so does a whole
  screen deleted by an edit that took too much. Two tests cover the common shapes: every
  address the page calls is asked of the daemon, and every `<name>Step` the wizard moves
  to is defined. Anything else has to be clicked.
- **`window.events.closing` runs on the thread that draws the window.** Anything that
  waits for the page to answer — `evaluate_js` above all — deadlocks there: the reply is
  delivered by the thread that is blocked waiting for it, and the window freezes with no
  way out but killing it. `Bridge.on_closing` hands that work to another thread. The same
  applies to `before_load`, `before_show` and `initialized`.
- **release-please publishes the notes it worked out when it opened the pull request**,
  not what `CHANGELOG.md` says. `release.yml` overwrites the body afterwards from the
  file, so the file is what a reader ends up with -- but between the two, the release
  briefly says something else.
- **A release pull request runs no checks.** Nothing triggers a workflow on a branch
  the Actions token pushed, so CI is silent on it and the version bump is unverified
  until it lands on `main`.
- **One package, twelve modules.** `packages/*` is gone: everything is `src/ml_stack/`
  under a single `pyproject.toml`. What used to be a package's dependencies is an extra
  of the same name, and `pip install ml-stack` still pulls in nothing.
- **The version goes on the end of a download's name**, never in the middle.
  `install.sh`, `install.ps1` and `updates.asset_for` all look for
  `ml-stack-<os>-<arch>` as a substring, and the copies already installed look for it
  in releases that do not exist yet.
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

Nothing is bumped or tagged by hand. Push to `main`; release-please opens a pull request
that raises the version in thirteen files and writes `CHANGELOG.md` from the commit
subjects since the last release. Merging it tags `vX.Y.Z`, publishes the release with
that changelog, and calls `release.yml` to build twelve wheels and three bundles and
attach them, adding the downloads to the bottom of the body.

`feat:` moves the minor, `fix:` the patch, `chore:` neither. A run of commits with no
prefix produces no release pull request, which is what an empty `/releases` page after a
week of work would mean.

`workflow_dispatch` on `release.yml` builds all three platforms and publishes nothing,
which is how to test a packaging change. A tag pushed by hand still works, and gets its
changelog from `git log` instead.

Actions are free: the repository is public and every runner is a standard GitHub-hosted
one. Larger runners would be billed; there are none.
