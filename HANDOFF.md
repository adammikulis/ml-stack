# Handoff

Pending work. Read `CLAUDE.md` first — its rules are not suggestions, and three of them
were written because they were broken in the session that produced v0.1.4.

`main` is the default branch and holds everything. `v0.1.4` is published with 17 assets.
The suite is 941 passing, `docs/verify_release.py` is 40/40.

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

1. **Thirteen files hold the version.** Twelve `packages/*/pyproject.toml` and
   `packaging/ml-stack-app.spec`, edited by hand at every release. `release-please`
   does exactly this, and would also write the changelog; it wants Conventional
   Commits, which nothing here forbids. The changelog is now built from the commits,
   so what is left to want from a tool is the version bump.
2. **Fleet-wide dataset catalogue.** `POST /fetch` moves files peer-to-peer; nothing
   indexes what datasets each machine holds. Browse every dataset on every machine,
   deduplicated by content digest, so three copies of one path that do not match show as
   three that do not match.
3. **Two more recipes** — image classification, and fine-tuning from an existing model.
   `text-lm` and `classify-text` work; the shape is the same.
4. **Local-SGD.** One training run split across machines, averaging safetensors with
   numpy so a Mac and an AMD box contribute to one model.
5. **Semantic search over conversations.** Search is keyword. `embed()`, `cosine()` and
   `top_k()` in `ml-stack-client` are ready; the cost is underneath them. llama-server
   cannot serve embeddings and chat at once — `backend.py` drops `--jinja` when
   `--embeddings` is set — so it needs a second server, a second model download, and a
   vectors sidecar recording the model and dimension it was built with.
6. **Sharing conversations across machines.** They stay where they were held.
7. **A machine holding only a draft model is never asked for it.** Drafts are left out
   of the listing a beacon carries, so `ensure_draft` asks the machines that hold the
   model itself, and falls back to the internet. A machine with the draft and not the
   model is invisible.
8. **`-ngl 99` is not a choice.** `ServerSpec` defaults `n_gpu_layers="auto"`, which is
   every layer, and the draft model gets `-ngld 99`. Pressing Run claims the whole GPU.
   The owner serves their GPU to something else; this should be a setting.

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

Bump thirteen files — twelve `packages/*/pyproject.toml` at line 7, plus
`packaging/ml-stack-app.spec` (`CFBundleShortVersionString`). Commit, `git tag vX.Y.Z`,
push the tag. `release.yml` builds twelve wheels and three bundles, writes the release
body from the tag range, and publishes only on a `refs/tags/` ref. `workflow_dispatch` on a branch builds all three platforms and
publishes nothing, which is how to test a packaging change without cutting a release.

Actions are free: the repository is public and every runner is a standard GitHub-hosted
one. Larger runners would be billed; there are none.
