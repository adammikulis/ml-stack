# Handoff

State as of the last commit on `main`, `b5f37b8 Version 0.1.3`.

Read `CLAUDE.md` first. It has two rules, both from the owner, both non-negotiable:
no rationale prose in code, and commit subjects that say what changed and nothing else.

## Released

`v0.1.3` is published with bundles for macOS (Apple silicon), Linux and Windows, plus
the twelve wheels and the two network installers. Verified by installing the published
artifact and training with it.

macOS Intel is not supported and its runner is out of the release matrix.

## Uncommitted work in progress

Four new files and seven modified, none committed. The full suite is green with all of
it: **736 passing**. Two features, both nearly done.

### 1. Inference proxy — working, tested, ready to commit

`packages/ml-stack-fleet/src/ml_stack/fleet/serving.py` (new) and a `_proxy` handler in
`daemon.py`.

A machine registers the port its llama.cpp server is on. The daemon forwards
`/infer/*` to `127.0.0.1:<port>`, so the model server stays on loopback and the only
LAN-exposed port is the one that already demands a token. Beacons carry which models a
machine is serving; `discover_serving(key)` returns `Endpoint`s, and
`Client(**endpoint.client_kwargs())` works with **no change to `Client`** — it already
accepts `api_key` and sends it as a bearer token.

`tests/test_fleet_serving.py` — 14 tests, all passing. The one that matters is
`test_a_streamed_completion_arrives_as_it_is_generated`: it caught a real bug where
`response.read(8192)` blocked until it had 8192 bytes, so a whole completion arrived at
once. Fixed with `read1`. Verified the test goes red against the old code.

### 2. Model catalog — working, one route missing

`packages/ml-stack-fleet/src/ml_stack/fleet/models.py` (new).

`Models.ensure(name)` gets a model, **preferring the network**: if a peer advertises it,
pull it peer-to-peer; otherwise download from `hf:owner/repo/file.gguf` or a URL.
Verified end to end — a 6 MB model pulled from one daemon to another over the LAN in
2.0s, and the internet fallback separately.

Daemon routes `GET /models`, `GET /models/{name}`, `POST /models/get` are wired.
`Peer.models()` and `Peer.get_model()` exist. `Peer.pull` gained a `route=` parameter so
it can pull from `/models/` as well as `/files/`.

`tests/test_fleet_models.py` — 17 tests, all passing.

**What is missing:** the `/ui/models` route in `ui.py`. Two attempts to insert it failed
silently because the anchor string had already changed. The Models screen in
`web/app.js` is written and calls `/ui/models`, so the screen currently returns
`{"error": "no such route"}`. Insert the handler before the `/ui/updates` route
(around line 445 of `ui.py`); the body was drafted and is straightforward:
`GET` returns `{here, elsewhere, free_gb, autodownload}`, `POST` calls
`ui.models.ensure(...)`. `ui.models` is already wired in `serve_forever`.

`Settings.autodownload_models` exists and defaults to True; the checkbox is in the
Settings screen.

### Unanswered question from the owner

**"How do we handle partial downloads?"** — asked, not yet answered. The honest state:

- **Peer-to-peer** (`Peer.pull`): lands in `.part`, resumes with `Range`, verifies the
  whole reassembled file against a sha256 header, `os.replace` only when complete. On a
  digest mismatch it deletes the `.part`, because resuming would preserve the
  corruption. A `.part` larger than the remote file is refused rather than spliced.
- **Internet** (`Models._from_internet`): lands in `.part`, sends `Range` if one exists,
  keeps the `.part` on a short read. **But** there is no digest to check against, and
  unlike `Peer.pull` it does not guard against resuming a `.part` that belongs to a
  different file of the same name. That is a real gap and the likely next fix.
- Nothing prunes abandoned `.part` files, and a long download reports no progress to the
  UI.

## Verifying

```
python -m pytest tests/ -q          # 736 tests, ~2m16s; exceeds a 120s Bash timeout
python docs/verify_release.py       # 29 checks against real daemons and real training
python packaging/build.py --bundle  # wheels plus a standalone app
```

`docs/FEATURES.md` describes every shipped feature and each has a check in
`verify_release.py`. It has a "Known limits" section — keep it honest.

## Things that will bite you

- **`packaging/build.py` used to ship stale code.** It reuses a build venv, and pip
  skips reinstalling an unchanged version. Fixed with `--force-reinstall`, but if a
  bundle behaves like code you already fixed, suspect this first.
- **Several daemons on one machine share the discovery UDP port and only one answers.**
  A real fleet is unaffected. It only shows up when simulating a cluster on one box,
  which every local demo does.
- **The full test suite takes longer than the 120s Bash timeout.** Run it with
  `run_in_background: true` and poll the output file.
- **`node --check` on `web/app.js` before trusting the UI.** A duplicate `const` at the
  top level stops the whole page loading and no Python test will catch it. There is now
  a test for this.
- **The bundle cannot train without the sidecar environment.** `Environment` builds a
  venv, downloading a standalone Python if the machine has none new enough — macOS ships
  3.9. `ml-stack-train` needs `ml-stack-contracts` and `numpy` declared; that was a real
  missing dependency, fixed.

## What the owner has asked for that is not built

In roughly the order it came up:

1. **Fleet-wide dataset catalogue.** `POST /fetch` moves files peer-to-peer already;
   there is no index of what datasets each machine holds. The owner asked for this
   specifically: browse every dataset on every machine, deduplicated by content digest
   so three copies of the same path that do not match show as three that do not match.
2. **Two more recipes** — image classification and fine-tuning from an existing model.
   `text-lm` and `classify-text` work; the shape is the same.
3. **Local-SGD.** Splitting one training run across machines, averaging safetensors with
   numpy so a Mac and an AMD box can contribute to one model. This was the original ask
   and is still not built.
4. **Progress for long downloads** in the UI, and pruning abandoned `.part` files.

## Owner preferences worth knowing

- Wants zero command line for end users. Any instruction that says "run this command"
  is a bug unless it is behind an "already have a terminal?" disclosure.
- Dislikes being told what used to be broken. Write for someone seeing it for the first
  time.
- Wants decisions surfaced as choices, not defaults chosen on their behalf — the wizard
  pre-ticks boxes and shows the reason beside each.
- Asked to keep fake device reports for testing. They exist only in test fixtures.
