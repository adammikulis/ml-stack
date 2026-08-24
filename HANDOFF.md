# Handoff

Pending work on `main`. Read `CLAUDE.md` first.

## Before tagging v0.1.4

The suite and `docs/verify_release.py` both pass, but neither has run a real
`llama-server`. Two things need a real machine:

- **`llama.py` asset names.** `_tokens()` guesses what the ggml-org release calls a
  build for each platform. Verified against a stubbed release, not a real one. Check the
  names on macOS arm64, Linux x64 and Windows x64 before trusting the Run button.
- **macOS quarantine.** A binary downloaded from the internet may refuse to run until
  the quarantine attribute is cleared. Not handled.

Then the release gate, which is two machines:

1. **The machine that installs nothing.** From the bundle only — no `llama-server`, no
   sidecar venv, no model store. Type the passphrase, open Chat, pick a model the other
   machine is serving, hold a conversation, reload, confirm it is still there. Check
   afterwards that no `llama-server` was fetched and no venv was built: hosting costs
   belong to the host.
2. **The machine that hosts.** From the bundle: fetch a model, press Run, confirm it
   appears to the first machine.

Then commit, `git tag v0.1.4`, push the tag. `release.yml` builds twelve wheels and three
bundles and publishes only on a `refs/tags/` ref. It sets `generate_release_notes: true`,
so GitHub builds the notes from commit subjects and the app shows that text to users —
the commit subject rule in `CLAUDE.md` is load-bearing.

## Asked for, not built

1. **Fleet-wide dataset catalogue.** `POST /fetch` moves files peer-to-peer; there is no
   index of what datasets each machine holds. Browse every dataset on every machine,
   deduplicated by content digest, so three copies of one path that do not match show as
   three that do not match.
2. **Two more recipes** — image classification, and fine-tuning from an existing model.
   `text-lm` and `classify-text` work; the shape is the same.
3. **Local-SGD.** One training run split across machines, averaging safetensors with
   numpy so a Mac and an AMD box contribute to one model.
4. **Progress for long downloads** in the interface, and pruning abandoned `.part`
   files. Both are listed under Known limits in `docs/FEATURES.md`.
5. **Semantic search over conversations.** Today the search is keyword. `embed()`,
   `cosine()` and `top_k()` in `ml-stack-client` are ready, and the cost is underneath
   them: llama-server cannot serve embeddings and chat at once, so it needs a second
   server, a second model download, and a vectors sidecar recording the model and
   dimension it was built with.
6. **Sharing conversations across machines.** They stay on the machine they were held
   on.

## Things that will bite you

- **`packaging/build.py` used to ship stale code.** It reuses a build venv and pip skips
  reinstalling an unchanged version. `--force-reinstall` handles it, but if a bundle
  behaves like code you already fixed, suspect this first.
- **Lazy imports are invisible to PyInstaller.** `ml_stack.serve` is reached only through
  a lazy import and is listed in `hidden` in both spec files. Anything else imported
  inside a function needs the same.
- **Several daemons on one machine share the discovery UDP port and only one answers.**
  A real fleet is unaffected. It shows up when simulating a cluster on one box, which is
  why the peer-copy check in `verify_release.py` is given an address rather than
  discovering one.
- **The full suite takes longer than the 120s Bash timeout.** Run it backgrounded and
  poll the output.
- **`node --check web/app.js` before trusting the interface.** A duplicate top-level
  `const` stops the whole page loading and no Python test catches it. There is a test for
  this, and another that checks every address the page calls is one the daemon answers.
- **The bundle cannot train without the sidecar environment.** `Environment` builds a
  venv, downloading a standalone Python when the machine has none new enough — macOS
  ships 3.9.

## Verifying

```
python -m pytest tests/ -q                    # exceeds a 120s Bash timeout
PYTHONPATH=$(ls -d packages/*/src | tr '\n' ':') python docs/verify_release.py
python packaging/build.py --bundle            # wheels plus a standalone app
```
