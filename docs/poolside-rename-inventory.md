# Poolside rename inventory

Measured on 0.2dev (16456955 at measurement; worktree base 25d7b312) by `git grep`.
Scope is `docs/poolside-refactor-plan.md`, "Naming scope": `cluster` and `fleet` become `pool`,
and `ml-stack` / `ml_stack` become `poolside`, in one combined rename after the structural work.

| word | files | lines |
|---|---|---|
| cluster (any case) | 239 | about 2,400 |
| fleet (any case) | 475 | 3,835 |
| ml-stack / ml_stack | 1,697 | 14,639 |

Distinct tokens: 154 `ML_STACK_*` and 30 `MLSTACK_*` environment variables; 38 `ml-stack*` console
scripts in `pyproject.toml`; 100 tracked files under `src/ml_stack/fleet` (31 in `onboard`, 47 web
components), 74 tests named or filed under fleet.

## 1. Old to new map

### cluster to pool

Rule: `cluster`->`pool`, `clusters`->`pools`, `Cluster`->`Pool`, `CLUSTER`->`POOL`.

- Console script `ml-stack-cluster` (`ml_stack.fleet.join:main`) -> `poolside-pool`.
- Modules: `fleet/automatic_clusters.py`, `fleet/cluster_modes.py`, `fleet/lan_clusters.py`,
  `fleet/onboard/clusters.py` -> `automatic_pools`, `pool_modes`, `lan_pools`, `onboard/pools`.
- Web components: `cluster-view.html`, `cluster-actions.html` -> `pool-view`, `pool-actions`;
  ids `cluster-stat`, `cluster-pause`, `cluster-cards`, `cluster-sweep`, `cluster-joined`,
  `cluster-token`, `setup-cluster-*`; JS `inCluster`, `clusterMode`, `drawClusters`, `clustersStep`;
  gallery step id `cluster` (`ui/assets/gallery.js`).
- Identifiers: `cluster_key(_path)`, `load_cluster_key`, `create_cluster_key`, `shared_cluster_key`,
  `mint_cluster`, `join_cluster`, `cluster_group`, `cluster_mode(s)`, `known_clusters`,
  `default_cluster`/`DEFAULT_CLUSTER`, `clusters_path`, `ClusterRoutes`, `cluster_action`,
  `authenticated_cluster`, `reconcile_cluster`, `train.create_cluster_key` export.
- Flags `--cluster-key`, `--no-cluster`, `--cluster`; routes `/ui/clusters`, `/ui/setup/clusters`.
- Env vars `ML_STACK_CLUSTER`, `ML_STACK_CLUSTER_KEY` (and `*_PAUSE`, `*_ID` as found).
- JSON fields `cluster`, `cluster_id`, `cluster_key`, `clusters` in project connections, whoami,
  beacons, onboarding bodies; Android `CompanionProtocolTest` asserts `cluster_key` is rejected.
- `docs/images/cluster.jpg`; honey file `cluster.key.old` (`sentinel/honey.py`).
- Tests: 15 files carry the word in their path; about 150 test function names.

### fleet to pool

Rule: `fleet`->`pool`, `Fleet`->`Pool`, `FLEET`->`POOL`. A pool-of-pools reading appears where code
says "fleet pool"; none found.

- Package `src/ml_stack/fleet` -> `src/poolside/pool`; imports `ml_stack.fleet.*` (144 sites).
- Modules: `workspace/fleet_routes.py`, `activity/fleet_routes.py`, `redteam/scenarios/fleet.py`
  (scenario name `"fleet"` and `TARGET`), `fleet/onboard/*`, `fleet/join.py` (`STARTED_FILE`).
- Components: `fleet-nav`, `fleet-model`, `fleet-onboard`, `fleet-benchmark`, `fleet-invites`,
  `fleet-update`, `fleet-select-model` (custom event), `fleet-chat-ready`; `window.fleetModel`,
  `FleetNav`, `FleetModel`, `FleetInvites`, `FleetBenchmark`.
- Identifiers: `fleet_join`, `fleet_peers`, `pause_fleet`, `join_fleet`, `fleet_planned`,
  `fleet_measure`, `FLEET_VIEWS`, `FLEET_STEPS`, `FLEET_PORT`, `FLEET_BASE`, `FLEET_WRITES`,
  `Capability.FLEET`, `FleetTransport`, `Join-Fleet` (install.ps1), `cmd_fleet`.
- Flags `--fleet`; route `/ui/fleet`; `ml-stack-fleet-bench`; walk view
  `fleet`.
- Tests: 74 files under fleet names (`test_fleet_join`, `test_fleet_daemon`, ...).
- Prose: README and docs; `docs/fleet.md` -> `docs/pool.md`.

### ml-stack / ml_stack to poolside

- Package dir `src/ml_stack` -> `src/poolside`; every `import ml_stack...` and `from ml_stack ...`.
- Distribution `name = "ml-stack"` -> `poolside`; `packaging/ml-stack.spec` -> `poolside.spec`;
  bundle `ml-stack-headless` -> `poolside-headless`; install scripts `packaging/install.sh`,
  `install.ps1`, `smoke.py`.
- Console scripts, 38 in `pyproject.toml` (`ml-stack`, `ml-stack-traind`, `-peers`, `-cluster`,
  `-serve`, `-bench`, `-workspace`, `-security`, ...) -> `poolside`, `poolside-<same suffix>`
  (`-cluster` becomes `-pool`). `ml-stack-traind` is also the service name (`autostart.py`).
- Env vars: 154 `ML_STACK_*` (`ML_STACK_HOME`, `ML_STACK_NONINTERACTIVE`, `ML_STACK_LIVE_API`, ...)
  and 30 `MLSTACK_*` (`MLSTACK_GUARD`, `MLSTACK_JOBS_HOME`, ...) -> `POOLSIDE_*`; the script
  reports pairs that become the same name.
- Hook names and files: `scripts/hooks/*` messages and env names, `.claude/settings.json` env and
  hook entries, entries written by `agent_hooks.py` into the user's `~/.claude/settings.json` and
  `~/.codex/config.toml` (command `ml-stack-workspace`), `harnesshook.py` message prefix.
- Runtime paths: state root `.ml-stack` (`home.DEFAULT_NAME`), `~/.ml-stack/llama.cpp`, `venv`,
  runtimes and launchers in the interpreter `bin` directory (all `ml-stack-*` entry points).
- Docs: README, AGENTS.md, CLAUDE.md, docs/*.md prose (14,639 lines overall); CHANGELOG.md is
  release-generated history and keeps the old names.
- Desktop and mobile: Tauri `productName` is already `Poolside` (`Poolside.app` keeps its name);
  `identifier` `com.ml-stack.app` -> `com.poolside.app`; Android package `com.mlstack.companion`
  is a decision (end of section 6).
- Service labels: `com.ml-stack.traind` (`autostart.py`), `com.ml-stack.llama-build`
  (`serve/build_persist.py`), systemd unit and Windows task of the same stem.

## 2. Collisions with existing `pool`

Whole-word `pool` already exists as a local variable or concept in:

- `geo.py`, `bench/show.py`, `bench/gathered.py`, `bench/profiles.py`, `bench/score.py`,
  `decide/guards/__init__.py`, `world/{building,story,sentences,simulate,kinds}.py`: a local
  `pool` holding a candidate list. None of these files mentions cluster or fleet, so no scope
  holds both. They stay: they are local names inside functions and the module path
  (`poolside.world.kinds`) is unrelated.
- `interventions.py`, `bench/measure.py`, `bench/speed.py`: `ThreadPoolExecutor(...) as pool`;
  local, renamed to `workers` if a function ever holds both.
- Prose: "buffer pool" (`graph/cypher.py`), "cache pool" (`bench/options.py`, `bench/speed.py`),
  "pooled" scores (`bench/show.py`), "pooling" (`gguf/convert.py`), pooled keys (`spec/qwen4.py`),
  "pool-adjacent-violators" (`decide/calibrate.py`). Kept; none is a device group. The user-facing
  help for "cache pool" reads next to "device pool" in `--help`, so rename that flag's wording to
  "shared cache" when the rename lands.
- `docs/poolside-refactor-plan.md` quotes the old words; the script skips it (a trial rewrote
  its line to "`pool` and `fleet` with `pool`").
- Cluster and fleet both map to `pool`, so names containing both collapse: the test fixture
  `fleet-cluster` -> `pool-pool` (reword by hand to `pool-two`). The script reports any target
  path or top-level name already taken before writing.

## 3. Kept-list (words that mean something else)

cluster:

- `bench/folding.py`: clusters of names that one entity folds into (data clustering).
- `graph/community.py`, `graph/web/components/graph-3d.html`: graph community clusters, selected
  cluster labels.
- `world/names.py`: consonant clusters (phonology).
- `client/embed.py`: embedding task string `clustering`.
- `guard/sqlscan.py`: SQL `CLUSTER` statement keyword.
- `guard/argscan.py`: `cluster` as a production-scope argument key (Kubernetes cluster).
- `HANDOFF.md:269`: "corpus with cluster structure" (graph benchmark).
- `docs/poolside-refactor-plan.md` and `AGENTS.md` section 5 ("Vocabulary"): they quote the old
  word on purpose; `CHANGELOG.md`: historical release text.

fleet:

- `world/catalogue.py`: "warehouse and fleet software", "fleet routing" (vehicle fleets).

ml-stack:

- Licences, `NOTICE`, and minified vendored assets under `src/ml_stack/ui/assets`.
- Repository URLs in CHANGELOG links stay until the owner renames the repository.

## 4. Persisted names

All paths below are under the state root. The root itself moves `~/.ml-stack` -> `~/.poolside`.
Nothing here is migrated by code in the package; one script outside the package does the move
(section 5), run by the owner, with every daemon stopped.

### 4a. Files and directories

| old | new | holds |
|---|---|---|
| `~/.ml-stack` | `~/.poolside` | everything below |
| `cluster.key` | `pool.key` | pool membership keys (secret) |
| `cluster.json` | `pool.json` | memberships |
| `cluster.passphrases` | `pool.passphrases` | keystore-wrapped passphrases |
| `cluster.group` | `pool.group` | listed in `fleet/uninstall.py` |
| `cluster.key.old` | `pool.key.old` | honey decoy, regenerated |
| `fleet-daemon.json` | `pool-daemon.json` | daemon start record |
| `fleet-seen.json` | `pool-seen.json` | pause state seen per peer |
| `onboard/` (`devices.json`, `peers.json`, `tls/`, `peer-downloads.jsonl`) | same | paired devices, pinned certificates |
| `workspace/` (board, graph store, `tokens/NAME`) | same | board and attribution evidence |
| `activity/u-<user>/activity.log` | same | activity log, encrypted (AAD below) |
| `requests/requests.enc` | same | pending person requests, encrypted |
| `keystore/credentials.json` | same | wrapped credentials |
| `authority.json`, `policy.toml`, `policy.lock`, `person/statements.log` | same | authority and statements |
| `bench/runs.ladybug`, `jobs/`, `train/`, `ingest/`, `memory`, `models/`, `llama.cpp/` | same | runs, jobs, checkpoints, memory vault, model files |
| user-made `cluster-recovery.json` exports | `pool-recovery.json` | recovery files |

Moves, with daemons stopped (`ml-stack-cluster leave` is not run; membership is kept):

```
mv ~/.ml-stack ~/.poolside
cd ~/.poolside
mv cluster.key pool.key && mv cluster.json pool.json && mv cluster.passphrases pool.passphrases
mv cluster.group pool.group 2>/dev/null; mv fleet-daemon.json pool-daemon.json
mv fleet-seen.json pool-seen.json
```

JSON fields renamed inside `pool.json`, `onboard/devices.json`, saved project connections and any
`connection.json` the workspace writes: `cluster`->`pool`, `cluster_id`->`pool_id`,
`cluster_key`->`pool_key`. The state-move script rewrites them in place with a backup copy.
Wire bodies (`cluster_id` in beacons and onboarding) change in the new daemon, so every device in
a pool must be moved before it is switched on.

### 4b. Keystore, keyring and service entries

Key material is machine-setting territory: the owner runs these at his own terminal; agents do not.

- OS keystore entries: service `ml-stack` (`keystore.SERVICE`, `credentials.KEYRING_SERVICE`),
  account is the per-user account `Keystore.account`. New service `poolside`.
  macOS, for each account listed by `security dump-keychain | grep -A4 '"ml-stack"'`:
  `security add-generic-password -s poolside -a ACCOUNT -w "$(security find-generic-password -s ml-stack -a ACCOUNT -w)"`
  then, after verification, `security delete-generic-password -s ml-stack -a ACCOUNT`.
  Windows Credential Manager and Secret Service entries are copied the same way by target name.

### 4c. Cryptographic constants that carry the product name

These bytes are inputs to key derivation, authenticated data or wire signatures. Renaming one
without re-encrypting makes existing data unreadable or breaks interoperability between devices:

- `keystore.py`: `b"ml-stack/keystore/v1"` (`_SALT` and the field prefix), wraps every keystore
  blob (passphrases, credentials, memory vault key).
- `activity/log.py` `_AAD`, `requests/store.py` `_AAD`, `workspace/filestore.py` `_AAD`,
  `fleet/onboard/signing.py` `b"ml-stack/v1"`, `memory/vault.py` key id label.
- `fleet/onboard/joining.py` salt `b"ml-stack-join-v1/" + group`: derives the pool key from the
  passphrase. A rename changes every derived key; all devices must change together, or the
  passphrase-derived pools are re-minted (the plan allows a fresh pool).
- `macauth.py` `INFO`, `b"ml-stack-hkdf-salt"`, `sealing.py` `INFO`: request signing and sealing
  between peers; renamed together or peers refuse each other.
- `fleet/calibration.py` `b"ml-stack-fleet-bench"`: a test block only.

Recommendation: rename all of them; the state-move script (not the package) holds the old
literals, decrypts and re-encrypts the AAD-bound stores and keystore-wrapped files. A fresh
store is acceptable only where no attribution evidence lives; the activity log and `workspace/`
hold it, so they are re-wrapped, not dropped.

### 4d. Services, hooks and runtimes outside the state root

- launchd `com.ml-stack.traind`, `com.ml-stack.llama-build` plists in `~/Library/LaunchAgents`;
  systemd unit `ml-stack-traind.service`; Windows task: unload, regenerate under the new label
  (`poolside-pool join --persist`), delete the old.
- Interpreter `bin` entry points `ml-stack-*` (pyenv, `~/.local/bin`) and the runtime trees that
  contain them; the old distribution is uninstalled only after the new one runs.
- `~/.claude/settings.json` and `~/.codex/config.toml` hook entries invoking
  `ml-stack-workspace`; the repo's `.claude/settings.json` and `.git/hooks` chain
  (`scripts/install-hooks.py` rewrites them).
- Tauri identifier `com.ml-stack.app`: a new identifier moves the app's WebView storage and
  Keychain access list; sign-ins are repeated once.
- Pip distribution: `pip uninstall ml-stack` after installing the built `poolside` wheel.

### 4e. Must not be discarded

Running jobs (`jobs/`, `MLSTACK_JOBS_HOME`) and train checkpoints (`train/`), `workspace/` (board,
graph, tokens), `activity/` logs, `requests.enc`, `person/statements.log`, `authority.json`,
the memory vault, `bench/runs.ladybug`, `keystore/` and the OS keystore entries, `onboard/` (paired
devices and pinned certificates). Moves keep inodes (`mv` on one volume); re-wrapping keeps a
`.pre-poolside` copy beside each file until the owner deletes it. Pool membership keys can be
re-minted, jobs and attribution cannot.

## 5. Order, script plan and invariants

Order, keeping the installed system usable:

1. Land or park the in-flight branches (section 6); the rename branches from the resulting tip
   and nothing else lands until it does.
2. `scripts/rename_poolside.py --dry-run` prints every rewrite and every collision; review the
   report, not the whole diff.
3. Run it in a sibling worktree: `git mv` paths, then the content rewrite, then
   `ruff check --select I --fix` for import order, then regenerate generated references
   (`scripts/redteam_coverage.py --write`, `docs/commands.md` via `cli/reference.py`,
   frontend assets via `scripts/frontend-assets.mjs`).
4. Build a wheel (`python packaging/build.py`) and install it as a new distribution `poolside`
   into a new runtime prefix. Old runtime keeps running.
5. Stop daemons; owner runs `scripts/poolside-state-move` (dry-run first): root move, file moves,
   JSON field rewrite, re-wrap with old tags, keystore copy.
6. Start the new daemon; verify (`poolside-doctor`, `poolside-pool status`, board read, a
   recovery import, one signed peer request); regenerate hooks and service labels.
7. Delete old keystore entries, old runtime, old launchers.

Scripts, kept in `scripts/`, reviewed before use, no aliases left in the package:

- `scripts/rename_poolside.py`: ordered substitution table (longest token first:
  `ML_STACK_CLUSTER_KEY` before `ML_STACK_`, `ml-stack-cluster` before `ml-stack-`), case-preserving,
  skip list (section 3 files and `CHANGELOG.md`, `NOTICE`, `LICENSE*`, `tests/known-fixtures.txt`
  entries, minified assets), path moves, collision report, `--check` mode that fails if any
  skipped-by-rule file would change.
- `scripts/poolside-state-move`: dry-run default; backs up each file it rewrites; contains the
  old crypto literals; refuses when a daemon holds a lock.

Invariants after the rename (grep, all must hold):

- `git grep -Iiw cluster` lists only the section 3 sites.
- `git grep -Ii fleet` lists only section 3 sites.
- `git grep -I 'ml_stack\|ml-stack\|ML_STACK\|MLSTACK'` is empty except CHANGELOG, NOTICE,
  licence text, repository URLs and the state-move script.
- `git ls-files | grep -i 'cluster\|fleet\|ml.stack'` is empty.
- `scripts/budgets` totals equal or lower (ruff F811 and pyright included); `tests-collected` equal.
- Every console script in `pyproject.toml` starts `poolside`; `tests/test_wiring.py` and
  `tests/test_layers.py` pass.
- The state-move script on a copy of a real state tree followed by a daemon start reads the
  same jobs, board and activity counts as before.

## 6. In-flight branches that will conflict

About 60 open worktrees have commits not in 0.2dev (`git diff --name-only 0.2dev...BRANCH`):
every one touches `src/ml_stack/` or `tests/`, so the `ml_stack` import rename conflicts with all
of them at the line level. Largest by changed files (files total / fleet-or-tests files):

- 270+ files: `fix/linux-immutable-runner`, `fix/linux-admission-stdio-current`,
  `feat/nats-current-development-integration` (131 fleet or test files each).
- 180 to 215 files: `fix/nats-preserved-history`, `fix/fleet-browser-admission`,
  `fix/poolside-nonbrowser-gate`, `fix/poolside-urgent-ui-integration`, `fix/browser-kernel-integration`,
  `fix/linux-runner-immutable`, `fix/model-work-attribution`, `fix/prework-runtime-integration`,
  `fix/immutable-compute-generations`, `fix/test-kernel-isolation`, `fix/broker-neutral-handoff`.
- 40 to 75 files: `fix/poolside-dm-heading-null`, `feat/poolside-studio-mlx`,
  `feat/board-history-pages`, `integrate/poolside-ui-publication`, `feat/nats-cpu-reviewed-integration`,
  `feat/dev-cpu-route`, `integrate/dev-cpu`, `feat/dev-cpu-verification-integration`,
  `fix/authenticated-agent-messages-nats`, `fix/poolside-chat-capabilities`, `feat/nats-profile-integration`.
- Under 40 files: about 35 more (`fix/remote-worker-controls`, `fix/durable-fleet-jobs`,
  `feat/agent-device-registration`, `fix/worker-reconnect`, `fix/source-recovery-grants`, ...), in
  `src/ml_stack/fleet/`, `src/ml_stack/workspace/` and `tests/test_fleet_*`.

Resolution: land the large integration branches first; for the rest, rebase after the rename by
re-running `rename_poolside.py` on the branch (it is idempotent) instead of resolving text
conflicts, then `git rebase` onto the renamed tip. Path-level conflicts (rename versus
edit of `src/ml_stack/fleet/*`) are resolved by `git mv` through the script, not by hand.
Locked agent worktrees must finish or hand off before the freeze.

Decisions the inventory cannot make: whether the Android package `com.mlstack.companion` is
renamed (it changes the installed app's identity), whether crypto constants are re-wrapped or
the stores started fresh (section 4c), and the repository URL rename.
