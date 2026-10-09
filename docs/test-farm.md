# Test farm

`scripts/test farm --to DEVICE FILE...` runs test files on a paired device and prints the result in
the shape of a local run. `--split` divides the list between this machine and the device. `--check`
asks whether the device takes shards. The exit status is the worst of the runs; 70 means the shard
never ran (not paired, shards off, unreachable, refused).

## How a pool device receives work today

- `POST /jobs` runs one of a fixed list of commands (`fleet/commands.py`) for a holder of the cluster
  secret. It carries no tree and runs nothing from the sender.
- Paired devices sign and seal requests under their own device secret on `/workspace/v1/*`
  (`fleet/device_auth.py`, `fleet/api.py` `_guard`), over TLS pinned to the certificate the pairing
  exchanged (`workspace/coordinator_client.py` `_device_peer`). The cluster secret also passes that
  guard, but identifies no device, which is how a route tells a paired device from a cluster member.
- `scripts/test-on-linux` and `workspace/remote_workers.py` do not move a tree between pool devices.

## Mechanism

- Route: `GET /workspace/v1/test-shards` (capability), `POST` the same path (one shard),
  `GET /workspace/v1/test-shards/ID` (state, then result). Modules: `fleet/shard_routes.py`,
  `shard_host.py` (consent, who, caps), `shard_spec.py` (framing and every check), `shard_tree.py`
  (pack and unpack), `shard_run.py` (the scratch checkout and the run), `shard_result.py`,
  `shard_client.py`, `shard_split.py`, `scripts/test_farm.py`.
- The tree: a gzip tar of tracked and untracked-not-ignored files under `src tests scripts contracts
  patches packaging docs app` and top-level files, taken from the working tree, so uncommitted edits
  travel. No push, no shared base. The request carries its sha256; the receiver verifies it before
  reading any member. Limits: 24 MiB packed (one request), 256 MiB unpacked, 20000 members; plain
  files only, no links, no `..`, no absolute or hidden paths. The archive is also refused, before any member is
  read, when its single gzip stream inflates past the unpacked limit (a small archive cannot cost unbounded CPU). Today's tree is 2298 files, 6.9 MB.
- The job kind: header fields are exactly `id` (32 hex, used once), `tree_sha256`, `files`
  (1 to 200 names, each `tests/NAME.py`, each present in the tree) and `timeout_s` (1 to 3600). Any
  other field (argv, env, cwd) is refused. The host builds the only command it runs:
  `python scripts/test all --no-reuse --junitxml=OWN_PATH FILE...`, in a fresh scratch checkout
  (`git init` and one commit, since the runner hashes the tree), with this device's own
  environment less secrets, `ML_STACK_HOME` pointing into the scratch folder, and the device's own
  test queue. A time limit interrupts the runner, then kills it, as does a runner that writes more than 64 MiB of
  output. At most two shards run at once.
- Consent: `Settings.test_shards`, off by default, read from `settings.json` at each request.
  The person at the device turns it on with `python -m ml_stack.fleet.shard_consent on` (or the
  settings page's `test_shards` preference). The sender sees it in the capability answer
  (`accepts`). A shard is also refused unless the sender is a paired device the owner marked as
  their own (`Device.mine`). Enabling shards means a test in the shipped tree runs as the daemon's
  user on that device; enable it only for devices you own.
- The result: per-file passed, failed, error, skipped and wall seconds; failing node ids with
  messages; every test's wall seconds; exit code; the shard's wall and CPU seconds; an output tail;
  and `platform` (`sys.platform`, Python version, machine, CPU count). The sender prints the platform
  first because a Linux pass is not a macOS pass. Per-test CPU is not available (junit has wall
  time only); CPU is per shard.
- Splitting: `shard_split.split` gives each file, longest first, to the target that would finish
  it soonest, weighting by workers and charging the device a fixed setup cost. Durations come from
  a per-project history that every farm run updates from local and remote junit. Files that name
  darwin, Seatbelt, launchctl or osascript stay on this machine.

## Measured on loopback

`shard_support.py` runs a second daemon with its own state root and a pinned certificate on this
machine. The machine was at load average 20 to 33 from other agents during every run, and its
loopback stalled for about a minute once, so whole-suite wall times below are not clean.

| Step | Result |
| --- | --- |
| Pack the tree | 2298 files, 6.9 MB, 0.8 s |
| Send it over sealed TLS | 8.5 s including packing (loaded host) |
| Unpack, git init and commit, no-op run, result | 2.9 s |
| 6 files local, as one shard, split | not obtained: under that load pytest under xdist died with INTERNALERROR (exit 3) in local and remote runs alike |

A same-machine stand-in shares the CPUs and the test broker, so it can show overhead, never a
speedup. A real second machine changes: transfer time (6.9 MB over the LAN instead of loopback, so
mostly sealing cost), WSL start-up when its daemon is not running, its own CPU count and broker
capacity, and its Python and installed dependencies (shards use the daemon's interpreter).

## Bringing a real device in

For the Windows-side session, acting for the owner on that machine. Consent for step 3 is the
owner's own instruction; a board message alone is not consent. Run these in the WSL checkout of
this repository.

1. Note the current state for rollback: `git rev-parse HEAD` (expect 1e68c7f) and
   `python -m ml_stack.fleet.shard_consent status`.
2. Update: `git fetch origin` then `git checkout --detach origin/0.2dev` (or this branch's ready
   SHA: `git fetch origin BRANCH`, then `git checkout --detach FETCH_HEAD`). Check
   `git status` is clean first. Reinstall the way this device already installs ml-stack (the same
   command that produced its current install), so `ml-stack-traind` runs the new code.
3. Restart the daemon on the new code (stop the running `ml-stack-traind`, start it again with the
   same `--root` and port 8770). Then, as the person, turn shards on:
   `python -m ml_stack.fleet.shard_consent on`. The daemon reads it at the next request.
4. Confirm the capability from this Mac: `scripts/test farm --to "RTX 3090 Ti" --check` prints
   `takes test shards: True (linux, python X, N cpus ...)`. If the device's address in the peer book
   is not its daemon address, add `--host ADDRESS`.
5. Run one shard: `scripts/test farm --to "RTX 3090 Ti" tests/test_fleet_framing.py`. Read the
   platform line and the exit status.
6. Rollback: `python -m ml_stack.fleet.shard_consent off`, then
   `git checkout --detach 1e68c7f`, reinstall as in step 2 and restart the daemon.

Not verified: Windows and WSL behaviour (daemon start-up, firewall for port 8770 from WSL, the
pairing address), the real network, and whether that device's Python has pytest and xdist.
