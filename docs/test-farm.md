# Test farm

`scripts/test TIER [FILE...] --on DEVICE` runs the tests on another device of the pool, so an agent on the Mac
tests Windows, WSL, Linux or another Mac without anyone at that device. It is the entry point the local tiers
use: same tier names, same printed lines, same content-keyed reuse, and `--on all` runs it on every other device.

```
scripts/test all tests/test_x.py --on "Windows PC"     those files on one device (name or fingerprint)
scripts/test quick --on all                            the files the change reaches, on every other device
scripts/test gate --on wsl-box                         the structural checks there
scripts/test full --on all --split                     divide the list between this machine and the devices
poolhouse-test-devices                                  the devices: platform, free slots out of two, last result
```

The exit status is the worst of the runs; 70 means a device never ran its part (not in the pool, shards off,
unreachable, refused, Python 3.13 missing). A refused or unreachable device is named on stderr with its reason.

## How the owner turns it on

Nothing runs on a device until its person says so, per device, and the default is off. There are two switches,
both off, and both audited:

1. The experimental feature `remote-tests` (`poolhouse.features`), on each machine that sends or takes tests:
   `poolhouse features enable remote-tests`. `scripts/test --on` refuses to start while it is off on the asking
   machine, and `consent on` refuses while it is off on the device. It is recorded in the authority audit log
   with who did it.
2. The device's own switch, at the device (its agent does it on the person's order; see `docs/windows-wsl.md`),
   from this repository's checkout, with the Python 3.13 the tests should use:

```
python -m poolhouse.testfarm.consent on --from "MAC NAME"     # allow DEVICE... | deny DEVICE... | off | status
```

A DEVICE is a name from `poolhouse-test-devices` or a fingerprint. **Being in the pool is not being allowed.**
The device takes tests only from the devices in its `allowed` list, which is empty until a person names some
(`--from`, `allow`), so a device that joined the pool by itself (for instance under policy `open` on the same
network) can ask whether tests are taken (`shard_caps`) and nothing else: `shard_put`, `shard_start`,
`shard_status` and `shard_cancel` are refused with `denied`, and the refusal is an audit entry on the pool board
(at most one a minute for each device). Allowing and denying are audit entries naming the session that did it and
the device. `deny`, and putting a member out of the pool (`member_revoke`), take it off the list, cancel its runs
that are going, and a revoked member is also refused at the connection.

`on` is `shard_consent` on the device's own node: it saves `<state>/shards.json` (on, the interpreter, the
checkout), refuses a Python that is not 3.13 and writes an `audit` entry to the pool board naming the session
that did it. `off` stops the next upload at once (the file is read at every request). Turning it on lets any
other device of the pool run a tree's tests here as this user, so do it only on a device whose pool you trust;
revoking a pool member (`member_revoke`) refuses it at its next request. `poolhouse features disable remote-tests` also switches this device's node off.

## What runs where

| step | where | what |
| --- | --- | --- |
| pick the devices | the Mac | the pool's other active members, from the local node; `all` or a name or a fingerprint |
| ask what they take | the Mac's node to each device | `shard_caps`: accepts or why not, platform (a WSL Linux says `wsl`), free slots |
| reuse | the Mac | a file that passed on that device with this content, platform and tier is not sent |
| pack and upload | the Mac to the device | the working tree as a gzip tar (uncommitted edits travel), 256 KiB at a time, over the node's TLS |
| run | the device | its node checks the digest, takes CPU slots from its lease service, starts the executor on the device's own Python 3.13 and checkout, which unpacks the tree into a scratch checkout and runs `scripts/test TIER --no-reuse` there |
| result | the device to the Mac | per-file counts, failing node ids with messages, platform, wall and CPU seconds, an output tail |
| record | the Mac | the printed lines, the per-device ledger, and a `test-result device=NAME ...` message on the project board |

## Mechanism

The node (`app/poolhouse-node/src/shard/`) is the transport, so a shard rides the pool's own membership: a
request is a peer op on a TLS 1.3 connection pinned to the certificate the pool record holds for the device, and
the sender is that certificate, never a field of the request. Revoked or unknown devices are refused at every
op. The local API has two methods: `shard_call` (a registered session asks one op of a pool device; the node adds
`op` and `by`, the session behind the token, as a label for the audit trail) and `shard_consent`.

- Peer ops, members only (`shard_caps` for any member; the rest only for a member on the allowed list): `shard_caps`, `shard_put` (one chunk, in order, from offset 0, 256 KiB at most, hex),
  `shard_start`, `shard_status`, `shard_cancel`. A shard belongs to the certificate that created it: another
  member gets "no such shard" for its status and its cancel.
- The job: `id` (32 hex), `tree_sha256`, `size`, `tier` (`all fast full gate slow`), `files` (0 to 200 names, each
  `tests/NAME.py`, none for the gate) and `timeout_s` (1 to 3600). Any other field (argv, env, cwd) is refused.
  The executor builds the only command it runs: `python scripts/test TIER --no-reuse --junitxml=OWN FILE...`.
- The tree: tracked and untracked-not-ignored files under `src tests scripts contracts patches packaging docs app`
  and top-level files. Limits: 24 MiB packed, 256 MiB unpacked, 20000 members; plain files only, no links, no `..`,
  no absolute, hidden, backslash or drive paths; a gzip stream that inflates past the limit is refused before a
  member is read. The node checks the sha256 of the upload before anything starts; the executor checks everything
  else again (`fleet/shard_spec.py`, `shard_tree.py`) because it trusts nothing it did not verify.
- Concurrency and the lease: two shards run at once; six are open at once (uploading or running), two uploading
  per sender; an upload idle for ten minutes is dropped. Each running shard holds the device's CPU slots (half
  its cores, at least one, class background) in the node's lease table until it ends, so it queues behind and
  yields to the device's own work like any other holder. `shard_start` and `shard_done` are `audit` entries on
  the pool board.
- Cancel and time: the executor enforces `timeout_s` on the runner (an interrupt, then a kill). Cancelling, a
  node stop or `timeout_s` plus 120 s makes the node send SIGTERM to the executor, which interrupts the runner;
  after 45 s it kills the executor's whole process group, and it always kills the group when the executor has
  gone, so no child outlives a shard. The runner shares the executor's group (the node leads it). Windows:
  `taskkill /T /F` on the tree.
- Output and results: a runner that writes more than 64 MiB is cancelled (exit 124); a result is at most 900 KiB
  (3000 slowest tests, 200 failures); the executor's own output tail is 8000 characters.
- Environment: the executor starts without `PYTHONPATH`, `PYTHONHOME` and the variables that mark an agent's
  shell (`CLAUDECODE`, `AI_AGENT`, `POOLHOUSE_WORKSPACE_*`, `POOLHOUSE_SESSION_*`, ...); the runner also loses
  anything that looks like a credential, and gets its own `POOLHOUSE_HOME` inside the scratch folder.
- Per-device results: the reuse key holds the device's fingerprint, its platform (`wsl` included), Python and the
  tier besides the file's content key, so a pass on Windows never satisfies macOS or WSL. Only a run that ended
  with pytest's exit 0 or 1 records passes, and only for files with passes and no failures. The ledger is
  `remote-results.json` in the project's reuse folder; `poolhouse-test-devices` reads the last result from it.
- Retired: the paired-device HTTP route (`/workspace/v1/test-shards`), `Settings.test_shards` and
  `fleet/shard_consent.py`. There is one remote path.

## Windows, WSL, Linux

Each of those is its own pool device with its own node, certificate and consent; WSL is not the Windows around
it. The Python used is the one recorded when the person turned shards on (`--python`, default the interpreter
running the command) and it must be 3.13: otherwise the node says "Python 3.13 is needed and PATH is X" and the
sender sees it as a refusal. Files are written byte for byte and the scratch checkout is made with
`core.autocrlf=false`, so line endings are the sender's. Paths in the tree are POSIX relative names; the device
joins them with its own separators, and a name with a backslash or a drive is refused.

## Tests

`app/poolhouse-node/tests/shard.rs` (two real devices, TLS on loopback, a shell stub for the interpreter): consent
off by default, membership and revocation, a forged requester, every malformed request, ordering and size of an
upload, the lease, two at once, cancel killing a grandchild. `tests/test_testfarm.py` (the executor as the node
starts it, over a real git tree whose `scripts/test` is a stub: hostile trees and jobs, the environment, the
ledger keys, the report, the command line) and `tests/test_testfarm_nodes.py` (two node processes on loopback
ports, no beacon, no multicast, with a real executor: consent, a run, reuse, a failure, the gate, an escaping
tree, cancel, the two-at-once cap, devices by name and fingerprint) cover the Python side.

## Not verified

Native Windows has not run it: the Rust side compiles for it (the process helpers use `taskkill`) and the
executor avoids POSIX-only calls, but `scripts/test` itself has only run on macOS and Linux, and WSL start-up,
the firewall and the real network are the device setup's (`docs/windows-wsl.md`). Neither has a speedup been
measured on real machines.
