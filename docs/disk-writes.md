# Disk writes

Measured 2026-10-08 on the owner's Mac, read-only, while several agents were working.

## How it was measured

- `psutil.Process.io_counters` does not exist on macOS, and `fs_usage` needs sudo (not available
  without a password), so per-process write bytes were **not available**. `/usr/bin/time -l` reports
  0 block output operations even for a store open that changes the file, so it is not usable either.
- Instead, `~/.poolhouse` (models, runtimes, llama.cpp, spec-venv, bench and _backups skipped) was
  walked every 2 s for 300 s and every file whose mtime changed was counted: touches per hour and
  size growth per hour. Bytes written by a rewrite are bounded by the file size, not measured.
- A `scripts/test all -n 1 tests/test_edit_guard.py` run was diffed over `~/.poolhouse`, the worktree
  and `$TMPDIR`; the result was swamped by other agents' concurrent runs and board traffic, so no
  per-test-run figure is claimed.

## Ranked findings (touches per hour, idle stack plus agents at work)

| rank | file | touches/h | size | write per touch |
| --- | --- | --- | --- | --- |
| 1 | `workspace/device-accounts.db` | 994 | 2.6 MB | a pure lookup (`account_for`) opened it read-write, which changed the file on every close |
| 2 | `traind/.../coordination.db` | 719 | 2.5 MB | heartbeats and claims (authoritative, kept); two read-only checks opened it read-write |
| 3 | `workspace-connections.json` | 635 | 30 KB | whole file rewritten by every `bind`, changed or not (about 19 MB/h) |
| 4 | `workspace-remote/*/worktree-lifecycle.db` | 515 | 2.9 MB | `remember` copied, rewrote, fsynced and promoted the whole store even when the record was identical |
| 5 | `guard/<tree>.json` (edit guard index) | per edit | 0.3-0.75 MB | rewritten on every edit anywhere in `src/poolhouse`; 78 files and 35 MB, never pruned |

Appended and kept: `audit.jsonl` (about 54 KB/h over the last 24 h, hash chained, fsynced),
`activity.log`, `sentinel/events.log` (HMAC chained). Not changed.

## Changed

- `device_accounts.account_for` opens the store read-only (falling back to a read-write open only
  when the schema is old) and creates nothing when the store is absent.
- `task_actions.working` and `resource_allocations.verified_binding` read the coordination store
  read-only.
- `project_connection.bind` writes the record only when the connections changed.
- `worktree_lifecycle.remember` reads first and writes only when the merged record differs.
- Tests: `tests/test_disk_writes.py` (unchanged input leaves the file byte and mtime identical;
  a changed input is still recorded).

## Kept (not weakened)

Signed journals and the hash chain (`workspace/chain.py` fsync on append), claims and leases,
authorization and keystore state, the sealed event log, and the lifecycle store's staged copy,
fsync and promote on a real change.

## What remains

- `scripts/hooks/claude-edit-guard` rebuilds and rewrites its whole index (4.5 s, 0.75 MB) after
  any `src/poolhouse` edit. The fix is an index keyed on `HEAD` plus a per-file (mtime, size) table,
  re-parsing only changed files in memory and rewriting only when `HEAD` moves, and pruning cache
  files of removed worktrees. Agents may not edit the guard (PROTECTED in the guard itself), so
  this is the owner's change.
- Bytes per SQLite/ladybug touch need `fs_usage` under sudo to quantify.
- Heartbeat interval for task leases (`HEARTBEAT_S`) was not changed; liveness semantics need a decision.
