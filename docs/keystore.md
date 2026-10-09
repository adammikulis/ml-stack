# The keystore

Poolhouse keeps one secret in the operating system's keystore (macOS Keychain, Windows
Credential Manager, Linux Secret Service): a random 32-byte master key, one item per OS user
(service `poolhouse`, account `master/<uid>:<login>`). Everything else that needs a key gets a
subkey cut from it, so a machine shows one item and one prompt instead of one per feature.
`poolhouse/keystore.py` is the only module that imports `keyring` (`tests/test_keystore_gate.py`
fails on any other).

## What uses it

| Purpose | Owner label | What is protected |
|---|---|---|
| `memory` | user / profile / store directory, plus the file's salt | the encrypted memory store (`docs/memory.md`) |
| `fleet-signing` | the state directory | the Ed25519 signing key, kept in `signing.key.wrapped` (`docs/onboarding.md`) |
| `credentials` | the credential name | values stored with `poolhouse credentials set NAME --keychain` (`docs/credentials.md`) |

A subkey is `HKDF-SHA256(master, info = purpose, owner, context)` with every field
length-prefixed. `wrap` and `unwrap` seal with AES-256-GCM under that subkey and bind purpose and
owner as AAD, so a value wrapped for one purpose or owner does not open for another.

## What a person sees

- Nothing at import, startup, `poolhouse credentials list`, an empty memory store, or
  `poolhouse-security keystore`.
- The first time a key is needed: one sentence ("Poolhouse will ask your computer to store one
  encryption key so your memory and fleet identity stay private. You will see one Keychain
  prompt.") and then the one OS prompt, which creates the master. A chat or tool reply gets the
  same sentence from `Keystore.pending_notice()`.
- After that, no prompt: the master is read at most once per process and kept in memory.

## Limits that keep it quiet

The ceilings stop prompt and store spam, not ordinary reads of an item the person already
authorised, so they are counted in two classes (`rate.json`, a list of times per class):

| Limit | Value |
|---|---|
| Reads of the existing master per user per hour | 600 (`READ_CEILING`); the next one raises `KeystoreBusy` and logs an event. From 80% of the ceiling each read first sleeps a random 50 to 250 ms, inside the lock, so a stampede slows down before it stops |
| Creates and deletes per user per hour, plus every read made after an earlier refusal (a retry) | 5 (`WRITE_CEILING`); the next one raises `KeystoreBusy` |
| After the OS refuses or the person declines | the process stops asking for good; other processes wait 10 minutes (`denied.json`); `KeystoreDenied` with one plain message. A read that succeeds again removes the mark |
| Processes starting together | one lock (`flight.lock`): each process reads the OS keystore itself, one at a time, polling the lock every 20 ms and backing off to 0.5 s. The master is never written to a file or shared between processes, so a swarm costs one backend read per process, not one per use |
| Background process (no terminal and no desktop session, a Windows session 0, or any agent marker: `POOLHOUSE_NONINTERACTIVE`, `POOLHOUSE_AGENT`, `CLAUDECODE`) | never creates the master; reads it only after `poolhouse-security unlock`; otherwise `KeystoreLocked` naming that command |

Why two classes: a swarm of subagents starts dozens of processes in an hour and each reads the
master once. One shared ceiling of 20 refused the twenty-first process (measured: 40 agents
starting together gave 22 `KeystoreBusy`). Reads of an item that exists never prompt, so they get
a ceiling a swarm cannot reach by accident (600) but a loop still hits. Anything that can make
the OS show a prompt or change the store (create, delete, retry after a refusal) keeps a ceiling
of a few per hour. `keystore-reset` keeps the counts, so reset and create in a loop is stopped by
the same 5.

Every backend call is a sentinel event `keystore.read`, `keystore.create` or `keystore.delete`
(subject `purpose:<label>`, evidence `{"outcome": ...}`), and a refusal is `keystore.denied` or
`keystore.refused`. No value is ever in an event, a log line or a state file.

| A backend call that does not return | bounded: 300 s for a person, 20 s for a background process (which on macOS also tells the Security framework to fail rather than show a dialog); the timeout latches the same ten-minute refusal |
| A master that vanished after it was made | not replaced by the next process; `KeystoreMissing`. Only `poolhouse-security unlock` starts over |
| A plaintext or null keyring backend (`keyrings.alt`) | refused: `KeystoreUnavailable` |

Platform findings and what is unverified: `docs/keystore-platform-audit-2026-10-08.md`.

## Commands

All are `poolhouse-security` subcommands; an agent's tool call naming them is refused.

```
poolhouse-security unlock          # a person at a terminal: make the master if missing, clear a refusal,
                                  # let background processes (daemons, jobs) read it from now on
poolhouse-security keystore        # provisioned? refused for how long? reads and writes this hour (no keystore call)
poolhouse-security keystore-reset  # delete the master after typing the account name back
```

## Reset

`keystore-reset` deletes the master item and the state files. Everything wrapped under it is
unreadable afterwards: memory stores, wrapped signing keys, wrapped credentials. Restore from the
passphrase fallback or start those stores again (`poolhouse-memory forget --all`, a new fleet key).
There is no way to rotate the master and keep the data.

## Items older versions kept

| Old item | Moved by | How |
|---|---|---|
| service `poolhouse-memory`, one JSON key per user/profile/directory | the first open of that memory store | every file of the store is re-sealed under the new subkey and read back; then the item is deleted. A file that does not verify keeps the item |
| service `poolhouse`, account `onboard-signing-<hash>` (the signing seed) | the first signing call | wrapped into `signing.key.wrapped` and unwrapped to the same key id; the item is deleted, then the record's `store` changes to `keystore`. Cut off at either point, the next call finishes |
| service `poolhouse`, account `<NAME>` (a credential) | the first `poolhouse-credentials` command a person runs that names it | wrapped into `credentials.json`, read back equal, then the item is deleted. Library code asking for a credential never probes the keystore for an old item |

## Passphrase fallback

With no usable keystore, `POOLHOUSE_MEMORY_KEYS=passphrase` and the signing key's encrypted file
keep working; their scrypt derivation is `keystore.scrypt_key`. A person chooses it; the keystore
never falls back to it by itself.

### Passphrases in the environment

`POOLHOUSE_SIGNING_PASSPHRASE` (unlocks `signing.key.enc`) and `POOLHOUSE_MEMORY_PASSPHRASE` (with
`POOLHOUSE_MEMORY_KEYS=passphrase`) exist for a machine with no keystore, such as a headless box or a
container. They stay, as an explicit choice, and the exposure is this: an environment variable is
readable by every process of the same user (`/proc/<pid>/environ` on Linux, `ps eww` or process
inspection elsewhere), it is inherited by every child the process starts (a hook, a build, an agent's
tool call), and it lands in whatever dumps an environment (a crash report, `env`, a CI log that echoes
it). Prefer the terminal prompt where a person is there. Where one is not, set the variable on the
one service that needs it (a systemd `EnvironmentFile=` with mode 0600, not the login shell), never
in a profile or an image, and treat the passphrase as exposed to anything running as that user. The
keystore master is never put in the environment.

## Where the files live, and who may change them at once

- The state root must be on a filesystem with working permissions and locks. A root on a Windows
  drive mounted into WSL (`POOLHOUSE_HOME=/mnt/c/...`) has neither, so the keystore refuses it with
  `KeystoreUnavailable` naming `POOLHOUSE_HOME` before it creates anything; use a path under the Linux
  home.
- On Windows the keystore directory is cut to the owner when it is made: `icacls /inheritance:r
  /grant:r <user>:(OI)(CI)F`, so SYSTEM, Administrators and Users inherit nothing. If `icacls` fails
  or `USERNAME` is unset the keystore logs a warning and the directory keeps its inherited access.
  Only the mocked call is tested; a real ACL is not read back (see the audit, section 5).
- `credentials.json` and the cluster passphrase file are changed under `<file>.lock` (`lock.rewriting`),
  so two commands at once both land. A holder that does not let go in 10 seconds makes the next
  command fail with a plain message (`CredentialError`; the passphrase save says it was not saved).

## macOS: why the single prompt stays

After a host Python upgrade, or a switch of interpreter, macOS asks once whether the new binary may
read the master, because the item trusts the binary that made it. Creating the item with the
`security` CLI and `-T` so that a "stable launcher" is trusted was considered and not done:

- A launcher that is a script is not what the Keychain checks: the kernel runs the interpreter named
  on its first line, and the access list is matched against that process. Trusting a script path
  trusts nothing useful; trusting a compiled launcher means shipping and signing one.
- The one stable binary that would work is Apple's `/usr/bin/security`: the master would be read by
  running it (`security find-generic-password -w`), and a Python upgrade would not matter. That
  binary is trusted by path and by Apple's signature, and **any process of your user can run it**:
  a shell script, a `curl | sh`, a browser extension's helper, with no prompt. Today the item trusts
  one Python binary, which stops code that is not running under that binary. So the change widens who
  reads the master silently from "processes under this interpreter" to "any process of this user"
  and gains one fewer prompt after an interpreter change. A replaced launcher is not the risk (a
  different binary has a different identity and is not trusted); the risk is that the trusted program
  is a general reader.
- Writing the master through `security add-generic-password -w` puts it on the command line, readable
  with `ps` by any local user; `security -i` (stdin) avoids that, as `scripts/encrypted-volume.sh`
  now does, but the read side stays as above.

One prompt per interpreter change is the smaller exposure, so it stays. If that changes, the
decision is the owner's (`docs/keystore-platform-audit-2026-10-08.md`, D-1). Not verified against a
real item: the owner's login keychain was not touched.

## A boot-time service with no unlocked keystore

`python -m poolhouse.fleet.autostart system` (a LaunchDaemon, a systemd unit with `User=`, a Windows task at
startup) prints a warning and carries on when the service's user has not run
`poolhouse-security unlock`, and on macOS always notes that the login keychain is locked until that
user logs in. It does not refuse, because the service does not run unprotected without the key: it is
a background process, so it never creates the master, every sealed store (memory, the request inbox,
the activity log, the reputation ledger, wrapped credentials) stays locked and the fleet signing key
is not unwrapped. Nothing is written in the clear. It comes back by itself after `unlock` and a
service restart.

## In a frozen app

The PyInstaller spec names `keyring.backends.macOS`, `.Windows` and `.SecretService` as hidden
imports. A frozen build (`packaging/build.py --bundle --no-window`) bundled them already through the
PyInstaller contrib hook, and `poolhouse-headless -m poolhouse.net.cli keystore` printed
`backend: keyring.backends.macOS.Keyring`. Naming them stops a change to that hook from taking the
keystore out of the app. The Windows and Secret Service backends were bundled but not run (this was
checked on a Mac).

## What this does not do

Code running as you can call the keystore like any program of yours can; the master protects
against a copied file or backup, not against execution as the user. The hourly ceiling and the
refusal latch limit how often such code can make the OS prompt, not whether it can read a key
that is already cached in a process.
