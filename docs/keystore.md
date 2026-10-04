# The keystore

ml-stack keeps one secret in the operating system's keystore (macOS Keychain, Windows
Credential Manager, Linux Secret Service): a random 32-byte master key, one item per OS user
(service `ml-stack`, account `master/<uid>:<login>`). Everything else that needs a key gets a
subkey cut from it, so a machine shows one item and one prompt instead of one per feature.
`ml_stack/keystore.py` is the only module that imports `keyring` (`tests/test_keystore_gate.py`
fails on any other).

## What uses it

| Purpose | Owner label | What is protected |
|---|---|---|
| `memory` | user / profile / store directory, plus the file's salt | the encrypted memory store (`docs/memory.md`) |
| `fleet-signing` | the state directory | the Ed25519 signing key, kept in `signing.key.wrapped` (`docs/onboarding.md`) |
| `credentials` | the credential name | values stored with `ml-stack credentials set NAME --keychain` (`docs/credentials.md`) |

A subkey is `HKDF-SHA256(master, info = purpose, owner, context)` with every field
length-prefixed. `wrap` and `unwrap` seal with AES-256-GCM under that subkey and bind purpose and
owner as AAD, so a value wrapped for one purpose or owner does not open for another.

## What a person sees

- Nothing at import, startup, `ml-stack credentials list`, an empty memory store, or
  `ml-stack-security keystore`.
- The first time a key is needed: one sentence ("ml-stack will ask your computer to store one
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
| Background process (no terminal and no desktop session, or `ML_STACK_NONINTERACTIVE`) | never creates the master; reads it only after `ml-stack-security unlock`; otherwise `KeystoreLocked` naming that command |

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

## Commands

All are `ml-stack-security` subcommands; an agent's tool call naming them is refused.

```
ml-stack-security unlock          # a person at a terminal: make the master if missing, clear a refusal,
                                  # let background processes (daemons, jobs) read it from now on
ml-stack-security keystore        # provisioned? refused for how long? reads and writes this hour (no keystore call)
ml-stack-security keystore-reset  # delete the master after typing the account name back
```

## Reset

`keystore-reset` deletes the master item and the state files. Everything wrapped under it is
unreadable afterwards: memory stores, wrapped signing keys, wrapped credentials. Restore from the
passphrase fallback or start those stores again (`ml-stack-memory forget --all`, a new fleet key).
There is no way to rotate the master and keep the data.

## Items older versions kept

| Old item | Moved by | How |
|---|---|---|
| service `ml-stack-memory`, one JSON key per user/profile/directory | the first open of that memory store | every file of the store is re-sealed under the new subkey and read back; then the item is deleted. A file that does not verify keeps the item |
| service `ml-stack`, account `onboard-signing-<hash>` (the signing seed) | the first signing call | wrapped into `signing.key.wrapped` and unwrapped to the same key id; the item is deleted, then the record's `store` changes to `keystore`. Cut off at either point, the next call finishes |
| service `ml-stack`, account `<NAME>` (a credential) | the first `ml-stack-credentials` command a person runs that names it | wrapped into `credentials.json`, read back equal, then the item is deleted. Library code asking for a credential never probes the keystore for an old item |

## Passphrase fallback

With no usable keystore, `ML_STACK_MEMORY_KEYS=passphrase` and the signing key's encrypted file
keep working; their scrypt derivation is `keystore.scrypt_key`. A person chooses it; the keystore
never falls back to it by itself.

## What this does not do

Code running as you can call the keystore like any program of yours can; the master protects
against a copied file or backup, not against execution as the user. The hourly ceiling and the
refusal latch limit how often such code can make the OS prompt, not whether it can read a key
that is already cached in a process.
