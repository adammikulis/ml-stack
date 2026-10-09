# Keystore platform audit, 2026-10-08

The owner asked for every use of the user's encrypted key store to be reviewed and made correct on
every platform. This is the inventory, the platform matrix, what was fixed, what is verified and
how, what is not, and what only the owner can decide. Method: code reading of `src`, `scripts`,
`packaging`, `tests` and `docs`, plus tests against fake keyring backends. No real OS keystore was
read, written, listed or prompted at any point; the owner's `~/.ml-stack/keystore` content was not
read (file names and sizes only).

## 1. Inventory

`src/ml_stack/keystore.py` is the only module that imports `keyring`
(`tests/test_keystore_gate.py` fails on any other). It holds one secret in the OS keystore:
service `ml-stack`, account `master/<uid>:<login>`, value `v1:<base64 of 32 random bytes>`. Every
other secret is wrapped (AES-256-GCM, HKDF subkey per purpose and owner) and kept in a file.

| Caller | What it protects | Where the wrapped value lives | Who may call it | May prompt | Failure |
|---|---|---|---|---|---|
| `keystore.Keystore` (master create, read) | the master | OS keystore | any ml-stack process; creating needs a person (`interactive()`), reading an existing one needs a person or a prior `ml-stack-security unlock` | yes: first create, and a read by a binary the item does not trust yet (macOS) | closed: `KeystoreLocked`, `KeystoreDenied` (10 minute latch), `KeystoreUnavailable`, `KeystoreMissing`, `KeystoreBusy`; never a plaintext fallback |
| `ml-stack-security unlock`, `keystore`, `keystore-reset` (`net/cli.py`) | create, status (no backend call), delete the master | OS keystore | a person at a terminal with no agent marker (`require_person`, `human.mint`) | yes (create) | closed |
| memory store (`memory/vault.py`, `store.py`) | fact store files | beside the store | the user's processes, hooks and daemons | via the master only | closed; `ML_STACK_MEMORY_KEYS=passphrase` is a person's explicit choice |
| fleet signing key (`fleet/onboard/signing.py`) | Ed25519 seed | `signing.key.wrapped` | controller processes | via the master only | no keystore: scrypt passphrase file `signing.key.enc` (encrypted, not plaintext; passphrase from a terminal or `ML_STACK_SIGNING_PASSPHRASE`) |
| credentials `--keychain` (`credentials/__init__.py`) | named tokens | `keystore/credentials.json` | a person's `ml-stack credentials` command; library lookups only unwrap | via the master only | closed (`CredentialError`) |
| cluster passphrase (`fleet/recovery.py`) | cluster passphrase | `<memberships>.passphrases` | join (CLI, web button) | via the master only | the join continues and says the passphrase was not saved |
| activity log, request inbox, workspace file store, reputation ledger (`activity/log.py`, `requests/store.py`, `workspace/filestore.py`, `reputation/sealed.py`) | sealed stores | beside the store | hooks, daemons, agents | via the master only | closed (the caller reports a locked store) |
| `scripts/release-key` | release signing key | `credentials --keychain` | the owner | via the master only | closed |
| `scripts/encrypted-volume.sh` | disk image passphrase | macOS login keychain, service `NAME-data`, through the `security` CLI | the owner, macOS only | yes | `set -eu`; see finding F-9 |
| legacy items (`ml-stack-memory`, `onboard-signing-<hash>`, `<NAME>`) | pre-master items | OS keystore | migrated and deleted on first use by a person's command | yes | kept when verification fails |
| Android `DeviceVault.java` | device grant | Android Keystore, user-authentication required | the companion app | biometric/PIN by design | closed |

Test isolation: `tests/conftest.py` sets `ML_STACK_NO_REAL_KEYSTORE=1` and
`PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring` for the run and every child, refuses the
real backends' `get/set/delete_password` in process, installs an in-memory ring and patches
`keystore.interactive` to true. `runtime_deploy._clean_environment` (the smoke run of a new
runtime) sets `ML_STACK_NO_REAL_KEYSTORE=1` and `ML_STACK_NONINTERACTIVE=1`.

## 2. Platform matrix

"Verified" is explained in section 4.

| Platform | Backend | Create the master | Read it later | Failure mode | Status |
|---|---|---|---|---|---|
| macOS, person at a terminal | `keyring.backends.macOS` (Security framework through ctypes, generic password in the login keychain, default access list: the creating binary) | one Keychain prompt, after the one-sentence notice | silent while the same interpreter runs; a changed interpreter binary prompts once | closed | code read; ACL behaviour NOT verified on a real item |
| macOS agent or hook (any agent marker) | same | refused (`KeystoreLocked`) since this audit; before it an agent on a desktop counted as a person | after `unlock`: silent if trusted; if the item does not trust the binary, `SecKeychainSetUserInteractionAllowed(false)` makes it fail instead of showing a dialog, and the call is bounded to 20 s | closed | verified by test (flag set, markers); dialog suppression NOT verified against a real item |
| macOS `--system` LaunchDaemon (`UserName`, no login) | same | background: refused | `XPC_SERVICE_NAME` is the label so `_desktop()` is false: background; the login keychain is locked or absent before login, so a read fails or is denied: latched `KeystoreDenied` or bounded timeout | closed | code read; NOT verified at boot |
| Windows desktop | `keyring.backends.Windows` (Credential Manager, DPAPI of the user profile) | silent (no dialog exists) | silent | closed | code read; NOT verified |
| Windows boot task / service (`schtasks /SC ONSTART /RU user`, session 0) | same | background since this audit (before it `_desktop()` was always true on Windows: a service could mint its own master under the service account) | the task's account has its own vault; a master made by the person at logon is only read if the task runs as that same user with a loaded profile | closed | session test with a fake `_session_id`; real session 0 NOT verified |
| Linux desktop (gnome-keyring, KWallet) | `SecretService` (jeepney), `kwallet`, `libsecret` | collection unlock dialog if locked; item created silently if unlocked | silent when unlocked | closed; a locked collection with no one to answer is bounded (300 s person, 20 s background) and latches | bounded call verified by test with a stuck fake; real D-Bus NOT verified |
| Linux headless, SSH, container, CI | no session bus: `SecretService.priority` raises, keyring falls to `fail.Keyring` (priority 0) | `KeystoreUnavailable`; signing key uses the scrypt passphrase file; memory needs `ML_STACK_MEMORY_KEYS=passphrase`; credentials `--keychain` refuses | n/a | closed, no plaintext | verified by test (unusable backend) |
| Linux with `keyrings.alt` installed | `PlaintextKeyring` or `EncryptedKeyring` would win | refused since this audit | n/a | closed | verified by test with a fake in the `keyrings.alt` namespace |
| Linux system unit (`User=`, `After=network-online`) | no session bus | background, `KeystoreUnavailable` or `KeystoreLocked` | n/a | closed | code read |
| WSL | like Linux headless unless `dbus` + `gnome-keyring` run inside the distro; `DISPLAY`/`WAYLAND_DISPLAY` are set by WSLg so `_desktop()` is true | per the backend present | the Windows Credential Manager is NOT used from WSL: a WSL master and a Windows master are different items, never shared | closed | code read; NOT verified |
| Frozen app (PyInstaller) | needs the keyring backends bundled | | | `KeystoreUnavailable` if absent | NOT verified |

## 3. The suspected problems

**(a) macOS access control per binary.** Refuted for the per-commit runtime, with a caveat. The
runtime tree is `python -m venv` (`fleet/runtime_wheel.py:130`); on macOS `bin/python` is a symlink to
`python3.13` and that to the host interpreter (here `~/.pyenv/versions/3.13.5/bin/python3.13`,
`pyvenv.cfg` `home =`). The kernel runs the resolved file and the Keychain access list names the code
identity of that file, not the venv path, so a new `~/.ml-stack/runtimes/<platform>/<commit>/<hash>`
tree reuses the host binary and its identity (read-only `codesign -d -r-` shows
`cdhash H"57855c9b..."`, ad-hoc linker-signed). Consequences: (1) nothing in the per-commit swap
causes a prompt; (2) the identity is a cdhash, so a host Python upgrade (pyenv, Homebrew) or a switch
to the standalone Python (`Environment.standalone_python`) is a new identity: the item still exists
but this binary is not on its access list, so macOS asks once (the dialog's allow-always choice needs the login
password) and a background process now fails closed instead of hanging; (3) an item is created through
`SecItemAdd`-style calls in `keyring.backends.macOS.api` with no access list argument, so the
creator binary alone is trusted; the user sees "python3.13 wants to use your confidential
information stored in 'ml-stack' in your keychain". Not verified: the exact dialog and whether the
error code under `SecKeychainSetUserInteractionAllowed(false)` is mapped by keyring to
`KeychainDenied`. Owner decision D-1 below.

**(b) `--system` and boot-time contexts.** Partly defined, now closed. The service environment
(`fleet/autostart.py:service_environment`) sets no agent marker, so the decision rests on
`interactive()`. macOS: `XPC_SERVICE_NAME` is the launchd label, so background. Linux: no `DISPLAY`
and no tty, background. Windows: was always interactive (defect F-2, fixed). Background processes
never create the master and read only after `unlock`; a read that the platform cannot serve is
refused (latched ten minutes) or times out after 20 s, never a hang. Gap that remains: nothing makes
a `--system` install print that the keystore is unavailable at boot; D-2.

**(c) Non-interactive callers.** Defect F-1 (fixed): `interactive()` ignored `CLAUDECODE` and
`ML_STACK_AGENT`. An agent's tool call has no stdin terminal, but on macOS and Windows `_desktop()` was
true, so it counted as a person and could create the master behind a Keychain prompt. Hooks set
`ML_STACK_NONINTERACTIVE` (`harnesshook.run`) so they were already safe. About the baseline plugin
on `test-agent-shell-baseline` (`scripts/testagentenv_pytest.py`): clearing the markers per test cannot
make an in-process test reach a real keychain, because `_keystore_has_a_person` already forces
`interactive()` true, the `MemoryRing` is installed, the real backends' methods raise, and
`ML_STACK_NO_REAL_KEYSTORE=1` and `PYTHON_KEYRING_BACKEND=...fail.Keyring` are in the environment every
child inherits. A test that builds a child environment from scratch without those two variables would
reach the keychain on a macOS desktop; I found no such test (`test_compare_harnesses`,
`test_embedding` build scratch environments but never touch the keystore). The sentence "ml-stack will
ask your computer to store one encryption key ... Keychain prompt" is `keystore.NOTICE`, printed by
`Keystore._tell` before the first backend call; in `test_the_join_button_runs_the_same_join` the call
is `fleet/recovery.remember` on the autouse fake ring with a fresh temporary state root (no
`noticed.json`), so the sentence is a test artefact and no prompt exists. In a real daemon the same
path (the web Join button) on a desktop session would show the sentence in the response and then the
OS prompt, which is right because a person clicked; in a background daemon `remember` gets
`KeystoreLocked` and reports "the passphrase was not saved".

**(d) Locks.** `flight.lock` and `state.lock` use `lock.only_one`: `flock` on POSIX and
`msvcrt.locking` on one byte at offset 2^30 on Windows, chosen at call time; both release when the
holder dies, so a stale file is harmless and needs no recovery. A waiter polls with backoff and gives up
after 180 s (`flight`) or 10 s (`state`) with `KeystoreBusy`. Verified by the existing
multi-process tests (six and forty processes). Not verified: `msvcrt` on Windows, and `flock` on
WSL2 drvfs (`/mnt/c`), where an `ENOSYS` would surface as a raw `OSError` (not `Busy`); the default
`~/.ml-stack` is on ext4 so only `ML_STACK_HOME` on `/mnt/c` is affected; D-4. Lock files are created
0644 inside the 0700 directory (now enforced).

**(e) Rotation and migration.** There is no master rotation; reset loses all wrapped data (documented
in `docs/keystore.md`). Defect F-4 (fixed): a master that vanished after `provisioned.json` was
written was silently replaced by a new random one (restored backup of `~/.ml-stack` without the
Keychain, a WSL shell with a different backend, an item deleted by hand), orphaning every wrapped file.
Now only `ml-stack-security unlock` by a person may start over. Runtime path changes do not matter
(master is keyed by service and account, not by interpreter path) except for (a).

**(f) Modes.** POSIX: state files are written through `files.writing` (`mkstemp`, 0600), the keystore
directory is created 0700 and now chmod 0700 if it existed with another mode (F-5); the salt file is
0600 `O_EXCL`; `credentials.json`, `signing.key.wrapped` chmod 0600; lock files 0644 inside the
private directory. Windows: `chmod` and `mkdir(mode)` are no-ops; files inherit the ACL of
`%USERPROFILE%\.ml-stack`, normally the user, SYSTEM and Administrators. Not verified; D-3 and the
`icacls` command in section 5.

**(g) Secrets in env, argv, logs.** Logs and events carry purpose and outcome only
(`tests/test_keystore.py::test_no_file_or_log_holds_the_master_a_subkey_or_a_wrapped_secret`).
`ml-stack credentials set` reads stdin or a prompt, never argv. The master is never in an env
variable. Env passphrases exist as an explicit choice: `ML_STACK_SIGNING_PASSPHRASE` and the memory
passphrase variable; an env var is readable by the same user (`/proc/<pid>/environ`, process
inspection on Windows) and inherited by every child, which `docs/keystore.md` should say (D-5).
`scripts/encrypted-volume.sh` passes the generated passphrase on argv (`security add-generic-password
-w "$PW"`), visible in `ps` to local users for the length of the call (F-9, fixed 2026-10-09: the script now
uses `security -i`).

## 4. Defects

| ID | Defect | Evidence | Status |
|---|---|---|---|
| F-1 | An agent marker other than `ML_STACK_NONINTERACTIVE` still counted as a person on macOS/Windows desktops | `keystore.interactive` read one variable; test `test_every_agent_marker_makes_a_process_background_...` failed against the old code | fixed |
| F-2 | Windows: `_desktop()` was always true, so a boot task or service in session 0 could create its own master | code; test with a fake session id | fixed |
| F-3 | A usable `keyrings.alt` plaintext backend would hold the master in a file in the clear | code (`_ring` accepted any priority above zero); test with a fake in that namespace failed before the fix | fixed |
| F-3b | `keyring.backends.libsecret` was not treated as the machine's own keystore by the test isolation | code | fixed |
| F-4 | A vanished master was replaced silently | test `test_a_lost_master_is_not_quietly_replaced_by_a_second_one` | fixed |
| F-5 | The keystore directory kept a looser mode if something else created it first | test | fixed |
| F-6 | A backend call stuck behind a dialog nobody could answer held `flight.lock` and the process forever; no timeout existed | test with a stuck fake: now refused after the wait and latched | fixed |
| F-7 | macOS background processes could pop a Keychain dialog (for example the first run under a new Python) | code | mitigated: dialogs forbidden for background processes (set-flag verified, effect not) |
| F-8 | `credentials.json` and the passphrase file are read-modify-write without a lock: two concurrent `credentials set --keychain` can lose one | code | fixed 2026-10-09 (D-6): `lock.rewriting`, two-process race tests |
| F-9 | `encrypted-volume.sh` puts a passphrase on argv | code | fixed 2026-10-09 (D-7): `security -i` on stdin, throwaway-keychain test |

Test status at the end: `tests/test_keystore.py`, `test_keystore_platform.py`, `test_keystore_gate.py`,
`test_redteam_keystore.py` and the memory, credentials, signing, recovery, request, reputation and
activity suites pass. Failing on `0.2dev` without these commits, so not caused here:
`tests/test_memory_vault.py::test_a_write_that_dies_midway_leaves_only_ciphertext` and two
Playwright tests in `tests/test_fleet_credentials.py` (a 30 s locator timeout).

## 5. Not verified, and the commands

None of these touches a secret value. Run them as yourself in a terminal, not through an agent.

**Windows machine** (PowerShell):

```
python -c "import keyring; print(keyring.get_keyring())"               # expect WinVaultKeyring, no prompt
ml-stack-security keystore                                              # state files only
ml-stack-security unlock                                                # creates the master; no dialog expected
cmdkey /list:ml-stack*                                                  # one entry named ml-stack:master/0:<login>
icacls "$env:USERPROFILE\.ml-stack\keystore"                            # expect only you, SYSTEM, Administrators
python -c "import ctypes;s=ctypes.c_ulong();k=ctypes.windll.kernel32;k.ProcessIdToSessionId(k.GetCurrentProcessId(),ctypes.byref(s));print(s.value)"   # a desktop shell: not 0
schtasks /Create /TN mlstack-keystore-probe /SC ONSTART /RU <you> /TR "cmd /c ml-stack-security keystore > %TEMP%\probe.txt" ; schtasks /Run /TN mlstack-keystore-probe   # then read probe.txt: interactive false, no hang
schtasks /Delete /F /TN mlstack-keystore-probe
```

**WSL device:**

```
python3 -c "import keyring; print(keyring.get_keyring())"               # fail.Keyring without dbus + gnome-keyring
ml-stack-security keystore
ML_STACK_MEMORY_KEYS=passphrase ml-stack-memory status                  # the passphrase path works without a keystore
echo "$DISPLAY $WAYLAND_DISPLAY $DBUS_SESSION_BUS_ADDRESS"              # WSLg sets display variables; dbus normally empty
python3 -c "import fcntl,os,tempfile;f=open('/mnt/c/Users/Public/lockprobe','w');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('flock ok on drvfs')"
```

**Linux box** (desktop, then over SSH with no display and then in a container):

```
python3 -c "import keyring; print(keyring.get_keyring())"               # SecretService / fail.Keyring
ml-stack-security keystore
ml-stack-security unlock                                                # desktop: one unlock dialog at most
stat -c '%a %n' ~/.ml-stack/keystore ~/.ml-stack/keystore/*             # 700 for the directory, 600 for json and wrapped files
sudo systemd-run --uid=$USER --wait --pipe ml-stack-security keystore   # a system-unit context: no hang, interactive false
pip list 2>/dev/null | grep -i keyrings.alt                             # must be absent or ml-stack refuses it
```

**macOS** (owner, one machine):

```
codesign -d -r- ~/.pyenv/versions/3.13.5/bin/python3.13                 # the identity the item trusts (cdhash)
ls -l ~/.ml-stack/runtimes/*/*/*/bin/python                             # symlink to the host binary
```

After a deliberate host Python change, run `ml-stack-security keystore` then any memory command in a
terminal and note whether a prompt appears; from an agent shell the same command must fail within 20 s
with a refusal, not hang.

## 6. Owner decisions, settled 2026-10-09

- D-1 macOS single prompt: kept; the `security -T` route was refused because the trusted program
  would be a general reader any process of the user can run (`docs/keystore.md`, "macOS: why the
  single prompt stays").
- D-2 `--system` with no unlocked keystore: warn and continue. The service fails closed (sealed
  stores locked, signing key not unwrapped, nothing in the clear), so it does not run unprotected
  (`fleet/autostart_keystore.py`).
- D-3 Windows: `icacls` tightening of `keystore/` at creation (`platform.private_dir`); tested with a
  mocked subprocess, a real ACL not read back.
- D-4 WSL: a state root on a Windows drive is refused (`KeystoreUnavailable`, names `ML_STACK_HOME`).
- D-5 Env passphrases documented; `ML_STACK_SIGNING_PASSPHRASE` kept.
- D-6 `credentials.json` and the passphrase file are changed under a lock (F-8 fixed).
- D-7 `scripts/encrypted-volume.sh` gives `security` the passphrase on stdin (F-9 fixed); tested
  against a throwaway keychain file on a Mac.
- D-8 The spec names the three keyring backends; a frozen macOS build bundled and selected them.
