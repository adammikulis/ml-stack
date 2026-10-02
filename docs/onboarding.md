# Onboarding devices on the local network

Status: the design is complete; the parts that can be tried on one machine are built and
tested (see "What is built"). Issue: *Zero-install onboarding of devices on the local network*.
Branch `agent/lan-onboarding`, which builds on `agent/hardening` (signed requests, pinned TLS,
loopback by default) and changes none of it.

The request: find new devices on my network without downloading ml-stack by hand, ask the
device's owner to accept, then fetch ml-stack peer to peer and connect.

## Decision record

1. First contact is a **PAKE with a code that exists only after the owner accepts**, not a MAC
   (offline-guessable at six digits) and not numeric comparison (needs a person at both screens).
   The PAKE is the maintained **`spake2` package**; nothing in this repository implements group
   arithmetic, and a test fails if something starts to (`tests/test_onboard_no_homemade_crypto.py`).
2. **Certificate fingerprints are the two identities in the exchange**, so the TLS handshake can
   be unverified at first contact and a relay still fails.
3. **Discovery is stdlib UDP behind a `Transport` seam**; mDNS is a transport, not a rewrite.
4. **Nothing is pushed.** A machine with nothing installed gets an opt-in address, a file
   bundle, or an SSH install the owner starts with their own keys and approves host by host;
   every route ends in a checked manifest, never `curl | sh`.
5. **Files travel by signed manifest**, chunk hashes and a scan seam. Each file has a sharing
   level: `open`, `owner` (a gated model: only the owner's own devices, only after their licence
   acceptance is on record) or `never`. Credentials never travel.
6. **The signing key is generated on the controller, kept in the OS keystore (`keyring`), and
   differs from the cluster key.** Signing is automatic; export, rotation and revocation need a
   person at a terminal. Manifests last days.
7. **The owner answers a dialog with buttons** (Decline, Accept as mine, Accept as someone
   else's); the code is in a second dialog after an Accept. Tests never raise a real one.
8. **Nothing in `agent/hardening` is weakened or replaced**: signed requests, pinned TLS and the
   beacon format are reused unchanged; the only change to an existing file is the subcommand
   hook in `fleet/join.py`, a row in the command table, the `fleet-onboard` extra and a guard in
   `tests/conftest.py`.

## What is possible, and what is not

| The new device has | What can be done | By whom |
|---|---|---|
| ml-stack already | find it on the LAN, ask its owner on its own screen, pair with a short code, join the cluster | automatic, with a person saying yes |
| nothing | **nothing can be pushed to it.** A machine that runs no agent of ours has no door; installing software on it without its owner doing something is what malware does, and ml-stack will not do it | the owner of the new device, opt in |
| nothing, owner willing | open an address on it, run one command, or hand it a file bundle, or let us install over SSH with the owner's own keys | see "Machines with nothing installed" |

So "add devices without manually downloading ml-stack" has an honest reading: the owner does
not *download* ml-stack by hand and trust what they got, because the controller serves it,
pinned and verified. They still have to do one thing on the new device.

## Machines that already run ml-stack

```
new device                                    owner's device (listening)
 |  announce? no: it browses (UDP) .............. announces: name, model, port, cert fingerprint
 |-- POST /onboard/v1/requests  (TLS, cert unverified) -------->  request pending, one per device
 |<-- 202 {id, server fingerprint}                               dialog: Decline / Accept as mine /
 |                                                                 Accept as someone else's
 |<-- GET .../<id>  state: accepted                              second dialog: the 6-digit code
 |  person types the code
 |-- SPAKE2 message (identities: both certificate fingerprints) ---->
 |<-- Y                                                          3 tries, then the request is dead
 |-- confirmation (proves the code) ------------------------->   checks it first
 |<-- confirmation, grant (cluster key, salt, certificate, signing key), tag
 |  verifies the confirmation and the tag, joins the cluster, pins the certificate
```

### Discovery

Choice: a small stdlib UDP announcement (`onboard/nearby.py`), behind a `Transport` seam, with an
optional mDNS/DNS-SD transport (`zeroconf`) not built.

| | stdlib UDP (built) | `zeroconf` (not built) |
|---|---|---|
| dependency | none | one package, pure Python, around a megabyte |
| hostile input | one JSON object, 1200 byte cap, fields validated | a DNS packet parser on port 5353 for hostile packets |
| firewalls | needs UDP 8773 and TCP 8772 allowed; **Windows blocks both inbound by default** (`discovery.windows_firewall_rules` shows how for the existing beacons) | UDP 5353 is usually already allowed: Bonjour on macOS, the mDNS rule on Windows, Avahi |
| interoperability | ml-stack only | any Bonjour/Avahi browser, including phones |
| same plumbing as the fleet's beacons | yes | no |

The Windows firewall point is the real argument for mDNS and is why the seam exists: a
`ZeroconfTransport` is two methods. Whether to ship it as an extra (`ml-stack[discover]`) is a
decision for the owner (below). An announcement is unauthenticated, so it is a hint: the
address is the datagram's source, text is cleaned to one printable line, the list is capped at
64 machines and 4 per source address, and entries age out after 30 s. A forged announcement can
put a wrong name in the list; it cannot get anybody paired.

### The request and the notification

A machine opens pairing deliberately (`ml-stack fleet listen --for 10m`); by default nothing
listens. A request carries name, hostname, model and the fingerprint of the certificate the new
device's daemon will serve. The owner sees all of it, plus the address the request came from and
the fingerprint of the certificate the new device was *seen* to present.

The owner is asked with **real buttons**, not told to run a command (`onboard/notify.py`):

* macOS: `osascript` `display alert` with `buttons {"Decline", "Accept as someone else's",
  "Accept as mine"}`, `default button "Decline"`, `cancel button "Decline"`,
  `giving up after 120`. The identity text is passed as script arguments after `--`; the answer
  is read from `button returned:... , gave up:...`. Escape and Decline are a decline; a timeout
  is a timeout. After an Accept a second dialog (one `OK`) shows the code. Both scripts compile
  under the real `osacompile`; the argument handling was checked against the real `osascript`
  with a hostile hostname (it comes back as data); a real dialog is shown only by an opt-in test
  (`ML_STACK_MANUAL_DIALOG=1`).
* Linux: `notify-send --action=... --wait` where libnotify supports actions, else `zenity
  --list`; argv and parsing are tested with shims, not on a Linux desktop.
* Windows: toast buttons need a registered app id; designed, not built.
* Without a desktop (`ML_STACK_NOTIFY=console`) the request is printed and `accept ID
  --mine|--other` / `decline ID` answer it; the answer goes through the same state machine
  either way (rate limits, one pending request per device, expiry).

"Mine" or "someone else's" is decided at the Accept click (or with `--mine` / `--other`; there is
no default) and stored with the device. It is shown in `requests` and `nearby`, and it is what
the sharing levels read.

**The pairing code is never in the first dialog**, and a test suite never raises a real
dialog: `tests/conftest.py` sets `ML_STACK_NOTIFY=console` for the session and puts shims named
`osascript`, `notify-send`, `zenity` and `kdialog` first on PATH (child processes inherit them);
a shim records and fails, and the run fails if anything was recorded. The owner's report that
tests opened Script Editor was this bug: `pick()` ignored any switch and every `listen` process
in the CLI tests used the real notifier.

### Choosing how the code is checked

Three schemes were considered.

1. **Beacon MAC plus pinned fingerprint** (what the fleet does for members). Needs a key the
   new device does not have. Not usable for the first contact.
2. **Numeric comparison** (Bluetooth style): both screens show six digits, a person confirms they
   match. Needs a screen and a person at *both* ends, and a person tends to press yes.
3. **A password-authenticated key exchange** with a code that is displayed on one side and typed
   on the other. **Chosen**, because the code only ever exists after the owner has accepted.

A six digit code is twenty bits. A confirmation of the form `HMAC(code, transcript)` on the wire
lets a listener, or a machine playing the other end, try every code offline in about a second.
In a PAKE nothing sent allows that, and a side's confirmation is sent only after the other's has
verified, so each guess costs one conversation with a person watching at one end.

**Which PAKE, evaluated with real installs (2026-10-02, Python 3.13.5):**

| | version, licence, last release | fit |
|---|---|---|
| **`spake2`** (python-spake2, used by magic-wormhole) | 0.9, MIT, 2024-09-25; pure Python; requires only `cryptography` | symmetric shared code, `idA`/`idB` bytes in the transcript (carry the two fingerprints), tiny API. **Chosen.** |
| `pysrp` | 1.1.0, MIT, 2021-06; pure Python | SRP-6a needs a verifier derived from the code and stored by the accepting side (twenty bits of offline target if it leaks); no identities in the transcript; unmaintained since 2021 |
| `opaque` (libopaque bindings) | 1.0.0, **GPLv3**, 2025-02; needs native libsodium and liboprf | OPAQUE wants a registration step and a long-lived server record; wrong shape for a one-time code, and a GPL dependency |
| `cryptography` | 50.0.2 | no PAKE; used for what surrounds it: Ed25519, scrypt, ChaCha20-Poly1305, certificates |

`spake2` is an Ed25519-group SPAKE2 with side-tagged messages. A full exchange of both sides
here (`pake.py`, including the HMAC-SHA256 confirmations) took a median 15.7 ms on this
laptop. It checks group membership: a zero element and an off-group point are refused (tested).
It is **pure Python and not constant time** (Python integers). That matters little here: the
exchange is interactive, a timing observer on the LAN gets at most three samples per accepted
request (each try is spent when the accepting side starts an exchange), each needs the owner to
have pressed Accept, the code lives 120 s, and network jitter is orders of magnitude above the
differences. A hardware-backed or constant-time PAKE would remove the question; none that is
maintained and pure Python exists. It is pure Python with no version pin; it was run here on 3.13.5 (the
project's) and 3.14; 3.11 and 3.12 were not run.

The transcript binds **both certificate fingerprints**: they are `spake2`'s `idA` and `idB`, and
the request id and nonce go in with the code, so an exchange replayed into another request, a
wrong code, and a relay that terminates TLS on both legs (a real one is in the tests, rewriting
the server field too) all fail the confirmation. Properties, each tested:

* right code: both confirmations verify; wrong code: neither does;
* a different certificate on either leg fails even with the right code;
* the accepting side reveals nothing checkable (no confirmation, grant or tag) before the asking
  side has proved the code, and a confirmation cannot be retried on one exchange;
* a request nobody accepted cannot start an exchange; opening exchanges without confirming runs
  out of tries too;
* three tries per request, then the request is closed, the fingerprint waits 15 minutes, and a
  `critical` event is raised. A stranger in the middle of an accepted request guesses the code
  with probability 3 in 10^6.

The library is an optional dependency behind the `fleet-onboard` extra (`pip install
'ml-stack[fleet-onboard]'`: `cryptography`, `spake2`, `keyring`); without it, `listen` fails at
once and `pair` says so, naming the extra.

What the owner is really approving: the default grant is the cluster key. **A cluster member is
fully trusted** (`docs/security.md`), so `listen --no-cluster` pairs a device without it.
Accepting a request is not enough to give anything away: the code must also be read to the device.

### Limits against spam and abuse

| Guard | Value |
|---|---|
| waiting time before a request expires | 300 s |
| life of a code | 120 s |
| tries at the code | 3 |
| active requests per fingerprint / per address | 1 / 1 |
| requests per address in 10 minutes | 5 |
| requests waiting for an answer, overall | 8 |
| wait after a decline | 60 s; three declines in an hour block that fingerprint for an hour |
| wait after a failed pairing | 15 min |
| bad requests (unknown id, bad body) per address | 20 a minute, then 2 minutes locked out (`macauth.Lockout`) |
| request body | 16 KiB; connections 64 (`framing.LimitedServer`) |

A request that is refused is answered quietly and does not notify anybody. Notifications are
sent when a request is created, once.

### Revocation

`ml-stack fleet revoke NAME-OR-FINGERPRINT` marks the device revoked: it cannot ask again, and
the key it signs file requests with (issued at pairing, one per device) stops working, so the
owner's machine serves it nothing. **It cannot take back the cluster key** the device was given:
the cluster has one shared key. The command says so. The re-keying flow (mint a new key and
salt, hand them to each remaining member over its pinned, signed channel) is designed and not
built. A device-only pairing (`--no-cluster`) has no cluster key to take back.

## Machines with nothing installed

Nobody pushes software to them. Opt-in paths, ranked by ease and then by safety:

1. **An address on the LAN** (`ml-stack fleet bootstrap --share DIR`). The controller serves for
   ten minutes, over HTTPS with a certificate made for the offer, an unguessable address
   `https://IP:PORT/b/TOKEN/#fp=CERT-FINGERPRINT` (`--advertise` names the address to print). The
   owner types it (or scans it as a QR code: drawing one needs a library, not built; the string
   is `Offer.url`) on the new device. The page says what will be installed (names, sizes,
   SHA-256), the signing key and manifest digest, and shows one command:
   `curl --fail -k --pinnedpubkey 'sha256//...' -o ml-stack-install.py https://.../install.py && python3 ml-stack-install.py`.
   `-k` is intentional: `--pinnedpubkey` is checked even then, and the certificate has no name a
   CA could vouch for. The installer is fixed text (under 90 lines), pins the certificate, and
   refuses unless the manifest matches the digest printed on the page and every file matches
   its size and SHA-256, before it hands anything to `pip`. Honest limits:
   * **A browser cannot pin a self-signed certificate.** The `#fp=` fragment is for a human or
     the installer. The *authoritative* command is the one printed on the controller's terminal.
   * `pip` fetches the wheel's dependencies from the package index under normal certificate
     checks and **without hashes**; a hash-locked requirements file in the manifest closes
     this (designed, not built).
   * Offers carry program files only: models and the cluster key are refused (tested).
2. **SSH install** (`ml-stack fleet bootstrap --ssh [user@]host --share DIR`, built, owner
   initiated, its own flag, never automatic). Properties, each a test:
   * the system `ssh` with the owner's keys, agent and config; `BatchMode=yes`, so a password is
     never asked for or seen; `StrictHostKeyChecking=yes` always, never `no`; no agent, X11 or
     other forwarding; timeouts; output cut at 64 KiB;
   * the target is checked character by character (`[user@]host`, no leading dash, no
     whitespace, no shell metacharacters; more than thirty hostile forms refused including
     `-oProxyCommand=...` and `host;rm`), and passed as one argv element after `--`;
     no shell string is built from input: both remote commands are constants;
   * the host key must already be in the owner's `known_hosts`, or `ssh-keyscan` fetches it, its
     SHA-256 fingerprint is shown, and the owner types it in full (or passes it with
     `--host-key-fingerprint` after reading it off that machine); only that key is then trusted,
     through a temporary `known_hosts` that is removed afterwards;
   * `--dry-run` prints the commands, the remote script in full (145 lines) with its SHA-256 and
     the files with theirs, and runs and contacts nothing and touches no key;
   * the payload goes over the SSH connection on stdin as a tar of plain files; the remote
     script (copied, then run) checks Python >= 3.11, macOS or Linux, not root, disk space, the
     manifest's **OpenSSH signature with `ssh-keygen -Y verify`** (the framing is written here,
     the Ed25519 signature is `cryptography`'s, and the real `ssh-keygen` verifies it in the
     tests; a pure-Python verifier would have been home-made crypto), that the key is the one
     the owner named, expiry, and the size and SHA-256 of every file, and that no other file came;
     only then a per-user venv (`~/.ml-stack/venv`, no sudo, no system Python touched) and
     `pip install`; it then starts `ml-stack-fleet listen` so the normal pairing flow with a
     code and pinned TLS completes. Models are not copied;
   * nothing is fetched from the internet by the remote side except pip's dependencies (the
     same gap as above); macOS and Linux targets only, Windows OpenSSH designed not built;
   * sentinel events for each step (`onboard.ssh.*`), no secrets in logs.
   Tested: an exact-argv shim for `ssh`, `ssh-keygen` and `ssh-keyscan`; the remote script run for
   real against a temporary HOME with good, tampered, swapped, wrong-key, extra-file and expired
   payloads; a real venv install of a tiny wheel (slow test); a localhost sshd test exists but
   runs only with `ML_STACK_TEST_SSHD=1`.
3. **A file bundle** (USB stick, AirDrop; the output of `bootstrap --share`'s manifest plus the
   installer): the manifest digest is read to the person from the owner's screen.

All three verify with the same two ingredients: a pinned channel or a typed digest, and a
manifest whose files' hashes are checked before anything runs. None is `curl | sh`.

## Peer to peer distribution

Once a device has paired it holds the cluster key, the certificate of the machine that took it
in, the cluster's signing public key, and a file-request key of its own. `onboard/transfer.py`:

* Any peer serves the signed manifest and, in ranges of one chunk, the files in it
  (`GET /onboard/v1/files/NAME`, `Range` required, signed by `macauth`, over the pinned
  certificate; `ShareServer`). A request signed with a device's own key says which device asks;
  one signed with only the cluster key is a member with no device identity.
* `Downloader` takes a manifest verified under the pinned key and a list of peers and fetches
  chunks from several at once. Every chunk is checked against its listed SHA-256 before it is
  kept; a peer that sends one that does not match is dropped after a second such answer, a
  `critical` event is raised and the chunk is fetched from another peer; if no peer is left
  nothing is staged. The finished file is hashed whole again.
* Resumable: verified chunks are listed beside the partial file and each is hashed from disk
  before it is trusted on the next run. Disk: the file plus a 1 GiB reserve must be free first.
* The finished file goes to a **staging directory**, not to its home, and `on_staged(path,
  entry)` is called: that is where the guarded pipeline's scan and sentinel's `manifest.pin` /
  quarantine step belong.

**Who may have which file** (`onboard/sharing.py`; the level is a signed field of each manifest
entry, so changing it breaks the signature):

| level | who gets it | for |
|---|---|---|
| `open` | any paired peer | programs, ungated models |
| `owner` | only a device the owner marked **mine** at pairing, and only once the owner's acceptance of that file's licence is on record (who, when, licence id, URL) | gated or licence-restricted models |
| `never` | nobody; each device downloads it itself from the entry's `source` | licences that forbid copies |

The owner who accepted a licence may put the model on their device #2; another person's device
in the same fleet may not have it. The first transfer of an `owner` file asks the person at the
terminal to type the licence id back, which writes the record; without a terminal nothing is
recorded and the file stays withheld. A file whose status is not known is `owner` and the owner
is asked; it is `open` only when it is known not to be gated and not to forbid copies
(`classify`). A manifest entry that does not say is read as `owner`. **Credentials (a Hub token,
an API key) are never copied between devices**: they are not files in a manifest or a field in
a grant, and a test fails if any onboarding module mentions one. The server refuses with a reason
(`403`), the client reports it without counting it against the peer, and `onboard.transfer.withheld`
is raised.

Never from an unauthenticated source: the manifest must verify under the pinned key (signature,
expiry, serial no lower than the last seen, entry names through `safenames`), and the peers are
cluster members over pinned TLS.

## Signing keys

One Ed25519 key per cluster, generated on the controller on first use, **separate from the
cluster key** (losing one is not losing the other). `onboard/signing.py`:

* **Stored in the OS keystore by default** through `keyring` (macOS Keychain, Windows
  Credential Manager, Linux Secret Service). With no usable keystore it is stored in a file
  encrypted under a passphrase (scrypt, ChaCha20-Poly1305, mode 0600) with a warning and a
  `warning` event; the passphrase comes from `ML_STACK_SIGNING_PASSPHRASE` or a prompt, and with
  neither it refuses rather than write a plaintext key. There is no plaintext path; tests scan
  every file under the state directory for the key's raw, base64 and hex forms.
* **Signing is automatic** when the owner shares or offers files: no prompt.
* **Export, rotation and revocation need a person at a terminal** (a human grant that follows
  sentinel's rule: stdin and stdout are terminals, no agent marker in the environment, the
  person types the key id back; it is kept as `onboard/human.py` until sentinel is merged).
  The commands are `ml-stack fleet signing show|export|rotate|revoke|confirm|accept`. An agent
  has no path to the key.
* **Manifests are short-lived** (3 days) and carry a serial that may not go backwards.
* **New devices pin the public key at pairing**; its fingerprint is printed on both machines
  (`accept` and `pair`) so a person can compare them. **Rotation is announced**, not applied:
  the old key signs a statement naming the new one, manifests carry the chain, and a member
  that sees a manifest signed by the successor reports `RotationAnnounced` and waits for `signing
  accept` (a human step) before it pins the new key. Revoked key ids travel in manifests.
* **High-assurance mode** (`signing confirm on`, a human step): each signing asks the person at the
  terminal first. A hardware key (a device that never releases the key) is the designed next
  step and is not built.

What this protects: the key leaking through a file copy, a backup or a repository. What it does
not: **code running as you can ask the keystore for the key** unless the operating system prompts
for each use; the keystore protects against theft of files, not against execution as you. If the
signing key is stolen anyway a thief can sign manifests members accept until the owner rotates
and each member runs `signing accept`; that recovery is by hand. What a signature vouches for:
the controller lists a file only after checking it itself (a wheel against the digest the release
or index publishes, through `httpguard`; a model through the guarded download and scan); it is
"the controller vouches for these bytes", not "these bytes are safe".

## Threat model

| Threat | Defence | Tested |
|---|---|---|
| Rogue device pretends to be a peer | pairing needs a code the owner reads out after accepting; members authenticate by signed requests over pinned TLS | yes |
| MITM during pairing | both certificate fingerprints in the SPAKE2 transcript; unverified handshake is acceptable only because of that; relay with a rewritten server field fails | yes |
| Replay of a join request or confirmation | nonce and id per request, an exchange is used once (409), a request id is single-use | yes |
| Notification spam / DoS | one active request per fingerprint and address, windows, caps, declines and failures make a device wait; quiet refusals | yes |
| Poisoned file from a peer | signed manifest, chunk and whole-file hashes, peer dropped, nothing staged on failure, then the caller's scan and sentinel | yes (scan hook seam only) |
| Code brute force | PAKE (no offline test), 3 tries, then closed and the device waits | yes |
| Accidental acceptance | the request shows name, hostname, model, address, fingerprint; accepting alone gives nothing until the code is read out | partly (text asserted; a person is not) |
| Hostile text in a name | one printable line, 64 characters, passed as an argument to `osascript`/`notify-send`; HTML escaped on the offer page | yes |
| Rogue or flooding announcements | hints only; caps per source and overall; age out | yes |
| Pairing over plain HTTP | refused (403) even from loopback | yes |
| Path tricks in manifests and requests | `safenames` on every name; `safe_join` resolves symlinks; no suffix ranges | yes |
| Stolen device | revoke blocks re-pairing; the cluster key must be rotated (not built) | partly |
| Wrong signing key | pinned key only; expiry; serial rollback; key revocation list | yes |
| A device of another person gets a gated model | `owner` level needs a device marked mine and the licence acceptance on record; the field is signed | yes |
| A stolen signing key | keystore, short expiry, serial, rotation announced and accepted by a person, revocation list | yes (the keystore does not stop code running as the user) |
| Dialogs or notifications raised by tests | env switch, PATH shims, session-end check | yes |
| Hostile SSH target or host | per-character validation, argv after `--`, no shell, host key typed in full | yes |
| Offer address guessing or reuse | 128-bit token, 10 minutes, lockout after 8 wrong in a minute, uniform 404 | yes |

### Out of scope

A compromised owner machine or signing key; physical access; what the upstream package index
or model host serve before the controller checked it; timing side channels in the pure Python
PAKE library (rate-limited and interactive; see above); IPv6 and multi-subnet
networks (TTL 1, IPv4); the internet; the Windows toast and Windows SSH targets; a web interface panel; phone
platforms; QR rendering; the cluster re-key flow; a hardware signing key; hash-locked dependency
install; mDNS.

## What is built

| Part | State |
|---|---|
| SPAKE2 pairing from the `spake2` package; three tries; confirmations in the right order | built, tested over real TLS sockets |
| Request state machine, limits, devices ledger (whose device, its file-request key), revoke | built, tested on real files, one test with a second process |
| Discovery announce / browse, injectable transport, real UDP on loopback | built; multicast and broadcast on a real LAN not verified |
| Dialog with buttons (macOS), second dialog for the code | built; scripts compile under the real `osacompile`; the click is read by a fake `osascript` in a three-process test; a real dialog only in the opt-in manual test |
| Dialog on Linux (`notify-send` actions, `zenity`) | argv and parsing only, no Linux desktop |
| Signed manifest (Ed25519), chunked and resumable multi-peer download, scan hook, sharing levels | built, two server processes on loopback, with pinned TLS |
| Signing key in the OS keystore, encrypted-file fallback, human-only export / rotate / revoke, confirm mode | built; the keystore is exercised through a real `keyring` backend over a file (the real Keychain is never touched by tests) |
| Bootstrap offer, pinned installer | built; installer run as a separate process; `curl --pinnedpubkey` checked against the real `curl` |
| SSH install | built; tested with recording shims for the argv, with the remote script run for real against a sandbox HOME, and a real venv install; **not run against a real sshd or another machine** (opt-in test exists) |
| CLI (`nearby listen pair requests accept decline revoke bootstrap share fetch signing`, all `--json`) | built; end-to-end tests across separate processes |
| Sentinel events | emitted for each step (list below); the adapter below was run against the real sentinel on a throwaway merge of `agent/sentinel-integ` earlier in this work |
| QR drawing, mDNS, Windows toast and SSH target, cluster re-key, hardware signing key, web panel, hashed dependencies | designed only |

Events (all on `onboard.events.BUS`; `Event(kind, severity, subject, evidence)`):
`onboard.request.received|accepted|declined|expired|refused`, `onboard.pair.wrong_code` (warning),
`onboard.pair.locked` (critical), `onboard.pair.succeeded`, `onboard.revoked`,
`onboard.notify.failed`, `onboard.nearby.flood`, `onboard.transfer.bad_chunk` (critical),
`onboard.transfer.bad_file` (critical), `onboard.transfer.peer_dropped`,
`onboard.transfer.withheld`, `onboard.transfer.staged`, `onboard.bootstrap.offered`,
`onboard.bootstrap.refused`, `onboard.signing.created|signed|exported|rotated|revoked|file_fallback`,
`onboard.ssh.started|hostkey_confirmed|hostkey_refused|copied|verified|installed|listening|failed`.
Evidence holds short facts, never a code, key or secret. Sentinel subscribes with a six-line
adapter on its own side (it imports none of its sources):

```python
def watch_onboarding(bus, sentinel, Event, Severity):
    return bus.subscribe(lambda e: sentinel.bus.emit(
        Event(e.kind, Severity.parse(e.severity), "onboard", e.subject, e.evidence, e.ts)))
```

### Gaps that matter when this is folded into the daemon

* The pairing listener and the file server run on **their own ports with their own certificate**
  (`<state>/onboard/tls`). In the finished design these routes live on the daemon's port under
  `/onboard/v1/`, with the daemon's certificate, so the certificate in a grant is the one the
  beacons carry. `trust.json` records the pairing partner's certificate and `fetch` uses it, but
  discovery does not yet consume it.
* `fetch` takes one peer (the machine it paired with). `Downloader` takes any number of peers and
  is tested with two; learning the others' certificates from beacons is the daemon-side work.
* One shared handler (`onboard/web.py`) serves pairing, sharing and the offer; it costs the repo's
  `http-servers` budget one handler. See "Decisions".
* `cryptography` is required; `spake2` and `keyring` are the other two members of the
  `fleet-onboard` extra.
* A request whose dialog is still open when the owner answers in the terminal (or the reverse)
  is answered once: the second answer finds the request already closed and is dropped.

## Tests and mutation checks

257 tests in 15 `tests/test_onboard_*.py` files (3 skipped by default: a localhost sshd, a real
dialog, and Linux-only paths), about 80 s with the slow ones on a loaded laptop, all against real
sockets, files and processes: TLS handshakes on loopback, a relay that terminates TLS, two peer
processes serving one file (one poisoned), a separate process running the installer, the real
`curl --pinnedpubkey`, the real `ssh-keygen -Y verify` and `osacompile`, the remote install
script run for real in a sandbox HOME including a real venv, and three or four CLI processes in
one conversation. Nothing uses the network beyond loopback, no real dialog or notification is
raised (a shim and an environment switch make that a failing condition, itself tested), and no
real device, sshd or Keychain is touched.

Mutation checks (a guard is broken, the targeted tests are run with `-x`): the first campaign
broke 79 guards, 63 killed, 16 survived, 13 of which were missing tests and were fixed (one
allowed unlimited guesses at the code on a single exchange). The campaign for this round broke
56 more (the PAKE wrapper, the signing key store and the human grant, rotation, the sharing
levels, device identity, the dialog parser and switch, the SSH target and host key checks, and the
remote script's checks): 49 killed, 7 survived. Two missing tests were found and written (a gated
file to a device second in the ledger; a manifest-listed file that is not a program file), and
those mutants killed. The rest are equivalent, each with a second guard behind it: the context
in the PAKE password (it is also in the confirmed transcript), the rotation chain's key-id
check (the signature check that follows rejects the same cases), a leading dash (the user and host
patterns reject it too), payload file names (read again through `safe_filename`), and the remote
size check (the SHA-256 follows). The first campaign's three equivalent survivors (the context in the old transcript, the
Content-Range check and the length check on a chunk) are the same kind; all are left in as depth.

## Commands

```
ml-stack fleet listen [--for 10m] [--port 8772] [--no-cluster]    open pairing here; asks with a dialog
ml-stack fleet requests                                            what is waiting
ml-stack fleet accept ID --mine|--other | decline ID               answer; accept prints the code
ml-stack fleet nearby                                              who is open to pairing
ml-stack fleet pair --host H [--port P] [--code C]                 ask to join
ml-stack fleet revoke NAME-OR-FINGERPRINT
ml-stack fleet bootstrap --share DIR [--valid 10m]                 offer ml-stack on an address
ml-stack fleet bootstrap --ssh [user@]host --share DIR [--dry-run] [--host-key-fingerprint SHA256:...]
ml-stack fleet share --dir DIR [--sharing NAME=open|owner|never] [--licence NAME=ID,URL] [--source NAME=URL]
ml-stack fleet fetch NAME... --from HOST:PORT                      fetch into staging (not installed)
ml-stack fleet signing show|export FILE|rotate|revoke KEYID|confirm on|off|accept
```

All take `--json` and `--state DIR`. State is under `<state root>/onboard` (requests, devices,
licences, this machine's pairing certificate, the public half of the signing key), every file
private; the signing key itself is in the OS keystore. `ML_STACK_NOTIFY=system|console|off` picks
how the owner is asked.

## Decisions for the owner

1. **Signing key** (decided: keystore, separate from the cluster key, signing automatic, human-only
   export / rotation / revocation, short manifests, high-assurance mode available). Still yours:
   whether to turn confirm-before-signing on by default, and when to build the hardware-key path.
   The honest limit: code running as you can use the keystore unless the OS prompts.
2. **Ship the address/QR bootstrap?** It is the easiest path and the weakest in a browser.
   Recommended: ship it with the terminal command as the authoritative one; QR left out.
3. **SSH install** is built, owner-initiated and off unless `--ssh` is given. It was tested
   without a real sshd; run it once on a spare machine with `--dry-run` first.
4. **Grant by default**: cluster key (every member is fully trusted) or device-only. As built
   the owner chooses with `--no-cluster`; the default gives the key.
5. **mDNS as an extra** (`ml-stack[discover]`) to avoid the Windows firewall step, or stdlib only.
6. **Budgets.** `http-servers` allows two handlers; onboarding adds one (`onboard/web.py`).
   Folding these routes into the daemon's handler is the clean answer and touches
   `fleet/api.py`; an agent cannot raise a budget.
7. **Python and dependencies**: `fleet-onboard` brings `cryptography`, `spake2` and `keyring`.
   The pyproject here pins Python 3.13; the extra is pure Python and was run on 3.13.5 and 3.14.
8. **Licence records**: the first transfer of a gated file asks the person at the terminal to
   type the licence id back. Deciding `gated` / `forbids_copies` from Hub metadata is not built:
   the owner names the level with `--sharing`.

