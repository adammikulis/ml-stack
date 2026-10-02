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
2. **Certificate fingerprints are the identities in the transcript**, so the TLS handshake
   can be unverified at first contact and a relay still fails.
3. **Discovery is stdlib UDP behind a `Transport` seam**; mDNS is a transport, not a rewrite.
4. **Nothing is pushed.** A machine with nothing installed gets an opt-in address, a file
   bundle, or (designed) an SSH install the owner approves; every route ends in a checked
   manifest, never `curl | sh`.
5. **Files travel by signed manifest**, chunk hashes and a scan seam; licence-restricted files
   are marked and never go peer to peer.
6. **Nothing in `agent/hardening` is weakened or replaced**: signed requests, pinned TLS and the
   beacon format are reused unchanged; the only change to an existing file is the subcommand
   hook in `fleet/join.py` and a row in the command table.

## What is possible, and what is not

| The new device has | What can be done | By whom |
|---|---|---|
| ml-stack already | find it on the LAN, ask its owner on its own screen, pair with a short code, join the cluster | automatic, with a person saying yes |
| nothing | **nothing can be pushed to it.** A machine that runs no agent of ours has no door; installing software on it without its owner doing something is what malware does, and ml-stack will not do it | the owner of the new device, opt in |
| nothing, owner willing | open an address on it, run one command, or hand it a file bundle, or (designed, not built) let us install over SSH | see "Machines with nothing installed" |

So "add devices without manually downloading ml-stack" has an honest reading: the owner does
not *download* ml-stack by hand and trust what they got, because the controller serves it,
pinned and verified. They still have to do one thing on the new device.

## Machines that already run ml-stack

```
new device                                    owner's device (listening)
 |  announce? no: it browses (UDP) .............. announces: name, model, port, cert fingerprint
 |-- POST /onboard/v1/requests  (TLS, cert unverified) -------->  request pending, one per device
 |<-- 202 {id, server fingerprint}                               notification + `requests`
 |                                                               owner: `accept` (decline)
 |<-- GET .../<id>  state: accepted                              6-digit code shown to the owner
 |  person types the code
 |-- SPAKE2 message X (binds both certificate fingerprints) ---->
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
listens. A request carries name, hostname, model and the fingerprint of the certificate the
new device's daemon will serve. The owner sees all of it, plus the address the request came
from and the fingerprint of the certificate the new device was *seen* to present (the pairing
step binds them).

Native notifications (`onboard/notify.py`): macOS `osascript`, Linux `notify-send`, Windows
not built (it needs a registered app id, and the PowerShell route builds XML from text, which
is an injection). A command-line program cannot attach an Accept button to a notification on
any of the three, so the notification says who is asking and the exact command; the answer is
typed in the terminal (`accept`, `decline`) or, when the web interface grows a panel (designed,
not built), clicked there. The text in a notification is a stranger's hostname; it is cleaned
and passed to the operating system as an argument, never spliced into a script (checked
against the real `osascript`: a hostname containing `" & (do shell script ...) & "` comes back
as data). **The pairing code is never in a notification.**

### Choosing how the code is checked

Three schemes were considered.

1. **Beacon MAC plus pinned fingerprint** (what the fleet does for members). Needs a key the
   new device does not have. Not usable for the first contact.
2. **Numeric comparison** (Bluetooth style): both screens show six digits derived from the two
   certificates and both nonces, a person confirms they match. Sound with commit-then-reveal,
   and nothing is typed. It needs a screen and a person at *both* ends, and a person tends to
   press yes.
3. **A password-authenticated key exchange** with a code that is displayed on one side and typed on
   the other. **Chosen**, because the code only ever exists after the owner has accepted, and
   because typing is the thing that makes a person look at the other screen.

A six digit code is twenty bits. A confirmation of the form `HMAC(code, transcript)` on the wire
lets a listener, or a machine playing the other end, try every code offline in about a second.
The PAKE removes that: the transcript is a function of the code that the guesser cannot
check without an answer from a machine that checks it, once per conversation. The scheme is
SPAKE2 over NIST P-256 (`onboard/spake.py`), with the blinding points hashed to the curve and
both certificate fingerprints in the transcript. Properties, each tested:

* right code: both confirmations verify; wrong code: neither does;
* a different certificate on either leg fails even with the right code: a relay that
  terminates TLS on both sides (tested with a real relay that also rewrites the server field)
  cannot complete a pairing and the real server records a wrong confirmation, not a success;
* the accepting side reveals nothing checkable (no confirmation, grant or tag) before the
  asking side has proved the code;
* a request nobody accepted cannot start an exchange; a recorded confirmation cannot be replayed;
* three tries per request, then the request is closed, the fingerprint waits 15 minutes, and a
  `critical` event is raised. The chance that a stranger who is in the middle of an accepted
  request guesses the code is 3 in 10^6 per accepted request.

Written from the standard SPAKE2 construction in about 200 lines of Python. It is not
RFC 9382 byte-compatible, which nothing needs, and it is **not constant time**; see "Out of
scope". The curve arithmetic is checked against the `cryptography` package in the tests. Python
has no PAKE in the standard library; the alternative is a dependency (`pyspake2` is
unmaintained) or `cryptography`, which has no SPAKE2. This is the one piece that deserves
independent review before a public release.

What the owner is really approving: the default grant is the cluster key. **A cluster member is
fully trusted** (`docs/security.md`: the daemon runs the command line a peer sends it), so the
prompt says so and `listen --no-cluster` pairs a device without giving it the key. Accepting a
request is not enough to give anything away: the code must also be read to the device, so a
stranger's request that the owner accepts by mistake gets nothing unless the owner then reads
out the code.

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
the event is recorded. **That is all it can do while the device holds the cluster key**, because
the cluster has one shared key. The command says so ("the cluster key must be changed on every
machine"). The re-keying flow (mint a new key and salt, hand them to each remaining member
over its pinned, signed channel, retire the old) is designed here and not built. The way to
make revocation real without re-keying is a per-device credential: `macauth.Authenticator`
already takes a callable returning the accepted secrets, so a controller could accept
`derive(device_key)` for each non-revoked device. That needs every member to know the list;
also designed, not built. Until one of them is, a device-only pairing (`--no-cluster`) is the
only kind whose revocation is complete, and it grants nothing yet.

## Machines with nothing installed

Nobody pushes software to them. Opt-in paths, ranked by ease and then by safety:

1. **An address on the LAN** (`ml-stack fleet bootstrap --share DIR`, built). The controller
   serves for ten minutes, over HTTPS with a certificate made for the offer, an unguessable
   address `https://IP:PORT/b/TOKEN/#fp=CERT-FINGERPRINT`. The owner types it (or scans it as a
   QR code: drawing one needs a library, not built; the string to encode is `Offer.url`) on the
   new device. The page says what will be installed (file names, sizes, SHA-256), the
   signing key and manifest digest, and shows one command:
   `curl --fail -k --pinnedpubkey 'sha256//...' -o ml-stack-install.py https://.../install.py && python3 ml-stack-install.py`.
   `-k` is intentional: `--pinnedpubkey` is checked even then, and the certificate has no name
   a CA could vouch for. The installer is fixed text (under 90 lines, tested not to contain
   `curl` or `| sh`), pins the certificate itself, and refuses unless the manifest matches the
   digest printed on the page and every file matches the manifest's size and SHA-256, before
   it hands anything to `pip`. `--dry-run` stops before pip. Honest limits:
   * **A browser cannot pin a self-signed certificate.** It shows a warning, and the `#fp=`
     fragment is for a human or for the installer. On a hostile LAN, the page a browser
     fetched could have been altered. The *authoritative* command is the one printed on the
     controller's own terminal; copy it from there. The QR path to a phone is weaker, and the
     page says so.
   * The installer verifies the wheel; `pip` then fetches that wheel's dependencies from the
     package index under normal certificate checking and **without hashes**. A hash-locked
     requirements file in the manifest (`pip install --require-hashes`) closes this; designed,
     not built.
   * Offers carry program files only: models and the cluster key are refused (tested). Joining
     a cluster is the pairing step, after install, with the code.
2. **SSH push** (designed, not built; `bootstrap --ssh` says so and exits 2). Only if the
   owner supplies the target and approves: key-based authentication only, never a password;
   the host key fingerprint shown and typed back in full by the owner (`yes` is not enough);
   a fixed, audited script (the same installer) run with the controller's offer as its source,
   its SHA-256 printed before it runs; nothing else executes; the connection is closed
   afterwards. Off by default, and a separate flag from the LAN offer.
3. **A file bundle** (USB stick, AirDrop; designed, built as far as `bootstrap --share`'s
   output). The same wheel, manifest and installer in a folder; the manifest digest is read
   to the person from the owner's screen, which is how a file moved by hand gets authenticated.

All three verify with the same two ingredients: a pinned channel or a typed digest, and a
manifest whose files' hashes are checked before anything runs. None is `curl | sh`.

## Peer to peer distribution

Once a device has paired it holds the cluster key, the certificate of the machine that took it
in and the cluster's signing public key. `onboard/transfer.py`:

* Any peer serves the signed manifest and, in ranges of one chunk, the files in it
  (`GET /onboard/v1/files/NAME`, `Range` required, signed by `macauth`, over the pinned
  certificate; `ShareServer`).
* `Downloader` takes a manifest verified under the pinned key and a list of peers and fetches
  chunks from several at once. Every chunk is checked against its listed SHA-256 before it is
  kept; a peer that sends one that does not match is dropped after a second such answer, a
  `critical` event is raised and the chunk is fetched from another peer; if no peer is left
  nothing is staged. The finished file is hashed whole again.
* Resumable: the verified chunks are listed beside the partial file, and each is hashed
  from disk before it is trusted on the next run (a damaged partial costs a re-fetch of the
  damaged chunk only).
* Disk: the file plus a 1 GiB reserve (a setting) must be free before the first request.
* The finished file goes to a **staging directory**, not to its home, and `on_staged(path,
  entry)` is called: that is where the guarded pipeline's scan and sentinel's
  `manifest.pin` / quarantine step belong. This branch implements the seam; the other
  branches' wiring is theirs to do.
* Licences: an entry with `shareable: false` (the signer sets it; a gated model is the case)
  is never asked of a peer and never served to one (`NotShareable` names its `source`, from
  which the new machine fetches with its own credentials). Deciding the flag from Hub
  metadata is designed, not built.

Never from an unauthenticated source: the manifest must verify under the pinned key (signature,
expiry, serial no lower than the last seen, entry names through `safenames`), and the peers are
cluster members over pinned TLS.

## Signing keys

An Ed25519 key per cluster, made on first use (`bootstrap`, `listen`) in
`<state root>/onboard/signing.key`, mode 0600, written exclusively (never overwritten). Its
public half rides in the grant. `key_id` is the SHA-256 of the public key and is on the
bootstrap page. What it vouches for: the controller lists a file only after checking it
itself (a wheel against the digest the release or index publishes, through `httpguard`; a
model through the guarded download and scan). Rotation: a new key signs a manifest and a
grant delivers the public key; an old key is refused once a member has pinned the new one
(`revoked_keys`); a manifest has a serial that may not go backwards and an expiry (a week).
The release-signing story for the wheel itself (minisign, a key held by the maintainer and
checked by the controller when it downloads) is outside this branch; the cluster key signs what the
cluster distributes, not who built it. **If the signing key is stolen** a thief can sign a
manifest that members accept: it is as sensitive as the cluster key, and is kept the same way.

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
| Offer address guessing or reuse | 128-bit token, 10 minutes, lockout after 8 wrong in a minute, uniform 404 | yes |

### Out of scope

A compromised owner machine or signing key; physical access; what the upstream package index
or model host serve before the controller checked it; timing side channels in the pure Python
curve arithmetic (rate-limited and interactive, accepted, review wanted); IPv6 and multi-subnet
networks (TTL 1, IPv4); the internet; the Windows toast; a web interface panel; phone
platforms; QR rendering; the re-key flow; per-device credentials; hash-locked dependency
install; mDNS.

## What is built

| Part | State |
|---|---|
| SPAKE2 pairing, three tries, confirmations in the right order | built, tested over real TLS sockets |
| Request state machine, limits, devices ledger, revoke | built, tested on real files, one test with a second process |
| Discovery announce / browse, injectable transport, real UDP on loopback | built; multicast and broadcast on a real LAN not verified |
| Notification adapters | macOS command shape verified against the real `osascript` (no notification shown); Linux by argv only; Windows not built |
| Signed manifest (Ed25519), chunked and resumable multi-peer download, scan hook | built, two server processes on loopback, with pinned TLS |
| Bootstrap offer, pinned installer | built; installer run as a separate process; `curl --pinnedpubkey` checked against the real `curl` |
| CLI (`nearby listen pair requests accept decline revoke bootstrap share fetch`, all `--json`) | built; end-to-end tests across separate processes: listen, pair, accept, fetch with a gated file refused, share refusing unsigned requests, a stale manifest refused |
| Sentinel events | emitted for each step (list below); adapter documented, not wired into sentinel here |
| SSH push, QR, mDNS, Windows toast, re-key, per-device credentials, web panel, hashed dependencies | designed only |

Events (all on `onboard.events.BUS`; `Event(kind, severity, subject, evidence)`):
`onboard.request.received|accepted|declined|expired|refused`, `onboard.pair.wrong_code` (warning),
`onboard.pair.locked` (critical), `onboard.pair.succeeded`, `onboard.revoked`,
`onboard.notify.failed`, `onboard.nearby.flood`, `onboard.transfer.bad_chunk` (critical),
`onboard.transfer.bad_file` (critical), `onboard.transfer.peer_dropped`,
`onboard.transfer.unshareable_asked`, `onboard.transfer.staged`, `onboard.bootstrap.offered`,
`onboard.bootstrap.refused`. Evidence holds short facts, never a code, key or secret.
Sentinel subscribes with a six-line adapter on its own side (it imports none of its sources):

```python
def watch_onboarding(bus, sentinel, Event, Severity):
    return bus.subscribe(lambda e: sentinel.bus.emit(
        Event(e.kind, Severity.parse(e.severity), "onboard", e.subject, e.evidence, e.ts)))
```

### Gaps in the prototype that matter when it is folded into the daemon

* The pairing listener and the file server run on **their own ports with their own
  certificate** (`<state>/onboard/tls`). In the finished design these routes live on the
  daemon's port under `/onboard/v1/`, with the daemon's certificate, so the certificate in a
  grant (`Grant.certificate`) is the one the beacons carry. Until then `trust.json` records the
  pairing partner's certificate and `fetch` uses it, but discovery does not yet consume it.
* `fetch` takes one peer from the command line (the machine it paired with, whose certificate
  it knows). `Downloader` takes any number of peers and is tested with two; learning the others'
  certificates from beacons is the daemon-side work.
* Routing the pairing, share and offer servers through one handler (`onboard/web.py`) costs the
  repo's `http-servers` budget one handler. See "Decisions".
* `cryptography` is required (the pinned TLS needs it or `openssl`; the manifest signature needs it).
* The sentinel adapter is documented and tested in `tests/test_onboard_sentinel.py`, which
  skips here and runs where sentinel is merged.

## Tests and mutation checks

126 tests in `tests/test_onboard_*.py`, about 35 s on a loaded 16-core laptop, all against real
sockets, files and processes: TLS handshakes on loopback, a relay that terminates TLS, two
peer processes serving the same file (one serving poisoned bytes), a separate process running
the installer, the real `curl --pinnedpubkey`, the real `osascript` (argument handling only),
and three or four CLI processes in one conversation. Nothing uses the real network beyond
loopback, no real notification is shown, and no real device is touched.

Mutation check: 79 guards were broken one at a time (a test run against each, `-x`). First
pass: 63 killed, 16 survived. Of the 16, 13 were missing tests, now written (for example: a
wrong confirmation could be retried on the same exchange, which would have allowed a million
guesses at the code on one exchange; an announcement list test that was an `or` and asserted
nothing; a manifest whose whole-file digest disagreed with its chunks); each of those mutants was
re-run and killed. The `share` and `fetch` commands added three more guards (stale manifest,
unsigned requests, private files): one survived for want of a test, which was written and
the mutant killed. Three survivors are equivalent: `context` in the SPAKE2 transcript (the
context is already in the code's scalar, so removing it changes nothing: kept as depth), and
the `Content-Range` and length checks on a chunk (a wrong range or length fails the chunk's
SHA-256 anyway; they exist for a clearer error and an earlier drop of the peer).

## Commands

```
ml-stack fleet listen [--for 10m] [--port 8772] [--no-cluster]    open pairing here
ml-stack fleet requests                                            what is waiting
ml-stack fleet accept ID | decline ID                              answer; accept prints the code
ml-stack fleet nearby                                              who is open to pairing
ml-stack fleet pair --host H [--port P] [--code C]                 ask to join
ml-stack fleet revoke NAME-OR-FINGERPRINT
ml-stack fleet bootstrap --share DIR [--valid 10m]                 offer ml-stack to a machine with none
ml-stack fleet share --dir DIR [--private NAME]                    serve files, signed, to paired machines
ml-stack fleet fetch NAME... --from HOST:PORT                      fetch into staging (not installed)
```

All take `--json` and `--state DIR`. State is under `<state root>/onboard` (requests, devices,
this machine's pairing certificate, signing key), every file private.

## Decisions for the owner

1. **Signing key**: where it lives (the controller's disk at 0600 as built, a hardware key, or
   offline with manifests signed by hand), who the controller is, and the rotation policy.
2. **Ship the address/QR bootstrap?** It is the easiest path and the one with the weakest
   authentication when opened in a browser. Recommended: ship it with the terminal command as
   the authoritative one, and the QR left out until a user asks.
3. **SSH push default**: recommended not to build it. If built, a separate flag, off, with the
   host key typed in full.
4. **Grant by default**: cluster key (every member is fully trusted) or device-only until
   per-device credentials exist. As built, the owner chooses with `--no-cluster`; the default
   gives the key.
5. **mDNS as an extra** (`ml-stack[discover]`) to avoid the Windows firewall step, versus
   staying stdlib.
6. **Budgets.** `http-servers` allows two handlers; onboarding adds one (`onboard/web.py`, shared
   by pairing, file sharing and the offer). Folding these routes into the daemon's handler on its
   port is the clean answer, and touches `fleet/api.py`, which the hardening branch is changing;
   an agent cannot raise a budget.
7. Whether `cryptography` becomes a hard dependency of the fleet (the signed manifest needs
   it; the pinned TLS needs it or `openssl`).
