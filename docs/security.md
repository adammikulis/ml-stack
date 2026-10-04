# Security

What ml-stack trusts, what it exposes, and what each kind of attacker can do. Reviewed
2026-10-02 on `agent/hardening` and carried onto the integration branch; the fix for each
finding is the commit named in the table. Other documents: `docs/credentials.md` (tokens and
keys), `docs/fleet.md` (the daemon), `docs/serving.md` and `docs/serve-admission.md` (model
servers and the broker), `docs/guardrails.md` and `docs/taint.md` (the agent's rails),
`docs/sentinel.md` (what watches for all of it), `docs/assistant-security.md` (the design contract
for anything that acts for a person, with what is built and what is not).

## The model

**Trusted.** The account that runs ml-stack, the files under `ML_STACK_HOME` and
`ML_STACK_CACHE`, every machine that holds the cluster key, `huggingface.co`, `github.com`
and the Python packages the user installed.

**Not trusted.** Anything that arrives over a network from a name or address the user did not
choose to trust: a web page, a model repository's file list, a downloaded archive, a
redirect, a GGUF file, a machine that has not proved it holds the cluster key, and anybody
else on the LAN.

**A cluster member is fully trusted.** The daemon runs the command line a peer sends it
(`POST /jobs`), downloads what a peer names (`POST /models/get`) and writes files a peer
sends (`PUT /files/*`). Holding the cluster key is the same as having a shell on every machine
in the cluster. What stands between the LAN and that is the cluster key, 256 random bits that
no passphrase derives. The passphrase is only a password for the join handshake, which locks out a
source that keeps failing, so it is at least 5 characters; `ml-stack-peers init` makes a key with
no passphrase. A job a peer submits is limited to an allowlist of ml-stack commands
(`docs/fleet.md`).

## What listens

| Surface | Default | Who can reach it |
|---|---|---|
| Fleet daemon, TCP 8770 | `127.0.0.1` (plain HTTP) until the machine joins a cluster, or `--lan`, `--host`, `--setup-from-lan`; beyond this machine it speaks TLS only | a LAN peer that pins its certificate and holds the cluster key (signed requests inside TLS); plain HTTP from another machine is dropped |
| Daemon web interface, `/ui` | this machine alone | other machines only with `--ui-from-lan`, and then over TLS only (it signs in with the passphrase) |
| Discovery beacons, UDP 8771, multicast `239.255.77.70`, TTL 1 | sent once a cluster is joined | the LAN segment |
| llama-server and other model servers | `127.0.0.1`, a random port | this machine; the daemon's `/infer` passes signed requests on to a fixed set of model-server paths |
| `ml-stack-graph` page | `127.0.0.1` | this machine |
| Pairing listener, TCP 8772, announcements UDP 8773 | off; only while `ml-stack fleet listen` runs | anyone on the LAN can send a request; nothing is given until the owner accepts and the code is typed. TLS only; plain HTTP is refused. `docs/onboarding.md` |
| Bootstrap offer, an HTTPS port chosen at the time | off; only while `ml-stack fleet bootstrap` runs, for ten minutes | whoever has the unguessable address; program files only |

A beacon carries the machine's name, port, device report and its TLS certificate, sealed with
AES-256-GCM under a key derived from the cluster key; the kind, a timestamp and the asker's nonce
are authenticated. It never carries the passphrase, the cluster key or the request secret, and a
capture holds nothing a passphrase guess can be tested against. The one unsealed datagram is a
request to join a cluster by name and its answer (a port and whether it speaks TLS).

**The join handshake.** A machine that has the passphrase and not the key asks a daemon in the
cluster for it. They run SPAKE2 with the passphrase as the password and the daemon's certificate
fingerprint as an identity, confirm to each other, and the daemon sends the key sealed under the
exchange's key. An eavesdropper or an impersonator gets at most one online guess per attempt. A
daemon counts every attempt per source address: five in ten minutes without a success locks that
address out for ten minutes, thirty from all addresses lock the handshake for ten minutes, and
each attempt is logged with its source and outcome. The passphrase is kept in the operating
system's keystore on each machine that joined with it, and only a machine that holds it takes
others in.

## Who can do what

**Somebody on the LAN who holds no key.** Can see that a daemon is there and hear beacons. Cannot
run a job, read or write a file, learn the name or device report from `/health`, or open the
web interface. Can send requests; one that is not correctly signed is refused, and an address
with ten failures in a minute is locked out for a minute. Can try passphrase guesses against the join
handshake, at five per ten minutes per address (see above); nothing it captures supports a
guess offline. Cannot read traffic between peers: request and response bodies and beacons are
sealed under a key derived from the cluster key, and connections are TLS to a certificate the
beacon vouched for. With `ML_STACK_FLEET_TLS=off` it can read file transfers and the streamed
output of the model proxy, which are not sealed by the application.

**A web page in the user's browser.** Cannot read the daemon: it listens on loopback, an API
request must be signed with a secret the page does not have, and `/ui` requires a custom header
that a cross-origin form cannot set. A page using DNS rebinding reaches loopback under its own
hostname: unsigned `/health` detail and the web interface setup guard check that the request
names an address, not a DNS name.

**A malicious model file or repository.** A repository's file names are only written under
`ML_STACK_HOME` after `safenames.safe_filename`; a download from a URL cannot reach a private or
metadata address at any redirect; a GGUF is parsed by llama.cpp, which is outside this library, and by a reader here that refuses a header that claims more than the file holds.
No model is loaded with a pickle (`torch.load`, `pickle`, `yaml.load` and `eval` are not used on
a file anywhere in the package), and `trust_remote_code` is never set.

**A malicious page being scraped or read.** `ml_stack.web.read` goes through `ml_stack.net`:
the address is resolved once and connected to, every redirect is re-checked, the body is
capped, hidden text and invisible characters are removed, and the result is fenced and marked
untrusted. A URL that only fetched content mentioned is not fetched by itself. The browser
checks every request it makes and downloads nothing. The remaining gap is a name that changes
answer between the browser's check and its load (`docs/internet.md`).

**A malicious release asset or archive.** Names are checked, links and devices refused,
entry count and unpacked size capped, and a release asset must carry a sha256 digest that
matches. The digest proves the bytes are what GitHub holds, not who built them.

## Default-on protections

These apply to every consumer of the library without a switch.

- A model server is started, reused and talked to through the machine's broker and nothing
  else (`ServerManager.lease` goes to the broker; `tests/test_serve_no_bypass.py` reads the
  source for any other way a process, a connection or the private start is reached). The broker
  admits a server only when the machine has the memory for it and queues generation requests
  one at a time per pool.
- An `Agent` runs with the built-in rails (tool policy, secrets, untrusted-text fencing, taint)
  and the model tier unless the caller passes `interventions=guard.off(because=...)`, which is
  logged. An empty list is refused.
- A GGUF or safetensors header is read by one bounded reader (`ml_stack.hub.modelfile`): every
  count and length is checked against the file's size, and a header is held to a pair cap, an
  item budget, a nesting depth, a dimension cap and a time budget.
- A server a `ServerManager` starts stops with the process that started it, including when that
  process is killed outright (`stop_on_exit`, a watchdog process, an orphan sweep on the next
  lease that stops only a server whose record proves its pid).
- Importing `ml_stack` registers no signal handler or exit hook, builds no manager, opens no
  socket and writes no file; `ml_stack.__version__` and a `NullHandler` are all it adds.
- The daemon listens on this machine until the machine joins a cluster; beyond this machine it
  is TLS to a pinned certificate or nothing (`ML_STACK_FLEET_TLS=off` is the one named switch,
  announced at every start); every request is signed, fresh and unseen, and its body is sealed
  with AES-256-GCM under a key derived separately from the signing secret; request framing,
  connection count and time are bounded.
- A cluster's key is random and is handed to a joining machine by a password-authenticated
  exchange; no key is derived from a passphrase, and a protocol 2 peer is ignored.
- `ml_stack.http.open_stream` and `build_request` open only `http` and `https`; messages about
  a URL carry no user, password or secret query parameter.
- Everything fetched from the internet goes through `ml_stack.net` (`docs/internet.md`): a
  host allow-list with recorded approvals; loopback, private, link-local, metadata,
  carrier-grade-NAT and multicast addresses refused at every redirect (an operator names an
  exception in `ML_STACK_FETCH_ALLOW_HOSTS`); no downgrade from HTTPS; size and time caps;
  a pinned SHA-256 where a manifest lists one, and none to download without one where a digest
  is required; staging in a private directory; GGUF, safetensors, PDF and archive checks;
  a virus scan where a scanner exists; provenance beside the file; a file that fails goes to
  sentinel's quarantine. `tests/test_net_no_bypass.py` fails when another module reaches out.
- llama-server is never handed an `hf:` reference, so it never downloads anything itself.
- Archives unpack through `safenames.unpack`; a remote file name goes through
  `safenames.safe_filename`.
- Child processes (model servers, a peer's job) do not inherit tokens and keys.
- An MCP server started over stdio runs under the sandbox (`ml_stack.sandbox`): it reads the
  interpreter and the project, writes a scratch directory, reaches loopback only and sees only
  the environment it was given. With no sandbox available it is not started unless the caller
  passes `AllowUnsandboxed(reason)`, which is logged and sent to the sentinel.
- A credential is read through `ml_stack.credentials`, from a file only its owner can read.
- State, token and credential files are written atomically with mode `0600`.
- Server logs are bounded in count, size and age.
- Decision models are the guard/decider layer, and their kind is a label, not a trust grant:
  `decision` in a listing (`hub.KINDS`) comes from the decider registry or from the words in a
  model's own name, so a file can claim it. The label changes no admission: such a model is
  still served only through a Broker lease, fitted against memory like any other, and the
  hashes in its decider config are checked when it loads.

## The OS keystore

One module, `ml_stack/keystore.py`, talks to the operating system's keystore, and
`tests/test_keystore_gate.py` fails if another does. It holds one master key per user; the
memory vault, the fleet signing key and wrapped credentials use subkeys cut from it with a
purpose label that is also authenticated data. It asks the OS lazily and at most once per
process, stops for good in a process after a refusal (and for ten minutes across processes),
allows twenty operations per hour, lets one process of several that start together do the
asking, and refuses a background process until a person runs `ml-stack-security unlock`.
Every operation is a sentinel event without a value. The design and the limits are in
`docs/keystore.md`; the attacks it is tested against are in `tests/test_redteam_keystore.py`.

## What is exempt from the net scan

`tests/test_net_no_bypass.py` fails for any module outside `ml_stack.net` that imports a network
library, opens a socket, runs `git`, `curl` or `pip install` against a remote, or calls
`ml_stack.http`'s request functions. The list of modules it lets through is short and each entry
carries its reason in the test. Three kinds, none of which can fetch from a public host on its
own:

- **The guard's judge model** is attacked directly and measured with a real model: results in
  `docs/redteam/findings.md` ("Judge hardening") and `docs/guardrails.md` ("Attacks on the judge itself").
- **Model servers a person pointed at.** `client/`, `bench/`, `fleet/*` peers, `serve/*` (a
  server's slots and props by local port), and the decide backend (`decide/logprob.py`,
  `decide/router.py`, the server in `ML_STACK_DECIDE_URL`, 127.0.0.1:8080 unless set). These
  talk to an inference server the person configured; nothing they get back is installed or kept.
  The decide backend is stricter than `client/`: its URL must be on this machine (127.0.0.1,
  `localhost`, `::1`) unless the operator names the host in `ML_STACK_FETCH_ALLOW_HOSTS`,
  because a decider sees every tool call it judges (`decide.logprob.require_decider_host`;
  `tests/test_decide_hosts.py`). `client/` may still name a remote host; that is the person's choice.
- **LAN onboarding** (`fleet/onboard/pairing.py`, `fleet/onboard/transfer.py`). A peer's
  certificate is pinned and never looked up in a trust store, so these cannot use the pipeline's
  client. What keeps them off the internet is `fleet/onboard/lan.py`: every connection first
  refuses an address that is public (loopback, private, link-local and carrier-grade NAT pass).
  A `PeerSource` needs a pinned TLS context, or `http` to this machine only. A test removes the
  check and watches it fail, and another asserts neither module builds a default-trust client.
- **Tailscale** (`fleet/tailnet.py`, `fleet/onboard/routes.py`). Detection runs only
  `tailscale status --json` (bounded output and time, scrubbed environment, tailnet-range
  addresses only, no auth field copied). `routes.py` checks a paired device's pinned certificate
  over its LAN or tailnet address after `lan.py` has refused public addresses; a tailnet address
  is a route, not a trust grant, and is learned only from the pairing exchange or a status peer
  whose certificate matched, never from an announcement. See `docs/onboarding.md`.
- **The red-team lab** (`redteam/tools.py`, `redteam/scenarios/fleet.py`,
  `redteam/scenarios/isolation.py`), which only connects to servers it started on 127.0.0.1.

A relative import such as `from .requests import ...` is a sibling module, not the `requests`
library, and is not flagged. Everything that does reach a public host (the Hub, GitHub, the
geocoder, pages, `huggingface_hub`) goes through `ml_stack.net`: the decider checkpoint and the
injection classifier now do too.

## Internet pipeline

`docs/internet.md` lists every path, what protected it, what protects it now and what is
left. Open in that table: `ddgs` makes its own requests; pip installs unpinned dependencies;
`lm-eval` fetches datasets; a name that changes answer between the browser's check and its
load; a well-formed model with poisoned weights or malware no signature knows passes a scan.

## Findings

Severity is what an attacker gets on the code as it stood at `agent/audit`. File and line are
at that commit.

| # | Sev. | Where | What an attacker does | Fixed in |
|---|---|---|---|---|
| 1 | High | `fleet/daemon.py:396`, `fleet/api.py:99` | A host on the LAN sniffs the static bearer token from plaintext HTTP and has code execution on every peer (`POST /jobs`) | `188ff67` signed requests: the secret is never sent, requests are fresh and single-use |
| 2 | High | `fleet/daemon.py:396` | The daemon listened on every interface by default, so a machine that never joined a cluster was reachable | `188ff67` loopback unless joined, `--lan`, `--host` or `--setup-from-lan` |
| 3 | High | `fleet/models.py:252`, `fleet/api.py:421` | A peer (or anything holding the token) names `http://169.254.169.254/...` or an internal URL as a model source and the daemon fetches it; redirects were not re-checked | `e12a6d2` downloads go through `httpguard`; `db89b03` the guarded fetch itself |
| 4 | High | `fleet/api.py:187,383,542` | `Content-Length` was read with a bare `int()`: a non-number raised in the handler, a negative one made `rfile.read(-1)` wait for EOF holding a thread, a huge one was read into memory | `188ff67` one validated reader, 413 above the cap |
| 5 | Medium | `fleet/api.py:546` | `Content-Range` parsed with a bare `int()`; offsets and lengths were never checked against each other | `188ff67` |
| 6 | Medium | `fleet/api.py` (server) | A client that dripped headers, or opened connections without limit, held a thread each (slowloris) | `188ff67` 15 s header deadline, 30 s socket timeout, 64 connections |
| 7 | Medium | `fleet/api.py:174` | The `/infer` proxy forwarded any path to the model server, including `POST /slots/N?action=erase` and `/props` | `22b0267` allow-listed paths; caches and properties are read-only |
| 8 | Medium | `fleet/api.py` (`/health`) | An unauthenticated caller read the machine name, device report and what it serves | `188ff67` |
| 9 | Medium | `fleet/ui.py` login | Another machine on the LAN opening the web interface sent the passphrase over plain HTTP | `0828ee2` `--ui-from-lan` is the named opt-in; now it answers other machines over TLS only |
| 10 | Medium | `fleet/discovery.py:86` | A 5 character minimum passphrase against offline grinding of a beacon's MAC | `7c13db7` 12 characters; superseded: beacons are sealed under a random cluster key and the passphrase only goes through the join handshake, so the minimum is 5 |
| 11 | Medium | `serve/manager.py` | A killed host left its llama-server holding GPU memory until the next lease on that port | `c367a8a` exit hook, signal chain and watchdog; `6f93a5a` verified orphan sweep |
| 12 | Medium | `serve/process.py:126,157` | Without psutil, `kill_process_tree` silently did nothing and records were dropped | `c157987` psutil is a core dependency |
| 13 | Medium | `fleet/updates.py:186` | A release asset's name from the API JSON was joined onto a directory (`../`), and an asset with no digest was accepted | `e12a6d2` |
| 14 | Medium | `fleet/updates.py:526` | A tracked repository or branch beginning with `-` or using the `ext::` transport reached `git` as an option or a program | `e12a6d2` validated, and passed after `--` |
| 15 | Medium | `serve/build_release.py:86`, `fleet/llama.py:136`, `fleet/updates.py:238` | Three archive extractors with no size or entry cap; a zip bomb filled the disk | `e12a6d2`, `982b8eb` one extractor, `safenames.unpack` |
| 16 | Medium | `fleet/autostart.py:140,187` | The boot job was staged in a user-writable directory and copied by the privileged command; a user-level process swapped it while the password dialog was open | `70a13f1` the command carries the text |
| 17 | Medium | `packaging/install.sh:112`, `install.ps1:129` | The installers ran what a release zip contained with no integrity check | `60df6a0` digest required and compared |
| 18 | Medium | `serve/binary.py:234`, `fleet/jobs.py:255`, `serve/python_engines.py:56` | A model server or a peer's job inherited `HF_TOKEN`, `ANTHROPIC_API_KEY` and every other secret in the environment | `0bc5597` |
| 19 | Low | `sources/html.py:239` | A document with a DOCTYPE and nested entities expanded without bound (`S314`) | `982b8eb` refused |
| 20 | Low | `fleet/autostart.py:109`, `scrape/browser.py:74` | A path or an application name containing `"` ended an AppleScript string | `70a13f1` escaped |
| 21 | Low | `http.py:130,140` | A URL with a password or `?token=` appeared in `ServerError` messages and so in logs; `file:` and `ftp:` were opened by `urllib` | `f36831e` |
| 22 | Low | `fleet/daemon.py:62` | The token file was written with the umask's mode and `chmod`ed afterwards | `188ff67` atomic `0600` |
| 23 | Low | `serve/backend.py:42` | One log per start accumulated forever | `086442d` |
| 24 | Low | `.github/workflows/*.yml` | Third-party actions pinned by tag, no `permissions:` on `ci.yml` and `release.yml` | `2478cfc` |
| 25 | Info | `credentials` (new) | Tokens were read ad hoc from the environment by each caller | `816f978` one resolver; `docs/credentials.md` |
| 26 | Info | `http.check` | The address check ran before the connection, so DNS could answer differently the second time | `db89b03` `httpguard.fetch` connects to the address it checked |
| 27 | Low | `serve/preflight.py:87`, `serve/mlx_tree.py:92` | A model file whose header declares a string, an array or a pair count of 2^40 made the preflight allocate or loop on it | `35846af` |
| 28 | Medium | `fleet/daemon.py`, `fleet/api.py` | Anyone on the segment read and altered peer-to-peer bodies (jobs, files, replies) | `d6ea366` |
| 29 | Medium | `fleet/discovery.py:_salt_for` | The passphrase salt was a hash of the cluster name, so a table precomputed for `ml-stack` fit every cluster of that name | `518a855` per-cluster random salt, protocol 2; superseded: no key is derived from a passphrase |
| 30 | Medium | `hub/header.py`, `hub/cards.py`, `serve/tensors.py` | Three more GGUF readers walked the counts and lengths a file claimed: a 2^60-byte string raised `MemoryError`, a 2^50-item or deeply nested array looped for as long as it liked, and a model list hung on one hostile file | integration: one reader, `hub/modelfile.py` |

`ruff --select S` (bandit) over `src`: 84 findings at `agent/audit`; the `ruff-security` budget
now stands lower by the sites fixed (S104, S202, S314, S310, S107, the sampling `S311`s) and the
rest are the `except ...: pass` sites listed in `HANDOFF.md`. `pip-audit` over the core
dependencies and the `store`, `hub`, `web`, `plot`, `graph`, `scrape`, `vision`, `pdf`, `gguf`,
`train`, `test`, `mcp`, `claude` and `privacy` extras, resolved on 2026-10-02, reports no known
vulnerabilities. There is no lockfile to audit, and `metal-smi` has no Linux distribution to
resolve.

### Supply chain gates

`audit.yml` fails on a known vulnerability in the Python extras (pip-audit), `app/package-lock.json` (npm audit) and
`app/src-tauri/Cargo.lock` (cargo audit), through `scripts/audit_gate.py` and the expiring allow-list
`.github/pip-audit-allow.json`; nothing in the workflow is `continue-on-error`, a report the tool did not produce fails
the job, and `tests/test_supply_chain_audit.py` parses the workflow YAML and runs the gate on recorded reports of all
three formats. `release.yml` builds `sbom.cdx.json` (CycloneDX 1.5, `scripts/sbom.py`, offline: the installed extras and the
`Cargo.lock` crates with their registry checksums) and attaches it to the release. Actions in both workflows are pinned to a
commit. Not covered: npm packages are not in the SBOM, and the audits need the advisory databases, so they run on the
runner, not here.

## Reviewed and clean

- Deserialization: no `pickle`, `marshal`, `yaml.load`, `torch.load`, `eval` or `exec` of data
  anywhere in `src`; weights are read as safetensors or GGUF; `from_pretrained` never sets
  `trust_remote_code`; `np.load` is called without `allow_pickle`.
- Shelling out: one `shell=True` (`checks.ask`, a fix line shown to and confirmed by the
  person); everything else is an argument list, and llama-server's arguments are built as a
  list from validated values.
- `fleet/files.py:safe_relpath` refuses absolute paths, `..`, odd characters and escapes after
  `resolve()`; tar extraction was already filtered.
- Constant-time comparison is used for every secret compare (`hmac.compare_digest`).
- Workflows: no `pull_request_target`; the one expression put in a `run:` is a commit SHA;
  `workflow_dispatch` inputs go through `env:`.

## Adding a device (`docs/onboarding.md`)

A device is added with a short code through a password-authenticated exchange (SPAKE2 from the
maintained `spake2` package, no group arithmetic of our own) with both certificate fingerprints
as its identities, so a wrong code and a machine in the middle
fail the same way, and a captured exchange gives nothing to test guesses against. The code
exists only after the owner accepts, never appears in a notification, lives 120 seconds and
has three tries; requests are limited per device, per address and overall, and a declined or
failed device waits. A cluster member is fully trusted, so pairing hands over the cluster key
unless told not to; `revoke` stops a device asking again but, while it holds the key, the
cluster key must be changed to lock it out (not automated yet). Files from peers are checked
against a manifest signed with the cluster's Ed25519 key (generated on the controller, kept in the
OS keystore, separate from the cluster key; export, rotation and revocation need a person at a
terminal; manifests last three days), chunk by chunk, and are staged for the scan rather than
installed. A gated model goes only to the devices the owner marked as theirs, after their licence
acceptance is on record; credentials never travel. Nothing pushes software to a machine that has none: the owner
opens an offer on it and runs a short, pinned installer, or starts `bootstrap --ssh` with their own
keys, a host key typed in full and a fixed script. Model downloads ask the paired devices before the
Hub (`docs/model-discovery.md`, "Peers first"): the digest a file must have is the Hub's listing,
never one a peer states; peers are reached only on private, loopback, link-local or tailnet
addresses over pinned TLS; the serving device applies the sharing level and withholds a quarantined
copy; a peer whose bytes fail a digest is dropped and reported to sentinel. Off with `--no-peers`,
`ML_STACK_NO_PEERS=1` or `ml-stack fleet peers off`. Pairing fills the peer book: each side's
share address, pinned certificate, signing key and request key arrive in the grant and in an
offer sealed under the exchange (a bad tag, or a certificate other than the one the exchange
bound, stores nothing); revoking removes the rows. `fleet share --models` serves only paths that
resolve inside the model roots at every request, never a quarantined copy, and a gated model only
to the owner's marked devices with the licence acceptance on record, which the downloading device
writes down (who, when, hash) in an append-only file. Routes to a paired device (this network,
then its tailnet address) always end in the pinned certificate after the public-address check; a
peer that stays under 2 MiB/s for 15 s is given up for the Hub. What this does not cover:
the pairing mathematics is Python integers (not constant time) and wants an independent review
before a public release; the re-key flow, per-device credentials and SSH push are designed and
not built.

## Open

These need a decision, or work that belongs to another branch (`HANDOFF.md`, ml-stack issue
#18).

- **Windows:** the mode and ownership checks on credential files do not apply, `icacls` is
  best effort, and the exit watchdog was not run on Windows.
- **`scripts/test-on-linux`** needs Docker's daemon; `docs/INTEGRATION-REPORT.md` says whether it
  ran for that integration.

## Requests

Everything that waits for a person (a tool call that asks first, a fact to remember, something
sentinel holds) is one request in an encrypted per-user store (`docs/requests.md`). It is answered
only by a person: at a terminal with a tty, or in the browser page behind a launch key, a session
cookie, a CSRF token and the loopback Host/Origin checks. An answer is bound to the fingerprint of
the words shown, the first answer wins, and expiry or a failing store denies. No token, tool, MCP
route or workspace message can answer, cancel or list the answers of a request.

## Sentinel

`ml_stack.sentinel` watches for the attacks above and holds what they touched; the design,
the measurements and the limits are in `docs/sentinel.md`. What it adds to this model, and
no more than what its tests show:

- **A pinned model, binary, adapter or config file that changes is found** at load and on a
  timer, by digest (and by link target), and in the default `guarded` mode moved aside into a
  `.ml-stack-quarantine` directory under the managed root it sits in. Nothing outside
  `ML_STACK_HOME`, `ML_STACK_CACHE` and the Hugging Face hub cache is ever moved. Release
  moves the file back byte for byte; only a person's confirmed purge deletes.
- **A peer that replays signed requests or sends forged ones is refused** once it crosses a
  threshold, through a wrapper around `macauth.Authenticator.check` that the fleet daemon
  installs (`tests/test_sentinel_wiring_serve.py` sends forged requests to a real handler).
- **Text a guard rail denies or reads as an instruction is held**, not shown again, and a
  summary that repeats a held sentence is held too. Nothing a model could be shown comes back
  except through a person's release at a terminal. Rewording defeats the summary check.
- **Release is human-only, and `ml-stack-security review` keeps it that way while making it
  easy.** A release or purge needs a grant that only `human.mint` or `human.mint_pressed` can
  make, and both refuse before reading a key unless stdin and stdout are terminals and no
  agent marker (`CLAUDECODE`, `ML_STACK_AGENT`, `ML_STACK_NONINTERACTIVE`) is set. The review
  screen lowers the cost (one confirming key to release, after showing what is unblocked)
  but not the bar: a purge still needs the id typed in full, the keys come only from the
  terminal, and everything printed from a held subject is escaped and bounded. Viewing
  (`review --list`, `status`) is open to agents.
- **The release dialog is the one release that takes a click instead of a terminal.** When a
  quarantine means something changed or was forged, one dialog lists it with `Release`,
  `Keep held` and `Later`. The release is made only by the process that put the dialog up
  (`heads_up.HeadsUp`, the only caller of `human.mint_clicked`), only for the ids the dialog
  listed, only when the answer it got back is exactly the `Release` label, and never in a
  process whose environment carries an agent marker; no tool, MCP endpoint, chat tool or
  command-line flag can show it or answer it. It defends against an agent or a held text
  that tries to release something: names in the dialog are escaped, the buttons are fixed,
  and a name that spells `Release` presses nothing. It does not defend against another
  process of the same user that drives the screen through the accessibility permission and
  presses the button; that is the same-user limit the marker check has, and the reason the
  dialog lists exactly what Release will unblock. A process that clears its own environment and drives a pseudo-terminal could
  still press the keys, as it could for `quarantine release`; that is the limit of a
  same-user marker (`docs/sentinel.md`, "Reviewing what is held").
- **Decoy credentials** planted in the state root on the first run of an agent or a daemon turn
  any read or send of them into a high-confidence event that freezes the session.
- **A bare `Agent` is screened without anyone calling sentinel:** a tool call a rail refuses
  is parked and counted, one that names a decoy or a sentinel path is refused, and a result
  that is held or carries a decoy value is replaced. Running without it takes
  `unwatched(because=...)` or `guard.off(because=...)` and is logged.
- **A server is started only for a model that matches its pin** (a model ml-stack pulled is pinned
  at pull time, `source=pull`; a model it did not pull is pinned on first use and logged as such; a signed
  owner manifest that this machine accepted and that names the file must agree with it, else the file is
  quarantined and an older manifest is refused), and a held model or server cannot be leased or restarted until a person
  releases it. The Broker and fleet daemons scan every pin on a timer
  (`ML_STACK_SENTINEL_SCAN`), and `ml-stack security status` says whether that loop is armed.
  Switching sentinel (`ML_STACK_SENTINEL=off`) or the scan off needs a reason in
  `ML_STACK_SENTINEL_BECAUSE` / `ML_STACK_SENTINEL_SCAN_BECAUSE`; without one the switch is
  ignored and logged. A start skips re-hashing a model whose size, mtime and inode are
  unchanged since its last full check; an in-place edit that restores them waits for a deep scan.
  `docs/sentinel.md`, "What is armed by default", lists what is not wired.
- **Repeated small signals add up.** Guard denials of a running agent, sandbox refusals during
  its tool calls, refused leases and resource-limit hits of a Broker caller are scored per
  session or caller: watch at 3, quarantine (a frozen session, a blocked caller) at 10 within
  ten minutes, released only by a person. Another session or caller is unaffected.
- **A model's reply is labelled (or, in `enforce`, held) when it carries a decoy value, held
  content or a secret-shaped string; a poisoned compaction summary is held and not fed back;
  a quarantined credential's variable is left out of the children the sandbox, `jobs.detach`,
  the server launcher and the MCP launcher start; a quarantined MCP server is not connected.**
- **A decoy HTTP endpoint on loopback** (Broker and fleet daemons) turns any request to it into
  a high-confidence event and freezes the sessions that had a tool running. Its address is only
  in `credentials.endpoint` under the state root. Off: `ML_STACK_SENTINEL_DECOY=off` with
  `ML_STACK_SENTINEL_DECOY_BECAUSE`.
- **Served models are probed on a schedule** (default hourly, 18 short requests each) against the
  answers recorded the first time: a drift is a watch, a large confirmed one quarantines the
  model. Off: `ML_STACK_SENTINEL_CANARY=off` with `ML_STACK_SENTINEL_CANARY_BECAUSE`. It
  detects change, not a model that was bad at the start.
- **The event log is hash-chained** and `ml-stack security verify` finds edits, cuts and
  reordering; the head can be written down elsewhere.
- **A model that answers a fixed probe set differently from how it did at install is
  flagged.** This is not backdoor detection. A model that was bad when it was pinned, or whose
  trigger no probe contains, passes.

Release, purge and mode changes need a person at a terminal; the check against an agent is on
strings and on the environment, and code running as the owner can bypass all of it. In
`observe` mode nothing is acted on; in `guarded` (the default) only the high-confidence
signals above act; `enforce` also acts on thresholds.

## Agent workspace

`ml-stack-workspace` (`src/ml_stack/workspace/`, design in `docs/workspace.md`) lets agent
processes on one machine exchange messages and notes and avoid each other's ports, worktrees
and scratch files. It is a channel from one model's output into another model's input, so it is
treated as untrusted input.

**What listens.** Nothing. The workspace is a directory (`ML_STACK_WORKSPACE_HOME`, else
`<state>/workspace`, mode 0700) read and written by the commands and MCP tools of processes of
one account. There is no socket and no LAN exposure.

| Attacker | What they try | What stops it |
| --- | --- | --- |
| An agent steered by injected text | send as someone else | the sender is the owner of the token; `send` has no sender argument, and the registry stores only a SHA-256 of each secret, so reading it yields no identity |
| the same | get a rule adopted, or a person's approval claimed | every item is delivered inside an `<untrusted>` fence labelled with sender, role and `no authority`; "approved", "authorize", "from now on all agents" and "add to CLAUDE.md" patterns quarantine the item; a rule note is advice with `binding: false`; no operation edits repository documents |
| the same | get a note believed | trust is set by the service from the token and from a run of an allow-listed command, never by the writer; a lower-trust note cannot supersede a higher one; facts carry a TTL and show as stale |
| the same | leak a secret or a private term through a note or message | the write is refused, naming the rule and not the match; the audit log keeps lengths, never text |
| the same | flood, fill the disk | per-sender sliding-window rate limit, body and subject caps, per-recipient inbox cap, per-agent note and scratch limits, retention |
| the same | escape its scratch folder | names are plain, resolved paths are checked after following symlinks, a symlinked agent or folder directory is refused |
| the same | hold the lead's ports and branches | claims have a TTL renewed by heartbeat and are released when the owning pid is gone |
| the same | run a command through a note | the service runs a note's command only for a lead or human token, only if the owner listed its leading words in `limits.json`, with no shell, a reduced environment and a timeout; the default list is empty |
| anyone | edit the history | every log is hash-chained; `audit-verify` finds the first broken row and accepts an external anchor for truncation |

**Not defended.** A hostile process of the same account can read the files, edit or delete the
logs (a full rewrite with recomputed hashes is undetectable) and read another process's
environment. Pattern screens miss paraphrase. The workspace labels and fences text; it does not
stop a model from obeying what it was shown.

**Operations.** `init` refuses to run when `CLAUDECODE`, `ML_STACK_AGENT` or
`ML_STACK_NONINTERACTIVE` is set. Minting, revoking, releasing quarantine, running a note's
command and `gc` are CLI-only and are not offered over MCP. Tokens come from
`ML_STACK_WORKSPACE_TOKEN` or `--token-file`, never from an argument.

**Private terms.** The denylist is a file outside the repository (`ML_STACK_WORKSPACE_DENYLIST`,
else `<state>/workspace/private-terms`), one term per line.

## Sandbox

`ml_stack.sandbox` confines a command to a deny-by-default policy (`docs/sandbox.md`): files it
may read and write, programs it may start, loopback or named ports or no network, an explicit
environment, and time, CPU, file-size and output limits. It is the second layer behind the
guard's rails and the taint rule: a command the guard allowed still cannot read a credential file
outside its allow-list, write outside its scratch directory or open a connection.

- **Where:** the shell tool (`ml_stack.sandbox.tools.SandboxedBash`), MCP servers started by
  `McpTools.stdio`, Claude Code's Bash tool in `ml_stack.harness`, and a model server when
  `ML_STACK_SANDBOX_SERVE=1` or `LlamaServerBackend(sandboxed=True)`.
- **Fail closed:** with no sandbox the command is not run. `AllowUnsandboxed(reason)` is the only
  way around it and is logged on every use.
- **Events:** a refusal is `sandbox.denied` on the sentinel bus (warning), with the operation and
  path; `sandbox.unavailable` and `sandbox.unsandboxed` are warnings too.
- **macOS uses `sandbox-exec`, which Apple has deprecated.** This is a recorded exception
  (`docs/sandbox.md`), confined to `sandbox/seatbelt.py`; GPU workloads have no replacement yet.
- **Linux** (bubblewrap) is built and checked by its argument list only.
- `ml-stack security sandbox status|test` shows the backend and runs the guarantees.

## Live Gym and training workspace

The Gym daemon starts an installed Python interpreter with an argument list and JSON session
settings. MetaDrive uses native Bullet/Panda3D; warehouse sessions use RWARE; traffic sessions
start SUMO, and procedural traffic generation starts `netgenerate`. Closing a session terminates
its worker process group, including child simulators. These processes run as the daemon's OS
account. They inherit its environment and are not enclosed in the agent shell sandbox. They
are therefore trusted installed programs, not an isolation boundary for hostile native code.
Gym sessions do not create containers or mount host directories into them. Monitored workspace
jobs run an installed `ml-stack-*` command with literal argument strings, without a shell.

Browser routes require the UI authentication and host checks. Request bodies must be JSON
objects; session configuration and control payloads must be objects. Manual scenario filenames
are relative to the daemon job queue's files root, which the daemon supplies. Resolved paths
must remain inside that root; traversal and symlink escapes are refused. SUMO XML rejects
entities, document types and includes. This does not make an arbitrary native simulator asset
safe: import only scenario files whose source you trust. Recording camera paths and review files
are confined to their session directory; dataset exports remain under the daemon files root.

Observations, actor identifiers, scenario metadata and recorded state reach decision models as
simulation data. A decision selects from native action labels; it does not grant shell, network,
credential, role or rule access. Decision inference runs in a separate process with one outstanding request. Auto accelerator loading first acquires the maintained broker’s exclusive GPU claim; old brokers without atomic GPU/model exclusion are refused. CPU can be selected explicitly. Loading, pending, stale and abstaining results apply the environment’s explicit braking or hold control while native physics continues; they never impersonate a trained model. Results retain their original observation, sequence, actor, model revision, probabilities and latency separately from the current transition. Reset, actor selection and controller changes invalidate old results. Pause cancels decision and vision processes, releasing their device claims and serving leases, and the existing process exit guard protects it when its owning physics worker exits. Missing verified model files appear as a readiness error. Optional drone vision uses the same bounded child transport and acquires a normal serving-broker lease; it does not force GPU admission. Only captured RGB and synthetic visible-surface thermal pixels reach the vision prompt. Actor ground truth and simulator detection labels are excluded. Frame IDs, image hashes, camera metadata and model revisions accompany results; stale or superseded results cannot control another actor. Apple FastVLM safetensors and MLX/CoreML weights remain explicitly unsupported by this llama.cpp path. The default maximum input age is one second; `decision_max_age_s` can be set between one and thirty seconds. Checkpoints and
scenario names are untrusted inputs: learning checkpoints must come from the trusted Gym
artifact directory, and decision weights use the model loader's integrity checks. Frozen and
online learning change policy updates, not the daemon account's privileges.

Gym controls are browser operations, not MCP or chat tools. Human-only credential changes,
keystore access, grant minting, quarantine release, rule adoption and approval of operating
system actions are not simulation actions. Any future agent-facing simulation tool must pass
through the roles, rules and human-only floor described in [agent roles](agent-roles.md).

The Tauri window capability allows core/window-state operations and the app's close-choice and
closing handlers for its main window. Remote IPC origins are loopback HTTP URLs. It declares
no filesystem, shell, process-spawn or credential plugin permission. The packaged daemon still
runs with the launching user's OS privileges; limiting webview IPC does not sandbox the daemon.

Board participation in Fleet uses the existing UI authorization: a signed-in cluster session
or a strictly local UI on an unjoined machine. Every API request requires the UI header;
posts also require a same-origin JSON request. The maintained Board service reads the private
person token on the server and verifies its human role. Browser requests cannot supply a
sender or token. Messages pass through workspace screening, permissions and rate limits,
and remain untrusted message content. Page posts cannot write announcements. The agent
directory exposes identifiers and roles, never credentials. Requests are capped at 32 KiB.
