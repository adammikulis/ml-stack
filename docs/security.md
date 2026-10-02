# Security

What ml-stack trusts, what it exposes, and what each kind of attacker can do. Reviewed
2026-10-02, branch `agent/hardening`; the fix for each finding is the commit named in the
table. Other documents: `docs/credentials.md` (tokens and keys), `docs/fleet.md` (the
daemon), `docs/serving.md` (model servers).

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
in the cluster. The passphrase is all that stands between the LAN and that, so it is at least
12 characters, and `ml-stack-peers init` makes a random key.

## What listens

| Surface | Default | Who can reach it |
|---|---|---|
| Fleet daemon, TCP 8770 | `127.0.0.1` until the machine joins a cluster, or `--lan`, `--host`, `--setup-from-lan` | a LAN peer holding the cluster key (signed requests only); anyone gets `{"ok": true}` from `/health` |
| Daemon web interface, `/ui` | this machine alone | other machines only with `--ui-from-lan` (it signs in with the passphrase over plain HTTP) |
| Discovery beacons, UDP 8771, multicast `239.255.77.70`, TTL 1 | sent once a cluster is joined | the LAN segment |
| llama-server and other model servers | `127.0.0.1`, a random port | this machine; the daemon's `/infer` passes signed requests on to a fixed set of model-server paths |
| `ml-stack-graph` page | `127.0.0.1` | this machine |

A beacon carries the machine's name, port, device report and an HMAC-SHA256 under the cluster
key. It never carries the passphrase, the cluster key or the request secret. It does let
anyone on the segment test passphrase guesses offline (they cost one scrypt, N=2^16, each),
which is why a passphrase has a 12 character minimum; see "Open".

## Who can do what

**Somebody on the LAN who holds no key.** Can see that a daemon is there and hear beacons. Cannot
run a job, read or write a file, learn the name or device report from `/health`, or open the
web interface. Can send requests; one that is not correctly signed is refused, and an address
with ten failures in a minute is locked out for a minute. Can try passphrase guesses against a
captured beacon (see above). Can read any traffic between peers: bodies are not encrypted
(see "Open").

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

- A server a `ServerManager` starts stops with the process that started it, including when that
  process is killed outright (`stop_on_exit`, a watchdog process, an orphan sweep on the next
  lease that stops only a server whose record proves its pid).
- Importing `ml_stack` registers no signal handler or exit hook, builds no manager, opens no
  socket and writes no file; `ml_stack.__version__` and a `NullHandler` are all it adds.
- The daemon listens on this machine until the machine joins a cluster; every request to it is
  signed, fresh and unseen; request framing, connection count and time are bounded.
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
- A credential is read through `ml_stack.credentials`, from a file only its owner can read.
- State, token and credential files are written atomically with mode `0600`.
- Server logs are bounded in count, size and age.

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
| 9 | Medium | `fleet/ui.py` login | Another machine on the LAN opening the web interface sent the passphrase over plain HTTP | `0828ee2` `--ui-from-lan` is the named opt-in |
| 10 | Medium | `fleet/discovery.py:86` | A 5 character minimum passphrase against offline grinding of a beacon's MAC | `7c13db7` 12 characters |
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

`ruff --select S` (bandit) over `src`: 84 findings at `agent/audit`; the `ruff-security` budget
now stands lower by the sites fixed (S104, S202, S314, S310, S107, the sampling `S311`s) and the
rest are the `except ...: pass` sites listed in `HANDOFF.md`. `pip-audit` over the core
dependencies and the `store`, `hub`, `web`, `plot`, `graph`, `scrape`, `vision`, `pdf`, `gguf`,
`train`, `test`, `mcp`, `claude` and `privacy` extras, resolved on 2026-10-02, reports no known
vulnerabilities. There is no lockfile to audit, and `metal-smi` has no Linux distribution to
resolve.

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

## Open

These need a decision, or work that belongs to another branch (`HANDOFF.md`, ml-stack issue
#18).

- **No encryption in transit.** Requests are authenticated, not hidden, and replies are not
  signed. A design that fits the codebase: a self-signed certificate per daemon whose SHA-256
  fingerprint rides in the beacon, which is already MAC'd with the cluster key; peers pin that
  fingerprint, so there is no certificate authority and no trust-on-first-use. The standard
  library cannot make a certificate; it needs `openssl` on the PATH or the `cryptography`
  package. Decision wanted: add `cryptography` to the fleet, or require `openssl`, or accept a
  signed-only fleet on trusted networks.
- **The salt of the passphrase-derived key is the cluster name.** Two clusters named
  `ml-stack` share one, so a precomputed table covers both. A random per-cluster salt minted by
  the first machine would travel in its beacon; a joiner would try each candidate against the
  beacon's MAC.
- **`web.py`, the scraper's browser and `ingest/run.py`** still use `http.check` (resolve, then
  fetch) rather than `httpguard.fetch`. A page redirect or sub-request inside the browser is
  unchecked.
- **`hub/` calls `huggingface_hub` without `token=`** so `ML_STACK_CREDENTIALS_FILE` and the
  keychain do not reach a download there.
- **Windows:** the mode and ownership checks on credential files do not apply, `icacls` is
  best effort, and the exit watchdog was not run on Windows.
- **`scripts/test-on-linux` was not run** for this pass.
