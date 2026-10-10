# Everything from the internet

Every request Poolhouse makes to a host outside this machine and its network, and every file or
page that comes back, goes through one module: `poolhouse.net`. This page lists each path,
what protected it before, what protects it now, and what is left.

```python
from poolhouse import net

page = net.default().get("https://huggingface.co/api/models/owner/repo")
done = net.download(url, dest, net.Want(sha256=digest, require_digest=True))
```

## What the pipeline does

| Step | What happens | Where |
|---|---|---|
| Policy | A host is reached only if it is on the allow-list (`huggingface.co`, `*.hf.co`, `github.com` and its release hosts, `nominatim.openstreetmap.org`, the host of `$HF_ENDPOINT`, hosts in `$POOLHOUSE_NET_ALLOW_HOSTS`) or a person approved it. An approval is a row in `net/approvals.jsonl` with who, when, why and an optional expiry. `NeedsApproval` names the host and the command. Web pages a model reads use the origin rule below instead. | `net/policy.py` |
| Address | The name is resolved once, every address must be public (no loopback, private, link-local, the cloud metadata address, carrier-grade NAT, multicast, or any of those wrapped in IPv6), the connection goes to the address that was checked, and the check repeats on every redirect hop. A redirect from HTTPS to HTTP is refused. | `httpguard.py` |
| TLS | Certificates are verified against the system store; there is no switch to turn that off. | `httpguard.py` |
| Credentials | A token goes to the host it was given for, over HTTPS (a private mirror the person named may use HTTP). `Authorization`, `Cookie`, `Proxy-Authorization` and `X-Api-Key` are dropped on a redirect to another host. No cookie jar exists. | `net/pipeline.py`, `httpguard.py` |
| Limits | Bytes (declared and actual, after decompression, with a ratio cap), redirects, a deadline for the whole transfer and a wait per socket read. A body that stops early is `Truncated`; a body over the cap is `TooLarge`. | `httpguard.py`, `net/download.py` |
| Staging | A download is written to `net/staging` under the state root (mode 0700), never to its final path. A partial file stays there and the next call resumes it (`Range` with `If-Range` when the server gave a validator; without one only when a SHA-256 is pinned). Staged partials older than a week are swept. | `net/download.py` |
| Digest | When a manifest lists a SHA-256 (Hugging Face LFS listings, GitHub release assets) the file must match. `require_digest` refuses a download that has none before any request is made. A digest is not a signature: it shows the bytes are the ones the release lists, not who built them. | `net/download.py` |
| Format | A `.gguf` must have GGUF magic, a known version and sane counts. A `.safetensors` must have a header that parses as JSON, known dtypes and tensor spans inside the data and not overlapping; tensors are never loaded. A PDF must have its header and a trailer, and is flagged for `/JavaScript`, `/Launch`, `/OpenAction` and embedded files. An archive is audited without extracting it: entry count, unpacked size, compression ratio, links and devices, absolute or climbing paths, and archives inside archives. Native-executable magic is refused for any data kind. Pickle suffixes (`.pt`, `.bin`, `.ckpt`, `.pkl`, ...) are refused as weights. A `text/html` answer for a binary kind is refused. | `net/sniff.py` |
| Scan | Every available scanner looks at the file (below). Infected files are never kept. | `net/scan.py`, `net/scanners.py` |
| Promotion | The file is moved to its final path in one step, and `<file>.provenance.json` is written beside it: URL, final URL, redirects, time, SHA-256, size, kind, a subset of response headers, the host, the scan line, warnings. The same row is appended to `net/downloads.jsonl`. | `net/provenance.py` |
| Failure | A file that fails a check is handed to sentinel's quarantine store (kind `artifact`, moved to `.poolhouse-quarantine/`), sentinel raises its event, and the failure is recorded in the index with the reason. | `net/hold.py` |

Nothing downloaded is executed, imported or loaded with `trust_remote_code` by the pipeline.
llama.cpp builds and the Python runtime are unpacked only by the command the person ran to
get them; llama-server is never given a reference to download (`hf:` references are fetched by
the pipeline and the server gets a path).

## Text a model reads from the web

`web.read`, `web.look`, search results and PDF text come back marked `untrusted: true` with an
origin (`web:<host>#<n>`), between fences that say the content is data. The label is the same
one `taint.Label(Level.UNTRUSTED, origin)` takes where the taint ledger is in use.

- HTML: comments, scripts, styles, templates, frames, `hidden` and `aria-hidden` elements, and
  elements styled invisible (display none, zero size or font, transparent, off-screen,
  clipped, white text without its own background) are removed before text is extracted. In a
  rendered page every element a browser computes as unseen is removed, including text the same
  colour as what is behind it.
- Text: tag characters (U+E0000 block), bidirectional controls, zero-width and other invisible
  format characters, control characters, markdown link titles and reference-style comments are
  stripped. A `<<<` in the text is defanged so it cannot close the fence.
- PDF: text in invisible render mode, white or under 2 points, off the page, at zero opacity or
  in a layer that is switched off is left out.
- Size: pages are capped at 8 MiB on the wire and cut to the caller's limit.
- A fetched document never fetches by itself. A URL is fetched when a person typed it, a search
  returned it, or it is the "next page" link of a page already read on the same host. A URL
  that only fetched content mentioned is fetched only when its host is allow-listed or
  approved. An agent loop calls `net.untrusted.shared().typed(text)` with what the person
  typed (`graph.conversation` does).
- The browser a model drives (`web.look`, rendered reads) runs with every request checked and
  downloads off.

## Every path

Paths are those found by searching `src/poolhouse` for `urlopen`, `urllib`, `http.client`,
`huggingface_hub`, `socket`, `subprocess` against `git`/`curl`/`pip`, `playwright`, `requests`
and URL literals. `tests/test_net_no_bypass.py` fails when a module outside `poolhouse.net`
reaches the internet on its own.

| # | Path | Module | Before | Now | What is left |
|---|---|---|---|---|---|
| 1 | Model download | `hub/transfer.py` | urllib, SSRF check by name, token dropped by hand on a CDN redirect, size and SHA-256 compared after the fact, resume in the destination folder | pipeline: staging, pinned digest from the LFS listing, GGUF/safetensors check, host policy, resume | AV cannot judge weights; a poisoned but well-formed model passes |
| 2 | Hub listing, search, README | `hub/remote.py`, `hub/listing.py`, `hub/cards.py`, `hub/drafts.py` | `huggingface_hub` (its own transport) and urllib | pipeline JSON and text; READMEs are cleaned of invisible characters | a README is the publisher's text and is read as such |
| 3 | `hub.fetch`, `hub.snapshot` | `hub/listing.py`, `hub/transfer.py`, `spec/engine.py` | `hf_hub_download`, `snapshot_download` | `hub.pull` into the Hugging Face hub cache (blobs named by digest, snapshot links, `refs`); a snapshot skips pickle weights | |
| 4 | Cached snapshot lookup | `serve/mlx_tree.py` | `snapshot_download(local_files_only=True)` | reads the folder, no network | |
| 5 | llama-server's own download | `serve/backend.py` | `--hf-repo`, `-hfd`, `--mmproj-url` made the server fetch | the server is handed paths only; `HF_TOKEN` is no longer put in its environment | |
| 6 | llama.cpp release download | `serve/build_release.py`, `fleet/llama.py`, `fleet/updates.py` | urllib stream, digest from GitHub, no staging, no scan | pipeline, `require_digest`, archive audit, scan | an archive no scanner could look at is refused until ClamAV is installed or `scan-policy archive warn` is set |
| 7 | llama.cpp source | `serve/build_source.py`, `serve/llamacpp_compile.py`, `serve/llamacpp_upstream.py`, `gguf/tools.py` | `git clone`/`fetch` by subprocess | `net.git`: policy, https only, no hooks, no submodules, `fsckObjects`, commit recorded | the cloned source is compiled or run (`convert_hf_to_gguf.py`) by the command that asked for it; `llama-cpp update` compiles it in the sandbox with no network, fixed cmake arguments and `.git` removed, and trusts the binary only after a smoke test |
| 8 | Self update | `fleet/updates.py` | GitHub API by urllib; `git ls-remote/fetch/pull`; `pip install -e .` | API and asset through the pipeline; git through `net.git`; the remote's host must be allowed | following a branch runs the new code's `pip install -e .`; only a person turns it on |
| 9 | Python runtime | `fleet/environment.py` | urllib stream, tar extracted with `filter="data"` | pipeline, digest required, archive audit, scan | pip installs below |
| 10 | pip | `fleet/environment.py`, `fleet/updates.py` | pip's own TLS to the index | unchanged | pip is not routed and installs unpinned dependencies |
| 11 | Web search | `web.py` | `ddgs` library (its own HTTP); SearXNG by urllib | SearXNG through the pipeline; titles and snippets cleaned | `ddgs` makes its own requests; its answers are untrusted text |
| 12 | Page reading | `web.py`, `scrape/polite.py` | urllib with a resolve-then-connect check, trafilatura's own fetcher | pipeline (address pinned, every hop checked), hidden-content removal, fence, label | |
| 13 | Rendered pages, screenshots | `web.py`, `scrape/browser.py` | Playwright navigates anywhere the page links | requests checked, downloads off | the browser resolves names itself, so a name that changes answer between the check and the load is not caught |
| 14 | PDF and file download | `scrape/download.py`, `web.py` | stream, type and first-bytes check | pipeline: staging, format, scan, provenance, quarantine | |
| 15 | Document download for ingest | `media/download.py`, `ingest/run.py` | urllib stream | pipeline | |
| 16 | Geocoding | `geo.py` | urllib | pipeline | place names leave the machine |
| 17 | Hub and release lookups | `fleet/catalogue.py`, `fleet/weights.py`, `fleet/llama.py` | urllib | pipeline | |
| 18 | Hash reputation lookup | `net/scanners.py` | none | off by default; sends only a SHA-256 to a VirusTotal-style service with a key from the credentials module; the host needs approval | |
| 19 | Installers | `packaging/install.sh`, `install.ps1` | `curl` and the release digest | unchanged (they run before Poolhouse exists) | |
| 20 | Benchmark datasets | `bench/standard.py` | `lm-eval` downloads through `datasets` | unchanged | the harness fetches datasets itself |
| 21 | Fleet peers | `fleet/*` | signed, TLS-pinned LAN traffic | unchanged | not the internet |
| 22 | Model servers and chat endpoints | `client/*`, `claude.py`, `bench/*` | urllib to the endpoint the person configured | unchanged | a remote provider's answers are model output |
| 23 | Metrics address | `fleet/ui.py` | a page a signed-in person typed | unchanged | any address, behind a session |
| 24 | Decider checkpoint files | `decide/fetch.py` | `hf_hub_download` (its own transport, hash checked after the file was kept) | pipeline: pinned size and SHA-256 required, laid out as the repository is, held when they differ | |
| 25 | Injection classifier model | `guard/classifier.py` | `snapshot_download` | pipeline, with the Hub's own size and SHA-256 per file | the Hub's hash is as good as the Hub |
| 26 | Onboarding: pairing, manifest and file transfer | `fleet/onboard/pairing.py`, `fleet/onboard/transfer.py` | own connections to a peer | unchanged transport (the peer's certificate is pinned), but a public address is refused at every connection (`fleet/onboard/lan.py`) | a peer on a tailnet or VPN counts as this network |
| 27 | Decide backend | `decide/logprob.py`, `decide/router.py` | `poolhouse.http` to `POOLHOUSE_DECIDE_URL` | loopback only unless the host is named in `POOLHOUSE_FETCH_ALLOW_HOSTS` | a decider sees every call it judges, so a remote one is an explicit choice |

## Scanning

`Scanner` is a protocol (`available()`, `scan(path)`) with these backends. A file is scanned by
every available one; a result is `clean`, `infected`, `error` or `no scanner available`, and
the last is never reported as clean.

| Backend | How | Verified here |
|---|---|---|
| ClamAV | `clamdscan --no-summary --fdpass` when its daemon answers, else `clamscan --no-summary`. Exit 0 is clean, 1 a signature matched, 2 an error. Files over 2000 MB are an error (the scanner's limit). A database older than a week is a warning. | macOS, ClamAV 1.5.4 installed with `brew install clamav`: the EICAR string is found |
| Windows Defender | `MpCmdRun.exe -Scan -ScanType 3 -File <path> -DisableRemediation`. Exit 0 is clean, 2 a threat. | command line and exit codes against a stand-in; not run on Windows |
| macOS | says `no malware scanner on macOS; install ClamAV (brew install clamav) for scanning`, adds the `com.apple.quarantine` flag and, for executables, what `spctl --assess` says, labelled as Gatekeeper and not a malware scan | macOS |
| Hash reputation | the `HashLookup` class takes a key (`VIRUSTOTAL_API_KEY` in the credentials module). It is not in `default_scanners()` and no environment variable or command turns it on yet. Only the SHA-256 is sent. An unknown hash is not clean. | against a local server |

Whether a file nobody could scan is kept is a policy per category, `poolhouse-security
scan-policy`: executables and archives are refused, data files and model weights are kept with
a warning. Model weights are not sent to a virus scanner by default (it cannot judge them and a
scan of a multi-gigabyte file is slow); `scan-policy scan_models on` sends them.

**What scanning cannot catch.** A well-formed model whose weights were poisoned or whose
behaviour was changed. Malware no signature knows yet. Scripts and documents that abuse
legitimate tools (living off the land). A page that is accurate and hostile to a model. Weights
are accepted as safetensors or GGUF only; pickle-based formats are refused.

## Commands

`poolhouse-security downloads` lists recent downloads with their provenance, scan status and
whether the file is still there and unchanged (`--verify` re-hashes). `hosts` lists the
allow-list and the approvals; `approve-host HOST` records one (a person at a terminal who types
the host back; refused when an agent started the process). `scanners` says which backends this
machine has. `scan-policy` shows or sets the unscanned-file policy.

State: `net/approvals.jsonl`, `net/downloads.jsonl`, `net/scan-policy.json`, `net/staging/`
under the state root. `POOLHOUSE_NET_ALLOW_HOSTS`, `POOLHOUSE_NET_UNSCANNED`
(`archive:warn,model:allow`) and `POOLHOUSE_NET_SCAN_MODELS` set the
same things for one run.
