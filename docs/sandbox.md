# Sandboxing and egress control for tool execution

Status: accepted 2026-10-02 (issue #27). One package, `ml_stack.sandbox`, runs a command under an
operating-system sandbox with a deny-by-default policy. Every place ml-stack starts an
agent-chosen command or an untrusted tool goes through it.

## Decision

A `Policy` names everything a process may touch. Anything not named is refused.

| Part | What it grants |
|---|---|
| `read` | trees it may read (plus the system trees every process reads: linker, frameworks, zoneinfo) |
| `write` | trees it may read and write |
| `exec` | the programs it may start; the command itself is added |
| `net` | `Net.deny()`, `Net.loopback()`, or `Net.only(*ports)` (loopback, those ports) |
| `env` | the whole environment; nothing is inherited |
| `limits` | wall clock (kills the process group), CPU seconds, file size, open files, output bytes |
| `gpu` | what Metal needs (below) and nothing else |

Every path is validated before it reaches a profile: absolute, existing, no control character,
no `..`, `.`, empty or trailing-slash segment, and no symlink on the way (`name the resolved
path`). The profile writer quotes each path as a string literal (quote and backslash escaped,
control characters refused), so a directory called `x") (allow file-read* (regex #".*"))`
is a directory, not a rule.

Backends sit behind one `Backend` protocol (`available`, `wrap`, `denials`):

| Backend | Where | State |
|---|---|---|
| `seatbelt` | macOS, `sandbox-exec -p <profile> -- argv` | implemented, tested against the real tool |
| `bubblewrap` | Linux, `bwrap --unshare-all --die-with-parent --new-session --clearenv ...` | argv built and tested; never run on Linux |
| `container` | any | stub that raises `NotImplementedError`; plan below |
| none | | `run` raises `SandboxUnavailable` and does not start the command |

Untrusted execution fails closed. Running without a sandbox takes an explicit
`AllowUnsandboxed(reason)` argument: the reason is logged at warning level and sent as a
`sandbox.unsandboxed` event with every run. There is no environment variable or global switch.

## The exception: `sandbox-exec`

The repository bans deprecated tools. `sandbox-exec` (Seatbelt) is deprecated by Apple and is
used here anyway.

- **Date and decision:** 2026-10-02, the owner, recorded on issue #27: on macOS it is the best
  available command-line sandbox.
- **Scope:** every use of the tool is in `src/ml_stack/sandbox/seatbelt.py`. Nothing else names
  it. Claude Code's own Bash sandbox (enabled by `ml_stack.harness`) is Anthropic's code and uses
  the same tool underneath.
- **Mitigations:** it logs one warning per process saying the tool is deprecated;
  `ml-stack security sandbox status` and `ml-stack security status` print it; the binary's
  existence is checked before each use; untrusted execution is refused when it is missing.
- **Replacement path:** a container backend (below) for CPU-only work, behind the same
  `Backend` protocol, so the change is one file.
- **GPU workloads have no non-deprecated replacement.** Containers and virtual machines on macOS
  cannot use Metal. A model server on the GPU can only be confined with Seatbelt today. This stays
  true until Apple ships GPU access for containers; nothing in this repository can change that.

## Evaluated alternatives

Measured on this Mac (macOS 26.6, Apple silicon) on 2026-10-02.

| Option | Version, licence, last release | Enforces | Platforms | Cost | Result here |
|---|---|---|---|---|---|
| Anthropic `sandbox-runtime` (`srt`) | 0.0.78, Apache-2.0, 2026-09-30; repo pushed 2026-10-02 | Seatbelt (macOS) or bubblewrap (Linux) from a JSON settings file; a domain-filtering HTTP/SOCKS proxy for network | macOS, Linux, Windows | Node 26 and 4 npm dependencies (16 MB installed); pre-1.0 | ran: `echo` in 0.25 s; a denied read and a write outside the allow-list were refused; **reads are allowed everywhere unless denied** (a home directory listing worked); **environment inherited**; proxy variables injected; no exec allow-list; no GPU rule |
| Apple `container` | 1.5.0, Apache-2.0, 2026-09-29 | one lightweight Linux VM per container | macOS 26, Apple silicon | 431 MB Homebrew bottle, a launchd service, a kernel and image downloads | installed and `--version` answered; `container system start` was refused by the permission system in this session, so startup latency, mounts and tool behaviour were **not measured**. No Metal in the guest |
| Lima 2.2.0 (Apache-2.0), Colima 0.10.3 (MIT), Podman 6.1.3 (Apache-2.0 and GPL-3.0+) | Homebrew formulae | a full Linux VM and a container runtime | macOS, Linux | a VM image, a daemon | not installed or run; same Metal limit; heavier than Apple `container` for the same result |
| bubblewrap 0.13.0 | LGPL-2.0+ | user, mount, pid, net namespaces | Linux | one setuid-free binary | not runnable on this Mac; the argv is tested only |
| nsjail | Apache-2.0 | namespaces, seccomp, rlimits | Linux | | no Homebrew formula; not run |
| Docker Desktop | installed on this Mac | container VM | | | daemon was not running; not used |

Why not `srt` as the macOS backend: it still calls `sandbox-exec`, so the deprecated tool does
not go away. Its read model (allow everywhere, deny some paths) is the opposite of the contract
asked for here, it passes the environment through, and it cannot express an exec list or the
GPU. It also adds a Node runtime to a Python library. Its useful part is the host-name egress
proxy; a host allow-list is not built here (web tools run inside the net pipeline,
`ml_stack.httpguard`), and an optional `srt` backend for that is the follow-up.

## What is enforced where

macOS (Seatbelt), checked by `tests/test_sandbox_seatbelt.py` against the real `sandbox-exec`:

- reads outside the allow-list fail; writes outside it fail; a symlink inside an allowed
  directory does not lead out;
- a child process inherits the sandbox;
- a program outside the exec list is not started;
- the network is refused unless the policy allows loopback or the named ports;
- the environment is exactly the policy's;
- the wall-clock limit kills the command's process group and no other process; CPU, file-size
  and output limits stop the command;
- a refusal is found in the system log (each run's profile carries a unique tag), becomes an
  `Result.denials` entry, a `sandbox.denied` event and a `SandboxViolation` naming the operation
  and path.

Known limits:

- Metadata (existence, size) is readable only for the allow-listed trees and their ancestor
  directories; contents outside the allow-list cannot be read.
- A process that calls `setsid` leaves the group the runner kills. It stays inside the sandbox.
- macOS does not enforce `RLIMIT_AS` or a useful `RLIMIT_NPROC` (the process limit is per user), so
  there is no memory or process-count ceiling.
- Detecting a refusal reads the system log (about 1.2 s) and only happens after a failed run or
  with `diagnose="always"`.
- Refusals that every confined process provokes at start-up (preference files, a few Mach
  services, `net.routetable`) are filtered out of `Result.denials`.

## GPU

`Policy(gpu=True)` adds two rules to the profile:

```
(allow iokit-open (iokit-user-client-class "IOGPUDeviceUserClient"))
(allow mach-lookup (global-name "com.apple.MTLCompilerService"))
```

Found by running `llama-server -ngl 99` (granite-4.0-h-350m, Q4_K_M) under the profile, starting
from a guessed list of 12 IOKit classes, 4 Mach services, 3 read paths and `iokit-get-properties`,
and removing one at a time while the server still loaded all 33 layers onto the GPU and answered.
Only the two above were needed. The server is up in 3.6 to 4.7 s with them. llama.cpp embeds its
Metal library, so it needs no shader-cache directory; `Policy.cache` names one directory that is
readable and writable for a stack that does (MLX, PyTorch). A model server started from a Homebrew prefix also needs that prefix readable (its libraries are
linked by absolute path) and a working directory it can read. The base profile also allows
`hw.pagesize_compat` (without it the allocator asks for 2^44 MiB and the server crashes).

## Where it is applied

| Call site | Policy | Layer behind it |
|---|---|---|
| `ml_stack.sandbox.tools.SandboxedBash`, the shell tool for an `Agent` | `bash`: read the project, write a scratch directory, no network, 120 s | the guard's tool-policy and taint rails (`bash` is an `exec` sink: text from outside cannot reach it) |
| `McpTools.stdio(...)` | `mcp_server`: read the interpreter, project and `reads`, write a scratch directory, loopback only, `env` as the whole environment | the guard; the server is not started when no sandbox is available |
| `ml_stack.harness` (Claude Agent SDK) | Claude Code's sandbox for Bash: on, no per-command opt-out, `failIfUnavailable`, no network domains | the guard's `PreToolUse` hooks; not driven end to end in this change (see below) |
| `LlamaServerBackend(sandboxed=True)` or `ML_STACK_SANDBOX_SERVE=1` | `model_server`: read the binary's directory and the files in the command line, loopback, GPU | opt-in; the Broker leases the server as before |

Not routed: `llama-server` and the Broker daemon by default (opt-in only), `checks.py`'s
`shell=True` fix line (typed by the person and confirmed), `subprocess` calls in `serve`, `bench`,
`fleet`, `gguf`, `speech` and `doctor` that run ml-stack's own commands with arguments the code
builds. Each is a candidate for review as it takes input from an agent.

Sandbox events go to the sentinel bus through `sentinel.adapters.sandbox_listener`:
`sandbox.denied` (warning), `sandbox.unavailable` (warning), `sandbox.unsandboxed` (warning),
`sandbox.timeout` (notice).

## Replacement path: container backend

Written when Apple's `container` (or a VM runner) is installed and measured:
`Container.wrap` turns the policy into `container run --rm --network none --read-only` with the
read trees mounted read-only, the write trees read-write, an explicit `--env` list and the limits
as `--cpus`/`--memory`, and a Linux build of the tool inside the image. It serves CPU-only work
(shells, MCP servers, converters). It cannot serve GPU work. Until then `Container.wrap` raises
`NotImplementedError` and `available()` is false.

## Linux

`bubblewrap.arguments` builds the argv: `--unshare-all`, `--share-net` only when the policy keeps
loopback, read-only binds for `read`/`exec`, binds for `write`, `--clearenv` with `--setenv` for
each variable. It is covered by argv tests only. Landlock and seccomp are a future option; they
are not written because they could not be verified here.

## Commands

`ml-stack security sandbox status` prints the backend, whether it can run and the deprecation
note. `ml-stack security sandbox test` runs five guarantees against real confined processes and
exits 1 when one fails.

## Sandboxed model server

`LlamaServerBackend(sandboxed=True)` or `ML_STACK_SANDBOX_SERVE=1` starts `llama-server` under
the `model_server` policy: loopback, the GPU, the binary's install tree, the directories of
the files on its command line, and the slot-save directory as the only writable path. A failed
start appends the sandbox's refusals to the `ServerFailed` message. It is opt-in; it stays off by
default until it has run a day of real workloads (speculative heads, mmproj, multi-shard models and
Hugging Face downloads were not tried; a Hugging Face reference needs the network and is not
supported while confined). The measured cost on a 350M model was within the 3.6 to 4.7 s start-up
of the unconfined server (not separately timed).

## Tests and measurements

- `tests/test_sandbox_policy.py` (pure), `tests/test_sandbox_seatbelt.py` (real `sandbox-exec`),
  `tests/test_sandbox_integration.py`, `tests/test_sandbox_redteam.py` and
  `tests/test_sandbox_gpu.py` (slow: a real `llama-server` offloading every layer inside the
  profile, plus a decoy read and a connection out refused). The Seatbelt tests skip cleanly
  where `sandbox-exec` is absent.
- Mutation check of the profile writer, validators, runner and bubblewrap argv: 45 hand-made
  mutations. The first pass left 7 alive (message-masked validation errors, signals, the
  operation parser); tests were tightened and all 45 are now killed.

Native managed-build success tests first run a confined system `true` command. They skip at the
test function when the backend is absent or the host denies namespace creation. Other probe
failures and malformed policies fail the tests. The Linux container runner keeps Docker's
default privileges; it does not enable privileged mode or relax seccomp to make a native
sandbox run. Fail-closed policy and managed-build refusal tests still run when native success
tests cannot.
