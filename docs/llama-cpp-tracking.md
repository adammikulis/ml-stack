# Following upstream llama.cpp

Status: accepted 2026-10-03. Owner requirement: llama.cpp is bleeding edge, so a new architecture
runs the day it merges. Homebrew's formula follows the stable `v0.x` tags (0.5.0, build 11146) and
trails master by days or weeks; the newest `bNNNN` tag is cut from master roughly hourly.

`ml-stack-serve llama-cpp ACTION` follows upstream with the checks below. `ml-stack-serve build`
(the older flow in [serving.md](serving.md)) stays for forks, release downloads and `--adopt`.

| Action | Does |
|---|---|
| `status [--json] [--offline]` | The active binary, its origin (env override, managed build, homebrew, PATH), version, build, commit; the newest build tag, the newest stable release and master upstream; `newer upstream commit: yes / no / unknown (offline)`. One check: three small GitHub API reads through `ml_stack.net`. |
| `update [--ref master\|TAG\|SHA] [--force] [--jobs N] [--model GGUF]` | Resolves the ref to a full commit, fetches that commit, builds it, smoke-tests it, pins it and makes it active. Default ref is master. |
| `rollback` | Back to the most recent earlier build that is still installed and still matches its pin. |
| `pin BUILD` / `pin --off` | Makes BUILD active and stops tracking: `update` refuses until `pin --off`. |
| `list` | The builds under `~/.ml-stack/llama.cpp/builds/`, which is active, previous or pinned, and the kept failures. |
| `prune [--keep N] [--yes]` | Lists the good tracked builds beyond the newest 3 (never the active, previous or pinned one, never `failed/`) and deletes them only after you type `yes`. Nothing prunes on its own. |

Exit codes: 0 done, 1 the new build failed its smoke test (the active build is unchanged), 2 an
error (missing tool, unknown ref, pinned), 3 the source host needs approval.

## Where things live

`~/.ml-stack/llama.cpp/builds/b<number>-<sha9>/` holds `llama-server` (one static binary, no
shared libraries) and `BUILD.json` (commit, build number, ref, `sha256`, the smoke result,
toolchain). `current` points at the active build; `track.json` records the previous builds and
the pin; `failed/<build>-<time>/` keeps a build that failed its smoke test, with the result in its
`BUILD.json`. The build number is the tag's when the commit is tagged `bNNNN`, otherwise the newest
tag's shifted by GitHub's commit count; `dev-<sha9>` when upstream cannot say.

## Decisions

**Source comes through `ml_stack.net` and git verifies it.** The ref is resolved to a full commit
by `api.github.com` (`commits/<ref>`). The commit is fetched with `net.git.run`: the host policy
applies (github.com is on the default allow-list; any other host ends in the needs-approval state
with the exact `ml-stack-security approve-host HOST` line), https only, hooks off
(`core.hooksPath` is an empty directory, so a template or global hook does not run),
`fetch.fsckObjects`, no submodules, `--depth 1`. `git rev-parse HEAD` must equal the commit that
was asked for; then `.git` is deleted, so the build has no repository to consult. Nothing in the
tree runs except cmake with fixed arguments (`llamacpp_compile.FIXED_FLAGS` plus the detected
backend and compilers). No `pip install`, no scripts.

**The compile runs in the sandbox.** Seatbelt can express a compiler toolchain on macOS once the
tools are named by their real paths instead of the `/usr/bin` shims: the shims ask `xcode-select`
and `xcrun`, which read `/var/db` and per-user caches the policy does not grant. The build resolves
`clang`, `clang++`, `make` and the SDK path with `xcrun` before the sandbox starts, passes them to
cmake as `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, `CMAKE_MAKE_PROGRAM` (ninja when present) and
`CMAKE_OSX_SYSROOT`, and the policy then reads only the source tree, the toolchain trees and
`/usr`, writes only one work directory, has no network, an empty environment and a wall clock. The
tests prove both limits by a CMake project that tries to write outside and to fetch a URL.
Linux uses the bubblewrap backend of `ml_stack.sandbox`; its argv is tested but it has not run on
Linux. Windows is refused with a pointer to `build --from release`.

**Backend is auto-detected and nothing is installed.** Metal on macOS, CUDA when `nvcc` is on PATH,
Vulkan when the SDK is, CPU otherwise (`build_platform.cmake_flags`). A missing cmake, git,
compiler or make ends with a message listing each missing tool and "ml-stack installs nothing".
The curl dependency is off (`-DLLAMA_CURL=OFF`): the server is given local files.

**A build is trusted only after a smoke test.** `llamacpp_smoke.run` starts the staged binary
through a `ServerManager` lease on the smallest local GGUF (`$ML_STACK_SMOKE_GGUF` names another),
and checks `/health`, a chat completion, `top_logprobs` and a slot save and restore. The estimator's
recorded-log sanity check is not part of it. Only a pass promotes the directory into `builds/`,
pins its sha256 in sentinel (`kind=binary`, `source=build`) and moves `current`. A failure leaves
the old build active and keeps the new one under `failed/`.

## Binary discovery

`find_binary("llama-server")` returns the first of:

1. an explicit path argument, then `$LLAMA_CPP_SERVER`, then `$LLAMA_CPP_DIR`;
2. a named build (`build=` or `$MLSTACK_LLAMA_BUILD`, see serving.md);
3. the managed active build, `~/.ml-stack/llama.cpp/current`;
4. a vendor directory and the cache directory;
5. Homebrew's and the login shell's directories, then PATH.

The managed active build is used only when its pin verifies (`llamacpp_trust.problem`): a pinned
binary must hash to its pin; a build `update` made and whose pin was lost must hash to the
`sha256` in its `BUILD.json`; any other managed binary is pinned on first use. A mismatch
quarantines the file like a model (sentinel moves it aside) and `find_binary` raises
`BinaryTampered` instead of falling through to Homebrew, for that call and the ones after it.
`llama-cpp rollback` switches to the previous build. An environment override is the owner's
explicit choice and is not checked.
