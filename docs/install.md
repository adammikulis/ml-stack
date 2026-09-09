# Installing

One script per platform, four modes. Re-running any of them upgrades in place.

**macOS and Linux:**

```
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh -s -- --headless
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sh -s -- --dev
curl -fsSL https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.sh | sudo sh -s -- --system
```

**Windows**, in PowerShell. `iex` runs a piped script with no arguments, so the mode is an
environment variable rather than a scriptblock incantation:

```
irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
$env:ML_STACK_MODE="headless"; irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
$env:ML_STACK_MODE="dev";      irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex
$env:ML_STACK_MODE="system";   irm https://raw.githubusercontent.com/adammikulis/ml-stack/main/packaging/install.ps1 | iex   # as administrator
```

| | what it installs | what it downloads | how it updates itself |
|---|---|---|---|
| **the app** (default) | the release zip for this machine, and a window | nothing, until you click: the first-run screen shows the models that fit, gemma-4-E2B suggested (2.6G, ~1.5s a question) beside E4B (4.4G, ~3s) and Flash-Next (104G, ~27s), the ones too big for the room greyed out | the newest published release, replacing the whole install — daemon, CLI and window — then restarting |
| `--headless` | a venv under `~/.ml-stack` (Windows: `%LOCALAPPDATA%\ml-stack`), console scripts on PATH, no window | `--models auto`: the best measured model this machine has room for, unless `ML_STACK_MODELS=none` | releases, the same way — or `main`, if you installed from `main` |
| `--dev` | a git checkout with `pip install -e .` | the same as headless | follows `main`: pulls, reinstalls if the packaging moved, restarts |
| `--system` | `--headless`, plus a service that starts at boot with nobody logged in. Needs `sudo` / an administrator | the same as headless | the same as headless |

**Per user or per machine.** The first three need no administrator and the daemon runs
while you are logged in; the fourth is per machine:

| | per user (app, headless, dev) | per machine (`--system`) |
|---|---|---|
| rights | none | `sudo`, or PowerShell as administrator |
| runs | while you are logged in | at boot, before anyone logs in |
| as | you | still you — see below |
| the Windows firewall | one approval prompt, once | the same, in the same step |
| the macOS wired limit | `sudo` once, offered by the app's first run and by `ml-stack-setup` | applied in the same step, since it already has the rights |
| the model cache | yours, `~/.cache/huggingface` | **the same one.** One cache per machine, shared, never duplicated |

That last row is why `--system` installs the service to run **as the account that ran it**
(a LaunchDaemon with `UserName`, a systemd unit with `User=`, a Scheduled Task with `/RU`)
rather than as root or SYSTEM. A service under another account would have its own empty
`~/.cache/huggingface` and download every model a second time; running as you, it opens the
one that is already there, in place, and nothing is copied, linked or fetched twice. If you
do point it at another account, the installer lists the models you have with their sizes and
says they would be downloaded again; `--adopt-cache` *moves* the cache to the shared path and
leaves a symlink behind, so your own tools keep working and every file still exists once.
Declining leaves your cache alone. It never copies.

Everything ml-stack keeps for itself is under `~/.ml-stack`: the measured records, the runs
store, the llama.cpp builds it made, the record of which model servers are running, and how
much of this machine it may take. `ML_STACK_HOME` moves the lot somewhere else -- a second
disk, a shared volume -- and each of `MLSTACK_BENCH_HOME`, `MLSTACK_INGEST_HOME`,
`MLSTACK_JOBS_HOME`, `MLSTACK_TRAIN_HOME`, `MLSTACK_FIT_FILE`, `MLSTACK_PROFILES_FILE` and
`MLSTACK_LIMITS_FILE` moves one corner of it on its own. What can be fetched or rebuilt --
downloads, one directory per long command's logs -- lives under `~/.cache/ml_stack`, which
`ML_STACK_CACHE` moves and which is safe to delete.

Unattended, for a machine you are setting up from a script: `ML_STACK_NAME`,
`ML_STACK_PASSPHRASE`, `ML_STACK_CLUSTER`, `ML_STACK_MODE`, `ML_STACK_MODELS`,
`ML_STACK_ADOPT_CACHE`, `ML_STACK_REF` answer every prompt, and a machine with no terminal
is never prompted at all. `ML_STACK_OFFLINE_ZIP` and `ML_STACK_OFFLINE_MODELS` install from
local files and skip every network step. `--uninstall` takes it off and leaves the model
cache where it is.

Past the install, every step is an ml-stack command rather than shell -- `ml-stack-setup`
(what this machine can do), `ml-stack-serve build` (llama.cpp), `ml-stack-models fetch`
(into the one cache, every download checked against its sha256), `ml-stack-fleet join
--persist`, and `ml-stack-doctor` at the end, whose lines it prints.

**Or download it yourself** from the [latest release](https://github.com/adammikulis/ml-stack/releases/latest):

| | |
|---|---|
| macOS | `ml-stack-macos-arm64-<version>.zip` — Apple silicon (M1 or later) |
| Windows | `ml-stack-windows-x86_64-<version>.zip` |
| Linux | `ml-stack-linux-x86_64-<version>.zip` |

Each download holds the app and `ml-stack-headless`, for a machine with no screen — the
same daemon, serving the interface to a browser on your network.

![Setting up a machine](images/setup.jpg)

Do the same on every machine you want to train with, typing the same passphrase. They
find each other on their own.

**On Windows** the same daemon runs, with the handful of things Windows does differently
decided in one place (`ml_stack/platform.py`): a job gets its own process group and is
stopped with a `CTRL_BREAK_EVENT` it can catch as `SIGBREAK`, the one-runner lock is
`msvcrt.locking` where POSIX has `flock`, the cluster key is made private with `icacls`
where `chmod 600` would only flip the read-only bit, and `ml-stack-traind --persist`
registers a Scheduled Task at logon (`com.ml-stack.traind.login`) the way `ml-stack-serve
build --persist` registers its weekly refresh. `--system` registers a second one
(`com.ml-stack.traind.system`) at `ONSTART` instead, so the machine is a peer before anyone
logs in -- with `/RU <you>` rather than `/RU SYSTEM`, because SYSTEM has its own profile and
would download every model again into an empty cache. It is a Scheduled Task rather than a
real service because a service needs a wrapper to hold a long-lived Python process, while a
startup task is one line and survives a reboot either way. Two things a Windows machine needs that the
others do not: llama.cpp comes from `ml-stack-serve build --from release` (a release zip,
since most Windows installs have no compiler), and **Windows Defender Firewall blocks the
daemon's TCP 8770 and its UDP 8771 beacons inbound by default**, so `ml-stack-peers ls` on
another machine sees nothing until, in a prompt opened as administrator:

```
netsh advfirewall firewall add rule name="ml-stack traind" dir=in action=allow protocol=TCP localport=8770 && netsh advfirewall firewall add rule name="ml-stack discovery" dir=in action=allow protocol=UDP localport=8771
```

`ml-stack-setup` prints that line as a finding until both rules exist. The first time on a
new Windows machine, in this order, each of which should say what follows it:
`pip install -e .` (an editable install, so `ml-stack-traind` is on PATH);
`ml-stack-setup` (the firewall finding, `!` until the rules are added, then `ok`);
`ml-stack-doctor` (the checkout and hooks);
`ml-stack-serve build --from release` ("current -> ...\builds\bNNNN" after "verifying");
`ml-stack-traind --persist` ("installed to start at login", then a `traind.log` under
`~\.ml-stack` that begins `ml-stack traind on http://0.0.0.0:8770`);
`ml-stack-peers ls` from another machine (the Windows box listed with its GPU). Everything
Windows-specific here was written against a faked `platform.system()` on a Mac -- the
Windows calls themselves run for the first time when that list does.

**If you write Python**:

```
pip install ml-stack            # all of it, and nothing else. No dependencies.
pip install ml-stack[train]     # and numpy and safetensors, to train
pip install ml-stack[all]       # and everything the rest of it can use
```

Building from source needs `pip install build`, then:

```
python packaging/build.py            # wheels into dist/
python packaging/build.py --bundle   # and the app for this platform (needs Rust, node)
```

Everything is adjustable later:

![Settings](images/settings.jpg)

