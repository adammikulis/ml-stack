> **READ THIS FIRST (2026-10-10).** The project was renamed from ml-stack to Poolhouse and the Mac's pool node was
> cut over to the new build. A device still on the old code cannot talk to it (the signing labels changed), so the
> two devices are NOT syncing right now, and that is the one thing that matters. On this device, in order:
>
> 1. `git remote set-url origin https://github.com/adammikulis/poolhouse.git`, then `git pull --ff-only` on `0.3dev`.
> 2. Run `scripts/cutover --plan`, then `scripts/cutover --yes` (the Windows steps use `schtasks.exe` and run
>    `poolhouse device-setup --yes`). If it stops, read the printed fix and run `scripts/cutover --resume`.
> 3. Make sure this device is awake and on the same network as the Mac (a sleeping laptop looks "offline" to the pool).
> 4. Run `python -m poolhouse.node_join_check` and commit its full output as `docs/windows-report.md` on `0.3dev`
>    and push, so the Mac side can read it without anyone typing anything.
>

# Setting up a new device (Windows, WSL, Linux, macOS)

For the agent on the new device. The owner synced the repository here, opened a session and said "connect to the
pool"; he has agreed to everything below, so you run it all and ask him to type nothing. The only thing he does
is press **Yes** on one Windows prompt (step 2), so tell him before it appears.

There is one command, the same on every platform:

```sh
python -m poolhouse.device_setup --yes
```

(from the repository root, with `PYTHONPATH=src` when Poolhouse is not installed; on Windows use `py` or `python`).
`--dry-run` lists the changes and makes none. It does only what this machine needs, so it is safe to rerun:

| Machine | What it may change |
| --- | --- |
| Windows | inbound firewall rules for TCP 7447 and UDP 7448 (one administrator prompt) |
| WSL | the same rules, plus `networkingMode=mirrored` in the Windows user's `.wslconfig` (the old file is kept beside it as `.wslconfig.poolhouse-backup-<time>`), then `wsl --shutdown` |
| every machine | the node: the selected runtime's, a release binary already in the repository, else a cargo build (Rust is installed first with the standard installer when cargo is missing); then it starts the node with its network on and runs the join check |

## Steps

1. **Find out where you are.** Run `python -m poolhouse.device_setup --dry-run` and read the list. A WSL machine
   in the default NAT mode ends its list at "restart WSL".
2. **Run it.** `python -m poolhouse.device_setup --yes`. On Windows or WSL, tell the owner just before: "A Windows
   prompt will appear; press Yes." That is the firewall step. If the prompt is declined it ends NOT READY and says
   to run again.
3. **After a WSL restart** your session ends (`wsl --shutdown` stops WSL). The owner reopens the session; run the
   same command again. It picks up where it stopped, because each step checks whether it is already done.
4. **Read the ending.** `READY` means this device and the others share a pool. `NOT READY at STEP. The one fix:
   ...` names the one change to make; make it (you may run the commands it prints) and run the command again.
   The steps and what a FAIL means:

   | Step | Fails when | Fix |
   | --- | --- | --- |
   | platform | no network, or WSL is on NAT (`172.16-31.x.x`) | connect wifi or cable; mirrored networking (step 2 and 3) |
   | binary | cargo is missing or the build failed | the printed rustup / `cargo build --release -p poolhouse-node` line (in `app/`) |
   | node, listening | the node did not start | read `node.log` in the state directory it names |
   | beacon-send, beacon-receive | a firewall blocks the beacon | step 2 again; a third-party firewall needs the same two rules |
   | peer-beacon | the other device is not beaconing | step 5 on the Mac, now |
   | enrolled | policy differs, or the device already has members | both devices on policy `open`; see "Secure pools" |
   | converge | the board message did not cross | rerun with the other device up |
5. **The Mac side.** The Mac must be on the same wifi or cable (not a guest network, not a VPN) with its node
   running and policy `open`: from the Mac's checkout, `python -m poolhouse.device_setup --yes` (a Mac agent does
   this; it changes nothing on the Mac except starting the node and turning the network on). Run the Windows
   command within a minute or two of it; the join check waits 90 s at each step that needs the other device.
6. **Confirm by exchanging a message.** The join check posts to the board `pool-check` and waits for the other
   device's post; `converge` PASS is that exchange. Then register yourself and say hello to the Mac's session:
   register under your own name as AGENTS.md, "Agent coordination and profiles" says, find the Mac session's
   name with `poolhouse-workspace digest --agent YOUR_NAME`, send it
   `poolhouse-workspace send MAC_SESSION_NAME status "connected from Windows" --agent YOUR_NAME`, and read the
   reply with `poolhouse-workspace inbox --agent YOUR_NAME`.
7. **Run tests on this device.** `scripts/test` on its own runs on the device you are on: `scripts/test quick
   tests/test_x.py` (WSL, Linux, macOS), with the explicit selectors AGENTS.md "Running the tests" asks for. On
   Windows the suite runs inside WSL (docs/windows-runtime.md); use the WSL checkout. Another device's agent can also
   run tests here with `--on` once you turn it on (the last section).

## Secure pools

Under policy `secure` nothing enrols without a pairing code. The Mac makes one and the Windows device uses it. The
code is valid for five minutes; you read it from the Mac's session, never from the owner:

```sh
# on the Mac (its agent runs this)
python -m poolhouse.node_join invite            # prints {"code": "...", "port": 7447, "expires_in_s": 300, ...}
# on this device (the Mac's agent sends you the code and the Mac's address over the board; the code is short-lived)
POOLHOUSE_PAIR_CODE=CODE python -m poolhouse.node_join pair --host MAC_ADDRESS
```

The code is in the environment, not the command line, so it is not left in a process list. Do not write it in a
commit, a file or a report; the pool does not keep it after use.

## What agents can do across devices

Found by reading `docs/workspace.md`, `poolhouse.jobs` and the node's ops:

- **Board messages across the pool** work: `send`, `inbox`, `thread`, `board read` reach the other device's sessions
  once they share a pool. Use them to ask the agent on the other device to do something.
- **`poolhouse-workspace remote-agent --device NAME`** starts a Board-message worker (a local Qwen) on another
  device. Its operations are bounded coordination ones; it does not run tests or commands (docs/workspace.md,
  "Explicit shared coordinator across devices").
- **`poolhouse-jobs`** runs long commands on the device you are on only.

Running tests on another device is `scripts/test TIER --on DEVICE` (docs/test-farm.md). It needs the person to turn
it on at the device that runs the tests, below.

## Letting the Mac run tests on this device

Only on the owner's order (for example "let the Mac test on this device"), because it lets any member of the pool
run a checkout's tests here as this user. The Windows and WSL sides are separate pool devices: each needs its own
node (this document's steps) and its own switch, in its own checkout.

1. Have Python 3.13 on this device (`python --version`; Windows: `py -3.13 --version`; WSL and Linux: `python3.13
   --version`). If it is missing, install it (Windows: `winget install Python.Python.3.13`; WSL: the distribution's
   package or pyenv) and give it the test dependencies the way this device already installs Poolhouse (`AGENTS.md`).
2. Turn the experimental feature on (`poolhouse features enable remote-tests`), then turn it on from the repository
   root, with that Python and `PYTHONPATH=src`:
   `python -m poolhouse.testfarm.consent on --from MAC_NAME` (Windows: `py -3.13 -m poolhouse.testfarm.consent on --from MAC_NAME`, with
   `$env:PYTHONPATH="src"`). It prints `test shards: on (python ..., checkout ...)`. If it says Python 3.13 is
   needed, run it with the 3.13 interpreter, or pass `--python PATH`; if it says the repo has no `shard_exec.py`,
   this checkout is older than the feature: update it, restart the node (`python -m poolhouse.node_launch swap`) and
   run it again. The node must be the new build, since it carries the shard ops.
   `MAC_NAME` is the Mac's name in `poolhouse-test-devices` (or its fingerprint). This device takes tests only from
   the devices it names; to add or remove one later: `python -m poolhouse.testfarm.consent allow NAME` or `deny NAME`.
   Never allow a device you did not mean to: it can run code here as this user. Joining the pool is not enough.
3. Ask the Mac's agent to run `poolhouse-test-devices` and then `scripts/test all tests/test_x.py --on THIS_DEVICE`;
   the result comes back to the Mac's session, and a `test-result` message is on the board.
4. To stop: `python -m poolhouse.testfarm.consent off`. It takes effect on the next upload; a run already going
   finishes or is cancelled from the Mac. Everything on and off is in the pool board's audit entries.

Failures say what to do: `Python 3.13 is missing` (install it and run step 2), `test shards are off on this
device` (step 2), `this device's CPU slots are in use` (wait), `no other active device ... is called` (the name
in `poolhouse-test-devices`).
