# Cutover to Poolhouse

## The one command

On each machine (this Mac, and the Windows device from its WSL or Git-Bash shell), once the rename has
landed on the branch the checkout follows:

```
cd ~/Documents/repos/ml-stack
scripts/cutover --plan     # what it will do and what is running; changes nothing
scripts/cutover            # shows the plan, asks one yes/no, then does it
```

`--yes` skips the question. It is POSIX shell using only git, python3, pip, pgrep and
launchctl/schtasks where present, so it works before the new package is installed. It does, in order, each
step idempotent and recorded in `.git/cutover.state`: 1 `git pull --ff-only`; 2 apply and commit
`docs/rename-protected.patch` (skipped when applied; if it cannot apply it says why and stops); 3 stop the old
build and verify `pgrep -fl 'ml-stack|ml_stack|poolside-node'` is empty (it prints what is left and stops);
4 remove a stray `~/.poolhouse` that holds only `activity` and `sentinel` (anything else is refused); 5 install
the new package and uninstall `ml-stack`; 6 `poolhouse migrate plan`, `run` and `verify`; 7 `poolhouse runtime ensure`,
`node build`, `scripts/land up`, `poolhouse-doctor` (on Windows `poolhouse device-setup --yes`, one UAC prompt).
It ends with `READY`, or with the step that failed, the fix, `scripts/cutover --resume` and the rollback line.
Tests drive it with `--dry-run-in DIR` (a fake HOME and stub commands under DIR; tests/test_cutover.py).
Still by hand afterwards: the login units again and the old `com.ml-stack.*` plists (step 6 below).

The manual steps below are the reference for what the script does.

The order matters: the new code must be installed before anything is restarted, and the old build must
be stopped before `poolhouse migrate run`, which refuses while a process of the old name is alive. Nothing
else moves live state. Run each step as the person who owns the machine. `NAME` below is the old
install's command prefix (`ml-stack`) and the new one (`poolhouse`).

## This Mac

1. **Apply the protected-file patch** (a person at a terminal; the guards refuse an agent):

   ```
   cd ~/Documents/repos/ml-stack
   git apply --check docs/rename-protected.patch && git apply docs/rename-protected.patch
   git add .claude/settings.json scripts/hooks/claude-bash-guard scripts/hooks/claude-edit-guard \
     scripts/hooks/person-consume scripts/hooks/rules_loader.py src/poolhouse/workspace/person_*.py
   git commit -m "chore: the guards, hook settings and person-record code take the Poolhouse names"
   ```

   Do this after the rename has landed on `0.3dev` and the checkout is on it. Until it is applied the
   workspace, chat, codex, harness and MCP commands fail to import and the guards still read the old
   variable names.
2. **Stop the old build**, newest layer first. Use the old commands (the installed `ml-stack-*` ones):

   ```
   scripts/land down                                    # the land runner and its supervisor (releases claim and lock)
   python -m ml_stack.node_launch stop                  # the LAN node and its supervisor
   launchctl bootout gui/$(id -u)/com.ml-stack.traind   # the pool daemon, if it is installed as a login unit
   launchctl bootout gui/$(id -u)/com.ml-stack.runtime-ensure
   launchctl bootout gui/$(id -u)/com.ml-stack.llama-build   # only if present
   ```

   Quit `Poolside.app` / `Poolhouse.app` from its menu. Agents that run from the old interpreter
   (`ml-stack-claude`, `ml-stack-agent`, `ml-stack-chat`) are stopped by their owners. Leave served models alone
   if you can: `migrate` does not touch `~/.ml-stack/models`, but a server whose files lie under `~/.ml-stack`
   counts as running there and blocks the move; stop that one with `ml-stack-serve down MODEL`.
   `pgrep -fl 'ml-stack|ml_stack|poolside-node'` should print nothing.
3. **Land** the rename (the lead's queue: `poolhouse-workspace land-request` once the new package is
   installed, or by hand on the primary checkout with the owner's authority).
4. **Install the new package** into the pyenv 3.13.5 interpreter, then remove the old one:

   ```
   cd ~/Documents/repos/ml-stack
   python -m pip install -e .
   python -m pip uninstall -y ml-stack
   ```

   `poolhouse migrate plan` now exists; the old `ml-stack*` launchers are gone with the old distribution.
5. **Look, then move:**

   ```
   poolhouse migrate plan      # what moves; lists any process still of the old name
   poolhouse migrate run       # exit 0 done, 1 old build alive, 2 a step failed, 3 both names exist
   poolhouse migrate verify    # exit 1 lists any dangling symlink or old-path venv file under the new directories
   ```

   `run` also repoints symlinks and rewrites venv scripts (`pyvenv.cfg`, `activate*`, shebangs, `*.pth`) that
   still named the old directories (originals kept under `~/.poolhouse/migrate-backup`, every change in
   `migrate.log`) and lists any other file that names an old path without changing it. A second run is a no-op.

   `run` renames `~/.ml-stack` to `~/.poolhouse`, `~/.cache/ml_stack` to `~/.cache/poolhouse`, moves the
   Keychain master key to the service `poolhouse` (macOS may ask for the login keychain password once) and
   renames `.ml-stack-project.json` in each connected checkout. It writes `~/.poolhouse/migrate.log`.
   Exit 3 means `~/.poolhouse` already exists: merge or remove one as the message says and run again.
6. **Restart, in this order:**

   ```
   poolhouse runtime ensure --checkout ~/Documents/repos/ml-stack   # builds and selects the runtime under the new name
   poolhouse node build                                             # Poolhouse.app (bundle id app.poolhouse.node, unchanged)
   poolhouse node permission                                        # only if macOS asks for Local Network again
   poolhouse-peers ls                                               # the node starts on the LAN and the pool is listed
   scripts/land up                                                  # the land runner under the new name
   poolhouse-doctor
   ```

   Install the login units again so launchd carries the new labels:
   `python -m poolhouse.fleet.autostart prepare --role pool-daemon --role runtime-ensure`, then
   `install --manifest <the path prepare printed>` (docs/install.md). Remove the old plists in
   `~/Library/LaunchAgents` named `com.ml-stack.*`.
7. **The desktop app**: its identifier is now `app.poolhouse.app`, so macOS keeps its data and Keychain
   access list apart from the old app; sign-ins are repeated once. The node's `app.poolhouse.node` is
   unchanged and keeps its Local Network grant.
8. Hooks and clients: regenerate `~/.claude/settings.json` and `~/.codex/config.toml` entries
   (`poolhouse-workspace` replaces `ml-stack-workspace`; `python scripts/install-hooks.py`), and set
   `POOLHOUSE_WINDOW_POSITION` where `ML_STACK_WINDOW_POSITION` was (the patch does it in the project settings).

## The Windows device (its agent runs these; each step needs the person's approval where it says so)

1. Stop the old build: `scripts\land down` is not needed there; run
   `python -m ml_stack.node_launch stop`, then
   `schtasks /End /TN com.ml-stack.traind.login` (and `com.ml-stack.traind.system` if the machine-wide task
   exists), then `schtasks /Delete /TN com.ml-stack.traind.login /F`. Close any `ml-stack-*` console.
2. Pull the landed `0.3dev`, then `python -m pip install -e .` and `python -m pip uninstall -y ml-stack`.
3. `poolhouse migrate plan`, then `poolhouse migrate run`. On Windows a directory in use cannot be renamed:
   if `run` stops on a permission error, the message names the holder; close it and run again.
4. `poolhouse device-setup --yes` re-creates the firewall rules and the node. **UAC:** adding or changing a
   firewall rule is one elevation prompt, approved by the person at the machine, the same as the first
   install; the old rules named `ml-stack traind` and `ml-stack discovery` are left behind, so delete them in
   the same prompt (`netsh advfirewall firewall delete rule name="ml-stack traind"`, likewise `discovery`)
   and let the new ones (`poolhouse traind`, `poolhouse discovery`) take their ports.
5. Register the login task again: `python -m poolhouse.fleet.autostart prepare --role pool-daemon`, then
   `install --manifest <path>` (one more elevation for a machine-wide task). The named pipe and stop event
   now carry `poolhouse-node-...`; a node started before the move cannot be stopped by the new code, which is
   why step 1 stops it first.
6. `poolhouse-peers ls` on this Mac and on the device should list each other once both nodes run.

## Rolling back

`~/.poolhouse/migrate.log` lists every move, one line each (`state: A to B: renamed`, `cache: ...`,
`moved the master key`, `project: PATH renamed`). To undo, stop the new build (`python -m poolhouse.node_launch stop`,
`scripts/land down`, the daemon unit) and reverse the lines:

```
mv ~/.poolhouse ~/.ml-stack
mv ~/.cache/poolhouse ~/.cache/ml_stack
# Keychain: copy the master key back to the old service, then delete the new item
KEY=$(security find-generic-password -s poolhouse -a "master/$(id -u):$(id -un)" -w)
security add-generic-password -U -s ml-stack -a "master/$(id -u):$(id -un)" -w "$KEY"
security delete-generic-password -s poolhouse -a "master/$(id -u):$(id -un)"
# each checkout named by a "project:" line
mv CHECKOUT/.poolhouse-project.json CHECKOUT/.ml-stack-project.json
```

Then reinstall the old package from the commit before the rename (`git checkout <that commit>`, `python -m
pip install -e .`), remove the patch commit, and start the old node, daemon and runner. A move that was
cut off on another volume leaves `~/.poolhouse.migrating` (a half copy, safe to delete) and the intact
`~/.ml-stack`; the next `migrate run` finishes or refuses, it never deletes an unverified source.
