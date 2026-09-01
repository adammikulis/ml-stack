#!/bin/sh
# Install the git hooks from scripts/hooks/ — the one place they live.
#
# A machine that wants the hooks to read a local database writes its own untracked
# wrapper in .git/hooks/ that exports NAMES_GRAPH / NAMES_SCRAPE and execs the script
# here; this installer leaves any hook it did not put there alone.
#
# scripts/hooks/claude-bash-guard is not a git hook and is not installed here. It is a
# Claude Code PreToolUse hook on Bash; a project wires it into its .claude/settings.json:
#   {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
#     "command": "$HOME/Documents/repos/ml-stack/scripts/hooks/claude-bash-guard"}]}]}}
set -e
cd "$(git rev-parse --show-toplevel)"
hooks="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$hooks"
for pair in "pre-commit no-real-names" "commit-msg commit-msg"; do
    hook=${pair% *}
    script=${pair#* }
    dst="$hooks/$hook"
    if [ -e "$dst" ] && [ "$(readlink "$dst" 2>/dev/null)" != "../../scripts/hooks/$script" ]; then
        echo "install-hooks: $dst exists and is not ours; leaving it alone" >&2
        continue
    fi
    ln -sf "../../scripts/hooks/$script" "$dst"
    echo "install-hooks: $hook -> scripts/hooks/$script"
done
