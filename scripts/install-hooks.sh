#!/bin/sh
# Install the git hooks from scripts/hooks/ — the one place they live.
#
# A machine that wants the hooks to read a local database writes its own untracked
# wrapper in .git/hooks/ that exports NAMES_GRAPH / NAMES_SCRAPE and execs the script
# here; this installer leaves any hook it did not put there alone.
#
# pre-push refuses a push unless ML_STACK_PUSH=yes is set for that command.
#
# pre-commit runs no-real-names then budgets; budgets refuses a staged file that adds a
# site to any metric in budgets.json (SKIP_BUDGETS=1 to override).
#
# scripts/hooks/claude-bash-guard is not a git hook and is not installed here. It is a
# Claude Code PreToolUse hook on Bash; a project wires it into its .claude/settings.json:
#   {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
#     "command": "$CLAUDE_PROJECT_DIR/scripts/hooks/claude-bash-guard"}]}]}}
#
# scripts/hooks/claude-edit-guard is the same kind of thing on the writing tools:
#   {"hooks": {"PreToolUse": [{"matcher": "Write|Edit|MultiEdit", "hooks": [{"type": "command",
#     "command": "$CLAUDE_PROJECT_DIR/scripts/hooks/claude-edit-guard"}]}]}}
set -e
cd "$(git rev-parse --show-toplevel)"
hooks="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$hooks"
for pair in "pre-commit pre-commit" "commit-msg commit-msg" "pre-push pre-push"; do
    hook=${pair% *}
    script=${pair#* }
    dst="$hooks/$hook"
    ours=no
    case "$(readlink "$dst" 2>/dev/null)" in
        ../../scripts/hooks/*) ours=yes ;;
    esac
    if [ -e "$dst" ] && [ "$ours" = no ]; then
        echo "install-hooks: $dst exists and is not ours; leaving it alone" >&2
        continue
    fi
    ln -sf "../../scripts/hooks/$script" "$dst"
    echo "install-hooks: $hook -> scripts/hooks/$script"
done
