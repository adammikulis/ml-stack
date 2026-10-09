"""The sandbox settings for each agent harness, derived from one description of the machine's layout."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

BOARD = "127.0.0.1:8770"

HOSTS = (
    "huggingface.co", "*.huggingface.co", "hf.co", "*.hf.co",
    "github.com", "api.github.com", "codeload.github.com", "objects.githubusercontent.com",
    "release-assets.githubusercontent.com", "raw.githubusercontent.com",
    "pypi.org", "files.pythonhosted.org", BOARD,
)
RUNTIME_HOSTS = ("api.anthropic.com",)

SCRUBBED_ENV = (
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "POOLHOUSE_HOME", "POOLHOUSE_CACHE", "POOLHOUSE_AGENT", "POOLHOUSE_WORKSPACE_AGENT",
    "POOLHOUSE_NET_ALLOW_HOSTS", "HF_ENDPOINT", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "GH_TOKEN", "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "SSH_AUTH_SOCK", "PYTHONPATH", "PYTHONSTARTUP",
    "PYTHON_KEYRING_BACKEND", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_COUNT",
    "GIT_SSH_COMMAND", "GIT_ASKPASS", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "BASH_ENV", "ENV",
)

STATE_DENY_READ = (
    "workspace/tokens", "workspace/local-agents", "workspace/device-accounts.db",
    "workspace/device-accounts.db.shadow", "workspace/invites.json", "workspace/shared-invites.json",
    "workspace-remote", "keystore", "server-keys.json", "cluster.key.old", "cluster.json",
    "credentials.endpoint", "credentials.toml.bak", "onboard",
)
STATE_ALLOW_READ = ("workspace/tokens/claude-code",)
STATE_ALLOW_WRITE = (
    "workspace", "logs", "activity", "requests", "holds", "hook-diagnostics",
    "broker.lock", "broker-leases.json", "servers.json", "servers.lock", "servers.admission.lock",
)
STATE_DENY_WRITE = (
    "workspace/tokens", "workspace/local-agents", "workspace/device-accounts.db",
    "workspace/device-accounts.db.shadow", "workspace/invites.json", "workspace/shared-invites.json",
    "keystore", "runtimes", "hooks", "gate", "guard", "machine-id", "wired-limit.json",
)
HOME_DENY_READ = (".ssh", ".aws", ".gnupg", ".config/gh", ".netrc", ".git-credentials", ".docker/config.json",
                  ".npmrc", ".pypirc")
HOME_DENY_WRITE = (
    ".claude", ".codex", ".ssh", ".aws", ".gnupg", ".config/gh", ".gitconfig", ".config/git",
    ".zshrc", ".zprofile", ".zshenv", ".bashrc", ".bash_profile", ".profile", ".config/fish",
    ".pyenv/versions", ".pyenv/shims", ".local/bin", ".config/systemd", ".config/autostart",
)
DARWIN_DENY_READ = ("Library/Application Support/Google/Chrome", "Library/Application Support/Firefox",
                    "Library/Safari", "Library/Keychains", "Library/Cookies")
DARWIN_DENY_WRITE = ("Library/LaunchAgents",)
LINUX_DENY_READ = (".config/google-chrome", ".config/chromium", ".mozilla")
LINUX_DENY_WRITE = (".config/systemd", ".config/autostart", ".local/share/systemd")
CACHE_ALLOW_WRITE = (".cache/dev-test-slots", ".cache/poolhouse", ".cache/huggingface", ".cache/pip",
                     ".cache/ms-playwright", "Library/Caches/ms-playwright", "Library/Caches/pip")
SHARED_GIT_DENY = ("config", "hooks", "info", "worktrees/*/config.worktree", "worktrees/*/hooks")
DEPLOY_COMMANDS = ("poolhouse runtime ensure", "poolhouse runtime rollback", "poolhouse runtime restart-host")
"""Run outside the sandbox because they write the runtimes root; the `runtime.deploy` authority gate is their guard."""
CHECKOUT_DENY = ("scripts/hooks", ".claude", ".mcp.json", ".git/config", ".git/hooks", ".githooks")


@dataclass(frozen=True)
class Layout:
    """Where everything lives on one machine."""

    home: Path
    state: Path
    primary: Path
    worktrees: tuple[Path, ...] = ()
    scratch: tuple[Path, ...] = ()
    platform: str = "darwin"
    system_paths: dict[str, str] = field(default_factory=dict)

    def checkouts(self) -> tuple[Path, ...]:
        """The primary checkout and every sibling worktree, without repeats."""
        return tuple(dict.fromkeys((self.primary, *self.worktrees)))


def _under(root: Path, names: tuple[str, ...]) -> list[str]:
    return [str(root / name) for name in names]


def write_roots(layout: Layout) -> list[str]:
    """Trees the agent's commands may write."""
    roots = [str(path) for path in layout.checkouts()] + [str(path) for path in layout.scratch]
    roots += _under(layout.state, STATE_ALLOW_WRITE)
    roots += _under(layout.home, CACHE_ALLOW_WRITE)
    roots.append(str(layout.primary / ".git"))
    return list(dict.fromkeys(roots))


def deny_write(layout: Layout) -> list[str]:
    """Trees no sandboxed command may write, whatever the allow list says."""
    out = _under(layout.home, HOME_DENY_WRITE) + _under(layout.state, STATE_DENY_WRITE)
    out += _under(layout.home, DARWIN_DENY_WRITE if layout.platform == "darwin" else LINUX_DENY_WRITE)
    out += [str(layout.primary / ".git" / name) for name in SHARED_GIT_DENY]
    for checkout in layout.checkouts():
        out += _under(checkout, CHECKOUT_DENY)
    return list(dict.fromkeys(out))


def deny_read(layout: Layout) -> list[str]:
    """Trees no sandboxed command may read."""
    out = _under(layout.home, HOME_DENY_READ) + _under(layout.state, STATE_DENY_READ)
    out += _under(layout.home, DARWIN_DENY_READ if layout.platform == "darwin" else LINUX_DENY_READ)
    return list(dict.fromkeys(out))


def allow_read(layout: Layout) -> list[str]:
    """Paths inside a denied tree that stay readable."""
    return _under(layout.state, STATE_ALLOW_READ)


def edit_rules(layout: Layout) -> list[str]:
    """Permission rules that refuse the file tools (outside the Bash sandbox) the same paths."""
    rules: list[str] = []
    for tool in ("Edit", "Write"):
        for path in deny_write(layout):
            rules += [f"{tool}(/{path})", f"{tool}(/{path}/**)"]
    return rules


def claude_settings(layout: Layout) -> dict:
    """The managed settings for Claude Code's built-in Bash sandbox."""
    network: dict = {"allowedDomains": list(HOSTS)}
    if layout.platform == "darwin":
        network["allowLocalBinding"] = True
    return {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "excludedCommands": list(DEPLOY_COMMANDS),
            "autoAllowBashIfSandboxed": False,
            "filesystem": {
                "allowWrite": write_roots(layout),
                "denyWrite": deny_write(layout),
                "denyRead": deny_read(layout),
                "allowRead": allow_read(layout),
            },
            "network": network,
            "credentials": {"envVars": {"deny": list(SCRUBBED_ENV)}},
        },
        "permissions": {"deny": edit_rules(layout)},
        "env": {"CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1"},
    }


def srt_settings(layout: Layout) -> dict:
    """Settings for sandbox-runtime wrapping a whole harness process, local models included."""
    return {
        "network": {"allowedDomains": [*HOSTS, *RUNTIME_HOSTS], "deniedDomains": []},
        "filesystem": {
            "denyRead": deny_read(layout),
            "allowWrite": write_roots(layout),
            "denyWrite": deny_write(layout),
        },
    }


def codex_toml(layout: Layout) -> str:
    """A Codex config.toml fragment: workspace-write with the checkouts and scratch dirs writable."""
    roots = write_roots(layout)
    lines = ['sandbox_mode = "workspace-write"', 'approval_policy = "on-request"', "",
             "[sandbox_workspace_write]", "network_access = false",
             "exclude_tmpdir_env_var = false", "exclude_slash_tmp = false", "writable_roots = ["]
    lines += [f"  {json.dumps(path)}," for path in roots]
    lines += ["]", ""]
    return "\n".join(lines)
