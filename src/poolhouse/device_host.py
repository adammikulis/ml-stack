"""What setting a device up does to the machine it runs on: Windows firewall rules, the WSL config, WSL restart, Rust.

`device_setup` decides; this acts. Windows things go through PowerShell (from WSL, the Windows `powershell.exe`).
Nothing here opens a socket or prints a secret.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

from poolhouse.home import user_home
from poolhouse.node_join import DEFAULT_BEACON_PORT, DEFAULT_PORT

HYPERV_ID = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
RULES = (("poolhouse-node", "TCP", DEFAULT_PORT), ("poolhouse-beacon", "UDP", DEFAULT_BEACON_PORT))
POWERSHELL = "powershell.exe"
RUSTUP_URL = "https://sh.rustup.rs"
TIMEOUT = 600

_HYPERV_PRESENT = "(Get-Command Get-NetFirewallHyperVVMSetting -ErrorAction SilentlyContinue)"


def host_kind(platform: str, proc_version: str) -> str:
    """`windows`, `wsl`, `mac` or `linux`."""
    if platform == "win32":
        return "windows"
    if platform == "darwin":
        return "mac"
    return "wsl" if "microsoft" in proc_version.lower() else "linux"


def current_kind() -> str:
    """The kind of the machine this runs on."""
    try:
        version = Path("/proc/version").read_text(encoding="utf-8")
    except OSError:
        version = ""
    return host_kind(sys.platform, version)


def _powershell(script: str) -> str:
    done = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True, timeout=TIMEOUT, check=False)
    return done.stdout.strip()


def rules_script() -> str:
    """PowerShell that adds each missing inbound rule and lets the WSL VM receive them."""
    adds = "; ".join(f"if (-not (Get-NetFirewallRule -DisplayName {name} -ErrorAction SilentlyContinue)) "
                     f"{{ New-NetFirewallRule -DisplayName {name} -Direction Inbound -Action Allow -Profile Private -Protocol {proto} -LocalPort {port} | Out-Null }}"
                     for name, proto, port in RULES)
    return f"{adds}; if ({_HYPERV_PRESENT}) {{ Set-NetFirewallHyperVVMSetting -Name '{HYPERV_ID}' -DefaultInboundAction Allow }}"


def firewall_missing() -> bool:
    """Whether the inbound rules (or the Hyper-V default) still need adding."""
    names = ",".join(f"'{name}'" for name, _, _ in RULES)
    script = (f"$m = @({names}) | Where-Object {{ -not (Get-NetFirewallRule -DisplayName $_ -ErrorAction SilentlyContinue) }}; "
              f"$h = {_HYPERV_PRESENT} -and ((Get-NetFirewallHyperVVMSetting -Name '{HYPERV_ID}' -ErrorAction SilentlyContinue).DefaultInboundAction -ne 'Allow'); "
              "if ($m -or $h) { 'missing' } else { 'ok' }")
    return _powershell(script) != "ok"


def add_firewall() -> None:
    """Add the rules in an elevated PowerShell: one Windows administrator prompt."""
    encoded = base64.b64encode(rules_script().encode("utf-16-le")).decode("ascii")
    _powershell(f"Start-Process {POWERSHELL} -Verb RunAs -Wait -ArgumentList '-NoProfile','-EncodedCommand','{encoded}'")


def wslconfig_path(kind: str) -> Path:
    """The Windows user's .wslconfig, seen from here."""
    if kind == "windows":
        return user_home() / ".wslconfig"
    profile = subprocess.run(["wslpath", "-u", _powershell("$env:USERPROFILE")], capture_output=True, text=True, timeout=TIMEOUT, check=False).stdout.strip()
    return Path(profile) / ".wslconfig"


def restart_wsl() -> None:
    """`wsl --shutdown`: ends this session when it runs from inside WSL."""
    subprocess.run(["wsl.exe", "--shutdown"], timeout=TIMEOUT, check=False)


def install_rust(kind: str) -> None:
    """Install Rust with the standard installer (rustup's script; winget's Rustup on Windows) and put cargo on this process's PATH."""
    if kind == "windows":
        subprocess.run(["winget", "install", "--id", "Rustlang.Rustup", "-e", "--accept-package-agreements", "--accept-source-agreements"], timeout=TIMEOUT, check=True)
    else:
        subprocess.run(["sh", "-c", f"curl --proto '=https' --tlsv1.2 -sSf {RUSTUP_URL} | sh -s -- -y --profile minimal"], timeout=TIMEOUT, check=True)
    os.environ["PATH"] = f"{user_home() / '.cargo' / 'bin'}{os.pathsep}{os.environ['PATH']}"
