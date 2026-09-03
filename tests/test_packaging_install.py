"""The network installers are what a person who never opens a terminal runs; what they
promise has to match what the code expects.

One script per platform, four modes each. These check the promises that can be checked
without a fresh machine to run them on: they parse, they offer every mode the README names,
the Windows one opens the firewall by the names and ports `discovery.py` uses, and neither
reimplements in shell what an ml-stack command already does.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from ml_stack.fleet.discovery import (
    DEFAULT_HTTP_PORT,
    FIREWALL_RULE_DISCOVERY,
    FIREWALL_RULE_HTTP,
    default_port,
)

PACKAGING = Path(__file__).resolve().parent.parent / "packaging"
SH = PACKAGING / "install.sh"
PS1 = PACKAGING / "install.ps1"


def test_the_windows_installer_opens_the_firewall_rules_the_fleet_needs():
    """Windows Defender Firewall blocks the daemon and its beacons inbound by default; the
    installer adds the two rules, by the names and ports discovery.py uses, so a machine
    joined from the app is not invisible to the rest of the fleet."""
    script = PS1.read_text(encoding="utf-8")
    rules = dict(re.findall(r'Name = "([^"]+)";\s+Protocol = "(?:TCP|UDP)"; Port = (\d+)', script))
    assert rules == {FIREWALL_RULE_HTTP: str(DEFAULT_HTTP_PORT),
                     FIREWALL_RULE_DISCOVERY: str(default_port())}
    assert "-Verb RunAs" in script, "the rules need an administrator's approval, once"
    assert "Get-NetFirewallRule" in script, "a rule already present is not added twice"


def test_the_shell_installer_parses():
    """It is piped straight into sh, so a syntax error reaches the person as half an
    install rather than as an error."""
    done = subprocess.run(["sh", "-n", str(SH)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_powershell_installer_parses():
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("no pwsh to parse with")
    check = ("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
             f"'{PS1}', [ref]$null, [ref]$e); if ($e.Count) {{ $e | ForEach-Object "
             "{ Write-Error $_ }; exit 1 }")
    done = subprocess.run([pwsh, "-NoProfile", "-Command", check],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]


@pytest.mark.parametrize("mode", ["headless", "dev", "system", "uninstall"])
def test_both_installers_offer_every_mode(mode):
    """The README promises four; a mode named there and missing here is a dead line."""
    assert f"--{mode}" in SH.read_text(encoding="utf-8")
    windows = PS1.read_text(encoding="utf-8")
    assert f"${mode.capitalize()}" in windows or f'"{mode}"' in windows


@pytest.mark.parametrize("command", ["ml-stack-serve", "ml-stack-setup", "ml-stack-models",
                                     "ml-stack-fleet", "ml-stack-doctor"])
def test_the_installers_call_the_commands_rather_than_redoing_them(command):
    """Sizing a machine, building llama.cpp, fetching a model and joining a fleet are each
    a command already. A shell reimplementation is a second answer that goes stale the
    first time the real one changes."""
    assert command in SH.read_text(encoding="utf-8")
    assert command in PS1.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["ML_STACK_PASSPHRASE", "ML_STACK_NAME", "ML_STACK_MODELS",
                                  "ML_STACK_OFFLINE_ZIP", "ML_STACK_MODE"])
def test_every_prompt_can_be_answered_from_the_environment(name):
    """A machine being set up by a script has no terminal to type at, and prompting one
    that cannot answer hangs the install rather than failing it."""
    assert name in SH.read_text(encoding="utf-8")
    assert name in PS1.read_text(encoding="utf-8")


def test_uninstalling_says_the_model_cache_is_left_alone():
    """Tens of gigabytes nobody asked to lose, and coming back downloads nothing again."""
    for script in (SH, PS1):
        assert "cache is left where it is" in script.read_text(encoding="utf-8"), script.name


def test_the_app_mode_downloads_no_model_behind_anybodys_back():
    """The first-run screen presents what fits and downloads on a click. Only the
    unattended modes -- which asked to be unattended -- fetch anything themselves."""
    body = SH.read_text(encoding="utf-8")
    assert 'if [ "$MODE" != app ]' in body
    assert re.search(r'fetch_models\n', body.split('if [ "$MODE" != app ]')[1])
