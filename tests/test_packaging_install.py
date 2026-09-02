"""The network installers are what a person who never opens a terminal runs; what they
promise has to match what the code expects."""

from __future__ import annotations

import re
from pathlib import Path

from ml_stack.fleet.discovery import (
    DEFAULT_HTTP_PORT,
    FIREWALL_RULE_DISCOVERY,
    FIREWALL_RULE_HTTP,
    default_port,
)

PACKAGING = Path(__file__).resolve().parent.parent / "packaging"


def test_the_windows_installer_opens_the_firewall_rules_the_fleet_needs():
    """Windows Defender Firewall blocks the daemon and its beacons inbound by default; the
    installer adds the two rules, by the names and ports discovery.py uses, so a machine
    joined from the app is not invisible to the rest of the fleet."""
    script = (PACKAGING / "install.ps1").read_text(encoding="utf-8")
    rules = dict(re.findall(r'Name = "([^"]+)";\s+Protocol = "(?:TCP|UDP)"; Port = (\d+)', script))
    assert rules == {FIREWALL_RULE_HTTP: str(DEFAULT_HTTP_PORT),
                     FIREWALL_RULE_DISCOVERY: str(default_port())}
    assert "-Verb RunAs" in script, "the rules need an administrator's approval, once"
    assert "Get-NetFirewallRule" in script, "a rule already present is not added twice"
