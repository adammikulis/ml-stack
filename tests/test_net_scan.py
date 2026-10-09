"""Scanners: the logic over fake scanners, the ClamAV and Defender command handling over scripts
that behave like the tools, ClamAV itself where installed, and the honest macOS answer."""

import json
import shutil
import stat
import sys

import pytest

from poolhouse import net
from poolhouse.httpguard import Limits
from poolhouse.net.scan import Outcome, ScanPolicy, ScanResult, scan_file, summarise
from poolhouse.net.scanners import ClamAV, HashLookup, MacNotice, WindowsDefender
from tests.net_site import EICAR

SCRIPT = """#!/bin/sh
# usage: tool [flags] FILE ; the last argument is the file
for last; do :; done
{body}
"""


def tool(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(SCRIPT.format(body=body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


CLAM_BODY = """
if [ "$1" = "--version" ]; then echo "ClamAV 1.4.1/27500/Tue Sep  2 08:23:11 2025"; exit 0; fi
if grep -q 'EICAR-STANDARD-ANTIVIRUS-TEST-FILE' "$last"; then echo "$last: Eicar-Signature FOUND"; exit 1; fi
exit 0
"""


class Told:
    def __init__(self, outcome, name="t"):
        self.name, self.outcome = name, outcome

    def available(self):
        return True

    def scan(self, path):
        return ScanResult(self.name, self.outcome)


class Missing:
    name = "missing"

    def available(self):
        return False

    def scan(self, path):  # pragma: no cover
        raise AssertionError("an unavailable scanner is not run")


def test_no_scanner_is_never_reported_as_clean(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"x")
    summary = scan_file(f, [Missing()])
    assert summary.outcome == Outcome.NO_SCANNER
    assert summary.line == "scanned: no scanner available"
    assert scan_file(f, []).outcome == Outcome.NO_SCANNER


def test_infected_beats_clean_and_clean_beats_error():
    assert summarise([ScanResult("a", Outcome.CLEAN), ScanResult("b", Outcome.INFECTED)]).outcome \
        == Outcome.INFECTED
    assert summarise([ScanResult("a", Outcome.ERROR), ScanResult("b", Outcome.CLEAN)]).outcome \
        == Outcome.CLEAN
    assert summarise([ScanResult("a", Outcome.ERROR)]).outcome == Outcome.ERROR
    assert summarise([ScanResult("a", Outcome.NO_SCANNER)]).outcome == Outcome.NO_SCANNER


def test_a_scanner_that_raises_oserror_is_an_error_result(tmp_path):
    class Broken(Told):
        def scan(self, path):
            raise PermissionError("denied")

    f = tmp_path / "f"
    f.write_bytes(b"x")
    out = scan_file(f, [Broken(Outcome.CLEAN, "broken")])
    assert out.outcome == Outcome.ERROR and "denied" in out.results[0].detail


@pytest.mark.parametrize(("kind", "unscanned_kept"), [
    ("gguf", True), ("safetensors", True), ("pdf", True), ("zip", False), ("tar", False),
    ("executable", False),
])
def test_the_default_policy_for_files_nobody_could_scan(kind, unscanned_kept):
    keep, _ = ScanPolicy().decide(kind, summarise([]))
    assert keep is unscanned_kept


def test_the_model_policy_says_why_a_virus_scanner_is_not_enough():
    _, why = ScanPolicy().decide("gguf", summarise([]))
    assert "weights cannot be judged" in why


def test_the_policy_can_be_set_by_the_environment(monkeypatch):
    monkeypatch.setenv("POOLHOUSE_NET_UNSCANNED", "archive:allow, model:refuse, data:bogus")
    policy = ScanPolicy().from_env()
    assert (policy.archive, policy.model, policy.data) == ("allow", "refuse", "warn")


def test_clamscan_exit_codes_map_to_outcomes(tmp_path):
    clean = tmp_path / "ok.txt"
    clean.write_text("hello")
    bad = tmp_path / "eicar.txt"
    bad.write_text(EICAR)
    scanner = ClamAV(clamscan=tool(tmp_path, "clamscan", CLAM_BODY), clamdscan="")
    assert scanner.scan(clean).outcome == Outcome.CLEAN
    hit = scanner.scan(bad)
    assert hit.outcome == Outcome.INFECTED and hit.signature == "Eicar-Signature"


def test_clamscan_exit_two_is_an_error_not_clean(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    scanner = ClamAV(clamscan=tool(tmp_path, "clamscan", 'echo "LibClamAV Error: no database" >&2; exit 2'),
                     clamdscan="")
    out = scanner.scan(f)
    assert out.outcome == Outcome.ERROR and "database" in out.detail


def test_clamdscan_is_used_when_the_daemon_answers_and_clamscan_when_it_does_not(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    marker = tmp_path / "which"
    up = tool(tmp_path, "clamdscan", f'echo daemon > {marker}; exit 0')
    down = tool(tmp_path, "clamdscan-down", 'echo "ERROR: Could not connect to clamd" >&2; exit 2')
    scan = tool(tmp_path, "clamscan", f'[ "$1" = "--version" ] && exit 0; echo scan >> {marker}; exit 0')
    assert ClamAV(clamscan=scan, clamdscan=up).scan(f).outcome == Outcome.CLEAN
    assert marker.read_text().split() == ["daemon"]
    marker.unlink()
    assert ClamAV(clamscan=scan, clamdscan=down).scan(f).outcome == Outcome.CLEAN
    assert marker.read_text().split() == ["scan"]


def test_a_daemon_error_with_no_fallback_is_an_error(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    down = tool(tmp_path, "clamdscan", 'echo "ERROR: Could not connect" >&2; exit 2')
    assert ClamAV(clamscan="", clamdscan=down).scan(f).outcome == Outcome.ERROR


def test_a_stale_signature_database_is_a_warning(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    scanner = ClamAV(clamscan=tool(tmp_path, "clamscan", CLAM_BODY), clamdscan="")
    out = scanner.scan(f)
    assert out.outcome == Outcome.CLEAN
    assert any("days old" in w for w in out.warnings)


def test_a_scan_that_runs_forever_is_an_error(tmp_path, monkeypatch):
    f = tmp_path / "f"
    f.write_text("x")
    monkeypatch.setattr("poolhouse.net.scanners.TIMEOUT_S", 0.3)
    slow = tool(tmp_path, "clamscan", "sleep 5")
    out = ClamAV(clamscan=slow, clamdscan="").scan(f)
    assert out.outcome == Outcome.ERROR and "timed out" in out.detail


def test_clamav_is_unavailable_when_neither_command_exists():
    assert ClamAV(clamscan="", clamdscan="").available() is False


@pytest.mark.skipif(not shutil.which("clamscan"), reason="ClamAV is not installed")
def test_real_clamav_finds_the_eicar_test_string(tmp_path):
    scanner = ClamAV()
    bad = tmp_path / "eicar.com.txt"
    bad.write_text(EICAR)
    ok = tmp_path / "ok.txt"
    ok.write_text("nothing here")
    assert scanner.scan(bad).outcome == Outcome.INFECTED
    assert scanner.scan(ok).outcome == Outcome.CLEAN


def test_defender_command_line_and_exit_codes(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    log = tmp_path / "args"
    body = f'echo "$@" > {log}; exit ${{FAKE_EXIT:-0}}'
    scanner = WindowsDefender(tool(tmp_path, "MpCmdRun.exe", body))
    assert scanner.scan(f).outcome == Outcome.CLEAN
    args = log.read_text().split()
    assert args[:5] == ["-Scan", "-ScanType", "3", "-File", str(f)] and "-DisableRemediation" in args


@pytest.mark.parametrize(("code", "outcome"), [(2, Outcome.INFECTED), (1, Outcome.ERROR)])
def test_defender_exit_two_is_a_threat_and_anything_else_an_error(tmp_path, monkeypatch, code, outcome):
    f = tmp_path / "f"
    f.write_text("x")
    scanner = WindowsDefender(tool(tmp_path, "MpCmdRun.exe", f"exit {code}"))
    assert scanner.scan(f).outcome == outcome


def test_defender_is_unavailable_off_windows():
    if sys.platform != "win32":
        assert WindowsDefender().available() is False


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macos_says_plainly_that_it_has_no_scanner(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"\x7fELF fake")
    out = MacNotice().scan(f)
    assert out.outcome == Outcome.NO_SCANNER
    assert "no malware scanner on macOS" in out.detail and "brew install clamav" in out.detail
    assert "XProtect" not in out.detail or "does not" in out.detail
    summary = scan_file(f, [MacNotice()])
    assert summary.outcome == Outcome.NO_SCANNER


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_macos_reports_the_quarantine_flag_and_gatekeeper_for_binaries(tmp_path):
    import subprocess

    f = tmp_path / "prog"
    f.write_bytes(b"#!/bin/sh\necho hi\n")
    f.chmod(0o755)
    subprocess.run(["xattr", "-w", "com.apple.quarantine", "0081;00000000;test;", str(f)],
                   check=True)
    out = MacNotice().scan(f)
    assert "com.apple.quarantine" in out.detail and "Gatekeeper (not a malware scan)" in out.detail


def test_mac_notice_is_not_available_elsewhere():
    assert MacNotice(platform="linux").available() is False


def _lookup(tmp_path, site, body, status=200):
    site.add("/files/" + "a" * 64, body, status=status, headers={"Content-Type": "application/json"})
    seen = {}

    def get(url, headers):
        seen["url"], seen["headers"] = url, headers
        got = net.Pipeline(
            policy=net.Policy(allowed=["127.0.0.1"], path=tmp_path / "a.jsonl"),
            limits=Limits(allow_hosts=frozenset({"127.0.0.1"}))).get(
                url, net.Ask(headers=headers))
        return got.status, got.body

    f = tmp_path / "f.bin"
    f.write_bytes(b"data")
    return HashLookup("key123", get=get, base=site.base + "/files/", digest=lambda p: "a" * 64), f, seen


def test_the_hash_lookup_sends_the_digest_and_nothing_else(tmp_path):
    from tests.net_site import Site

    stats = {"data": {"attributes": {"last_analysis_stats": {"malicious": 0, "suspicious": 0}}}}
    with Site() as site:
        scanner, f, seen = _lookup(tmp_path, site, json.dumps(stats).encode())
        out = scanner.scan(f)
        route = site.routes["/files/" + "a" * 64]
    assert out.outcome == Outcome.CLEAN and "reputation only" in out.detail
    assert seen["url"].endswith("a" * 64) and "data" not in seen["url"]
    assert route.seen[0]["x-apikey"] == "key123"


def test_a_flagged_hash_is_infected_and_an_unknown_hash_is_not_clean(tmp_path):
    from tests.net_site import Site

    bad = {"data": {"attributes": {"last_analysis_stats": {"malicious": 5}}}}
    with Site() as site:
        scanner, f, _ = _lookup(tmp_path, site, json.dumps(bad).encode())
        assert scanner.scan(f).outcome == Outcome.INFECTED
    with Site() as site:
        scanner, f, _ = _lookup(tmp_path, site, b"{}", status=404)
        assert scanner.scan(f).outcome == Outcome.NO_SCANNER
    with Site() as site:
        scanner, f, _ = _lookup(tmp_path, site, b"nonsense")
        assert scanner.scan(f).outcome == Outcome.ERROR


def test_the_hash_lookup_is_off_without_a_key():
    assert HashLookup("", get=lambda url, headers: (500, b"")).available() is False
