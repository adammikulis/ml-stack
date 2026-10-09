"""Poolhouse.app: its Info.plist, designated requirement, launch and signing command lines, and the probe's result. No sockets, no keychain."""

from __future__ import annotations

import plistlib
from pathlib import Path

from poolhouse import node_app, node_permission, node_signing

SHA1 = "4E2F3980AC138E86B7532FF4E3460EF53D0E7B23"


def test_the_bundle_id_is_fixed():
    assert node_app.BUNDLE_ID == "app.poolhouse.node"  # a new id is a new app to macOS and loses every grant


def test_info_plist_carries_the_identity_and_the_prompt_keys():
    plist = plistlib.loads(plistlib.dumps(node_app.info_plist("abc123")))
    assert plist["CFBundleIdentifier"] == "app.poolhouse.node"
    assert plist["CFBundleName"] == "Poolhouse"
    assert plist["CFBundleExecutable"] == "poolhouse-node"
    assert plist["LSUIElement"] is True
    assert plist["NSLocalNetworkUsageDescription"] == "Poolhouse finds and pairs with your other devices on this network."
    assert plist["NSBonjourServices"] == ["_poolhouse._tcp"]
    assert plist["CFBundleIconFile"] == "Poolhouse"
    assert "CFBundleIconFile" not in node_app.info_plist(icon=False)


def test_requirement_names_the_id_and_the_certificate_never_a_cdhash():
    text = node_app.requirement(SHA1)
    assert text == f'designated => identifier "app.poolhouse.node" and certificate leaf = H"{SHA1.lower()}"'
    assert "cdhash" not in text
    assert node_app.ad_hoc_requirement() == 'designated => identifier "app.poolhouse.node"'


def test_launch_command_runs_the_executable_inside_the_bundle():
    bundle = Path("/x/Poolhouse.app")
    assert node_app.launch_command(bundle, ["run", "--lan"]) == ["/x/Poolhouse.app/Contents/MacOS/poolhouse-node", "run", "--lan"]


def test_only_a_lan_start_on_macos_goes_through_the_bundle(monkeypatch):
    monkeypatch.setattr(node_app.sys, "platform", "darwin")
    assert node_app.uses_bundle(["--lan"]) and not node_app.uses_bundle(["--listen", "127.0.0.1:0"])
    monkeypatch.setattr(node_app.sys, "platform", "linux")
    assert not node_app.uses_bundle(["--lan"])


def test_sign_command_uses_the_identifier_and_requirement():
    argv = node_signing.sign_command(Path("/x/P.app"), "app.poolhouse.node", node_app.requirement(SHA1))
    assert argv[:4] == ["codesign", "--force", "--sign", "Poolhouse Local Signing"]
    assert argv[4:6] == ["--identifier", "app.poolhouse.node"]
    assert f"-r={node_app.requirement(SHA1)}" in argv
    assert node_signing.sign_command(Path("/x/P.app"), "i", "r", identity="-")[3] == "-"


def test_certificate_request_is_a_code_signing_leaf():
    words = " ".join(node_signing.certificate_request())
    assert "/CN=Poolhouse Local Signing" in words and "codeSigning" in words and "CA:false" in words


def test_parse_sha1_reads_security_output():
    out = f"SHA-256 hash: AA\nSHA-1 hash: {SHA1}\nkeychain: x\n"
    assert node_signing.parse_sha1(out) == SHA1.lower()
    assert node_signing.parse_sha1("nothing") == ""


def test_probe_command_goes_through_launch_services():
    argv = node_permission.probe_command(Path("/x/P.app"), Path("/t/o.json"), 30, ["192.168.2.1"])
    assert argv[:5] == ["open", "-W", "-n", "-a", "/x/P.app"]
    assert argv[5:] == ["--args", "probe", "--seconds", "30", "--out", "/t/o.json", "--peer", "192.168.2.1"]


def test_verdicts():
    assert node_permission.verdict(None) == node_permission.NOT_ASKED
    assert node_permission.verdict({"heard_self": True, "sent": 4, "send_errors": 2}) == node_permission.GRANTED
    assert node_permission.verdict({"heard_self": False, "sent": 4}) == node_permission.DENIED
    assert node_permission.verdict({"heard_self": True, "sent": 0}) == node_permission.DENIED
    assert node_permission.verdict({"error": "boom"}) == node_permission.DENIED


def test_parse_result_refuses_what_is_not_an_object():
    assert node_permission.parse_result('{"heard_self": true}') == {"heard_self": True}
    assert node_permission.parse_result("") is None and node_permission.parse_result("[1]") is None


def test_gateway_is_read_from_route_output():
    out = "   route to: default\n    gateway: 192.168.2.1\n  interface: en0\n"
    assert node_permission.gateway(out) == "192.168.2.1"
    assert node_permission.gateway("no route") == ""


def test_the_supervisor_runs_the_bundle_for_a_lan_start_and_the_bare_binary_otherwise(monkeypatch):
    from poolhouse import node_supervise

    monkeypatch.setattr(node_app.sys, "platform", "darwin")
    monkeypatch.setattr(node_app, "bundled", lambda binary, sha: Path("/x/Poolhouse.app/Contents/MacOS/poolhouse-node"))
    assert node_supervise._program(Path("/r/poolhouse-node"), "ab", ["--lan"]) == Path("/x/Poolhouse.app/Contents/MacOS/poolhouse-node")
    assert node_supervise._program(Path("/r/poolhouse-node"), "ab", []) == Path("/r/poolhouse-node")


def test_a_hostile_path_is_one_argument_in_every_command_line_the_app_builds():
    hostile = "/x/$(touch pwned); `id` | rm -rf y/Poolhouse.app"
    sign = node_signing.sign_command(Path(hostile), "app.poolhouse.node", node_app.requirement(SHA1))
    assert sign[-1] == hostile and len(sign) == 9
    probe = node_permission.probe_command(Path(hostile), Path(hostile + ".json"), 5, ["192.168.2.1; id"])
    assert probe[4] == hostile and probe[8] == "5" and probe[10] == hostile + ".json" and probe[-1] == "192.168.2.1; id"
    assert node_app.launch_command(Path(hostile), ["run"])[0].startswith(hostile)
