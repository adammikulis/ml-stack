"""Prejoin hints authenticate only when the person confirms a named cluster."""

import base64
import json
import socket

import pytest

from ml_stack.fleet import discovery, lan_clusters, peers
from ml_stack.fleet.onboard import joining
from ml_stack.fleet.ui import UI

WORDS = "nine blue lanterns together"


def test_native_named_hint_then_authenticated_join(tmp_path, monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(port))
    from test_fleet_handshake import Machine

    host = Machine(tmp_path, port)
    host.advertiser.joinable = True
    try:
        assert lan_clusters.nearby(timeout_s=.4, port=port) == [{"group": "lab"}]
        client = UI(name="desk", cluster_key_path=tmp_path / "client.key")
        remembered = []
        monkeypatch.setattr("ml_stack.fleet.ui.recovery.remember", lambda *args: remembered.append(args))
        with pytest.raises(discovery.DiscoveryError):
            client.join("the wrong words entirely", "lab", "local", existing=True)
        assert not discovery.memberships(client.cluster_key_path) and not remembered
        client.join("quince larch marlow", "lab", "local", existing=True)
        assert discovery.load_cluster_key(client.cluster_key_path) == host.member.key
        assert remembered[0][:2] == ("quince larch marlow", "lab")
    finally:
        host.stop()


def test_selected_missing_cluster_never_creates_membership(tmp_path, monkeypatch):
    monkeypatch.setattr(joining, "find_joiners", lambda *args, **kwargs: [])
    client = UI(name="desk", cluster_key_path=tmp_path / "client.key")
    with pytest.raises(discovery.DiscoveryError, match="no longer available"):
        client.join(WORDS, "Cedar lab", "local", existing=True)
    assert not discovery.memberships(client.cluster_key_path)


@pytest.mark.parametrize("name", [None, "", "  ", False, [], {}, "a\nname", "\ud800", "x" * 65])
def test_names_required_before_any_key_is_written(tmp_path, name):
    for make in (joining.join_by_passphrase,):
        with pytest.raises(discovery.DiscoveryError, match="cluster name"):
            make(WORDS, group=name, path=tmp_path / "client.key")
    with pytest.raises(discovery.DiscoveryError, match="cluster name"):
        discovery.create_cluster_key(tmp_path / "client.key", group=name)
    assert not discovery.memberships(tmp_path / "client.key")


@pytest.mark.redteam
def test_malformed_stale_and_oversized_hints_are_ignored():
    packet = {"v": discovery.PROTOCOL, "kind": "join", "nonce": "fresh", "group": "Cedar lab",
              "method": "passphrase", "port": 8770}
    assert lan_clusters._hint(json.dumps(packet).encode(), "fresh") == {"group": "Cedar lab"}
    assert lan_clusters._hint(json.dumps(packet).encode(), "stale") is None
    for bad in ([], None, {**packet, "group": False}, {**packet, "port": []}, {**packet, "method": "shell"}):
        assert lan_clusters._hint(json.dumps(bad).encode(), "fresh") is None
    assert lan_clusters._hint(b" " * 2049, "fresh") is None


def test_cli_creation_requires_an_explicit_cluster_name(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert peers.main(["--cluster-key", str(tmp_path / "key"), "setup", "--passphrase", WORDS]) == 2
    with pytest.raises(SystemExit):
        peers.main(["--cluster-key", str(tmp_path / "key"), "init"])
    assert not discovery.memberships(tmp_path / "key")


@pytest.fixture
def serving(tmp_path):
    from test_fleet_ui import Serving

    server = Serving(tmp_path)
    yield server
    server.close()


@pytest.mark.parametrize("name", [None, "", "   ", False, [], {}])
def test_setup_api_rejects_missing_or_nonstring_names(serving, name):
    status, body, _ = serving.call("/ui/setup/join", method="POST", body={"group": name, "passphrase": WORDS})
    assert status == 400 and "cluster name" in body["error"]
    assert not discovery.memberships(serving.keyfile)


@pytest.mark.redteam
def test_prejoin_discovery_obeys_setup_boundary(serving, monkeypatch):
    monkeypatch.setattr(lan_clusters, "nearby", lambda **kwargs: [{"group": "Cedar lab"}])
    status, body, _ = serving.call("/ui/setup/clusters")
    assert status == 200 and body["clusters"] == [{"group": "Cedar lab"}]
    status, _, _ = serving.call("/ui/setup/clusters", ui_header=False)
    assert status == 403
    status, _, _ = serving.call("/ui/setup/clusters", headers={"Host": "hostile.invalid"})
    assert status == 403


@pytest.fixture
def password_cluster(tmp_path, serving):
    from test_fleet_handshake import Machine

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    serving.ui.discovery_port = port
    host = Machine(tmp_path, port)
    host.advertiser.joinable = True
    try:
        yield host
    finally:
        host.stop()


@pytest.mark.slow
def test_browser_named_choices_confirmation_refresh_and_manual(serving, password_cluster, monkeypatch, playwright):
    from playwright.sync_api import expect

    monkeypatch.setattr("ml_stack.fleet.ui.recovery.remember", lambda *args: None)
    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"http://127.0.0.1:{serving.port}/ui/")
        wizard = page.locator("first-run")
        wizard.get_by_role("button", name="Continue", exact=True).click()
        expect(wizard.get_by_text("lab", exact=True)).to_be_visible()
        wizard.get_by_role("button", name="Join", exact=True).click()
        expect(wizard.locator("#g")).to_have_value("lab")
        assert wizard.locator("#g").get_attribute("readonly") is not None
        expect(wizard.get_by_role("button", name="Join lab", exact=True)).to_be_disabled()
        wizard.get_by_role("button", name="Enter a name manually", exact=True).click()
        wizard.locator("#p1").fill("quince larch marlow")
        expect(wizard.get_by_role("button", name="Create or join manually", exact=True)).to_be_disabled()
        wizard.get_by_role("button", name="Refresh nearby clusters", exact=True).click()
        wizard.get_by_role("button", name="Join", exact=True).click()
        wizard.locator("#p1").fill("quince larch marlow")
        wizard.get_by_role("button", name="Join lab", exact=True).click()
        expect(wizard.get_by_text("Joined. Use this cluster name and passphrase on your other machines.", exact=True)).to_be_visible()
        assert discovery.memberships(serving.keyfile)[0].group == "lab"
        assert discovery.load_cluster_key(serving.keyfile) == password_cluster.member.key
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path="/private/tmp/ml-stack-named-lan-mobile.png", full_page=True)
        assert not errors


def test_prejoin_scan_never_touches_the_keystore(serving, monkeypatch):
    from ml_stack import keystore

    def forbidden():
        raise AssertionError("discovery must not request the keystore")

    monkeypatch.setattr(keystore, "default", forbidden)
    monkeypatch.setattr(lan_clusters, "nearby", lambda **kwargs: [{"group": "Cedar lab"}])
    assert serving.call("/ui/setup/clusters")[0] == 200


@pytest.mark.parametrize("selection", ["true", None, 1, []])
def test_existing_selection_requires_a_boolean(serving, selection):
    status, body, _ = serving.call("/ui/setup/join", method="POST", body={
        "group": "Cedar lab", "passphrase": WORDS, "existing": selection})
    assert status == 400 and "selection" in body["error"]
    assert not discovery.memberships(serving.keyfile)


@pytest.fixture
def random_cluster(tmp_path, serving, monkeypatch):
    from ml_stack.fleet import recovery

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    serving.ui.discovery_port = port
    keyfile = tmp_path / "random-host.key"
    key = discovery.create_cluster_key(keyfile, group="Pine workshop").encode()
    tell = discovery.Advertiser(discovery.Beacon(name="tower"), key, port=port)
    tell.cluster = "Pine workshop"
    exported = tmp_path / "pine.recovery"
    recovery.export_recovery(exported, path=keyfile)
    with tell:
        yield exported.read_text(), key


@pytest.mark.redteam
def test_random_key_hint_and_upload_authenticate_without_keystore(serving, random_cluster, monkeypatch):
    from ml_stack import keystore

    def forbidden():
        raise AssertionError("random-key onboarding must not touch the keystore")

    monkeypatch.setattr(keystore, "default", forbidden)
    text, key = random_cluster
    status, body, _ = serving.call("/ui/setup/clusters")
    assert status == 200 and body["clusters"] == [{"group": "Pine workshop", "method": "recovery"}]
    status, body, headers = serving.call("/ui/setup/recovery", method="POST", body={
        "group": "Pine workshop", "recovery": text})
    assert status == 200 and body["group"] == "Pine workshop" and "Set-Cookie" in headers
    assert discovery.load_cluster_key(serving.keyfile) == key
    assert key.decode() not in json.dumps(body)
    assert serving.call("/ui/setup/recovery", method="POST", body={
        "group": "Pine workshop", "recovery": text})[0] == 401


@pytest.mark.redteam
@pytest.mark.parametrize("bad", [None, [], {}, "null", "[]", "{}", "x" * 8193])
def test_hostile_recovery_upload_never_writes_membership(serving, bad):
    status, _, _ = serving.call("/ui/setup/recovery", method="POST", body={
        "group": "Pine workshop", "recovery": bad})
    assert status == 400 and not discovery.memberships(serving.keyfile)


@pytest.mark.redteam
def test_wrong_name_wrong_key_and_cross_origin_cannot_import(serving, random_cluster):
    text, _ = random_cluster
    body = {"group": "Other workshop", "recovery": text}
    assert serving.call("/ui/setup/recovery", method="POST", body=body)[0] == 400
    data = json.loads("\n".join(line for line in text.splitlines() if not line.startswith("#")))
    data["key"] = base64.urlsafe_b64encode(b"x" * 32).decode().rstrip("=")
    body = {"group": "Pine workshop", "recovery": json.dumps(data)}
    assert serving.call("/ui/setup/recovery", method="POST", body=body)[0] == 400
    body["recovery"] = text
    assert serving.call("/ui/setup/recovery", method="POST", body=body, ui_header=False)[0] == 403
    assert serving.call("/ui/setup/recovery", method="POST", body=body,
                        headers={"Host": "hostile.invalid"})[0] == 403
    assert not discovery.memberships(serving.keyfile)


@pytest.mark.slow
def test_browser_random_cluster_recovery_picker(serving, random_cluster, playwright):
    from playwright.sync_api import expect

    text, key = random_cluster
    with playwright.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(f"http://127.0.0.1:{serving.port}/ui/")
        wizard = page.locator("first-run")
        wizard.get_by_role("button", name="Continue", exact=True).click()
        wizard.get_by_role("button", name="Join with recovery file", exact=True).click()
        expect(wizard.get_by_role("button", name="Join Pine workshop with recovery file", exact=True)).to_be_disabled()
        wizard.locator("#cluster-recovery-file").set_input_files({
            "name": "pine.recovery", "mimeType": "text/plain", "buffer": text.encode()})
        wizard.get_by_role("button", name="Join Pine workshop with recovery file", exact=True).click()
        expect(wizard.get_by_text("Joined Pine workshop.", exact=True)).to_be_visible()
        assert discovery.load_cluster_key(serving.keyfile) == key
        assert key.decode() not in page.content()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path="/private/tmp/ml-stack-random-lan-mobile.png", full_page=True)


@pytest.mark.redteam
@pytest.mark.parametrize("payload", [
    {"group": "Other lab", "key": "eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHg"},
    {"group": "lab", "key": []},
    {"group": "lab", "key": "not-a-key"},
])
def test_authenticated_pake_answer_must_contain_selected_name_and_valid_key(
        serving, password_cluster, monkeypatch, payload):
    from ml_stack import sealing

    host = password_cluster
    finish = host.joining._finish

    def malformed(body, source):
        held = host.joining._pending[body["id"]]
        result = finish(body, source)
        result["sealed"] = sealing.seal(held.session.key("join"), json.dumps(payload).encode(), held.context).hex()
        return result

    monkeypatch.setattr(host.joining, "_finish", malformed)
    status, body, _ = serving.call("/ui/setup/join", method="POST", body={
        "group": "lab", "passphrase": "quince larch marlow", "existing": True})
    assert status == 400 and "authenticate" in body["error"]
    assert not discovery.memberships(serving.keyfile)


@pytest.mark.redteam
def test_pake_server_confirmation_is_required_before_membership(serving, password_cluster, monkeypatch):
    host = password_cluster
    finish = host.joining._finish

    def unconfirmed(body, source):
        result = finish(body, source)
        result["confirmation"] = "0" * 64
        return result

    monkeypatch.setattr(host.joining, "_finish", unconfirmed)
    status, body, _ = serving.call("/ui/setup/join", method="POST", body={
        "group": "lab", "passphrase": "quince larch marlow", "existing": True})
    assert status == 400 and "passphrase" in body["error"]
    assert not discovery.memberships(serving.keyfile)


def test_join_method_reads_wrapped_record_presence_without_unwrapping(tmp_path, monkeypatch):
    from ml_stack import keystore
    from ml_stack.files import write_json
    from ml_stack.fleet import recovery

    def forbidden():
        raise AssertionError("join discovery must not open the keystore")

    monkeypatch.setattr(keystore, "default", forbidden)
    path = tmp_path / "cluster.key"
    assert not recovery.has_passphrase("lab", path)
    write_json(recovery.passphrases_path(path), {"lab": "wrapped-ciphertext"})
    assert recovery.has_passphrase("lab", path)
    assert not recovery.has_passphrase("other", path)
