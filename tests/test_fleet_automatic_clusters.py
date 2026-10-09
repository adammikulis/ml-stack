"""Development cluster admission and deterministic convergence."""

import hashlib

import pytest

from poolhouse.fleet import automatic_clusters as automatic, discovery
from poolhouse.fleet.onboard import joining

KEY = b"a" * 43
NONCE = "c" * 32
FINGERPRINT = "d" * 64


def offer(key=KEY):
    return {"mode": "dev", "method": "automatic", "tls": True,
            "fingerprint": FINGERPRINT, "cluster_id": hashlib.sha256(key).hexdigest(),
            "group": "development", "port": 8770}


@pytest.mark.redteam
@pytest.mark.parametrize("changed", [{"nonce": "e" * 32}, {"key": "b" * 43},
                                     {"mode": "prod"}, {"group": "elsewhere"}])
def test_receive_rejects_response_identity_mismatch(monkeypatch, changed):
    monkeypatch.setattr(automatic.secrets, "token_hex", lambda count: NONCE)
    answer = {"mode": "dev", "group": "development", "key": KEY.decode(), "nonce": NONCE}
    answer.update(changed)
    monkeypatch.setattr(joining._Call, "post", lambda self, step, body: (200, answer))
    with pytest.raises(discovery.DiscoveryError):
        automatic.receive("127.0.0.1", offer())


@pytest.mark.redteam
@pytest.mark.parametrize("key", [b"!" * 43, b"=" * 43])
def test_receive_refuses_malformed_key_with_matching_advertised_digest(monkeypatch, key):
    monkeypatch.setattr(automatic.secrets, "token_hex", lambda count: NONCE)
    answer = {"mode": "dev", "group": "development", "key": key.decode(), "nonce": NONCE}
    monkeypatch.setattr(joining._Call, "post", lambda self, step, body: (200, answer))
    with pytest.raises(discovery.DiscoveryError, match="invalid key"):
        automatic.receive("127.0.0.1", offer(key))


@pytest.mark.redteam
def test_ensure_preserves_membership_after_malformed_automatic_key(tmp_path, monkeypatch):
    path = tmp_path / "device.key"
    member = discovery.adopt(discovery.Membership("development", KEY), path)
    key = b"!" * 43
    offered = offer(key)
    monkeypatch.setattr(automatic, "offers", lambda port: [("127.0.0.1", offered)])
    monkeypatch.setattr(automatic.secrets, "token_hex", lambda count: NONCE)
    answer = {"mode": "dev", "group": "development", "key": key.decode(), "nonce": NONCE}
    monkeypatch.setattr(joining._Call, "post", lambda self, step, body: (200, answer))
    assert offered["cluster_id"] < hashlib.sha256(member.key).hexdigest()
    assert automatic.ensure(path) == member
    assert discovery.memberships(path) == [member]


def test_receive_pins_advertised_certificate_and_binds_nonce(monkeypatch):
    captured = []
    monkeypatch.setattr(automatic.secrets, "token_hex", lambda count: NONCE)

    def post(call, step, body):
        captured.append((call.seen, call.joiner.tls, call.timeout, step, body))
        return 200, {"mode": "dev", "group": "development", "key": KEY.decode(), "nonce": body["nonce"]}

    monkeypatch.setattr(joining._Call, "post", post)
    member, _ = automatic.receive("127.0.0.1", offer())
    assert member == discovery.Membership("development", KEY)
    assert len(captured) == 1
    pin, tls, timeout, step, body = captured[0]
    assert (pin, tls, step) == (FINGERPRINT, True, "automatic")
    assert {k: body[k] for k in ("group", "nonce")} == {"group": "development", "nonce": NONCE}
    assert body["cert"] == joining.local_identity().beacon
    assert 0 < timeout <= 5.0


@pytest.mark.redteam
def test_pinned_certificate_change_closes_connection(monkeypatch):
    class Socket:
        def getpeercert(self, *, binary_form):
            return b"different-certificate"

    class Connection:
        sock = Socket()
        closed = False

        def connect(self):
            pass

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(joining, "require_local", lambda host, port: None)
    monkeypatch.setattr(joining.http.client, "HTTPSConnection", lambda *args, **kwargs: connection)
    call = joining._Call(joining.Joiner("127.0.0.1", 8770, True), 1)
    call.seen = FINGERPRINT
    with pytest.raises(discovery.DiscoveryError, match="certificate"):
        call._connect()
    assert connection.closed


@pytest.mark.redteam
@pytest.mark.parametrize("mode,tls,status", [("dev", False, 403), ("dev", True, 200),
                                            ("prod", False, 403), ("prod", True, 403)])
def test_automatic_admission_requires_transport_tls_and_dev_mode(mode, tls, status):
    member = discovery.Membership("development", KEY, mode=mode)
    keeper = joining.Joining(lambda: [member], fingerprint=lambda: FINGERPRINT, log=lambda text: None)
    code, answer = keeper.handle(joining.API + "/automatic", {"group": member.group, "nonce": NONCE},
                                 "127.0.0.1", transport_tls=tls)
    assert code == status
    if code == 200:
        assert answer["key"] == KEY.decode()
        assert answer["nonce"] == NONCE
    else:
        assert "key" not in answer


@pytest.mark.redteam
@pytest.mark.parametrize("nonce", [None, "", "f" * 31, "z" * 32, "f" * 33])
def test_automatic_admission_rejects_invalid_nonce(nonce):
    keeper = joining.Joining(lambda: [discovery.Membership("development", KEY)],
                             fingerprint=lambda: FINGERPRINT, log=lambda text: None)
    code, answer = keeper.handle(joining.API + "/automatic", {"group": "development", "nonce": nonce},
                                 "127.0.0.1", transport_tls=True)
    assert code == 400
    assert "key" not in answer


def test_singletons_converge_on_same_cluster_regardless_of_offer_order(tmp_path, monkeypatch):
    members = [discovery.Membership("development", value * 43) for value in (b"a", b"b", b"c")]
    winner = min(members, key=lambda member: hashlib.sha256(member.key).hexdigest())
    by_identity = {hashlib.sha256(member.key).hexdigest(): member for member in members}
    monkeypatch.setattr(automatic, "receive", lambda host, row: (by_identity[row["cluster_id"]], ""))
    for index, member in enumerate(members):
        path = tmp_path / f"device-{index}.key"
        discovery.adopt(member, path)
        rows = [("127.0.0.1", offer(other.key)) for other in reversed(members)]
        monkeypatch.setattr(automatic, "offers", lambda port, rows=rows: rows)
        assert automatic.ensure(path) == winner
        assert discovery.memberships(path) == [winner]


def test_unreachable_offers_fall_back_to_existing_membership(tmp_path, monkeypatch):
    members = [discovery.Membership("development", value * 43) for value in (b"a", b"b")]
    current = max(members, key=lambda member: hashlib.sha256(member.key).hexdigest())
    other = next(member for member in members if member != current)
    path = tmp_path / "device.key"
    discovery.adopt(current, path)
    monkeypatch.setattr(automatic, "offers", lambda port: [("127.0.0.1", offer(other.key))])
    calls = []

    def unreachable(host, row):
        calls.append(host)
        raise discovery.DiscoveryError("unreachable")

    monkeypatch.setattr(automatic, "receive", unreachable)
    assert automatic.ensure(path) == current
    assert calls == ["127.0.0.1"]


def test_production_membership_never_attempts_automatic_discovery(tmp_path, monkeypatch):
    member = discovery.Membership("production", KEY, mode="prod")
    path = tmp_path / "device.key"
    discovery.adopt(member, path)

    def unexpected(port):
        pytest.fail("Production started automatic discovery")

    monkeypatch.setattr(automatic, "offers", unexpected)
    assert automatic.ensure(path) == member


def test_production_advertiser_answers_manual_join_discovery(monkeypatch):
    import json

    advertiser = discovery.Advertiser(discovery.Beacon(name="device", port=8770), KEY,
                                      cluster="production")
    advertiser.mode = "prod"
    advertiser.joinable = True
    packet = json.dumps({"v": discovery.PROTOCOL, "kind": "join?", "group": "production",
                         "nonce": NONCE}).encode()

    class Socket:
        reads = 0

        def settimeout(self, timeout):
            pass

        def recvfrom(self, size):
            self.reads += 1
            if self.reads == 1:
                return packet, ("127.0.0.1", 8771)
            raise OSError("closed")

    answered = []
    monkeypatch.setattr(discovery, "_socket", lambda **kwargs: Socket())
    monkeypatch.setattr(advertiser, "_tell_join", lambda sock, nonce, address: answered.append(nonce))
    advertiser._serve()
    assert answered == [NONCE]


@pytest.mark.redteam
@pytest.mark.parametrize("changed", [{"tls": False}, {"mode": "prod"}, {"method": "passphrase"},
                                     {"fingerprint": "short"}, {"cluster_id": "short"}])
def test_offers_ignore_nonautomatic_or_unpinned_advertisements(monkeypatch, changed):
    rejected = offer()
    rejected.update(changed)
    monkeypatch.setattr(joining, "_ask_join", lambda *args, **kwargs:
                        [("127.0.0.1", rejected), ("127.0.0.2", offer())])
    assert automatic.offers() == [("127.0.0.2", offer())]


def test_failed_candidate_does_not_block_next_cluster(tmp_path, monkeypatch):
    rows = [("127.0.0.1", offer(b"a" * 43)), ("127.0.0.2", offer(b"b" * 43))]
    rows.sort(key=lambda item: item[1]["cluster_id"])
    member = discovery.Membership("development", (b"a" if rows[1][0] == "127.0.0.1" else b"b") * 43)
    calls = []
    monkeypatch.setattr(automatic, "offers", lambda port: list(reversed(rows)))

    def receive(host, row):
        calls.append(host)
        if host == rows[0][0]:
            raise discovery.DiscoveryError("unreachable")
        return member, ""

    monkeypatch.setattr(automatic, "receive", receive)
    assert automatic.ensure(tmp_path / "device.key") == member
    assert calls == [row[0] for row in rows]


def test_membership_offer_identifies_the_cluster_key():
    import hashlib
    import json

    advertiser = discovery.Advertiser(discovery.Beacon(name="device", port=8770), KEY,
                                      cluster="development")
    packets = []
    class Socket:
        def sendto(self, packet, address):
            packets.append((json.loads(packet), address))
    advertiser._tell_join(Socket(), NONCE, ("127.0.0.1", 8771))
    assert len(packets) == 1
    offered, address = packets[0]
    assert offered["cluster_id"] == hashlib.sha256(KEY).hexdigest()
    assert offered["mode"] == "dev"
    assert offered["nonce"] == NONCE
    assert address == ("127.0.0.1", 8771)


def test_manual_dev_selection_survives_restart_and_project_settle(tmp_path, monkeypatch):
    from poolhouse.workspace.automatic_connection import settle

    path = tmp_path / "device.key"
    member = discovery.mint_cluster("chosen-dev", path, selection="manual")
    monkeypatch.setattr(automatic, "offers", lambda port: pytest.fail("manual selection discovered another cluster"))
    assert discovery.memberships(path)[0].selection == "manual"
    assert automatic.ensure(path, mode="dev") == member
    assert settle(discovery.memberships(path)[0], path) == member


def test_auto_action_releases_manual_selection_and_converges(tmp_path, monkeypatch):
    path = tmp_path / "device.key"
    discovery.adopt(discovery.Membership("chosen-dev", KEY, selection="manual"), path)
    target = discovery.Membership("automatic-dev", b"b" * 43)
    candidates = sorted((KEY, target.key), key=lambda key: hashlib.sha256(key).hexdigest())
    discovery.adopt(discovery.Membership("chosen-dev", candidates[1], selection="manual"), path)
    target = discovery.Membership("automatic-dev", candidates[0])
    monkeypatch.setattr(automatic, "offers", lambda port: [("127.0.0.1", offer(target.key))])
    monkeypatch.setattr(automatic, "receive", lambda host, offered: (target, ""))
    assert automatic.select_automatic(path) == target
    assert discovery.memberships(path)[0] == target


def test_passphrase_creation_marks_dev_manual(tmp_path, monkeypatch):
    path = tmp_path / "device.key"
    monkeypatch.setattr(joining, "find_joiners", lambda *args, **kwargs: [])
    member = joining.create_by_passphrase("trusted words", "chosen-dev", path, mode="dev")
    assert member.selection == "manual"
    assert discovery.memberships(path) == [member]


@pytest.mark.redteam
def test_automatic_action_refuses_production_and_invalid_selection(tmp_path):
    path = tmp_path / "device.key"
    member = discovery.mint_cluster("production", path, mode="prod")
    with pytest.raises(discovery.DiscoveryError, match="Development"):
        automatic.select_automatic(path)
    assert discovery.memberships(path) == [member]
    with pytest.raises(ValueError, match="selection"):
        discovery.Membership("dev", KEY, selection="invalid")


def test_manual_join_marks_received_dev_membership(tmp_path, monkeypatch):
    path = tmp_path / "device.key"
    received = discovery.Membership("chosen-dev", KEY)
    monkeypatch.setattr(joining, "_shake", lambda *args: (received, ""))
    member = joining._accept([joining.Joiner("127.0.0.1", 8770, True)],
                             received.group, "shared secret", path, mode="dev")
    assert member.key == received.key
    assert member.selection == "manual"
    assert discovery.memberships(path) == [member]


def test_project_settle_respects_manual_choice_made_during_discovery(tmp_path, monkeypatch):
    from poolhouse.workspace.automatic_connection import settle

    original = discovery.Membership("development", KEY)
    chosen = discovery.Membership("chosen-dev", b"b" * 43, selection="manual")
    monkeypatch.setattr(automatic, "ensure", lambda *args, **kwargs: chosen)
    monkeypatch.setattr(automatic, "offers", lambda port: pytest.fail("manual selection discovered another cluster"))
    assert settle(original, tmp_path / "device.key") == chosen
