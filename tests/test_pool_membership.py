"""Each device is its own certificate; a cluster lists the devices in it; a device put out is refused."""

import base64
import http.client as httpclient
import json
import ssl
import threading
from dataclasses import dataclass

import pytest

from ml_stack import http, macauth
from ml_stack.fleet import discovery, membership, tls
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.discovery import Beacon, derive_token, memberships
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.pool_roster import Pool

SPARE = macauth.PREFIX + "a-secret-no-cluster-key-made"
"""A request secret the daemon accepts that is derived from no cluster, so only the certificate decides."""
GONE = (ssl.SSLError, OSError, httpclient.HTTPException, http.ServerUnreachable)


@pytest.fixture(autouse=True)
def forget_pins():
    yield
    http._PINNED.clear()


@dataclass
class Device:
    ident: tls.Identity
    name: str

    @property
    def cert(self) -> str:
        return self.ident.beacon


def device(tmp_path, name):
    return Device(tls.identity(tmp_path / name / "tls", name), name)


@dataclass
class Served:
    server: Device
    port: int
    keyfile: object
    pool: Pool
    token: str
    group: str = "lab"

    def connect(self, who: Device | None, *, ceiling=None) -> httpclient.HTTPSConnection:
        context = tls.pinned_context(self.server.cert, who=who.ident) if who else _anonymous(self.server.cert)
        if ceiling is not None:
            context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
            context.maximum_version = ceiling
        return httpclient.HTTPSConnection("127.0.0.1", self.port, context=context, timeout=10)

    def get(self, conn, path, *, secret=None):
        url = f"https://127.0.0.1:{self.port}{path}"
        conn.request("GET", path, headers=macauth.sign(secret or self.token, "GET", url, None))
        reply = conn.getresponse()
        return reply.status, json.loads(reply.read() or b"{}")

    def post(self, caller: Device | None, path, payload):
        """A signed, sealed POST through the library's own client, pinned to the server."""
        context = tls.pinned_context(self.server.cert, who=caller.ident) if caller else _anonymous(self.server.cert)
        http.pin(f"127.0.0.1:{self.port}", context)
        try:
            return 200, http.request_json(f"https://127.0.0.1:{self.port}{path}", payload=payload, token=self.token)
        except http.ServerError as error:
            if error.status is None:
                raise
            return error.status, json.loads(error.body) if error.body.startswith("{") else {}


def _anonymous(server_cert: str) -> ssl.SSLContext:
    """A client that pins the server and shows no certificate of its own."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(base64.b64decode(server_cert)))
    return ctx


@pytest.fixture
def served(tmp_path):
    """A real fleet daemon handler behind its real member context, on TLS, in a temp root."""
    keyfile = tmp_path / "cluster.key"
    key = discovery.mint_cluster("lab", keyfile, mode="prod").key
    server = device(tmp_path, "server")
    pool = Pool(keyfile)
    pool.enrol("lab", server.cert, "server", "self")
    root = tmp_path / "daemon"
    (root / "files").mkdir(parents=True)
    runner = JobRunner(root)
    daemon = Daemon(runner, root / "files", derive_token(key), name="server", cluster_key_path=keyfile,
                    tokens=lambda: {derive_token(m.key) for m in memberships(keyfile)} | {SPARE}, members=pool)
    handler = make_handler(daemon)
    handler.protocol_version = "HTTP/1.1"
    httpd = LimitedServer(("127.0.0.1", 0), handler, tls=tls.member_context(server.ident, pool))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Served(server, httpd.server_port, keyfile, pool, derive_token(key))
    runner.shutdown()
    httpd.shutdown()
    httpd.server_close()


def enrolled(served, tmp_path, name):
    one = device(tmp_path, name)
    served.pool.enrol(served.group, one.cert, name, "test")
    return one


# -- who is who -----------------------------------------------------------------------------


def test_two_devices_with_different_keys_are_told_apart(served, tmp_path):
    a, b = enrolled(served, tmp_path, "alpha"), enrolled(served, tmp_path, "beta")
    assert a.ident.fingerprint != b.ident.fingerprint and a.ident.der != b.ident.der
    seen = {}
    for one in (a, b):
        conn = served.connect(one)
        status, answer = served.get(conn, "/fleet/v1/self")
        assert status == 200
        seen[one.name] = answer
        conn.close()
    assert seen["alpha"]["fingerprint"] == a.ident.fingerprint
    assert seen["beta"]["fingerprint"] == b.ident.fingerprint
    assert seen["alpha"]["groups"] == ["lab"]


def test_a_device_that_was_never_paired_is_refused(served, tmp_path):
    stranger = device(tmp_path, "stranger")
    conn = served.connect(stranger)
    with pytest.raises(GONE):
        served.get(conn, "/jobs")


def test_a_caller_with_no_certificate_cannot_use_the_cluster_secret(served):
    conn = served.connect(None)
    status, answer = served.get(conn, "/jobs")
    assert status == 401 and "certificate" in answer["error"]
    status, _ = served.get(conn, "/workspace/v1/projects")
    assert status == 401


def test_a_member_of_another_cluster_cannot_use_this_clusters_secret(served, tmp_path):
    other = discovery.mint_cluster("other", served.keyfile, mode="prod")
    one = enrolled(served, tmp_path, "alpha")
    conn = served.connect(one)
    status, _ = served.get(conn, "/fleet/v1/self", secret=derive_token(other.key))
    assert status == 403
    conn = served.connect(one)
    assert served.get(conn, "/fleet/v1/self")[0] == 200


# -- revocation -----------------------------------------------------------------------------


def test_a_revoked_device_is_refused_at_its_next_request_on_a_live_session(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    conn = served.connect(one)
    assert served.get(conn, "/fleet/v1/self")[0] == 200
    assert served.pool.revoke(one.ident.fingerprint, "owner") == ["lab"]
    status, answer = served.get(conn, "/fleet/v1/self")
    assert status == 403 and "not a member" in answer["error"]


def test_the_certificate_alone_refuses_a_revoked_device_when_the_secret_names_no_cluster(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    conn = served.connect(one)
    assert served.get(conn, "/fleet/v1/self", secret=SPARE)[0] == 200
    served.pool.revoke(one.ident.fingerprint, "owner")
    status, answer = served.get(conn, "/fleet/v1/self", secret=SPARE)
    assert status == 403 and "not a member" in answer["error"]


def test_a_revoked_device_fails_the_next_handshake_too(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    first = served.connect(one)
    assert served.get(first, "/fleet/v1/self")[0] == 200
    served.pool.revoke(one.ident.fingerprint, "owner")
    with pytest.raises(GONE):
        served.get(served.connect(one), "/fleet/v1/self")


def test_a_revoked_device_is_out_everywhere_it_is_checked(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    beacon = Beacon(name="alpha", host="127.0.0.1", port=1234, cert=one.cert)
    devices = membership.roster(memberships(served.keyfile)[0].key)
    assert discovery._trusted(beacon, devices) is True and "127.0.0.1:1234" in http._PINNED
    served.pool.revoke(one.ident.fingerprint, "owner")
    assert not devices.is_active(one.ident.fingerprint)
    assert served.pool.groups_of(one.ident.fingerprint) == frozenset()
    assert one.ident.fingerprint not in {membership.fingerprint_of(c) for c in _pem_certs(served.pool.certs_pem())}
    assert discovery._trusted(beacon, devices) is False and "127.0.0.1:1234" not in http._PINNED
    with pytest.raises(GONE):
        served.get(served.connect(one), "/fleet/v1/members")


def _pem_certs(pem: str) -> list[str]:
    out, block = [], []
    for line in pem.splitlines():
        block.append(line)
        if line.startswith("-----END"):
            out.append(base64.b64encode(ssl.PEM_cert_to_DER_cert("\n".join(block))).decode())
            block = []
    return out


def test_a_revoked_certificate_cannot_be_enrolled_again(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    served.pool.revoke(one.ident.fingerprint, "owner")
    with pytest.raises(membership.Revoked):
        served.pool.enrol("lab", one.cert, "alpha", "pairing")
    assert not served.pool.is_active(one.ident.fingerprint)


def test_a_stale_peer_cannot_bring_a_revoked_device_back(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    stale = [d.public() for d in served.pool.roster("lab").devices()]
    served.pool.revoke(one.ident.fingerprint, "owner")
    assert served.pool.merge("lab", stale, "peer") == 0
    later = [{**row, "at": row["at"] + 3600} for row in stale]
    assert served.pool.merge("lab", later, "peer") == 0
    assert not served.pool.is_active(one.ident.fingerprint)


def test_a_revocation_heard_from_a_member_is_applied_and_kept(served, tmp_path):
    one, spreader = enrolled(served, tmp_path, "alpha"), enrolled(served, tmp_path, "beta")
    revoked = {**served.pool.roster("lab").get(one.ident.fingerprint).public(), "status": "revoked"}
    status, answer = served.post(spreader, "/fleet/v1/members", {"group": "lab", "rows": [revoked]})
    assert status == 200 and any(r["status"] == "revoked" for r in answer["rows"])
    assert not served.pool.is_active(one.ident.fingerprint)
    with pytest.raises(GONE):
        served.get(served.connect(one), "/fleet/v1/self")


def test_only_a_member_of_the_cluster_exchanges_membership(served, tmp_path):
    stranger = device(tmp_path, "stranger")
    with pytest.raises(GONE):
        served.post(stranger, "/fleet/v1/members", {"group": "lab", "rows": []})
    status, _ = served.post(None, "/fleet/v1/members", {"group": "lab", "rows": []})
    assert status == 401
    other = discovery.mint_cluster("elsewhere", served.keyfile, mode="prod")
    one = enrolled(served, tmp_path, "alpha")
    status, _ = served.post(one, "/fleet/v1/members", {"group": "elsewhere", "rows": []})
    assert status == 403 and other.group == "elsewhere"


# -- the floor and the pin ------------------------------------------------------------------


def test_a_downgrade_to_tls_12_is_refused_even_for_a_member(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    conn = served.connect(one, ceiling=ssl.TLSVersion.TLSv1_2)
    with pytest.raises(GONE):
        served.get(conn, "/fleet/v1/self")


def test_a_client_that_pins_another_certificate_than_the_servers_is_refused(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    impostor = device(tmp_path, "impostor")
    context = tls.pinned_context(impostor.cert, who=one.ident)
    conn = httpclient.HTTPSConnection("127.0.0.1", served.port, context=context, timeout=10)
    with pytest.raises(GONE):
        served.get(conn, "/fleet/v1/self")


def test_a_pin_with_a_byte_changed_is_not_the_server(served, tmp_path):
    one = enrolled(served, tmp_path, "alpha")
    der = bytearray(served.server.ident.der)
    der[len(der) // 2] ^= 1
    with pytest.raises((ssl.SSLError, ValueError)):
        context = tls.pinned_context(base64.b64encode(bytes(der)).decode(), who=one.ident)
        conn = httpclient.HTTPSConnection("127.0.0.1", served.port, context=context, timeout=10)
        served.get(conn, "/fleet/v1/self")


def test_a_roster_row_whose_fingerprint_is_not_its_certificates_is_dropped(served, tmp_path):
    one, other = device(tmp_path, "alpha"), device(tmp_path, "other")
    roster = membership.roster(memberships(served.keyfile)[0].key)
    roster.enrol(other.cert, "other", "test")
    raw = json.loads(roster.file.read_text())
    for row in raw["devices"]:
        if row["fingerprint"] == other.ident.fingerprint:
            row["cert"] = one.cert
    roster.file.write_text(json.dumps(raw))
    assert not roster.is_active(other.ident.fingerprint) and not roster.is_active(one.ident.fingerprint)
    with pytest.raises(GONE):
        served.get(served.connect(one), "/fleet/v1/self")
    forged = [{"fingerprint": one.ident.fingerprint, "cert": other.cert, "status": "active", "name": "x"}]
    assert served.pool.merge("lab", forged, "peer") == 0


def test_a_beacon_whose_certificate_was_never_enrolled_is_not_pinned(served, tmp_path):
    stranger = device(tmp_path, "stranger")
    devices = membership.roster(memberships(served.keyfile)[0].key)
    beacon = Beacon(name="stranger", host="127.0.0.1", port=4321, cert=stranger.cert)
    assert discovery._trusted(beacon, devices) is False and "127.0.0.1:4321" not in http._PINNED
    assert discovery._trusted(beacon, None) is True
