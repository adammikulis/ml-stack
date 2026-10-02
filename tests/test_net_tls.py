"""TLS is verified and never stepped down: a certificate the system does not trust ends the
request, and a redirect from HTTPS to HTTP is refused."""

import base64
import ssl

import pytest

from ml_stack import net
from ml_stack.fleet import tls
from ml_stack.httpguard import Limits, Refused
from tests.net_site import Site, gguf_bytes


@pytest.fixture
def secure(tmp_path):
    ident = tls.identity(tmp_path / "tls", "127.0.0.1")
    context = tls.server_context(ident)
    trusting = tls.pinned_context(base64.b64encode(ident.der).decode())
    return context, trusting


def pipeline(tmp_path, context=None, hosts=("127.0.0.1",)):
    return net.Pipeline(
        policy=net.Policy(allowed=list(hosts), path=tmp_path / "a.jsonl"),
        limits=Limits(allow_hosts=frozenset(hosts), context=context, timeout=3.0, deadline_s=10.0),
        scanners=[])


def test_a_certificate_the_system_does_not_trust_ends_the_request(tmp_path, secure):
    context, _trusting = secure
    with Site(context) as site:
        route = site.add("/m.gguf", gguf_bytes())
        with pytest.raises(Refused):
            net.download(site.base + "/m.gguf", tmp_path / "m.gguf", None, pipeline(tmp_path))
    assert route.seen == []
    assert not (tmp_path / "m.gguf").exists()


def test_a_certificate_the_caller_pinned_is_accepted(tmp_path, secure):
    context, trusting = secure
    with Site(context) as site:
        site.add("/m.gguf", gguf_bytes())
        done = net.download(site.base + "/m.gguf", tmp_path / "m.gguf", None,
                            pipeline(tmp_path, trusting))
    assert done.final_url.startswith("https://") and done.size


def test_a_redirect_from_https_to_http_is_refused(tmp_path, secure):
    context, trusting = secure
    with Site(context) as secure_site, Site() as plain:
        plain.add("/m.gguf", gguf_bytes())
        secure_site.redirect("/go", plain.base + "/m.gguf")
        with pytest.raises(Refused, match="https to plain http"):
            net.download(secure_site.base + "/go", tmp_path / "m.gguf", None,
                         pipeline(tmp_path, trusting))
        assert plain.routes["/m.gguf"].seen == []


def test_a_token_goes_over_tls_and_not_to_the_next_host(tmp_path, secure):
    context, trusting = secure
    answers = {"a.test": ["127.0.0.1"], "b.test": ["127.0.0.1"]}
    pipe = net.Pipeline(
        policy=net.Policy(allowed=["a.test", "b.test"], path=tmp_path / "a.jsonl"),
        limits=Limits(allow_hosts=frozenset({"a.test", "b.test"}), context=trusting,
                      resolver=lambda host, port: answers[host], timeout=3.0, deadline_s=10.0),
        scanners=[])
    with Site(context) as site:
        first = site.redirect("/go", f"https://b.test:{site.port}/m.gguf")
        last = site.add("/m.gguf", gguf_bytes())
        net.download(f"https://a.test:{site.port}/go", tmp_path / "m.gguf",
                     net.Want(token="secret", headers={"Cookie": "s=1"}), pipe)
    assert first.seen[0]["authorization"] == "Bearer secret"
    assert "authorization" not in last.seen[0] and "cookie" not in last.seen[0]


def test_the_default_context_verifies_certificates_and_hostnames():
    context = ssl.create_default_context()
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    assert Limits().context is None
