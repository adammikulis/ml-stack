"""The download pipeline against a real local HTTP server: staging, hashes, formats, resumption,
redirects, limits, scanning and the quarantine of whatever fails."""

import hashlib
import io
import struct
import tarfile
import zipfile
from pathlib import Path

import pytest

from ml_stack import files, home, net
from ml_stack.httpguard import Limits, Refused, TooLarge
from ml_stack.net import provenance
from ml_stack.net.hold import staging_dir
from ml_stack.net.scan import Outcome, ScanPolicy, ScanResult
from tests.net_site import EICAR, Site, gguf_bytes, safetensors_bytes

SHA = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731


class Fake:
    """A scanner that answers what it was told to."""

    def __init__(self, outcome=Outcome.CLEAN, name="fake"):
        self.outcome, self.name, self.seen = outcome, name, []

    def available(self):
        return True

    def scan(self, path):
        self.seen.append(Path(path).name)
        return ScanResult(self.name, self.outcome, "told so")


class Eicar:
    """A scanner that matches the EICAR test string in the file's bytes."""

    name = "eicar-matcher"

    def available(self):
        return True

    def scan(self, path):
        hit = EICAR.encode() in Path(path).read_bytes()
        return ScanResult(self.name, Outcome.INFECTED if hit else Outcome.CLEAN, "", "Eicar-Test")


@pytest.fixture
def site():
    with Site() as s:
        yield s


@pytest.fixture
def pipe(tmp_path, site):
    return net.Pipeline(
        policy=net.Policy(allowed=["127.0.0.1"], path=tmp_path / "approvals.jsonl"),
        limits=Limits(allow_hosts=frozenset({"127.0.0.1"}), timeout=2.0, deadline_s=8.0),
        scanners=[Fake()], scan_policy=ScanPolicy())


def pull(pipe, site, path, name, want=None, **hooks):
    return net.download(site.base + path, name, want, pipe, net.Hooks(**hooks))


def test_a_good_gguf_is_staged_checked_scanned_and_promoted(tmp_path, site, pipe):
    body = gguf_bytes(extra=4096)
    site.add("/m.gguf", body, headers={"Content-Type": "application/octet-stream"})
    out = pull(pipe, site, "/m.gguf", tmp_path / "models" / "m.gguf", net.Want(sha256=SHA(body)))
    assert (tmp_path / "models" / "m.gguf").read_bytes() == body
    assert out.sha256 == SHA(body) and out.size == len(body) and out.kind == "gguf"
    assert not list(staging_dir().glob("*__m.gguf"))
    record = files.read_json(provenance.sidecar(tmp_path / "models" / "m.gguf"), {})
    assert record["url"] == site.base + "/m.gguf" and record["sha256"] == SHA(body)
    assert record["headers"]["content-type"] == "application/octet-stream"
    assert "set-cookie" not in record["headers"]
    assert record["host_source"] == "127.0.0.1" and record["version"] == provenance.VERSION
    assert provenance.recent(1)[0]["sha256"] == SHA(body)


def test_nothing_reaches_the_final_path_before_the_checks_pass(tmp_path, site, pipe):
    body = gguf_bytes()
    site.add("/m.gguf", body)
    where = tmp_path / "m.gguf"
    seen = []

    class Looks(Fake):
        def scan(self, path):
            seen.append(where.exists())
            return super().scan(path)

    pipe.scanners = [Looks()]
    pipe.scan_policy = ScanPolicy(scan_models=True)
    pull(pipe, site, "/m.gguf", where)
    assert seen == [False] and where.exists()


def test_the_staging_directory_is_private(tmp_path):
    assert staging_dir().stat().st_mode & 0o777 == 0o700


def test_a_pinned_hash_that_does_not_match_holds_the_file_and_promotes_nothing(tmp_path, site, pipe):
    body = gguf_bytes()
    site.add("/m.gguf", body)
    with pytest.raises(net.ChecksumMismatch) as caught:
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(sha256="0" * 64))
    assert not (tmp_path / "m.gguf").exists()
    assert caught.value.held and Path(caught.value.held).read_bytes() == body
    row = provenance.recent(1)[0]
    assert row["outcome"] == "held" and "sha256" in row["reason"]


def test_a_download_that_must_be_pinned_is_refused_before_any_request(tmp_path, site, pipe):
    route = site.add("/tool", b"x")
    with pytest.raises(net.NoDigest):
        pull(pipe, site, "/tool", tmp_path / "tool", net.Want(require_digest=True))
    assert route.seen == []


def test_a_gguf_name_on_other_bytes_is_held(tmp_path, site, pipe):
    site.add("/m.gguf", b"<html>not a model</html>", headers={"Content-Type": "text/html"})
    with pytest.raises(net.Blocked) as caught:
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    assert "GGUF" in str(caught.value) or "served as" in str(caught.value)
    assert not (tmp_path / "m.gguf").exists()


def test_a_safetensors_header_is_parsed_without_loading_tensors(tmp_path, site, pipe):
    good = safetensors_bytes()
    bad = safetensors_bytes({"w": ("F32", [4], 0, 9999)})
    site.add("/g.safetensors", good)
    site.add("/b.safetensors", bad)
    pull(pipe, site, "/g.safetensors", tmp_path / "g.safetensors")
    with pytest.raises(net.Blocked, match="outside the data"):
        pull(pipe, site, "/b.safetensors", tmp_path / "b.safetensors")


def test_pickle_weights_are_refused(tmp_path, site, pipe):
    site.add("/w.pt", b"\x80\x04\x95" + b"\0" * 64)
    with pytest.raises(net.Blocked, match="pickle"):
        pull(pipe, site, "/w.pt", tmp_path / "w.pt")


def test_a_pdf_must_be_a_pdf(tmp_path, site, pipe):
    site.add("/a.pdf", b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    site.add("/b.pdf", b"MZ\x90\x00 this is a program")
    site.add("/c.pdf", b"%PDF-1.7\nstream cut off")
    pull(pipe, site, "/a.pdf", tmp_path / "a.pdf")
    with pytest.raises(net.Blocked):
        pull(pipe, site, "/b.pdf", tmp_path / "b.pdf")
    with pytest.raises(net.Blocked, match="cut short"):
        pull(pipe, site, "/c.pdf", tmp_path / "c.pdf")


def test_a_pdf_with_active_content_is_kept_with_a_warning(tmp_path, site, pipe):
    site.add("/a.pdf", b"%PDF-1.7\n<< /OpenAction << /S /JavaScript /JS (x) >> >>\n%%EOF\n")
    out = pull(pipe, site, "/a.pdf", tmp_path / "a.pdf")
    assert any("/JavaScript" in w for w in out.warnings)


def _zip(entries):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return out.getvalue()


def _tar(members):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tf:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            if kind == "link":
                info.type, info.linkname = tarfile.SYMTYPE, data
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    return out.getvalue()


@pytest.mark.parametrize("case", [
    ("bomb.zip", _zip({"z": b"\0" * 40_000_000}), "compresses"),
    ("nested.zip", _zip({"inner.zip": b"PK\x05\x06" + b"\0" * 18}), "inside the archive"),
    ("abs.zip", _zip({"/etc/cron.d/x": b"boom"}), "plain relative path"),
    ("climb.zip", _zip({"../../x": b"boom"}), "plain relative path"),
    ("link.tar", _tar([("l", "link", "/etc/passwd")]), "link"),
    ("nested.tar", _tar([("a/inner.tar", "file", b"x")]), "inside the archive"),
])
def test_hostile_archives_are_held(tmp_path, site, pipe, case):
    name, body, why = case
    site.add("/" + name, body)
    pipe.scan_policy = ScanPolicy(archive="allow")
    with pytest.raises(net.Blocked, match=why):
        pull(pipe, site, "/" + name, tmp_path / name)
    assert not (tmp_path / name).exists()


def test_a_plain_archive_is_kept(tmp_path, site, pipe):
    site.add("/ok.zip", _zip({"a/b.txt": b"hello"}))
    pipe.scanners = [Fake()]
    assert pull(pipe, site, "/ok.zip", tmp_path / "ok.zip").kind == "zip"


def test_eicar_is_held_by_a_scanner_that_matches_it(tmp_path, site, pipe):
    site.add("/t.txt", EICAR.encode())
    pipe.scanners = [Eicar()]
    with pytest.raises(net.Blocked, match="infected") as caught:
        pull(pipe, site, "/t.txt", tmp_path / "t.txt")
    assert not (tmp_path / "t.txt").exists()
    assert Path(caught.value.held).read_bytes() == EICAR.encode()


def test_an_infected_file_is_never_kept_whatever_the_policy(tmp_path, site, pipe):
    site.add("/t.txt", b"harmless text")
    pipe.scanners = [Fake(Outcome.INFECTED)]
    pipe.scan_policy = ScanPolicy(data="allow")
    with pytest.raises(net.Blocked):
        pull(pipe, site, "/t.txt", tmp_path / "t.txt", net.Want(allow_unscanned=True))


def test_unscanned_models_are_kept_with_a_warning_and_unscanned_archives_are_not(tmp_path, site, pipe):
    site.add("/m.gguf", gguf_bytes())
    site.add("/a.zip", _zip({"x": b"y"}))
    pipe.scanners = []
    kept = pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    assert kept.scan.startswith("scan skipped: model weights")
    pipe.scan_policy = ScanPolicy(scan_models=True)
    asked = pull(pipe, site, "/m.gguf", tmp_path / "m2.gguf")
    assert asked.scan.startswith("scanned: no scanner available")
    assert "weights cannot be judged" in asked.scan
    pipe.scan_policy = ScanPolicy()
    with pytest.raises(net.Blocked, match="not kept unscanned"):
        pull(pipe, site, "/a.zip", tmp_path / "a.zip")
    allowed = pull(pipe, site, "/a.zip", tmp_path / "a.zip", net.Want(allow_unscanned=True))
    assert allowed.scan.startswith("scanned: no scanner available")


def test_a_scanner_that_errors_is_not_a_clean_bill(tmp_path, site, pipe):
    site.add("/a.zip", _zip({"x": b"y"}))
    pipe.scanners = [Fake(Outcome.ERROR)]
    with pytest.raises(net.Blocked, match="error"):
        pull(pipe, site, "/a.zip", tmp_path / "a.zip")


def test_a_truncated_download_resumes_from_the_bytes_already_there(tmp_path, site, pipe):
    body = gguf_bytes(extra=200_000)
    route = site.add("/m.gguf", body, headers={"ETag": '"v1"'}, ranges=True, cut_at=50_000)
    with pytest.raises(net.Truncated):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    assert not (tmp_path / "m.gguf").exists()
    assert any(p.stat().st_size == 50_000 for p in staging_dir().glob("*.part"))
    route.cut_at = None
    out = pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(sha256=SHA(body)))
    assert (tmp_path / "m.gguf").read_bytes() == body and out.sha256 == SHA(body)
    assert route.seen[-1]["range"] == "bytes=50000-" and route.seen[-1]["if-range"] == '"v1"'


def test_a_server_that_ignores_range_restarts_the_file(tmp_path, site, pipe):
    body = gguf_bytes(extra=100_000)
    route = site.add("/m.gguf", body, headers={"ETag": '"v1"'}, cut_at=30_000)
    with pytest.raises(net.Truncated):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    route.cut_at = None
    pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(sha256=SHA(body)))
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_a_resumed_file_that_changed_on_the_server_fails_its_hash(tmp_path, site, pipe):
    old = gguf_bytes(extra=100_000)
    new = bytes([old[0] ^ 0]) + old[1:-1] + b"Z"
    route = site.add("/m.gguf", old, headers={"ETag": '"v1"'}, ranges=True, cut_at=30_000)
    with pytest.raises(net.Truncated):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    route.body, route.cut_at = new, None
    with pytest.raises(net.ChecksumMismatch):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(sha256=SHA(new)[:-1] + "0"))


def test_an_endless_body_stops_at_the_cap_and_leaves_nothing_behind(tmp_path, site, pipe):
    site.add("/forever.gguf", endless=True)
    with pytest.raises(TooLarge):
        pull(pipe, site, "/forever.gguf", tmp_path / "f.gguf", net.Want(max_bytes=500_000))
    assert not (tmp_path / "f.gguf").exists()
    assert not list(staging_dir().glob("*.part*"))


def test_a_declared_size_over_the_cap_is_refused_before_the_body(tmp_path, site, pipe):
    site.add("/big.gguf", gguf_bytes(extra=300_000))
    with pytest.raises(TooLarge):
        pull(pipe, site, "/big.gguf", tmp_path / "b.gguf", net.Want(max_bytes=1000))


def test_a_slow_loris_server_hits_the_deadline(tmp_path, site, pipe):
    site.add("/slow.gguf", gguf_bytes(extra=300), drip_s=0.05)
    with pytest.raises(Refused, match="out of time"):
        pull(pipe, site, "/slow.gguf", tmp_path / "s.gguf", net.Want(deadline_s=1.0))


def test_a_host_that_is_not_listed_needs_a_recorded_approval(tmp_path, site, pipe):
    site.add("/m.gguf", gguf_bytes())
    pipe.policy = net.Policy(allowed=["example.org"], path=tmp_path / "ap.jsonl")
    with pytest.raises(net.NeedsApproval) as caught:
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    assert caught.value.host == "127.0.0.1"
    pipe.policy.approve("127.0.0.1", by="test person", note="local site")
    pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    row = pipe.policy.approvals()[0]
    assert (row.host, row.by, row.note) == ("127.0.0.1", "test person", "local site")


def test_an_approval_can_lapse(tmp_path):
    clock = [1000.0]
    policy = net.Policy(allowed=[], path=tmp_path / "ap.jsonl", clock=lambda: clock[0])
    policy.approve("a.example", by="p", for_s=60)
    assert policy.approved("a.example")
    clock[0] += 61
    assert not policy.approved("a.example")


def test_the_policy_is_asked_again_at_every_redirect_hop(tmp_path, site):
    answers = {"first.test": ["127.0.0.1"], "second.test": ["127.0.0.1"]}
    pipe = net.Pipeline(
        policy=net.Policy(allowed=["first.test"], path=tmp_path / "a.jsonl"),
        limits=Limits(allow_hosts=frozenset({"first.test", "second.test"}),
                      resolver=lambda host, port: answers[host], timeout=2.0, deadline_s=6.0),
        scanners=[])
    target = site.add("/m.gguf", gguf_bytes())
    site.redirect("/go", f"http://second.test:{site.port}/m.gguf")
    with pytest.raises(net.NeedsApproval) as caught:
        net.download(f"http://first.test:{site.port}/go", tmp_path / "m.gguf", None, pipe)
    assert caught.value.host == "second.test" and target.seen == []
    pipe.policy.approve("second.test", by="person")
    net.download(f"http://first.test:{site.port}/go", tmp_path / "m.gguf", None, pipe)
    assert (tmp_path / "m.gguf").is_file()


def test_a_kept_file_can_be_read_by_other_programs(tmp_path, site, pipe):
    site.add("/m.gguf", gguf_bytes())
    pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    assert (tmp_path / "m.gguf").stat().st_mode & 0o044 == 0o044


def test_a_redirect_to_a_host_the_policy_does_not_admit_is_refused(tmp_path, site, pipe):
    site.redirect("/go", f"http://localhost:{site.port}/m.gguf")
    site.add("/m.gguf", gguf_bytes())
    with pytest.raises(Refused):
        pull(pipe, site, "/go", tmp_path / "m.gguf")


@pytest.mark.parametrize("target", [
    "http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/x", "http://127.0.0.2/x",
    "http://[::1]/x", "http://192.168.1.1/x", "file:///etc/passwd",
])
def test_a_redirect_into_a_private_or_metadata_address_is_refused(tmp_path, site, pipe, target):
    pipe.policy = net.Policy(allowed=["127.0.0.1", "169.254.169.254", "10.0.0.5", "127.0.0.2",
                                      "::1", "192.168.1.1"], path=tmp_path / "ap.jsonl")
    site.redirect("/go", target)
    with pytest.raises(Refused):
        pull(pipe, site, "/go", tmp_path / "x.gguf")


def test_every_hop_is_resolved_once_and_connected_to_that_answer(tmp_path, site):
    calls = []

    def resolver(host, port):
        calls.append(host)
        return ["127.0.0.1"] if len(calls) == 1 else ["169.254.169.254"]

    pipe = net.Pipeline(
        policy=net.Policy(allowed=["rebind.test"], path=tmp_path / "ap.jsonl"),
        limits=Limits(allow_hosts=frozenset({"rebind.test"}), resolver=resolver, timeout=2.0,
                      deadline_s=6.0), scanners=[Fake()])
    site.add("/m.gguf", gguf_bytes())
    out = net.download(f"http://rebind.test:{site.port}/m.gguf", tmp_path / "m.gguf", None, pipe)
    assert out.size and calls == ["rebind.test"]


def test_a_rebind_to_a_private_address_on_the_second_hop_is_refused(tmp_path, site):
    answers = {"first.test": ["127.0.0.1"], "second.test": ["169.254.169.254"]}
    pipe = net.Pipeline(
        policy=net.Policy(allowed=["first.test", "second.test"], path=tmp_path / "ap.jsonl"),
        limits=Limits(allow_hosts=frozenset({"first.test"}),
                      resolver=lambda host, port: answers[host], timeout=2.0, deadline_s=6.0),
        scanners=[Fake()])
    site.redirect("/go", f"http://second.test:{site.port}/m.gguf")
    with pytest.raises(Refused, match="not on the public internet"):
        net.download(f"http://first.test:{site.port}/go", tmp_path / "m.gguf", None, pipe)


def test_credentials_and_cookies_do_not_cross_hosts(tmp_path, site):
    answers = {"a.test": ["127.0.0.1"], "b.test": ["127.0.0.1"]}
    pipe = net.Pipeline(
        policy=net.Policy(allowed=["a.test", "b.test"], path=tmp_path / "ap.jsonl"),
        limits=Limits(allow_hosts=frozenset({"a.test", "b.test"}),
                      resolver=lambda host, port: answers[host], timeout=2.0, deadline_s=6.0),
        scanners=[Fake()])
    first = site.redirect("/go", f"http://b.test:{site.port}/m.gguf")
    target = site.add("/m.gguf", gguf_bytes())
    sent = {"Authorization": "Bearer secret", "Cookie": "session=abc"}
    net.download(f"http://a.test:{site.port}/go", tmp_path / "m.gguf", net.Want(headers=sent), pipe)
    assert first.seen[-1]["authorization"] == "Bearer secret"
    assert "cookie" not in target.seen[-1] and "authorization" not in target.seen[-1]


def test_a_token_is_never_sent_over_plain_http_to_the_internet(tmp_path, pipe):
    pipe.policy = net.Policy(allowed=["example.org"], path=tmp_path / "a.jsonl")
    with pytest.raises(Refused, match="https only"):
        net.download("http://example.org/m.gguf", tmp_path / "m.gguf", net.Want(token="secret"),
                     pipe)
    with pytest.raises(Refused, match="https only"):
        pipe.get("http://example.org/x", net.Ask(token="secret"))


def test_a_token_goes_to_a_private_mirror_the_person_named(tmp_path, site, pipe):
    route = site.add("/m.gguf", gguf_bytes())
    pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(token="secret"))
    assert route.seen[0]["authorization"] == "Bearer secret"


def test_a_name_that_escapes_is_refused(tmp_path, site, pipe):
    site.add("/m.gguf", gguf_bytes())
    from ml_stack.safenames import Unsafe

    with pytest.raises(Unsafe):
        net.download(site.base + "/m.gguf", tmp_path / "nul", None, pipe)


def test_failures_are_moved_into_sentinels_quarantine_with_an_event(tmp_path, site, pipe):
    from ml_stack import sentinel

    body = gguf_bytes()
    site.add("/m.gguf", body)
    with pytest.raises(net.ChecksumMismatch):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(sha256="1" * 64))
    records = sentinel.default().store.records(kind="artifact")
    assert records and records[0].state.value == "quarantined"
    assert home.home().resolve() in Path(records[0].action["held"]).resolve().parents
    log = sentinel.default().bus.log.path.read_text(encoding="utf-8")
    assert "quarantine" in log


def test_stale_partial_files_are_swept(tmp_path):
    import os
    import time

    part = staging_dir() / "old.part"
    part.write_bytes(b"x")
    os.utime(part, (time.time() - 30 * 86400,) * 2)
    fresh = staging_dir() / "new.part"
    fresh.write_bytes(b"x")
    from ml_stack.net.download import sweep

    assert sweep() == 1 and fresh.exists() and not part.exists()


def test_a_server_that_resumes_at_the_wrong_offset_is_not_trusted(tmp_path, site, pipe):
    body = gguf_bytes(extra=100_000)
    route = site.add("/m.gguf", body, headers={"ETag": '"v1"'}, ranges=True, cut_at=30_000)
    with pytest.raises(net.Truncated):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    route.cut_at, route.shift = None, 7
    with pytest.raises(net.Truncated, match="another offset"):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    assert not list(staging_dir().glob("*.part"))
    route.shift = 0
    pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(sha256=SHA(body)))
    assert (tmp_path / "m.gguf").read_bytes() == body


def test_a_file_of_another_size_than_the_manifest_lists_is_held(tmp_path, site, pipe):
    body = gguf_bytes(extra=1000)
    site.add("/m.gguf", body)
    with pytest.raises(net.Blocked, match="expected 5"):
        pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(size=5))
    assert not (tmp_path / "m.gguf").exists()
    assert pull(pipe, site, "/m.gguf", tmp_path / "m.gguf", net.Want(size=len(body)))


def test_a_set_cookie_header_is_never_kept_in_the_record(tmp_path, site, pipe):
    site.add("/m.gguf", gguf_bytes(), headers={"Set-Cookie": "session=secret", "ETag": '"v"',
                                               "Server": "test"})
    done = pull(pipe, site, "/m.gguf", tmp_path / "m.gguf")
    assert "set-cookie" not in done.headers and "secret" not in str(done.headers)
    assert done.headers["etag"] == '"v"' and done.headers["server"] == "test"


def test_other_ways_a_file_can_be_the_wrong_thing(tmp_path, site, pipe):
    garbage = b"XXXX" + struct.pack("<IQQ", 3, 0, 0) + b"\0" * 40
    site.add("/wrong-magic.gguf", garbage)
    site.add("/future.gguf", b"GGUF" + struct.pack("<IQQ", 99, 0, 0) + b"\0" * 40)
    site.add("/prog.dat", b"\x7fELF\x02\x01\x01" + b"\0" * 64)
    site.add("/note.pdf", b"just some text, no header, and %%EOF\n")
    both = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
    site.add("/page.pdf", both, headers={"Content-Type": "text/html; charset=utf-8"})
    for name, why in (("wrong-magic.gguf", "no GGUF magic"), ("future.gguf", "version 99"),
                      ("prog.dat", "native executable"), ("note.pdf", "no %PDF- header"),
                      ("page.pdf", "served as text/html")):
        with pytest.raises(net.Blocked, match=why):
            pull(pipe, site, "/" + name, tmp_path / name)
        assert not (tmp_path / name).exists()


def test_overlapping_safetensors_tensors_are_refused(tmp_path, site, pipe):
    site.add("/o.safetensors", safetensors_bytes(
        {"a": ("F32", [4], 0, 16), "b": ("F32", [4], 8, 24)}, data=24))
    with pytest.raises(net.Blocked, match="overlap"):
        pull(pipe, site, "/o.safetensors", tmp_path / "o.safetensors")


def test_an_archive_over_the_entry_or_size_limit_is_refused(tmp_path):
    from ml_stack.net.sniff import audit_archive

    path = tmp_path / "many.zip"
    path.write_bytes(_zip({f"f{n}": b"x" * 10 for n in range(6)}))
    assert audit_archive(path) == []
    assert any("entries" in p for p in audit_archive(path, max_entries=3))
    assert any("unpacks to" in p for p in audit_archive(path, max_bytes=30))
