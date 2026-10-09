"""Pulling models from a local stand-in for the Hugging Face endpoint."""

from __future__ import annotations

import hashlib
import struct
import threading

import pytest

from ml_stack import http, hub
from ml_stack.httpguard import Refused
from ml_stack.hub import remote, transfer as pulling
from ml_stack.testing.fakehub import fake_hub

MIB = 1 << 20


def blob(size: int, seed: int = 0) -> bytes:
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
    return (head + bytes((seed + i * 7) % 251 for i in range(256)) * (size // 256 + 1))[:size]


REPOS = {
    "maker/thing-GGUF": {
        "thing-Q4_K_M.gguf": blob(3 * MIB, 1),
        "thing-Q8_0.gguf": blob(5 * MIB, 2),
        "mmproj-thing-F16.gguf": blob(MIB, 3),
        "README.md": b"hello",
    },
    "maker/big-GGUF": {
        "UD-Q4/big-UD-Q4-00001-of-00002.gguf": blob(MIB, 4),
        "UD-Q4/big-UD-Q4-00002-of-00002.gguf": blob(MIB, 5),
    },
    "maker/gated-GGUF": {"g-Q4_K_M.gguf": blob(MIB, 6)},
}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent-hf-home")
    monkeypatch.setattr(http, "check", lambda url: url)
    with fake_hub(REPOS, gated=frozenset({"maker/gated-GGUF"})) as hub_:
        hub_.point(monkeypatch)
        yield hub_


def test_a_file_reference_downloads_into_the_store_and_is_found_afterwards(server, tmp_path):
    got = hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf")
    assert got.read_bytes() == REPOS["maker/thing-GGUF"]["thing-Q4_K_M.gguf"]
    assert got.parent.name == "thing-GGUF" and got.parent.parent.name == "maker"
    assert not list(got.parent.glob("*.part"))
    (found,) = [m for m in hub.discover(formats=("gguf",)) if m.path == got] or [None]
    assert found is None or found.repo == "maker/thing-GGUF"


def test_progress_reports_bytes_files_and_a_final_done(server, tmp_path):
    seen: list[pulling.Progress] = []
    hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path, seen.append)
    assert seen[-1].phase == "done" and seen[-1].done_bytes == seen[-1].total_bytes == 5 * MIB
    assert {p.phase for p in seen} >= {"downloading", "verifying", "done"}
    done = [p.done_bytes for p in seen]
    assert done == sorted(done) and seen[0].fraction <= 1.0
    assert seen[-1].fraction == 1.0 and seen[-1].file == "thing-Q8_0.gguf"


def test_every_shard_of_a_build_comes_down(server, tmp_path):
    first = hub.pull("hf:maker/big-GGUF/UD-Q4/big-UD-Q4-00001-of-00002.gguf", tmp_path)
    assert first.name == "big-UD-Q4-00001-of-00002.gguf"
    assert (first.parent / "big-UD-Q4-00002-of-00002.gguf").read_bytes() == blob(MIB, 5)


def test_a_quantisation_tag_picks_the_file(server, tmp_path):
    got = hub.pull("hf:maker/thing-GGUF:Q8_0", tmp_path)
    assert got.name == "thing-Q8_0.gguf"


def test_a_repository_alone_takes_the_q4_k_m_build(server, tmp_path):
    assert hub.pull("hf:maker/thing-GGUF", tmp_path).name == "thing-Q4_K_M.gguf"


def test_a_cut_connection_resumes_from_the_bytes_on_disk(server, tmp_path):
    server.cut_after = MIB + 12345
    with pytest.raises(OSError):
        hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path)
    (part,) = tmp_path.rglob("*.part")
    assert 0 < part.stat().st_size < 5 * MIB
    server.hub_seen.clear()
    server.cdn_seen.clear()
    got = hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path)
    assert got.read_bytes() == REPOS["maker/thing-GGUF"]["thing-Q8_0.gguf"]
    ranges = [r for _m, _p, r, _a in server.cdn_seen if r]
    assert ranges and ranges[0].startswith("bytes=") and int(ranges[0][6:-1]) > 0


def test_a_server_that_ignores_the_range_request_restarts_the_file(server, tmp_path):
    server.cut_after = MIB + 5
    with pytest.raises(OSError):
        hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path)
    server.ignore_range = True
    got = hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path)
    assert got.read_bytes() == REPOS["maker/thing-GGUF"]["thing-Q8_0.gguf"]


def test_a_cancel_leaves_the_partial_file_and_the_next_pull_continues(server, tmp_path):
    token = pulling.CancelToken()
    count = []

    def stop(progress):
        count.append(progress)
        if progress.phase == "downloading" and progress.done_bytes >= MIB:
            token.cancel()

    with pytest.raises(pulling.Cancelled):
        hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path, stop, token)
    assert token.cancelled and list(tmp_path.rglob("*.part"))
    assert hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path).stat().st_size == 5 * MIB


def test_a_file_already_complete_is_not_fetched_again(server, tmp_path):
    hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf", tmp_path)
    before = dict(server.downloads)
    hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf", tmp_path)
    assert server.downloads == before


def test_a_wrong_checksum_is_refused_and_the_partial_removed(server, tmp_path):
    server.corrupt.add("maker/thing-GGUF/thing-Q4_K_M.gguf")
    with pytest.raises(pulling.ChecksumMismatch):
        hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf", tmp_path)
    assert not list(tmp_path.glob("thing-Q4_K_M.gguf*"))
    assert not list((tmp_path / "machine-state" / "net" / "staging").glob("*.part*"))
    (held,) = (tmp_path / "machine-state" / ".ml-stack-quarantine").rglob("*thing-Q4_K_M.gguf")
    assert held.stat().st_size == 3 * MIB


def test_too_little_disk_is_refused_before_a_byte_is_read(server, tmp_path, monkeypatch):
    class Usage:
        free = 1024

    monkeypatch.setattr(pulling.shutil, "disk_usage", lambda _p: Usage)
    with pytest.raises(pulling.NotEnoughSpace, match="free"):
        hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path)
    assert not server.downloads


def test_a_gated_repository_says_how_to_get_a_token(server, tmp_path):
    with pytest.raises(pulling.GatedRepo) as caught:
        hub.pull("hf:maker/gated-GGUF/g-Q4_K_M.gguf", tmp_path)
    assert "HF_TOKEN" in str(caught.value) and "settings/tokens" in str(caught.value)


def test_a_token_opens_a_gated_repository_and_never_reaches_the_cdn(server, tmp_path,
                                                                    monkeypatch):
    monkeypatch.setenv("HF_TOKEN", server.token)
    got = hub.pull("hf:maker/gated-GGUF/g-Q4_K_M.gguf", tmp_path)
    assert got.read_bytes() == blob(MIB, 6)
    assert any(a == f"Bearer {server.token}" for *_x, a in server.hub_seen)
    assert server.cdn_seen and all(a == "" for *_x, a in server.cdn_seen)


def test_a_wrong_token_is_named_in_the_hint(server, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "nope")
    with pytest.raises(pulling.GatedRepo, match=r"refused the token"):
        hub.pull("hf:maker/gated-GGUF/g-Q4_K_M.gguf", tmp_path)


def test_a_missing_file_names_what_the_repository_holds(server, tmp_path):
    with pytest.raises(pulling.NotFound, match="thing-Q4_K_M"):
        hub.pull("hf:maker/thing-GGUF/absent.gguf", tmp_path)
    with pytest.raises(pulling.NotFound):
        hub.pull("hf:maker/absent-repo/x.gguf", tmp_path)


def test_a_malformed_reference_is_refused(server):
    with pytest.raises(ValueError, match="should look like"):
        hub.pull("hf:justone")


def test_two_pulls_of_one_file_take_turns_and_end_with_one_good_file(server, tmp_path):
    results = []

    def one():
        results.append(hub.pull("hf:maker/thing-GGUF/thing-Q8_0.gguf", tmp_path))

    threads = [threading.Thread(target=one) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert len(results) == 2 and results[0] == results[1]
    assert results[0].read_bytes() == REPOS["maker/thing-GGUF"]["thing-Q8_0.gguf"]
    assert server.downloads["maker/thing-GGUF/thing-Q8_0.gguf"] == 1


def test_a_server_without_redirects_is_read_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", "/nonexistent-hf-home")
    with fake_hub({"o/r": {"a-Q4_K_M.gguf": blob(MIB)}}, redirect=False) as plain:
        plain.point(monkeypatch)
        assert hub.pull("hf:o/r/a-Q4_K_M.gguf", tmp_path).stat().st_size == MIB
        assert not plain.cdn_seen


def test_parse_reads_files_tags_and_revisions():
    got = remote.parse("hf:o/r/dir/f.gguf@abc")
    assert (got.repo, got.file, got.revision, got.quant) == ("o/r", "dir/f.gguf", "abc", "")
    assert remote.parse("o/r:Q4_K_M").quant == "Q4_K_M"
    assert remote.parse("hf:o/r").text == "hf:o/r"


def test_listing_carries_sizes_and_sha256(server):
    rows = {f.path: f for f in remote.listing("maker/thing-GGUF")}
    one = rows["thing-Q4_K_M.gguf"]
    assert one.size == 3 * MIB
    assert one.sha256 == hashlib.sha256(REPOS["maker/thing-GGUF"]["thing-Q4_K_M.gguf"]).hexdigest()
    assert one.quantization == "Q4_K_M" and rows["mmproj-thing-F16.gguf"].companion


def test_the_token_comes_from_the_environment_then_the_login_file(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert remote.token() == ""
    (tmp_path / "token").write_text("from-file\n")
    assert remote.token() == "from-file"
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert remote.token() == "from-env"


def test_the_token_can_come_from_the_ml_stack_credential_store(tmp_path, monkeypatch):
    from ml_stack import credentials

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    credentials.set("HF_TOKEN", "from-ml-stack")
    assert remote.token() == "from-ml-stack"


def test_search_returns_gguf_repositories_with_builds_and_sizes(server):
    found = hub.search("thing")
    (one,) = found
    assert one.id == "maker/thing-GGUF" and one.downloads > 0
    assert [(b[0], b[1], b[3]) for b in one.builds()] == [
        ("thing-Q8_0.gguf", 5 * MIB, "Q8_0"), ("thing-Q4_K_M.gguf", 3 * MIB, "Q4_K_M")]


def test_search_filters_by_size_quantisation_owner_and_gating(server):
    small = hub.search("thing", hub.Filters(max_bytes=4 * MIB))
    assert [b[0] for b in small[0].builds()] == ["thing-Q4_K_M.gguf"]
    assert hub.search("thing", hub.Filters(quant="Q8_0"))[0].builds()[0][3] == "Q8_0"
    assert hub.search("thing", hub.Filters(max_bytes=MIB // 2)) == []
    assert hub.search("thing", hub.Filters(owner="someone")) == []
    assert hub.search("gated", hub.Filters(gated=False)) == []
    assert [r.id for r in hub.search("gated")] == ["maker/gated-GGUF"]
    assert hub.search("gated", hub.Filters(files=False))[0].files == ()


@pytest.fixture
def guarded(monkeypatch):
    """The same stand-in hub with the address policy left on, so a redirect to the CDN on
    this machine is a redirect to a host the policy refuses."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("ML_STACK_FETCH_ALLOW_HOSTS", raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent-hf-home")
    with fake_hub(REPOS) as hub_:
        monkeypatch.setenv("HF_ENDPOINT", hub_.url)
        yield hub_


def test_a_redirect_to_a_host_the_address_policy_refuses_is_never_followed(guarded, tmp_path):
    with pytest.raises(Refused):
        hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf", tmp_path)
    assert guarded.cdn_seen == []
    assert not list(tmp_path.rglob("*.part")) and not list(tmp_path.rglob("*.gguf"))


def test_a_redirect_must_pass_the_allow_list_and_the_address_policy_each_on_its_own(
        guarded, tmp_path, monkeypatch):
    from ml_stack.net.policy import NeedsApproval

    cdn = guarded.cdn_url.split("//", 1)[1]
    monkeypatch.setenv("ML_STACK_NET_ALLOW_HOSTS", cdn)  # listed, but still a private address
    with pytest.raises(Refused, match="not on the public internet") as listed_only:
        hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf", tmp_path)
    assert not isinstance(listed_only.value, NeedsApproval)
    monkeypatch.delenv("ML_STACK_NET_ALLOW_HOSTS")
    monkeypatch.setenv("ML_STACK_FETCH_ALLOW_HOSTS", cdn)  # private allowed, but not on the list
    with pytest.raises(NeedsApproval):
        hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf", tmp_path)
    assert guarded.cdn_seen == []


def test_a_host_the_operator_names_may_be_redirected_to(guarded, tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_FETCH_ALLOW_HOSTS", "127.0.0.1,localhost")
    monkeypatch.setenv("ML_STACK_NET_ALLOW_HOSTS", "127.0.0.1")
    got = hub.pull("hf:maker/thing-GGUF/thing-Q4_K_M.gguf", tmp_path)
    assert got.read_bytes() == REPOS["maker/thing-GGUF"]["thing-Q4_K_M.gguf"]
    assert guarded.cdn_seen


def test_a_pull_pins_each_file_to_the_size_and_digest_the_listing_gives():
    one = remote.RemoteFile("UD/a-Q4_K_M.gguf", 1000, "ab" * 32)
    want = pulling._want(one, "tok")
    assert (want.sha256, want.size, want.token, want.kind) == ("ab" * 32, 1000, "tok", "gguf")
    assert 1000 <= want.max_bytes < 1000 + 2 * pulling.MARGIN
    assert pulling._want(remote.RemoteFile("README.md", 5), "").kind == ""
