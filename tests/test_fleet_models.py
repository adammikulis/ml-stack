"""Model files, and getting one from the network before the internet."""

from __future__ import annotations

import http.server
import json
import os
import socket
import threading
from pathlib import Path

import pytest
from ml_stack.fleet.models import ModelError, Models, _resolve


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def store(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return Models([d], d)


def a_model(folder, name="qwen3-4b-q4.gguf", mb=2):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(os.urandom(mb * 1024 * 1024))
    return path


class TestFinding:
    def test_it_lists_model_files_and_ignores_everything_else(self, store, tmp_path):
        a_model(tmp_path / "models")
        (tmp_path / "models" / "notes.txt").write_text("hello")
        assert [m.name for m in store.all()] == ["qwen3-4b-q4.gguf"]

    def test_a_tiny_file_is_not_a_model(self, store, tmp_path):
        (tmp_path / "models" / "stub.gguf").write_bytes(b"x" * 100)
        assert store.all() == []

    def test_it_matches_on_part_of_the_name(self, store, tmp_path):
        a_model(tmp_path / "models", "Qwen3-4B-Instruct-Q4_K_M.gguf")
        assert store.find("qwen3") is not None
        assert store.find("llama") is None

    def test_the_beacon_carries_names_and_sizes_only(self, store, tmp_path):
        a_model(tmp_path / "models")
        row = store.public()[0]
        assert set(row) == {"name", "size", "modified"}
        assert "path" not in row

    def test_the_digest_is_computed_once_per_file(self, store, tmp_path):
        a_model(tmp_path / "models")
        model = store.all()[0]
        assert store.digest(model) == store.digest(model)
        assert len(store.digest(model)) == 64


class TestSources:
    def test_a_hugging_face_reference_becomes_a_url(self):
        url = _resolve("hf:Qwen/Qwen3-4B-GGUF/qwen3-4b-q4.gguf")
        assert url.startswith("https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/")
        assert url.endswith("qwen3-4b-q4.gguf?download=true")

    @pytest.mark.parametrize("bad", ["just-a-name", "hf:owner", "hf:owner/repo",
                                     "ftp://somewhere/x.gguf"])
    def test_something_that_is_not_a_source_is_refused(self, bad):
        with pytest.raises(ModelError):
            _resolve(bad)


class TestGetting:
    def test_a_model_already_here_is_not_fetched_again(self, store, tmp_path):
        a_model(tmp_path / "models")
        got = store.ensure("qwen3", source="http://127.0.0.1:1/never.gguf")
        assert got.name == "qwen3-4b-q4.gguf"

    def test_with_automatic_downloading_off_it_refuses(self, store):
        with pytest.raises(ModelError, match="automatic downloading is off"):
            store.ensure("absent.gguf", autodownload=False)

    def test_with_nobody_holding_it_and_no_source_it_says_so(self, store):
        with pytest.raises(ModelError, match="no machine on this network"):
            store.ensure("absent.gguf")

    def test_it_downloads_when_no_machine_has_it(self, store, tmp_path):
        payload = os.urandom(2 * 1024 * 1024)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            got = store.ensure(
                "far.gguf",
                source=f"http://127.0.0.1:{srv.server_address[1]}/far.gguf")
        finally:
            srv.shutdown()

        assert got.path.read_bytes() == payload
        assert got.path.parent == store.store

    def test_a_short_download_is_left_to_resume_from(self, store):
        payload = os.urandom(1024 * 1024)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                # Claims more than it sends.
                self.send_header("Content-Length", str(len(payload) * 2))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with pytest.raises(ModelError, match="resume"):
                store.ensure("short.gguf",
                             source=f"http://127.0.0.1:{srv.server_address[1]}/s.gguf")
        finally:
            srv.shutdown()
        assert not (store.store / "short.gguf").exists()
        assert (store.store / "short.gguf.part").exists()


class TestResuming:
    """A .part on disk is a claim about a file. Each test breaks that claim."""

    def serve(self, handler):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_a_server_that_ignores_range_does_not_get_spliced_onto_the_part(
            self, store):
        payload = os.urandom(512 * 1024)
        part = store.store / "m.gguf.part"
        part.write_bytes(b"\xff" * 4096)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                # Answers 200 with the whole file even though Range was asked for.
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = self.serve(H)
        try:
            got = store.ensure(
                "m.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert got.path.read_bytes() == payload
        assert got.size == len(payload)
        assert not part.exists()

    def test_a_part_left_by_a_different_file_is_discarded(self, store):
        payload = os.urandom(256 * 1024)
        part = store.store / "m.gguf.part"
        part.write_bytes(b"\x00" * 8192)
        Path(str(part) + ".from").write_text(json.dumps(
            {"url": "http://elsewhere.invalid/other.gguf", "validator": '"x"'}))
        seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen.append(self.headers.get("Range"))
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = self.serve(H)
        try:
            got = store.ensure(
                "m.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert seen == [None], "asked to resume a part belonging to another file"
        assert got.path.read_bytes() == payload

    def test_a_part_longer_than_the_file_is_refused_not_promoted(self, store):
        part = store.store / "m.gguf.part"
        part.write_bytes(b"\x01" * (64 * 1024))

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(416)
                self.send_header("Content-Range", "bytes */1024")
                self.send_header("Content-Length", "0")
                self.end_headers()

        srv = self.serve(H)
        try:
            with pytest.raises(ModelError, match="discarded"):
                store.ensure(
                    "m.gguf",
                    source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert not (store.store / "m.gguf").exists()
        assert not part.exists()

    def test_a_genuine_resume_asks_for_the_rest_and_keeps_what_it_had(self, store):
        head, tail = b"A" * 4096, b"B" * 4096
        part = store.store / "m.gguf.part"
        part.write_bytes(head)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                assert self.headers.get("Range") == "bytes=4096-"
                self.send_response(206)
                self.send_header("Content-Range", "bytes 4096-8191/8192")
                self.send_header("Content-Length", str(len(tail)))
                self.end_headers()
                self.wfile.write(tail)

        srv = self.serve(H)
        try:
            got = store.ensure(
                "m.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
        finally:
            srv.shutdown()

        assert got.path.read_bytes() == head + tail
        assert not part.exists()


class TestProgress:
    def test_the_two_kinds_of_progress_do_not_share_a_callback(self, store):
        """ensure() reports a stage as text and bytes as two numbers. One callback
        taking both would be called with a string and then with a pair."""
        payload = os.urandom(512 * 1024)

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        notes, seen = [], []
        try:
            store.ensure("m.gguf",
                         source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf",
                         on_note=notes.append,
                         on_progress=lambda done, total: seen.append((done, total)))
        finally:
            srv.shutdown()

        assert notes == ["Downloading m.gguf"]
        assert seen, "no byte counts arrived"
        assert all(isinstance(d, int) and isinstance(t, int) for d, t in seen)
        assert seen[-1][0] == len(payload)
        assert seen[-1][1] == len(payload)


class TestGettingInTheBackground:
    def serve(self, payload, delay=0.0):
        import time as clock

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                for i in range(0, len(payload), 65536):
                    self.wfile.write(payload[i:i + 65536])
                    self.wfile.flush()
                    if delay:
                        clock.sleep(delay)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def waited(self, downloads, gid, want, timeout=20.0):
        import time as clock

        until = clock.monotonic() + timeout
        while clock.monotonic() < until:
            row = next(g for g in downloads.active() if g.id == gid)
            if row.state == want:
                return row
            clock.sleep(0.05)
        raise AssertionError(f"still {row.state} after {timeout}s: {row.error}")

    def test_a_download_runs_without_holding_the_caller(self, store):
        from ml_stack.fleet.models import Downloads

        payload = os.urandom(1024 * 1024)
        srv = self.serve(payload)
        downloads = Downloads(store)
        try:
            started = downloads.start(
                "big.gguf",
                source=f"http://127.0.0.1:{srv.server_address[1]}/big.gguf")
            assert started.state == "getting"
            done = self.waited(downloads, started.id, "done")
        finally:
            srv.shutdown()

        assert done.total == len(payload)
        assert done.done == len(payload)
        assert (store.store / "big.gguf").read_bytes() == payload

    def test_how_far_along_it_is_can_be_read_while_it_runs(self, store):
        """Bigger than one CHUNK, or the whole body arrives in a single read and
        the only counts ever seen are nothing and everything."""
        from ml_stack.fleet.models import CHUNK, Downloads

        payload = os.urandom(4 * CHUNK)
        srv = self.serve(payload, delay=0.01)
        downloads = Downloads(store)
        try:
            started = downloads.start(
                "slow.gguf",
                source=f"http://127.0.0.1:{srv.server_address[1]}/slow.gguf")
            import time as clock
            seen = []
            for _ in range(200):
                row = next(g for g in downloads.active() if g.id == started.id)
                seen.append(row.done)
                if row.state != "getting":
                    break
                clock.sleep(0.05)
            self.waited(downloads, started.id, "done")
        finally:
            srv.shutdown()

        assert any(0 < d < len(payload) for d in seen), (
            f"only ever saw {sorted(set(seen))}")

    def test_a_failure_is_reported_rather_than_raised_into_nowhere(self, store):
        from ml_stack.fleet.models import Downloads

        downloads = Downloads(store)
        started = downloads.start("nope.gguf", source="http://127.0.0.1:1/nope.gguf")
        row = self.waited(downloads, started.id, "failed")
        assert row.error
        assert not (store.store / "nope.gguf").exists()

    def test_asking_twice_for_the_same_model_does_not_start_it_twice(self, store):
        from ml_stack.fleet.models import Downloads

        payload = os.urandom(512 * 1024)
        srv = self.serve(payload, delay=0.05)
        downloads = Downloads(store)
        try:
            source = f"http://127.0.0.1:{srv.server_address[1]}/twice.gguf"
            first = downloads.start("twice.gguf", source=source)
            again = downloads.start("twice.gguf", source=source)
            assert again.id == first.id
            self.waited(downloads, first.id, "done")
        finally:
            srv.shutdown()
        assert len([g for g in downloads.active() if g.name == "twice.gguf"]) == 1


class TestUnfinishedDownloads:
    def test_a_part_still_being_written_is_not_offered_for_discard(self, store):
        (store.store / "busy.gguf.part").write_bytes(b"x" * 1024)
        assert store.unfinished() == []

    def test_one_nothing_has_touched_for_an_hour_is(self, store):
        import os
        import time

        part = store.store / "stopped.gguf.part"
        part.write_bytes(b"x" * 2048)
        old = time.time() - 7200
        os.utime(part, (old, old))

        found = store.unfinished()
        assert [r["name"] for r in found] == ["stopped.gguf.part"]
        assert found[0]["size"] == 2048

    def test_discarding_takes_the_part_and_what_it_recorded(self, store):
        part = store.store / "gone.gguf.part"
        part.write_bytes(b"x" * 16)
        stamp = Path(str(part) + ".from")
        stamp.write_text(json.dumps({"url": "http://x/y.gguf", "validator": "t"}))

        assert store.discard("gone.gguf.part") == ["gone.gguf.part"]
        assert not part.exists()
        assert not stamp.exists()

    def test_discarding_leaves_finished_models_alone(self, store, tmp_path):
        a_model(tmp_path / "models", name="keep.gguf")
        (store.store / "drop.gguf.part").write_bytes(b"x")
        store.discard("drop.gguf.part")
        assert [m.name for m in store.all()] == ["keep.gguf"]

    @pytest.mark.parametrize("bad", ["../keep.gguf", "keep.gguf", "a/b.part"])
    def test_discard_reaches_nothing_outside_the_store(self, store, tmp_path, bad):
        a_model(tmp_path / "models", name="keep.gguf")
        outside = store.store.parent / "keep.gguf"
        outside.write_bytes(b"important")
        assert store.discard(bad) == []
        assert outside.exists()
        assert (store.store / "keep.gguf").exists()


class TestRemoving:
    def test_only_models_this_machine_downloaded_can_be_removed(self, tmp_path):
        """A model in someone's own folder is theirs, not this program's to delete."""
        elsewhere = tmp_path / "theirs"
        a_model(elsewhere, "theirs.gguf")
        store = Models([elsewhere, tmp_path / "ours"], tmp_path / "ours")

        assert store.find("theirs") is not None
        assert store.remove("theirs") is False
        assert (elsewhere / "theirs.gguf").exists()

    def test_one_it_downloaded_can_be(self, store, tmp_path):
        a_model(tmp_path / "models")
        assert store.remove("qwen3") is True
        assert store.all() == []


class TestOverHTTP:
    """The routes a peer uses: GET /models and POST /models/get."""

    @pytest.fixture
    def served(self, tmp_path):
        from http.server import ThreadingHTTPServer

        from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
        from ml_stack.fleet.remote import Peer

        root = tmp_path / "traind"
        files = root / "files"
        files.mkdir(parents=True)
        shelf = tmp_path / "shelf"
        a_model(shelf)
        token = load_or_create_token(root)
        runner = JobRunner(root)
        port = free_port()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", port),
            make_handler(runner, files, token,
                         models=Models([shelf], shelf),
                         cluster_key_path=tmp_path / "cluster.key"))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            yield Peer(f"http://127.0.0.1:{port}", token), shelf
        finally:
            runner.shutdown()
            httpd.shutdown()
            httpd.server_close()

    def test_a_peer_lists_the_models_a_machine_holds(self, served):
        peer, _ = served
        rows = peer.models()
        assert [r["name"] for r in rows] == ["qwen3-4b-q4.gguf"]
        assert "path" not in rows[0]

    def test_a_peer_asks_a_machine_for_a_model_it_already_has(self, served):
        peer, shelf = served
        got = peer.get_model("qwen3")
        assert got["name"] == "qwen3-4b-q4.gguf"
        assert got["size"] == (shelf / "qwen3-4b-q4.gguf").stat().st_size

    def test_asking_for_one_nobody_has_is_refused_not_crashed(self, served):
        from ml_stack.fleet.remote import PeerError

        peer, _ = served
        with pytest.raises(PeerError):
            peer.get_model("nothing-like-this.gguf")
