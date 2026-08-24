"""Model files, and getting one from the network before the internet."""

from __future__ import annotations

import http.server
import os
import socket
import threading

import pytest
from ml_stack.fleet.models import Model, ModelError, Models, _resolve


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
