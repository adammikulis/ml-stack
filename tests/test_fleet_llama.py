"""Getting a llama.cpp server without opening a terminal."""

from __future__ import annotations

import hashlib
import http.server
import json
import socket
import threading
import zipfile

import pytest
from ml_stack.fleet import llama


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def a_release(names):
    return {"tag_name": "b1234",
            "assets": [{"name": n, "browser_download_url": f"http://x/{n}",
                        "size": 0} for n in names]}


class TestPickingABuild:
    def test_it_takes_the_zip_for_this_machine(self, monkeypatch):
        monkeypatch.setattr(llama, "_tokens", lambda: ("macos-arm64",))
        got = llama.asset_for_this_machine(a_release([
            "llama-b1234-bin-ubuntu-x64.zip",
            "llama-b1234-bin-macos-arm64.zip",
            "llama-b1234-bin-win-cpu-x64.zip"]))
        assert got["name"] == "llama-b1234-bin-macos-arm64.zip"

    def test_a_release_with_no_build_for_this_machine_is_none(self, monkeypatch):
        monkeypatch.setattr(llama, "_tokens", lambda: ("macos-arm64",))
        assert llama.asset_for_this_machine(
            a_release(["llama-b1234-bin-ubuntu-x64.zip"])) is None

    def test_it_ignores_things_that_are_not_archives(self, monkeypatch):
        monkeypatch.setattr(llama, "_tokens", lambda: ("macos-arm64",))
        assert llama.asset_for_this_machine(
            a_release(["llama-b1234-bin-macos-arm64.tar.gz"])) is None

    def test_every_platform_asks_for_something(self):
        assert llama._tokens()


class TestFindingOneAlreadyHere:
    def test_a_downloaded_server_is_used_rather_than_fetched_again(self, tmp_path):
        vendor = llama.cache_dir(tmp_path)
        vendor.mkdir(parents=True)
        (vendor / llama.SERVER).write_text("#!/bin/sh\n")
        assert llama.find_server(vendor) == (vendor / llama.SERVER).resolve()

    def test_with_nothing_anywhere_it_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(llama.shutil, "which", lambda *_: None)
        assert llama.find_server(llama.cache_dir(tmp_path)) is None


class TestUnpacking:
    def test_an_archive_that_escapes_its_directory_is_refused(self, tmp_path):
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../escaped.txt", "no")
        with pytest.raises(llama.LlamaError, match="escapes"):
            llama._unzip(bad, tmp_path / "out")
        assert not (tmp_path.parent / "escaped.txt").exists()


class TestFetching:
    @pytest.fixture
    def releases(self, tmp_path):
        """A stand-in for GitHub: one release, one real zip holding a server."""
        payload = b"#!/bin/sh\necho llama-server\n"
        blob = tmp_path / "build.zip"
        with zipfile.ZipFile(blob, "w") as zf:
            zf.writestr(f"build/bin/{llama.SERVER}", payload)
            zf.writestr("build/bin/libggml.dylib", b"lib")
        raw = blob.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        port = free_port()
        body = {"tag_name": "b1234", "assets": [{
            "name": "llama-b1234-bin-test.zip",
            "browser_download_url": f"http://127.0.0.1:{port}/build.zip",
            "size": len(raw), "digest": f"sha256:{digest}"}]}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                out = raw if self.path.endswith(".zip") else json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            yield f"http://127.0.0.1:{port}/releases", payload
        finally:
            srv.shutdown()

    def test_it_downloads_unpacks_and_makes_it_runnable(self, tmp_path, releases,
                                                        monkeypatch):
        url, payload = releases
        monkeypatch.setattr(llama, "API", url)
        monkeypatch.setattr(llama, "_tokens", lambda: ("test",))
        monkeypatch.setattr(llama.shutil, "which", lambda *_: None)

        got = llama.ensure_server(tmp_path)

        assert got.read_bytes() == payload
        assert got.stat().st_mode & 0o111, "not executable"
        assert (got.parent / "libggml.dylib").exists(), "left its libraries behind"

    def test_a_second_call_does_not_download_again(self, tmp_path, releases,
                                                   monkeypatch):
        url, _ = releases
        monkeypatch.setattr(llama, "API", url)
        monkeypatch.setattr(llama, "_tokens", lambda: ("test",))
        monkeypatch.setattr(llama.shutil, "which", lambda *_: None)
        first = llama.ensure_server(tmp_path)

        def refuse(*a, **k):
            raise AssertionError("went back to the network")

        monkeypatch.setattr(llama, "latest", refuse)
        assert llama.ensure_server(tmp_path) == first

    def test_a_corrupted_download_is_refused(self, tmp_path, monkeypatch):
        port = free_port()
        body = {"tag_name": "b1", "assets": [{
            "name": "llama-b1-bin-test.zip",
            "browser_download_url": f"http://127.0.0.1:{port}/build.zip",
            "size": 4, "digest": "sha256:" + "0" * 64}]}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                out = b"junk" if self.path.endswith(".zip") else json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            monkeypatch.setattr(llama, "API", f"http://127.0.0.1:{port}/releases")
            monkeypatch.setattr(llama, "_tokens", lambda: ("test",))
            monkeypatch.setattr(llama.shutil, "which", lambda *_: None)
            with pytest.raises(llama.LlamaError, match="digest"):
                llama.ensure_server(tmp_path)
        finally:
            srv.shutdown()
        assert llama.find_server(llama.cache_dir(tmp_path)) is None

    def test_a_release_with_nothing_for_this_machine_says_which_machine(
            self, tmp_path, releases, monkeypatch):
        url, _ = releases
        monkeypatch.setattr(llama, "API", url)
        monkeypatch.setattr(llama, "_tokens", lambda: ("plan9-vax",))
        monkeypatch.setattr(llama.shutil, "which", lambda *_: None)
        with pytest.raises(llama.LlamaError, match="plan9-vax"):
            llama.ensure_server(tmp_path)
