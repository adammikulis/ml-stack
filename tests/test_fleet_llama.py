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

    def test_it_takes_the_tarball_platforms_ship(self, monkeypatch):
        """Windows builds are zipped; macOS and Linux ones are tarred."""
        monkeypatch.setattr(llama, "_tokens", lambda: ("macos-arm64",))
        got = llama.asset_for_this_machine(a_release([
            "llama-b1234-bin-ubuntu-x64.tar.gz",
            "llama-b1234-bin-macos-arm64.tar.gz"]))
        assert got["name"] == "llama-b1234-bin-macos-arm64.tar.gz"

    def test_it_ignores_things_that_are_not_archives(self, monkeypatch):
        monkeypatch.setattr(llama, "_tokens", lambda: ("macos-arm64",))
        assert llama.asset_for_this_machine(
            a_release(["llama-b1234-bin-macos-arm64.txt",
                       "llama-b1234-bin-macos-arm64.sha256"])) is None

    def test_every_platform_asks_for_something(self):
        assert llama._tokens()


# The assets of a real llama.cpp build release, b10612. Recorded rather than
# fetched so the test does not need the network.
REAL = [
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.4-arm64.zip",
    "llama-b10612-bin-android-arm64.tar.gz",
    "llama-b10612-bin-macos-arm64.tar.gz",
    "llama-b10612-bin-macos-x64.tar.gz",
    "llama-b10612-bin-ubuntu-arm64.tar.gz",
    "llama-b10612-bin-ubuntu-rocm-7.14-x64.tar.gz",
    "llama-b10612-bin-ubuntu-s390x.tar.gz",
    "llama-b10612-bin-ubuntu-vulkan-x64.tar.gz",
    "llama-b10612-bin-ubuntu-x64.tar.gz",
    "llama-b10612-bin-win-cpu-arm64.zip",
    "llama-b10612-bin-win-cpu-x64.zip",
    "llama-b10612-bin-win-cuda-12.4-x64.zip",
    "llama-b10612-bin-win-vulkan-x64.zip",
    "llama-b10612-xcframework.zip",
    "llama-b10612-ui.tar.gz",
]


class TestAgainstARealRelease:
    @pytest.mark.parametrize("tokens,want", [
        (("macos-arm64",), "llama-b10612-bin-macos-arm64.tar.gz"),
        (("macos-x64",), "llama-b10612-bin-macos-x64.tar.gz"),
        (("ubuntu-x64", "ubuntu-vulkan-x64"), "llama-b10612-bin-ubuntu-x64.tar.gz"),
        (("ubuntu-arm64",), "llama-b10612-bin-ubuntu-arm64.tar.gz"),
        (("win-cpu-x64", "win-x64"), "llama-b10612-bin-win-cpu-x64.zip"),
        (("win-cpu-arm64", "win-arm64"), "llama-b10612-bin-win-cpu-arm64.zip"),
    ])
    def test_every_platform_resolves_to_its_build(self, tokens, want, monkeypatch):
        monkeypatch.setattr(llama, "_tokens", lambda: tokens)
        assert llama.asset_for_this_machine(a_release(REAL))["name"] == want

    def test_the_build_releases_are_prereleases_so_the_list_is_read(self, monkeypatch):
        """/releases/latest skips prereleases, and every binary build is one, so
        it answers with a release carrying no binaries at all."""
        monkeypatch.setattr(llama, "_tokens", lambda: ("macos-arm64",))
        tagged = a_release(["nightly-tag.txt"])
        tagged["tag_name"] = "v0.2.0"
        builds = a_release(REAL)
        builds["prerelease"] = True

        assert llama.asset_for_this_machine(tagged) is None
        got = llama._first_with_a_build([tagged, builds])
        assert got is builds

    def test_a_draft_is_passed_over(self, monkeypatch):
        monkeypatch.setattr(llama, "_tokens", lambda: ("macos-arm64",))
        draft = a_release(REAL)
        draft["draft"] = True
        real = a_release(REAL)
        assert llama._first_with_a_build([draft, real]) is real


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
    def test_a_zip_that_escapes_its_directory_is_refused(self, tmp_path):
        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../escaped.txt", "no")
        with pytest.raises(llama.LlamaError, match="escapes"):
            llama._unpack(bad, tmp_path / "out")
        assert not (tmp_path.parent / "escaped.txt").exists()

    def test_a_tarball_that_escapes_its_directory_is_refused(self, tmp_path):
        import io
        import tarfile

        bad = tmp_path / "bad.tar.gz"
        with tarfile.open(bad, "w:gz") as tf:
            info = tarfile.TarInfo("../escaped.txt")
            info.size = 2
            tf.addfile(info, io.BytesIO(b"no"))
        with pytest.raises(llama.LlamaError, match="escapes"):
            llama._unpack(bad, tmp_path / "out")
        assert not (tmp_path.parent / "escaped.txt").exists()

    def test_a_tarball_unpacks(self, tmp_path):
        import io
        import tarfile

        good = tmp_path / "good.tar.gz"
        with tarfile.open(good, "w:gz") as tf:
            info = tarfile.TarInfo(f"build/bin/{llama.SERVER}")
            info.size = 4
            tf.addfile(info, io.BytesIO(b"body"))
        llama._unpack(good, tmp_path / "out")
        assert (tmp_path / "out" / "build" / "bin" / llama.SERVER).read_bytes() == b"body"


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
