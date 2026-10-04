"""Release downloads and tracked commits are accepted only when signed by the pinned key.

The keys are generated here with ssh-keygen, the assets are served from a local HTTP
server, and the commits are made and fetched by real git.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

from ml_stack.fleet import signing, updates


def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("ml-stack", text * 100)
    return buffer.getvalue()


ZIP = _zip("the bundle")
OTHER_ZIP = _zip("another bundle")
NAME = "ml-stack-linux-x86_64-v9.9.9.zip"


def _keygen(where: Path, name: str) -> tuple[Path, str]:
    key = where / name
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "fixture", "-f", str(key)],
                   check=True)
    return key, key.with_suffix(".pub").read_text().strip()


def _sign(path: Path, key: Path, namespace: str = signing.NAMESPACE) -> str:
    subprocess.run(["ssh-keygen", "-q", "-Y", "sign", "-f", str(key), "-n", namespace, str(path)],
                   check=True)
    return Path(str(path) + ".sig").read_text()


@pytest.fixture
def keys(tmp_path, monkeypatch):
    release, public = _keygen(tmp_path, "release")
    other, _ = _keygen(tmp_path, "other")
    monkeypatch.setattr(signing, "RELEASE_KEY", public)
    return release, other


@pytest.fixture
def serve(loopback_net):
    servers = []

    def start(files: dict[str, bytes]) -> updates.Release:
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = files.get(self.path.lstrip("/"))
                self.send_response(200 if body is not None else 404)
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                self.wfile.write(body or b"")

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        assets = tuple({"name": n, "size": len(b), "browser_download_url": f"{base}/{n}",
                        "digest": "sha256:" + hashlib.sha256(b).hexdigest()}
                       for n, b in files.items())
        return updates.Release("9.9.9", "", "", assets, 0)

    yield start
    for srv in servers:
        srv.shutdown()


def _fetch(release: updates.Release, tmp_path: Path) -> Path:
    return updates.download_release(release, updates.asset_for(release, "linux-x86_64"),
                                    tmp_path / "dl")


class TestReleaseAssets:
    def test_a_correctly_signed_asset_is_kept(self, tmp_path, keys, serve):
        asset = tmp_path / NAME
        asset.write_bytes(ZIP)
        signature = _sign(asset, keys[0])
        release = serve({NAME: ZIP, NAME + ".sig": signature.encode()})

        got = _fetch(release, tmp_path)

        assert got.read_bytes() == ZIP

    def test_a_tampered_asset_is_refused_and_removed(self, tmp_path, keys, serve):
        asset = tmp_path / NAME
        asset.write_bytes(ZIP)
        signature = _sign(asset, keys[0])
        release = serve({NAME: OTHER_ZIP, NAME + ".sig": signature.encode()})

        with pytest.raises(updates.UpdateError, match="does not match"):
            _fetch(release, tmp_path)
        assert not list((tmp_path / "dl").iterdir())

    def test_a_signature_from_another_key_is_refused(self, tmp_path, keys, serve):
        asset = tmp_path / NAME
        asset.write_bytes(ZIP)
        signature = _sign(asset, keys[1])
        release = serve({NAME: ZIP, NAME + ".sig": signature.encode()})

        with pytest.raises(updates.UpdateError, match="different key"):
            _fetch(release, tmp_path)
        assert not list((tmp_path / "dl").iterdir())

    def test_a_signature_for_another_purpose_is_refused(self, tmp_path, keys, serve):
        asset = tmp_path / NAME
        asset.write_bytes(ZIP)
        signature = _sign(asset, keys[0], namespace="file")
        release = serve({NAME: ZIP, NAME + ".sig": signature.encode()})

        with pytest.raises(updates.UpdateError, match="something else"):
            _fetch(release, tmp_path)

    def test_an_asset_with_no_signature_is_refused(self, tmp_path, keys, serve):
        release = serve({NAME: ZIP})

        with pytest.raises(updates.UpdateError, match="no signature"):
            _fetch(release, tmp_path)
        assert not (tmp_path / "dl").exists()

    def test_garbage_in_place_of_a_signature_is_refused(self, tmp_path, keys, serve):
        release = serve({NAME: ZIP, NAME + ".sig": b"not a signature"})

        with pytest.raises(updates.UpdateError, match="not an ssh signature"):
            _fetch(release, tmp_path)

    def test_no_pinned_key_refuses_everything(self, tmp_path, keys, serve, monkeypatch):
        asset = tmp_path / NAME
        asset.write_bytes(ZIP)
        signature = _sign(asset, keys[0])
        release = serve({NAME: ZIP, NAME + ".sig": signature.encode()})
        monkeypatch.setattr(signing, "RELEASE_KEY", "")

        with pytest.raises(updates.UpdateError, match="no release key"):
            _fetch(release, tmp_path)

    def test_the_signature_file_is_never_chosen_as_the_download(self):
        release = updates.Release("9.9.9", "", "", (
            {"name": NAME + ".sig"}, {"name": NAME}), 0)
        assert updates.asset_for(release, "linux-x86_64")["name"] == NAME


def _git(where: Path, *args: str, key: Path | None = None) -> str:
    sign = ["-c", "gpg.format=ssh", "-c", f"user.signingkey={key}"] if key else []
    done = subprocess.run(["git", "-C", str(where), "-c", "user.name=Test",
                           "-c", "user.email=test@example.invalid", *sign, *args],
                          capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _commit(where: Path, name: str, key: Path | None) -> None:
    (where / name).write_text(name)
    _git(where, "add", name)
    _git(where, "commit", "-m", name, *(["-S"] if key else []), key=key)


@pytest.fixture
def checkouts(tmp_path, keys, monkeypatch):
    monkeypatch.setattr(updates, "check_remote", lambda url, branch: None)
    remote, local = tmp_path / "remote", tmp_path / "local"
    remote.mkdir()
    _git(remote, "init", "-q", "-b", "main")
    _commit(remote, "first", None)
    subprocess.run(["git", "clone", "-q", str(remote), str(local)], check=True)
    return remote, local


def _plain_git(local: Path):
    def run(args):
        done = subprocess.run(["git", "-C", str(local), *args], capture_output=True, text=True)
        return done.returncode, f"{done.stdout}{done.stderr}".strip()

    return run


def _track(remote: Path, local: Path) -> updates.Pulled:
    return updates.track_once(str(remote), "main", local, git=_plain_git(local),
                              pip=lambda where: (0, ""), restart=lambda: "restarted")


class TestTrackedCommits:
    def test_a_commit_signed_by_the_key_is_pulled(self, checkouts, keys):
        remote, local = checkouts
        _commit(remote, "second", keys[0])

        got = _track(remote, local)

        assert got.pulled and not got.error
        assert (local / "second").exists()

    def test_an_unsigned_commit_is_not_pulled(self, checkouts):
        remote, local = checkouts
        _commit(remote, "second", None)

        got = _track(remote, local)

        assert not got.pulled and "not signed" in got.error
        assert not (local / "second").exists()

    def test_a_commit_signed_by_another_key_is_not_pulled(self, checkouts, keys):
        remote, local = checkouts
        _commit(remote, "second", keys[1])

        got = _track(remote, local)

        assert not got.pulled and "not signed" in got.error
        assert not (local / "second").exists()

    def test_without_a_pinned_key_nothing_is_pulled(self, checkouts, keys, monkeypatch):
        remote, local = checkouts
        _commit(remote, "second", keys[0])
        monkeypatch.setattr(signing, "RELEASE_KEY", "")

        got = _track(remote, local)

        assert not got.pulled and "no release key" in got.error
