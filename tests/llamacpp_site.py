"""A stand-in for GitHub's llama.cpp repository on loopback: a real git repository served by
``git http-backend`` over TLS, with the few API routes the track flow reads. Its commits hold
a CMake project that builds a stub ``llama-server`` shell script."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from poolhouse import net
from poolhouse.fleet import tls
from poolhouse.http import Server
from poolhouse.httpguard import Limits
from poolhouse.serve.llamacpp_compile import Toolchain, find_make
from poolhouse.serve.llamacpp_upstream import Upstream

CMAKE = """cmake_minimum_required(VERSION 3.14)
project(stub NONE)
add_custom_target(llama-server ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/server.sh ${CMAKE_BINARY_DIR}/bin/llama-server)
"""
SERVER = """#!/bin/sh
case "$1" in
  --version) echo "version: {build} (build {build}, commit {short})" ;;
  --help) echo "-m, --model FNAME   model path"; echo "--port N   port" ;;
esac
exit 0
"""
GIT = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]


def git(*args: str, cwd: Path) -> str:
    done = subprocess.run(["git", *GIT, *args], cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout.strip()


class UpstreamSite:
    """The repository, its API and its TLS endpoint, until ``close``."""

    def __init__(self, base: Path) -> None:
        self.bare = base / "srv" / "ggml-org" / "llama.cpp"
        self.work = base / "work"
        self.bare.mkdir(parents=True)
        self.work.mkdir()
        git("init", "-q", "--bare", cwd=self.bare)
        git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=self.bare)
        git("init", "-q", "-b", "master", cwd=self.work)
        git("remote", "add", "origin", str(self.bare), cwd=self.work)
        self.releases: list[dict] = []
        self.stable = ""
        self.hits: list[str] = []
        ident = tls.identity(base / "tls", "127.0.0.1")
        self.context = tls.pinned_context(base64.b64encode(ident.der).decode())
        self.server = Server(("127.0.0.1", 0), self._handler())
        self.server.socket = tls.server_context(ident).wrap_socket(self.server.socket, server_side=True)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    @property
    def base(self) -> str:
        return f"https://127.0.0.1:{self.port}"

    @property
    def upstream(self) -> Upstream:
        return Upstream(git_base=self.base, api_base=self.base)

    def pipeline(self, path: Path, allowed: tuple[str, ...] = ("127.0.0.1",)) -> net.Pipeline:
        return net.Pipeline(
            policy=net.Policy(allowed=list(allowed), path=path / "approvals.jsonl"),
            limits=Limits(allow_hosts=frozenset({"127.0.0.1"}), context=self.context, timeout=5.0, deadline_s=20.0),
            scanners=[])

    def commit(self, number: int, *, tag: bool = True, release: bool = True, note: str = "") -> str:
        """A commit whose stub server reports build ``number``; tagged ``b<number>`` and listed
        as the newest release when asked."""
        (self.work / "CMakeLists.txt").write_text(CMAKE)
        (self.work / "server.sh").write_text(SERVER.format(build=number, short="stub"))
        (self.work / "server.sh").chmod(0o755)
        (self.work / "NOTE").write_text(note or str(number))
        git("add", "-A", cwd=self.work)
        git("commit", "-q", "-m", f"build {number}", cwd=self.work)
        sha = git("rev-parse", "HEAD", cwd=self.work)
        git("push", "-q", "-f", "origin", "master", cwd=self.work)
        if tag:
            git("tag", f"b{number}", sha, cwd=self.work)
            git("push", "-q", "origin", f"b{number}", cwd=self.work)
        if release:
            self.releases.insert(0, {"tag_name": f"b{number}", "target_commitish": sha})
        return sha

    def _api(self, path: str) -> tuple[int, object]:
        route = path.removeprefix("/repos/ggml-org/llama.cpp/")
        if route.startswith("releases?"):
            return 200, self.releases[:1]
        if route == "releases/latest":
            return (200, {"tag_name": self.stable}) if self.stable else (404, {})
        if route.startswith("compare/"):
            a, b = route.removeprefix("compare/").split("...")
            ahead = int(git("rev-list", "--count", f"{a}..{b}", cwd=self.bare))
            behind = int(git("rev-list", "--count", f"{b}..{a}", cwd=self.bare))
            return 200, {"status": "ahead" if ahead and not behind else "behind" if behind and not ahead
                         else "identical" if not ahead else "diverged", "ahead_by": ahead, "behind_by": behind}
        if route.startswith("commits/"):
            try:
                return 200, {"sha": git("rev-parse", f"{route.removeprefix('commits/')}^{{commit}}", cwd=self.bare)}
            except subprocess.CalledProcessError:
                return 404, {}
        return 404, {}

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _send(self, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def handle_any(self) -> None:
                site.hits.append(f"{self.command} {self.path}")
                if self.path.startswith("/repos/"):
                    status, payload = site._api(self.path)
                    self._send(status, [("Content-Type", "application/json")], json.dumps(payload).encode())
                    return
                path, _, query = self.path.partition("?")
                length = int(self.headers.get("Content-Length") or 0)
                env = {**os.environ, "GIT_PROJECT_ROOT": str(site.bare.parent.parent), "GIT_HTTP_EXPORT_ALL": "1",
                       "PATH_INFO": path, "QUERY_STRING": query, "REQUEST_METHOD": self.command,
                       "CONTENT_TYPE": self.headers.get("Content-Type", ""), "CONTENT_LENGTH": str(length),
                       "GIT_PROTOCOL": self.headers.get("Git-Protocol", ""), "REMOTE_ADDR": "127.0.0.1"}
                if self.headers.get("Content-Encoding"):
                    env["HTTP_CONTENT_ENCODING"] = self.headers["Content-Encoding"]
                body = self.rfile.read(length) if length else b""
                done = subprocess.run(["git", "http-backend"], input=body, capture_output=True, env=env, check=False)
                head, _, rest = done.stdout.partition(b"\r\n\r\n")
                status, headers = 200, []
                for line in head.decode("latin-1").split("\r\n"):
                    name, _, value = line.partition(": ")
                    if name.lower() == "status":
                        status = int(value.split()[0])
                    elif name:
                        headers.append((name, value))
                self._send(status, headers, rest)

            do_GET = do_POST = handle_any

        return Handler

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def toolchain() -> Toolchain:
    """A toolchain for the stub project, which needs cmake and no compiler."""
    import shutil

    cmake = os.path.realpath(shutil.which("cmake") or "")
    tree = str(Path(cmake).parent.parent)
    make = find_make()
    return Toolchain(cmake, "/usr/bin/cc", "/usr/bin/c++", "",
                     tuple({tree, "/usr", str(Path(make).parent.parent)}), make)

