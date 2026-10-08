"""Materialize and seal container test assets before immutable image creation."""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def verify_source(expected):
    digest = hashlib.sha256()
    manifest = set()
    for source in sorted(Path("/src").rglob("*")):
        if source.is_symlink():
            raise RuntimeError("container seal: redirected source")
        if not source.is_file():
            continue
        relative = source.relative_to("/src")
        manifest.add(relative)
        content = source.read_bytes()
        digest.update(str(relative).encode() + b"\0" + content)
        if (Path("/work") / relative).read_bytes() != content:
            raise RuntimeError("container seal: copied source changed during setup")
    actual = set()
    for node in Path("/work").rglob("*"):
        if node.is_symlink():
            raise RuntimeError("container seal: redirected copied source")
        if node.is_file():
            actual.add(node.relative_to("/work"))
    if actual != manifest:
        raise RuntimeError("container seal: copied source inventory changed")
    if digest.hexdigest() != expected:
        raise RuntimeError("container seal: source proof changed")


def materialize_dependencies():
    sites = list(Path("/priv-venv/lib").glob("python*/site-packages/shared-env.pth"))
    if len(sites) != 1:
        raise RuntimeError("container seal: ambiguous dependency site")
    site = sites[0]
    shared = Path(site.read_text().strip())
    if not shared.is_relative_to("/venv") or shared.name != "site-packages" or shared.parent.parent.parent.name != "env":
        raise RuntimeError("container seal: invalid cached dependency environment")
    environment = shared.parent.parent.parent
    destination = Path("/runtime-deps")
    lock = environment.parent.with_suffix(".lock")
    with lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        shutil.copytree(environment, destination, symlinks=True)
    site.write_text(str(destination / shared.relative_to(environment)) + "\n")
    for executable in (destination / "bin").iterdir():
        if executable.is_symlink() or not executable.is_file():
            continue
        content = executable.read_bytes()
        prefix = b"#!" + str(environment).encode() + b"/bin/"
        if content.startswith(prefix):
            executable.write_bytes(content.replace(str(environment).encode(), str(destination).encode(), 1))
    return environment, destination


def finalize_assets(environment, destination):
    saved = Path("/test-env")
    text = saved.read_text().replace(str(environment), str(destination))
    text = text.replace("/home/tester/.cache/ms-playwright", "/opt/test-browsers")
    browsers = Path("/home/tester/.cache/ms-playwright")
    if browsers.exists():
        shutil.move(str(browsers), "/opt/test-browsers")
    text = text.replace("HOME=/home/tester", "HOME=/tmp/test-home")
    saved.write_text(text)
    subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "init", "-q", "/work"], check=True)
    subprocess.run(["git", "-C", "/work", "-c", "core.hooksPath=/dev/null", "add", "--", "."], check=True)
    subprocess.run(["git", "-C", "/work", "-c", "core.hooksPath=/dev/null", "-c", "user.name=test-agent",
                    "-c", "user.email=test-agent@example.invalid", "commit", "-qm", "test source snapshot"], check=True)
    for root in (Path("/work"), Path("/priv-venv"), destination, saved):
        for node in [root, *root.rglob("*")] if root.is_dir() else [root]:
            os.chown(node, 0, 0, follow_symlinks=False)
            if not node.is_symlink():
                node.chmod(node.stat().st_mode & ~0o022)


def seal(expected):
    verify_source(expected)
    environment, destination = materialize_dependencies()
    finalize_assets(environment, destination)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("container seal: expected source digest")
    seal(sys.argv[1])
