"""Supervisor-owned adversarial Git inventory observations."""
from __future__ import annotations

import hashlib
import json
import secrets
import select
import subprocess
import sys
import time
from pathlib import Path

from poolhouse.activity.source_snapshot import SourceSnapshot, git_output, tree_digest


def source_tree(root: Path, storage: Path) -> str:
    snapshot = SourceSnapshot(root, storage)
    try:
        return snapshot.tree_hash()
    finally:
        snapshot.close()


def report(process) -> bytes:
    deadline = time.monotonic() + 15
    output = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
            raise TimeoutError("test confinement: supervisor Git observation deadline")
        block = process.stdout.read1(4097 - len(output))
        if not block:
            break
        output.extend(block)
        if len(output) > 4096:
            raise RuntimeError("test confinement: supervisor Git observation overflow")
    process.wait(timeout=max(.001, deadline - time.monotonic()))
    return bytes(output)


def observe(control: Path) -> dict:
    """Verify source inventory against hostile Git configuration before collection."""
    root = control / "git-observation"
    root.mkdir(mode=0o700)
    repo = root / "repository"
    repo.mkdir(mode=0o700)
    template = root / "template"
    template.mkdir(mode=0o700)
    git_output(["init", "--template=" + str(template), str(repo)], cwd=root)
    source = repo / "tracked.py"
    source.write_bytes(b"value = 1\n")
    git_output(["add", "tracked.py"], cwd=repo)
    marker = root / "executed"
    helper = root / "hostile-helper"
    helper.write_text("#!/bin/sh\nprintf executed > '" + str(marker) + "'\nexit 1\n")
    helper.chmod(0o700)
    hostile = "[core]\n fsmonitor = " + str(helper) + "\n[filter \"hostile\"]\n clean = " + str(helper) + "\n"
    (repo / ".git/config").write_text(hostile)
    (repo / ".gitattributes").write_text("*.py filter=hostile\n")
    global_config = root / "global-config"
    global_config.write_text(hostile)
    source_root = Path(__file__).resolve().parents[1]
    source_identity = source_tree(source_root, control)
    clone = root / "snapshot.py"
    clone.write_bytes((source_root / "src/poolhouse/activity/source_snapshot.py").read_bytes())
    clone.chmod(0o400)
    nonce = secrets.token_hex(32)
    code = ("import runpy,sys,json; sys.path.insert(0,sys.argv[5]); m=runpy.run_path(sys.argv[1]); "
            "s=m['SourceSnapshot'](m['Path'](sys.argv[2]),m['Path'](sys.argv[3])); "
            "print(json.dumps({'nonce':sys.argv[4],'tracked':[x.decode() for x in s.tracked],"
            "'tree':s.tree_hash()}));s.close()")
    environment = {"PATH": "/usr/bin:/bin", "GIT_CONFIG_GLOBAL": str(global_config),
                   "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.fsmonitor",
                   "GIT_CONFIG_VALUE_0": str(helper), "GIT_TRACE2_EVENT": str(root / "trace")}
    expected = {}
    for name, data in ((b"tracked.py", b"value = 1\n"), (b".gitattributes", b"*.py filter=hostile\n")):
        expected[name] = b"100644", hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).digest()
    process = subprocess.Popen([sys.executable, "-I", "-S", "-c", code, str(clone), str(repo), str(root), nonce,
                                str(source_root / "src")],
                               env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, close_fds=True)
    try:
        output = report(process)
        if process.returncode or len(output) > 4096:
            raise RuntimeError("test confinement: supervisor Git observation failed")
        value = json.loads(output)
        if (not isinstance(value, dict) or set(value) != {"nonce", "tracked", "tree"}
                or value["nonce"] != nonce or value["tracked"] != ["tracked.py"]
                or value["tree"] != tree_digest(expected).hex()
                or marker.exists() or (root / "trace").exists()
                or source.read_bytes() != b"value = 1\n"
                or source_tree(source_root, control) != source_identity):
            raise RuntimeError("test confinement: invalid supervisor Git observation")
        return {"nonce": nonce, "source": hashlib.sha256(clone.read_bytes()).hexdigest(), "tree": value["tree"],
                "source_tree": source_identity}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
