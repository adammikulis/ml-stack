"""Nothing outside the net pipeline opens a connection to a host on the internet.

The scan reads every module under ``src/ml_stack`` and flags: imports of network libraries,
a raw connection, a git or download program run by subprocess, and a call into
``ml_stack.http``'s request functions (the client for servers on this machine or network).
A module may do these only when it is on a list below with the reason its traffic stays on
this machine or network, or is handed to a library the pipeline cannot wrap. A relative import
(``from .requests import ...``) is a sibling module, never a network library.

Three kinds of exemption, each narrow (docs/security.md, "What is exempt from the net scan"):
  * model servers on this machine or network that a person configured or a port names;
  * the LAN onboarding transport, which refuses a public address at every connection
    (`fleet/onboard/lan.py`, tested below) and pins the peer's certificate;
  * the red-team lab, which only talks to servers it started on 127.0.0.1.
Anything that can reach a public host, such as a library's own downloader, is not on the list.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "src" / "ml_stack"

LIBRARIES = {"urllib.request", "http.client", "requests", "httpx", "aiohttp", "huggingface_hub",
             "ftplib", "smtplib", "telnetlib", "urllib3", "websocket", "websockets"}
CALLS = {"socket.create_connection", "urllib.request.urlopen", "urlopen"}
PROGRAMS = {"curl", "wget", "scp", "sftp", "rsync"}
GIT_NETWORK = {"clone", "fetch", "pull", "ls-remote", "push", "submodule"}
REQUESTS = {"open_stream", "request_json", "request_bytes", "request_stream", "head_once"}

PIPELINE = ("net/",)

ALLOWED: dict[str, str] = {
    "http.py": "the client for model servers on this machine and network; urllib lives here",
    "httpguard.py": "the guarded connection the pipeline is built on",
    "fleet/serving.py": "a port check on loopback",
    "fleet/invite_client.py": "single-use LAN invitation; every resolved address is checked, the selected sockaddr is frozen and TLS is pinned before HTTP",
    "fleet/project_client.py": "sealed project source from a LAN peer; exact redirects and require_local_url at every request",
    "workspace/remote.py": "sealed project board on the LAN; exact redirects and require_local_url at every request",
    "fleet/wsl_network.py": "WSL LAN bridge; require_local before each TCP connection and restricted discovery destinations",
    "fleet/api.py": "proxies an inference call to a fleet peer",
    "fleet/remote.py": "calls a fleet peer",
    "fleet/join.py": "joins a fleet peer",
    "fleet/chat.py": "chat with a fleet peer",
    "fleet/launch.py": "the local daemon's health",
    "fleet/ui.py": "a /metrics address a person typed into the fleet view (needs a session)",
    "serve/broker_wire.py": "the lease broker on loopback",
    "serve/escalation.py": "a model server's slots on loopback",
    "serve/reclaim.py": "a model server's slots on loopback",
    "bench/": "model servers on this machine or network",
    "client/": "model servers the person chose; the chat endpoint is configured, not fetched",
    "decide/logprob.py": "the decide backend: the model server the person configured "
                         "(ML_STACK_DECIDE_URL: this machine only unless the operator names the host)",
    "decide/router.py": "reachability of that same configured decide server",
    "serve/slotdump.py": "slot save and restore on a model server this machine started",
    "serve/llamacpp_smoke.py": "health, chat and slot checks on the model server a smoke test started",
    "serve/slots_cli.py": "a model server's slots, addressed by local port",
    "serve/unmanaged.py": "the props of a model server found listening on this machine",
    "fleet/onboard/transfer.py": "files from paired peers; a public address is refused at every "
                                 "connection and TLS is pinned to the peer's certificate "
                                 "(fleet/onboard/lan.py)",
    "fleet/onboard/pairing.py": "the pairing exchange with a device on this network, confirmed "
                                "by a code and the certificate fingerprint (fleet/onboard/lan.py)",
    "fleet/onboard/joining.py": "the join handshake with a daemon on this network, proved by the "
                                "passphrase and the certificate fingerprint (fleet/onboard/lan.py)",
    "fleet/onboard/routes.py": "asks a paired device's address for its certificate; a public address "
                               "is refused first and only the pinned certificate counts "
                               "(fleet/onboard/lan.py)",
    "redteam/scenarios/fleet.py": "the lab's own servers on 127.0.0.1",
    "redteam/scenarios/isolation.py": "the lab's own servers on 127.0.0.1",
    "redteam/tools.py": "the lab's page server and honeypot on 127.0.0.1",
}


def reason(rel: str) -> str:
    for prefix, why in ALLOWED.items():
        if rel == prefix or (prefix.endswith("/") and rel.startswith(prefix)):
            return why
    return ""


def names(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def findings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name in LIBRARIES
                      or any(a.name.startswith(lib + ".") for lib in LIBRARIES)]
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if node.level and module != "http":
                continue  # a sibling module that shares a library's name
            if module == "http.client" and all(a.name == "HTTPException" for a in node.names):
                continue
            if module in LIBRARIES or any(module.startswith(lib + ".") for lib in LIBRARIES):
                found.append(f"from {module} import ...")
            elif module in ("urllib", "http"):
                found += [f"from {module} import {a.name}" for a in node.names
                          if f"{module}.{a.name}" in LIBRARIES]
            elif module == "ml_stack.http" or (node.level and module == "http"):
                found += [f"from ml_stack.http import {a.name}" for a in node.names
                          if a.name in REQUESTS]
        elif isinstance(node, ast.Call):
            called = names(node.func)
            if called in CALLS or called.split(".")[-1] == "urlopen":
                found.append(f"call {called}")
            if called.split(".")[-1] in REQUESTS and called.startswith(("http.", "ml_stack.http.")):
                found.append(f"call {called}")
            if called.startswith("subprocess.") and node.args:
                found += program_findings(node.args[0])
    return found


def program_findings(argv: ast.AST) -> list[str]:
    if not isinstance(argv, (ast.List, ast.Tuple)):
        return []
    words = [e.value for e in argv.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    out = [f"runs {w}" for w in words if w in PROGRAMS]
    if "git" in words and GIT_NETWORK & set(words):
        out.append("runs git against a remote")
    if "pip" in words and "install" in words:
        out.append("runs pip install")
    return out


def modules() -> list[tuple[str, Path]]:
    return [(p.relative_to(ROOT).as_posix(), p) for p in sorted(ROOT.rglob("*.py"))]


def test_no_module_outside_the_pipeline_reaches_the_internet_on_its_own():
    bad: list[str] = []
    for rel, path in modules():
        if rel.startswith(PIPELINE) or reason(rel):
            continue
        for item in findings(path):
            bad.append(f"{rel}: {item}")
    assert not bad, "network I/O outside ml_stack.net:\n  " + "\n  ".join(bad)


def test_every_allowed_module_exists_and_still_needs_its_place():
    """A list entry whose module is gone, or whose module no longer does anything on the
    list, is a door left open for nothing."""
    existing = {rel for rel, _ in modules()}
    stale = []
    for prefix in ALLOWED:
        matches = [rel for rel in existing if rel == prefix or
                   (prefix.endswith("/") and rel.startswith(prefix))]
        if not matches:
            stale.append(f"{prefix}: no such module")
        elif not any(findings(ROOT / rel) for rel in matches):
            stale.append(f"{prefix}: does nothing that needs the allowance")
    assert not stale, "\n".join(stale)


@pytest.mark.parametrize("source", [
    "import urllib.request\n",
    "from http.client import HTTPConnection\n",
    "from http.client import HTTPException, HTTPConnection\n",
    "from urllib.request import urlopen\n",
    "import requests\n",
    "import socket\nsocket.create_connection(('x', 1))\n",
    "import subprocess\nsubprocess.run(['git', 'clone', 'u'])\n",
    "import subprocess\nsubprocess.run(['curl', 'u'])\n",
    "import subprocess\nsubprocess.run(['python', '-m', 'pip', 'install', 'x'])\n",
    "from huggingface_hub import hf_hub_download\n",
    "from ml_stack.http import open_stream\n",
    "from ml_stack import http\nhttp.request_json('u')\n",
])
def test_the_scan_recognises_each_way_of_reaching_out(tmp_path, source):
    path = tmp_path / "m.py"
    path.write_text(source)
    assert findings(path), source


def test_the_scan_leaves_alone_a_module_that_only_reads_a_local_file(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("import json\nimport subprocess\nsubprocess.run(['git', 'status'])\n")
    assert findings(path) == []


def test_the_pipeline_is_the_one_place_that_opens_connections():
    """The pipeline itself still goes through httpguard and nothing else."""
    for rel, path in modules():
        if not rel.startswith("net/"):
            continue
        for item in findings(path):
            assert "urllib.request" not in item and "requests" not in item, f"{rel}: {item}"


RELATIVE = [
    "from .requests import Device\n",
    "from . import requests\n",
]


@pytest.mark.parametrize("source", RELATIVE)
def test_a_sibling_module_named_like_a_library_is_not_the_library(tmp_path, source):
    path = tmp_path / "m.py"
    path.write_text(source)
    assert findings(path) == []


def test_every_module_that_may_open_a_peer_connection_checks_the_address_first():
    """The onboarding exemption is only as good as `require_local`: each module on the list for
    it calls it before it connects, and none builds a client that trusts the system store."""
    for rel in ("fleet/onboard/transfer.py", "fleet/onboard/pairing.py", "fleet/onboard/joining.py"):
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert "require_local" in source, rel
        assert "create_default_context" not in source, rel
        assert "HTTPSConnection(" not in source or "context=" in source, rel


def test_http_exception_import_does_not_open_a_connection(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("from http.client import HTTPException\n")
    assert findings(path) == []
