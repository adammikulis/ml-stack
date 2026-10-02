"""Giving a machine ml-stack over SSH, when its owner asks us to (``bootstrap --ssh HOST``).

Opt-in and owner-initiated only. The system's OpenSSH client does the work with the owner's
keys, agent and config: we never see a password (BatchMode) or a private key, never turn host key
checking off, never build a shell string from input. The target is validated and passed as one
argv element after ``--``. The host key must be in the owner's known_hosts already, or its
fingerprint is shown and the owner types it in full, and then only that key is trusted for this
run. The remote side runs one fixed script (`remote_install.txt`, printed in full with its
SHA-256 by ``--dry-run``) on files we copy: the wheel, the signed manifest, an OpenSSH signature
over it. It checks the signature and every SHA-256 before pip, installs into a per-user venv
and starts ``ml-stack-fleet listen``. Models are not copied; Windows targets are not built.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.safenames import safe_filename

from .events import BUS, Bus
from .manifest import Entry, Signer

__all__ = ["NAMESPACE", "REMOTE_RUN", "REMOTE_SCRIPT", "REMOTE_UNPACK", "Runtime", "Signed", "SshRefused",
           "Target", "bootstrap_over_ssh", "build_payload", "parse_target", "plan",
           "program_entries"]

NAMESPACE = "ml-stack-bootstrap"
MOST_OUTPUT = 64 * 1024
OPTIONS = ("-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
           "-o", "ForwardX11=no", "-o", "ClearAllForwardings=yes", "-o", "PermitLocalCommand=no")
"""Always on. Nothing here can name ``StrictHostKeyChecking=no``, a password prompt or a forward."""
REMOTE_UNPACK = ('umask 077 && mkdir -p -- "$HOME/.ml-stack-bootstrap" && cd -- '
                 '"$HOME/.ml-stack-bootstrap" && rm -rf -- inbox && mkdir inbox && '
                 'tar -xf - -C inbox')
REMOTE_RUN = 'exec python3 "$HOME/.ml-stack-bootstrap/inbox/install_remote.py"'
REMOTE_SCRIPT = Path(__file__).with_name("remote_install.txt")
USER = re.compile(r"[A-Za-z0-9._][A-Za-z0-9._-]{0,31}")
HOST = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")
V6 = re.compile(r"[0-9A-Fa-f:]{2,45}")
PROGRAM_SUFFIXES = {".whl": "wheel", ".gz": "sdist", ".zip": "sdist"}

Result = tuple[int, bytes, bytes]
Runner = Callable[[Sequence[str], bytes | None, float], Result]


class SshRefused(RuntimeError):
    """The target or the situation is not one we will go ahead with; the message says why."""


@dataclass(frozen=True, slots=True)
class Signed:
    """What the controller sends alongside the files: the signed manifest, an OpenSSH signature
    over its bytes, the key as an allowed_signers line, and the key's id."""

    manifest: bytes
    sshsig: str
    allowed_signers: str
    key_id: str


@dataclass(frozen=True, slots=True)
class Target:
    user: str
    host: str
    port: int = 22

    @property
    def spec(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def known_name(self) -> str:
        """How known_hosts names it."""
        return self.host if self.port == 22 else f"[{self.host}]:{self.port}"


def parse_target(text: str, port: int = 22) -> Target:
    """``[user@]host`` checked character by character; `SshRefused` for anything else (a
    leading dash, whitespace, shell metacharacters, an option such as ``-oProxyCommand=...``)."""
    if not isinstance(text, str) or text != text.strip() or not text or text.startswith("-"):
        raise SshRefused("the target is [user@]host, with nothing else around it")
    user, at, host = text.rpartition("@")
    if (at or user) and not USER.fullmatch(user):
        raise SshRefused("that user name has characters a user name does not")
    if not (HOST.fullmatch(host) or (V6.fullmatch(host) and host.count(":") >= 2)):
        raise SshRefused("that host is not a host name or an address")
    if host.startswith("-") or not 1 <= port <= 65535:
        raise SshRefused("that host or port is not usable")
    return Target(user, host, port)


def real_run(argv: Sequence[str], data: bytes | None, timeout: float) -> Result:
    """Run ``argv`` (no shell) with ``data`` on stdin, output cut at 64 KiB."""
    try:
        done = subprocess.run(list(argv), input=data, capture_output=True, timeout=timeout,
                              check=False)
    except subprocess.TimeoutExpired:
        return 124, b"", b"timed out"
    except OSError as exc:
        return 127, b"", type(exc).__name__.encode()
    return done.returncode, done.stdout[:MOST_OUTPUT], done.stderr[:MOST_OUTPUT]


def script_text() -> str:
    return REMOTE_SCRIPT.read_text()


def script_sha256() -> str:
    return hashlib.sha256(REMOTE_SCRIPT.read_bytes()).hexdigest()


def ssh_argv(target: Target, remote: str, extra: Sequence[str] = ()) -> list[str]:
    """The exact argv: options, port, the target after ``--``, then one constant remote
    command. ``extra`` is for the pinned known_hosts file."""
    return ["ssh", *OPTIONS, *extra, "-p", str(target.port), "--", target.spec, remote]


def program_entries(share: Path) -> list[Entry]:
    return [Signer.entry_for(f, kind=PROGRAM_SUFFIXES[f.suffix])
            for f in sorted(share.iterdir()) if f.is_file() and f.suffix in PROGRAM_SUFFIXES]


def plan(target: Target, share: Path) -> dict[str, Any]:
    """Everything ``--dry-run`` shows and nothing it does: the commands, the script in full and
    its SHA-256, the files and their digests."""
    entries = program_entries(share)
    return {"target": target.spec, "port": target.port,
            "commands": [ssh_argv(target, REMOTE_UNPACK), ssh_argv(target, REMOTE_RUN)],
            "script_sha256": script_sha256(), "script": script_text(),
            "files": [{"name": e.name, "size": e.size, "sha256": e.sha256} for e in entries],
            "runs_nothing": True}


def build_payload(share: Path, signed: Signed) -> bytes:
    """The tar the remote script expects: the program files, the signed manifest, an OpenSSH
    signature over it, the signing key as an allowed_signers line and its id, and the script.
    Plain files only, fixed modes, no owners or times."""
    members: dict[str, bytes] = {
        "manifest.json": signed.manifest,
        "manifest.sig": signed.sshsig.encode(),
        "allowed_signers": signed.allowed_signers.encode(),
        "KEYID": (signed.key_id + "\n").encode(),
        "install_remote.py": REMOTE_SCRIPT.read_bytes()}
    for e in json.loads(signed.manifest)["manifest"]["entries"]:
        members[safe_filename(e["name"])] = (share / safe_filename(e["name"])).read_bytes()
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(safe_filename(name))
            info.size, info.mode = len(data), 0o700 if name == "install_remote.py" else 0o600
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def host_key_pin(target: Target, run: Runner, offered: Callable[[list[str]], str],
                 bus: Bus) -> tuple[list[str], Path | None]:
    """The ssh options that make the host key checked, and the temporary known_hosts holding it
    when the owner's own file does not know the host. `SshRefused` unless the owner types the
    fingerprint in full."""
    if run(["ssh-keygen", "-F", target.known_name], None, 15)[0] == 0:
        return [], None
    code, found, _ = run(["ssh-keyscan", "-T", "10", "-p", str(target.port), "-t",
                          "ed25519,ecdsa,rsa", "--", target.host], None, 30)
    lines = [ln for ln in found.decode(errors="replace").splitlines()
             if ln and not ln.startswith("#") and len(ln.split()) == 3]
    if code != 0 or not lines:
        raise SshRefused(f"could not read {target.host}'s host key")
    prints: dict[str, str] = {}
    for line in lines:
        with tempfile.NamedTemporaryFile("w", suffix=".pub") as pub:
            pub.write(line + "\n")
            pub.flush()
            out = run(["ssh-keygen", "-l", "-E", "sha256", "-f", pub.name], None, 15)[1].decode()
        found_print = re.search(r"SHA256:[A-Za-z0-9+/]{43}", out)
        if found_print:
            prints[found_print[0]] = line
    typed = offered(sorted(prints))
    if typed not in prints:
        bus.emit("onboard.ssh.hostkey_refused", "warning", f"host:{target.host}")
        raise SshRefused("the fingerprint typed is not one that host presented; nothing was run")
    pin = Path(tempfile.mkdtemp(prefix="ml-stack-hostkey-")) / "known_hosts"
    pin.write_text(f"{target.known_name} " + " ".join(prints[typed].split()[1:]) + "\n")
    bus.emit("onboard.ssh.hostkey_confirmed", "notice", f"host:{target.host}", fingerprint=typed)
    return ["-o", f"UserKnownHostsFile={pin}", "-o", "GlobalKnownHostsFile=/dev/null"], pin


@dataclass(slots=True)
class Runtime:
    """How commands are run and reported: swapped in tests, and by nothing else."""

    run: Runner = real_run
    timeout: float = 600.0
    bus: Bus = BUS


def bootstrap_over_ssh(target: Target, share: Path, signed: Signed,
                       offered: Callable[[list[str]], str],
                       runtime: Runtime | None = None) -> dict[str, Any]:
    """Copy and run. ``signed`` is the manifest, its OpenSSH signature and the key. Returns
    what the remote script reported; raises `SshRefused` with a clean message on any failure."""
    rt = runtime or Runtime()
    run, timeout, bus = rt.run, rt.timeout, rt.bus
    for tool in ("ssh", "ssh-keygen", "ssh-keyscan"):
        if shutil.which(tool) is None:
            raise SshRefused(f"{tool} is not on PATH; this needs the system OpenSSH client")
    subject = f"host:{target.host}"
    bus.emit("onboard.ssh.started", "notice", subject, user=target.user, port=target.port)
    extra, pin = host_key_pin(target, run, offered, bus)
    try:
        payload = build_payload(share, signed)
        code, _, err = run(ssh_argv(target, REMOTE_UNPACK, extra), payload, timeout)
        if code != 0:
            raise SshRefused(f"copying failed ({code}): {err.decode(errors='replace')[:200]}")
        bus.emit("onboard.ssh.copied", "info", subject, bytes=len(payload))
        code, out, err = run(ssh_argv(target, REMOTE_RUN, extra), None, timeout)
    finally:
        if pin is not None:
            shutil.rmtree(pin.parent, ignore_errors=True)
    reports = [json.loads(ln) for ln in out.decode(errors="replace").splitlines()
               if ln.startswith("{") and ln.endswith("}")]
    final = reports[-1] if reports else {}
    if code != 0 or final.get("ok") is False:
        why = final.get("refused") or err.decode(errors="replace")[:200] or f"exit {code}"
        bus.emit("onboard.ssh.failed", "warning", subject, reason=str(why)[:200])
        raise SshRefused(f"the machine refused or failed: {why}")
    for step in reports:
        bus.emit(f"onboard.ssh.{step.get('step', 'reported')}", "info", subject)
    return {"target": target.spec, "steps": [r.get("step") for r in reports],
            "listen_port": final.get("port"), "fingerprint": final.get("fingerprint")}
