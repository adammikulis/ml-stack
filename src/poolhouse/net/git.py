"""Running git against a remote: the host policy applies, only https (and ssh when the caller
names it) is spoken, hooks and submodules are off, and the commit that came down is recorded."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.parse
from collections.abc import Sequence
from pathlib import Path

from poolhouse.httpguard import Refused
from poolhouse.net import policy as hosts, provenance
from poolhouse.net.hold import staging_dir

__all__ = ["NETWORK", "GitFailed", "clone", "environment", "guarded", "head", "run"]

NETWORK = frozenset({"clone", "fetch", "pull", "ls-remote", "push", "submodule"})
TIMEOUT_S = 1800.0
SSH = re.compile(r"(?:ssh://)?[A-Za-z0-9_.-]+@([A-Za-z0-9][A-Za-z0-9.-]*)[:/]")


class GitFailed(RuntimeError):
    """git ran and failed, or is not installed."""


def environment(protocols: str, hooks: Path) -> dict[str, str]:
    """The environment git runs in: no prompts, only ``protocols`` allowed, no system config,
    and the options below set through ``GIT_CONFIG_*`` so the command line stays as written."""
    env = dict(os.environ)
    # The caller chooses the repository through cwd, never inherited Git selectors.
    for name in ("GIT_EXTERNAL_DIFF", "GIT_SSH_COMMAND", "GIT_PROXY_COMMAND", "GIT_ASKPASS",
                 "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
                 "GIT_PREFIX", "GIT_NAMESPACE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        env.pop(name, None)
    env.update(GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL=protocols, GIT_CONFIG_NOSYSTEM="1")
    options = {"protocol.ext.allow": "never", "protocol.file.allow": "never",
               "protocol.git.allow": "never", "core.fsmonitor": "false",
               "core.hooksPath": str(hooks), "transfer.fsckObjects": "true",
               "fetch.fsckObjects": "true", "submodule.recurse": "false"}
    env["GIT_CONFIG_COUNT"] = str(len(options))
    for number, (key, value) in enumerate(options.items()):
        env[f"GIT_CONFIG_KEY_{number}"], env[f"GIT_CONFIG_VALUE_{number}"] = key, value
    return env


def _host(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https":
        return hosts.host_of(url)
    found = SSH.match(url)
    return found.group(1).lower() if found else ""


def guarded(url: str, policy: hosts.Policy | None = None, protocols: str = "https") -> str:
    """The URL when git may talk to it: https (or ssh when ``protocols`` allows it), no
    credentials in it, and a host the policy admits. `Refused` otherwise."""
    if not url or url.startswith("-") or any(ord(ch) < 33 for ch in url):
        raise Refused("that is not a git address")
    scheme = urllib.parse.urlsplit(url).scheme
    ssh = bool(SSH.match(url)) and scheme in ("", "ssh")
    if scheme != "https" and not (ssh and "ssh" in protocols.split(":")):
        raise Refused(f"git speaks {protocols} here, not {scheme or 'a bare path'}")
    if urllib.parse.urlsplit(url).password:
        raise Refused("a git address does not carry a password")
    (policy or hosts.default()).admit(f"https://{_host(url)}/", "git")
    return url


def _git() -> str:
    found = shutil.which("git")
    if found is None:
        raise GitFailed("git is not on PATH")
    return found


def run(args: Sequence[str], *, cwd: Path | None = None, url: str = "",
        policy: hosts.Policy | None = None,
        protocols: str = "https") -> subprocess.CompletedProcess[str]:
    """``git *args``; a network command needs ``url`` (or an ``origin`` in ``cwd``) the policy
    admits. `GitFailed` when git exits non-zero."""
    if args and args[0] in NETWORK:
        target = url or _origin(cwd)
        guarded(target, policy, protocols)
    hooks = staging_dir() / "no-hooks"
    hooks.mkdir(exist_ok=True)
    try:
        done = subprocess.run([_git(), *args], cwd=cwd, capture_output=True, text=True,
                              timeout=TIMEOUT_S, env=environment(protocols, hooks), check=False)
    except subprocess.TimeoutExpired as exc:
        raise GitFailed(f"git {args[0] if args else ''} timed out") from exc
    if done.returncode != 0:
        raise GitFailed(f"git {' '.join(args)} failed: {(done.stderr or '').strip()}")
    return done


def blobs(digests: list[str], *, cwd: Path, limit: int, total: int) -> dict[str, bytes]:
    """Read bounded Git blobs in two batches without checkout filters."""
    if len(digests) > 10_000 or any(not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", d) for d in digests):
        raise GitFailed("Invalid Git blob identifiers")
    identifiers = list(dict.fromkeys(digests))
    if not identifiers:
        return {}
    query = ("\n".join(identifiers) + "\n").encode()
    hooks = staging_dir() / "no-hooks"
    env = environment("", hooks)
    checked = subprocess.run([_git(), "cat-file", "--batch-check"], cwd=cwd, input=query,
                             capture_output=True, timeout=30, env=env, check=False)
    rows = checked.stdout.splitlines()
    if checked.returncode or len(rows) != len(identifiers):
        raise GitFailed("Git source objects could not be sized")
    sizes = []
    for digest, row in zip(identifiers, rows, strict=True):
        parts = row.decode().split()
        if len(parts) != 3 or parts[:2] != [digest, "blob"] or not parts[2].isdigit():
            raise GitFailed("Git source object is not a blob")
        sizes.append(int(parts[2]))
    if any(size > limit for size in sizes) or sum(sizes) > total:
        raise GitFailed("Git source blobs exceed source size limits")
    done = subprocess.run([_git(), "cat-file", "--batch"], cwd=cwd, input=query,
                          capture_output=True, timeout=30, env=env, check=False)
    if done.returncode or len(done.stdout) > total + len(identifiers) * 100:
        raise GitFailed("Git source blobs could not be read")
    offset, result = 0, {}
    for digest, size in zip(identifiers, sizes, strict=True):
        newline = done.stdout.find(b"\n", offset)
        expected = f"{digest} blob {size}".encode()
        if newline < offset or done.stdout[offset:newline] != expected:
            raise GitFailed("Git source blob changed while reading")
        offset = newline + 1
        result[digest] = done.stdout[offset:offset + size]
        offset += size
        if done.stdout[offset:offset + 1] != b"\n":
            raise GitFailed("Truncated Git source blob")
        offset += 1
    if offset != len(done.stdout):
        raise GitFailed("Unexpected Git source blob bytes")
    return result


def source_index(entries: list[dict], *, cwd: Path) -> None:
    """Add verified source files to a fresh index without filters."""
    if not entries:
        return
    hooks = staging_dir() / "no-hooks"
    env = environment("", hooks)
    paths = ("\n".join(entry["path"] for entry in entries) + "\n").encode()
    hashed = subprocess.run([_git(), "hash-object", "-w", "--no-filters", "--stdin-paths"],
                            cwd=cwd, input=paths, capture_output=True, timeout=30, env=env, check=False)
    digests = hashed.stdout.decode().splitlines()
    if hashed.returncode or len(digests) != len(entries) or any(not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", d) for d in digests):
        raise GitFailed("Verified source could not be indexed")
    rows = [f"{'100755' if entry['executable'] else '100644'} {digest}\t{entry['path']}\n"
            for entry, digest in zip(entries, digests, strict=True)]
    indexed = subprocess.run([_git(), "update-index", "--index-info"], cwd=cwd,
                             input="".join(rows).encode(), capture_output=True, timeout=30, env=env, check=False)
    if indexed.returncode:
        raise GitFailed("Verified source index could not be written")


def _origin(cwd: Path | None) -> str:
    if cwd is None:
        raise Refused("a git network command needs the address it talks to")
    done = subprocess.run([_git(), "-C", str(cwd), "remote", "get-url", "origin"],
                          capture_output=True, text=True, timeout=30, check=False)
    return done.stdout.strip()


def head(cwd: Path) -> str:
    """The commit checked out in ``cwd``."""
    return run(["rev-parse", "HEAD"], cwd=cwd).stdout.strip()


def clone(url: str, dest: Path, *, branch: str = "", depth: int = 1,
          policy: hosts.Policy | None = None) -> str:
    """Shallow-clone ``url`` into ``dest`` and return the commit; the commit is recorded in
    the download index with the address it came from. Nothing in the tree is run."""
    args = ["clone", "--depth", str(depth), "--no-tags", "--no-recurse-submodules"]
    if branch:
        args += ["--branch", branch]
    run([*args, "--", url, str(dest)], url=url, policy=policy)
    commit = head(dest)
    provenance.record(provenance.Provenance(
        url=url, final_url=url, path=str(dest), sha256=commit, size=0, kind="git",
        fetched_at=provenance.stamp(), host_source=hosts.host_of(f"https://{_host(url)}/"),
        scan="not scanned: source tree", outcome="kept"))
    return commit
