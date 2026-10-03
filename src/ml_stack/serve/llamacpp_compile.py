"""Fetching one llama.cpp commit and compiling it under the sandbox."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ml_stack import net, sandbox
from ml_stack.httpguard import Refused
from ml_stack.net import git as netgit
from ml_stack.sandbox.policies import SYSTEM_EXEC, system_env
from ml_stack.sandbox.policy import Limits, Net, Policy
from ml_stack.serve.build_paths import BuildFailed
from ml_stack.serve.build_platform import cmake_flags, server_name
from ml_stack.serve.llamacpp_upstream import Upstream

__all__ = ["CMAKE_TARGET", "Job", "Toolchain", "checkout", "compile_source", "toolchain"]

CMAKE_TARGET = "llama-server"
WALL_SECONDS = 4 * 3600.0
FIXED_FLAGS = ("-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF",
               "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_CURL=OFF")
"""The cmake arguments every build gets: one static server, nothing that fetches or tests."""


@dataclass(frozen=True, slots=True)
class Toolchain:
    """What a build runs: cmake, the two compilers and, on macOS, the SDK; ``reads`` are the
    trees the sandbox lets them read and start."""

    cmake: str
    cc: str
    cxx: str
    sysroot: str
    reads: tuple[str, ...]
    make: str = ""

    def arguments(self) -> list[str]:
        out = [f"-DCMAKE_C_COMPILER={self.cc}", f"-DCMAKE_CXX_COMPILER={self.cxx}"]
        if self.make:
            out += [f"-DCMAKE_MAKE_PROGRAM={self.make}"]
            out += ["-GNinja"] if Path(self.make).name == "ninja" else ["-GUnix Makefiles"]
        return out + ([f"-DCMAKE_OSX_SYSROOT={self.sysroot}"] if self.sysroot else [])


def _tree(path: str) -> str:
    """The install prefix of the program at ``path`` (the tree above its ``bin``)."""
    real = Path(os.path.realpath(path))
    return str(real.parent.parent if real.parent.name == "bin" else real.parent)


def _xcode(*query: str) -> str:
    done = subprocess.run(["/usr/bin/xcrun", *query], capture_output=True, text=True, timeout=60,
                          check=False)
    return done.stdout.strip() if done.returncode == 0 else ""


def find_make() -> str:
    """The build program cmake drives: ninja when present, else the real ``make`` (on macOS
    the one inside the developer tools, since ``/usr/bin/make`` is a shim that asks
    ``xcode-select``, which the sandbox cannot let it do)."""
    if ninja := shutil.which("ninja"):
        return os.path.realpath(ninja)
    if platform.system() == "Darwin" and (found := _xcode("-f", "make")):
        return found
    return os.path.realpath(shutil.which("make") or "")


def toolchain() -> Toolchain:
    """The tools a build needs, found on this machine. `BuildFailed` names each one missing;
    nothing is installed."""
    missing = [name for name in ("git", "cmake") if shutil.which(name) is None]
    if platform.system() == "Windows":
        raise BuildFailed("building llama.cpp from source is not supported on Windows; "
                          "`ml-stack-serve build --from release` downloads a release instead")
    sysroot = ""
    if platform.system() == "Darwin":
        cc, cxx = _xcode("-f", "clang"), _xcode("-f", "clang++")
        sysroot = _xcode("--show-sdk-path")
        if not (cc and cxx and sysroot):
            missing.append("a C/C++ compiler (the Xcode command line tools: `xcode-select --install`)")
    else:
        cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang") or ""
        cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++") or ""
        if not (cc and cxx):
            missing.append("a C/C++ compiler (cc and c++)")
    make = find_make()
    if not make:
        missing.append("make or ninja")
    if missing:
        raise BuildFailed("cannot build llama.cpp: missing " + ", ".join(missing)
                          + "; ml-stack installs nothing, install them and run it again")
    cmake = os.path.realpath(shutil.which("cmake") or "")
    trees = {_tree(cmake), _tree(cc), _tree(cxx), _tree(make), "/usr", "/Library/Developer"}
    if sysroot:
        trees.add(_tree(sysroot) if "/Developer/" not in sysroot else sysroot.split("/Developer/")[0] + "/Developer")
    if shutil.which("nvcc"):
        trees.add(_tree(shutil.which("nvcc") or ""))
    return Toolchain(cmake, cc, cxx, sysroot,
                     tuple(sorted(t for t in trees if t not in ("/", "") and Path(t).is_dir())), make)


def checkout(upstream: Upstream, sha: str, dest: Path, pipeline: net.Pipeline) -> None:
    """``sha`` of the upstream repository in ``dest``, fetched through the host policy with
    hooks and submodules off, then checked to be that commit; ``.git`` is removed afterwards,
    so nothing in the tree is a repository the build could ask anything of."""
    def git(*args: str, url: str = "") -> subprocess.CompletedProcess[str]:
        try:
            return netgit.run(args, cwd=dest, url=url, policy=pipeline.policy)
        except net.NeedsApproval:
            raise
        except (netgit.GitFailed, Refused) as exc:
            raise BuildFailed(str(exc)) from exc

    dest.mkdir(parents=True, exist_ok=True)
    git("init", "-q")
    git("remote", "add", "origin", upstream.git_url)
    git("fetch", "-q", "--depth", "1", "--no-tags", "--no-recurse-submodules", "origin", sha,
        url=upstream.git_url)
    git("checkout", "-q", "--detach", "FETCH_HEAD")
    head = git("rev-parse", "HEAD").stdout.strip()
    if head != sha:
        raise BuildFailed(f"the checkout is {head}, not the commit {sha} that was asked for")
    shutil.rmtree(dest / ".git")


def _policy(tools: Toolchain, source: Path, work: Path, wall: float) -> Policy:
    env = system_env(HOME=str(work), TMPDIR=str(work))
    env["PATH"] = os.pathsep.join([*(str(Path(t) / "bin") for t in tools.reads), env["PATH"]])
    return Policy("llama-build", read=(str(source), *tools.reads), write=(str(work),),
                  exec=(*SYSTEM_EXEC, *tools.reads), net=Net.deny(), env=env,
                  limits=Limits(wall_seconds=wall, output_bytes=50_000_000))


@dataclass(frozen=True, slots=True)
class Job:
    """One compile: the source tree, the one directory it may write, the build number and
    commit the binary reports, and how it runs."""

    source: Path
    work: Path
    number: int | None
    commit: str
    jobs: int
    tools: Toolchain
    wall_seconds: float = WALL_SECONDS


def compile_source(job: Job, say: Callable[[str], None] = lambda _text: None) -> Path:
    """Configure and build ``job.source`` with fixed cmake arguments inside the sandbox: it reads
    the source tree and the toolchain, writes only ``job.work``, and has no network. Returns the
    built server binary; `BuildFailed` carries the tail of the output otherwise."""
    tools = job.tools
    work, source = Path(os.path.realpath(job.work)), Path(os.path.realpath(job.source))
    policy = _policy(tools, source, work, job.wall_seconds)
    build = work / "build"
    version = [f"-DLLAMA_BUILD_NUMBER={job.number}"] if job.number is not None else []
    steps = (
        ("configure", [tools.cmake, "-S", str(source), "-B", str(build), *FIXED_FLAGS, *tools.arguments(),
                       *cmake_flags(), "-DLLAMA_CURL=OFF", *version, f"-DLLAMA_BUILD_COMMIT={job.commit[:9]}"]),
        ("compile", [tools.cmake, "--build", str(build), "--config", "Release", "--target",
                     CMAKE_TARGET, "-j", str(job.jobs)]),
    )
    for name, argv in steps:
        say(f"  {name} (sandboxed: no network, writes only {work})")
        done = sandbox.run(argv, policy, cwd=str(work))
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip()[-3000:]
            raise BuildFailed(f"{name} failed ({'timed out' if done.timed_out else done.returncode}):\n{tail}")
    built = build / "bin" / server_name()
    if not built.is_file():
        raise BuildFailed(f"the build produced no {server_name()} in {built.parent}")
    return built
