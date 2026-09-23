"""Building llama-server from master (or downloading a release), trusted only once measured.

The homebrew bottle lags master by an architecture or two -- gemma4 and qwen3moe shipped in
a release before qwen4exp did -- and a hand-built binary sitting in one person's home
directory fixed that for one machine, selected by one env var, silently, while every other
bench run kept loading the stale bottle. These tests fake git, cmake, the GitHub releases
API and the download it drives, the same way the rest of this suite fakes `subprocess.run`
directly (see ``test_setup.py``) -- but the *built binary itself* is a real, tiny,
executable shell script, so `flags_of` and `--version` run for real against it, and real
``strings`` genuinely reads the fake dylibs these tests write. Nothing here compiles
anything or reaches the network.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml_stack.serve.binary as binary_module
from ml_stack.serve import (
    build,
    build_persist,
    build_platform,
    build_release,
    build_report,
    build_source,
    build_verify,
)

_REAL_RUN = subprocess.run   # captured before any test patches subprocess.run

_ok = lambda stdout="", stderr="", returncode=0: SimpleNamespace(  # noqa: E731
    returncode=returncode, stdout=stdout, stderr=stderr)

HELP = "-m,    --model FNAME   model path\n-c,    --ctx-size N    context size\n"
VERSION = "llama-server 0.3.0-dev (build 1, commit abc1234)"


def _fake_server_script(path: Path, *, help_text: str = HELP, version: str = VERSION) -> None:
    """A real, tiny executable standing in for the compiled binary -- real enough that
    `flags_of` and `--version` run it for real rather than being told what it says."""
    path.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        f"  --help) cat <<'HELP'\n{help_text}HELP\n  ;;\n"
        f"  --version) echo '{version}' ;;\n"
        "esac\nexit 0\n")
    path.chmod(0o755)


def _fake_libllama(path: Path, arches: set[str]) -> None:
    """A file real ``strings`` will read the given architecture words out of."""
    path.write_bytes(("\n".join(sorted(arches)) + "\n").encode())


class FakeToolchain:
    """Stands in for git and cmake: records every command, and does just enough on disk
    for the next step -- a source tree, a build/bin/llama-server -- to find what it needs.
    Anything that is not git or cmake (the compiled binary answering --help/--version) is
    run for real, against a real executable."""

    def __init__(self, *, commit: str = "abc1234", arches: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.commit = commit
        self.arches = arches if arches is not None else {"gemma4", "qwen4exp"}

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        prog = Path(argv[0]).name
        if prog == "git":
            return self._git(argv[1:], kwargs.get("cwd"))
        if prog == "cmake":
            return self._cmake(argv[1:], kwargs.get("cwd"))
        return _REAL_RUN(argv, **kwargs)

    def _git(self, args, cwd):
        if args[0] == "clone":
            dest = Path(args[-1])
            (dest / ".git").mkdir(parents=True, exist_ok=True)
            (dest / "src").mkdir(exist_ok=True)
            return _ok()
        if args[0] == "rev-parse":
            return _ok(stdout=self.commit + "\n")
        return _ok()          # fetch, checkout, merge: nothing on disk is needed

    def _cmake(self, args, cwd):
        bin_dir = Path(cwd) / "build" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        if args and args[0] == "-B":
            return _ok()
        # --build: drop the binary and a library carrying the architecture words in place.
        _fake_server_script(bin_dir / build_platform.server_name())
        _fake_libllama(bin_dir / "libllama.0.3.0.dylib", self.arches)
        return _ok()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Every managed path lives under tmp_path, never a real ~/.ml-stack."""
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "ml-stack"))
    monkeypatch.setattr(build_persist, "PERSIST_PLIST",
                        tmp_path / "Library" / "LaunchAgents" / f"{build_persist.PERSIST_LABEL}.plist")
    monkeypatch.setattr(build_report, "find_binary", lambda *a, **k: None)
    monkeypatch.setattr(build_release, "arches_at", lambda ref, **k: set())
    # No patches, unless a test points this at a directory holding some.
    monkeypatch.setenv("MLSTACK_LLAMA_PATCHES", str(tmp_path / "no-patches"))
    yield


def _args(**over):
    base = {"commit": "", "jobs": 0, "source": "", "force": False, "check": False,
            "rollback": False, "persist": False, "source_kind": "source", "adopt": "",
            "repo": "", "ref": "", "tag": "", "name": "", "list": False}
    base.update(over)
    return SimpleNamespace(**base)


class TestSourceBuild:
    def test_a_fresh_checkout_is_cloned_then_configured_then_compiled_then_installed(
            self, monkeypatch):
        chain = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain.run)

        code = build.cmd_build(_args())
        assert code == 0

        assert Path(chain.calls[0][0]).name == "git" and chain.calls[0][1] == "clone"
        cmake_calls = [c for c in chain.calls if Path(c[0]).name == "cmake"]
        assert cmake_calls[0][1] == "-B"
        assert "--target" in cmake_calls[1] and "llama-server" in cmake_calls[1]
        assert chain.calls.index(cmake_calls[0]) > chain.calls.index(
            next(c for c in chain.calls if Path(c[0]).name == "git" and c[1] == "clone"))

        dest = build.builds_dir() / "abc1234"
        assert (dest / build_platform.server_name()).is_file()
        assert (dest / "libllama.0.3.0.dylib").is_file()
        manifest = json.loads((dest / "BUILD.json").read_text())
        assert manifest["commit"] == "abc1234"
        assert manifest["source"] == "source"
        assert "built_at" in manifest and manifest["version"]

    def test_an_existing_checkout_is_fetched_and_fast_forwarded_not_cloned(
            self, monkeypatch):
        (build.src_dir() / ".git").mkdir(parents=True)
        chain = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain.run)

        assert build.cmd_build(_args()) == 0
        git_calls = [c[1:] for c in chain.calls if Path(c[0]).name == "git"]
        assert ["fetch", "--depth", "1", "origin", "master"] in git_calls
        assert ["checkout", "master"] in git_calls
        assert ["reset", "--hard", "origin/master"] in git_calls
        assert not any(c[0] == "clone" for c in git_calls)

    def test_a_fresh_clone_is_shallow(self, monkeypatch):
        """llama.cpp's full history is a large, slow clone that buys nothing here --
        matching how `ensure_converter` elsewhere in this codebase already clones."""
        chain = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain.run)
        assert build.cmd_build(_args()) == 0
        clone_call = next(c[1:] for c in chain.calls
                          if Path(c[0]).name == "git" and c[1] == "clone")
        assert "--depth" in clone_call and "1" in clone_call

    def test_a_commit_is_fetched_by_name_and_checked_out_after_syncing(self, monkeypatch):
        """A shallow clone has only master's tip -- a requested commit is fetched by name
        rather than assumed to already be present locally."""
        chain = FakeToolchain(commit="deadbee")
        monkeypatch.setattr(subprocess, "run", chain.run)

        assert build.cmd_build(_args(commit="deadbee")) == 0
        git_calls = [c[1:] for c in chain.calls if Path(c[0]).name == "git"]
        assert ["fetch", "--depth", "1", "origin", "deadbee"] in git_calls
        assert ["checkout", "FETCH_HEAD"] in git_calls
        assert (build.builds_dir() / "deadbee").is_dir()

    def test_an_already_built_commit_is_not_rebuilt_without_force(self, monkeypatch):
        chain = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain.run)
        assert build.cmd_build(_args()) == 0

        chain2 = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain2.run)
        assert build.cmd_build(_args()) == 0
        assert not any(Path(c[0]).name == "cmake" for c in chain2.calls), \
            "already built at this commit -- nothing should be compiled again"

        chain3 = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain3.run)
        assert build.cmd_build(_args(force=True)) == 0
        assert any(Path(c[0]).name == "cmake" for c in chain3.calls), \
            "--force must rebuild the same commit"


class TestVerificationGatesTheSwitch:
    def test_a_build_that_reads_fewer_architectures_than_the_current_one_is_refused(
            self, monkeypatch, capsys):
        """The whole point: a new build must not cost an architecture the old one read."""
        baseline_dir = build.root().parent / "baseline"
        baseline_dir.mkdir(parents=True)
        _fake_server_script(baseline_dir / build_platform.server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4", "qwen4exp", "phi9"})
        build_verify.relink(build.current_link(), baseline_dir)

        chain = FakeToolchain(arches={"gemma4", "qwen4exp"})   # missing phi9
        monkeypatch.setattr(subprocess, "run", chain.run)

        code = build.cmd_build(_args())
        assert code == 2
        err = capsys.readouterr().err
        assert "phi9" in err
        assert build.current_link().resolve() == baseline_dir.resolve(), \
            "a build that lost an architecture must not become current"

    def test_a_build_that_answers_no_help_is_refused(self, monkeypatch, capsys):
        chain = FakeToolchain()

        def silent_cmake(args, cwd):
            bin_dir = Path(cwd) / "build" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            if args and args[0] == "-B":
                return _ok()
            path = bin_dir / build_platform.server_name()
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
            return _ok()

        monkeypatch.setattr(chain, "_cmake", silent_cmake)
        monkeypatch.setattr(subprocess, "run", chain.run)

        code = build.cmd_build(_args())
        assert code == 2
        assert "did not answer --help" in capsys.readouterr().err
        assert not build.current_link().exists() and not build.current_link().is_symlink()

    def test_a_build_that_reads_a_superset_becomes_current(self, monkeypatch, capsys):
        baseline_dir = build.root().parent / "baseline"
        baseline_dir.mkdir(parents=True)
        _fake_server_script(baseline_dir / build_platform.server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4"})
        build_verify.relink(build.current_link(), baseline_dir)

        chain = FakeToolchain(arches={"gemma4", "qwen4exp"})
        monkeypatch.setattr(subprocess, "run", chain.run)

        assert build.cmd_build(_args()) == 0
        assert build.current_link().is_symlink()
        assert build.current_link().resolve() == (build.builds_dir() / "abc1234").resolve()
        out = capsys.readouterr().out
        assert "superset of the current 1" in out

    def test_a_dylib_string_that_only_shares_a_family_prefix_does_not_block_the_switch(
            self, monkeypatch):
        """Measured for real: `phi4` turns up in libllama and reads exactly like an
        architecture, but master's own llama-arch.cpp defines no LLM_ARCH_PHI4 -- it names
        a chat template, not a model architecture. A source checkout is what tells the
        difference; without one this heuristic cannot, so this is exactly what the source
        checkout `_arches_from_source` reads is for."""
        (build.src_dir() / "src").mkdir(parents=True)
        (build.src_dir() / "src" / "llama-arch.cpp").write_text(
            '{ LLM_ARCH_GEMMA4,   "gemma4"   },\n'
            '{ LLM_ARCH_QWEN4EXP, "qwen4exp" },\n')

        baseline_dir = build.root().parent / "baseline"
        baseline_dir.mkdir(parents=True)
        _fake_server_script(baseline_dir / build_platform.server_name())
        # The baseline has the false-positive string too -- exactly the real machine.
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4", "phi4"})
        build_verify.relink(build.current_link(), baseline_dir)

        chain = FakeToolchain(arches={"gemma4", "qwen4exp"})   # no "phi4" -- correctly so
        monkeypatch.setattr(subprocess, "run", chain.run)

        assert build.cmd_build(_args()) == 0
        assert build.current_link().resolve() == (build.builds_dir() / "abc1234").resolve()


class TestRollback:
    def test_rolling_back_points_current_at_the_previous_build(self, monkeypatch):
        for commit in ("first01", "second2"):
            chain = FakeToolchain(commit=commit)
            monkeypatch.setattr(subprocess, "run", chain.run)
            assert build.cmd_build(_args()) == 0

        assert build.current_link().resolve() == (build.builds_dir() / "second2").resolve()
        assert build.cmd_build(_args(rollback=True)) == 0
        assert build.current_link().resolve() == (build.builds_dir() / "first01").resolve()

    def test_rolling_back_with_nothing_earlier_fails_rather_than_doing_nothing_silently(
            self, monkeypatch, capsys):
        chain = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain.run)
        assert build.cmd_build(_args()) == 0

        code = build.cmd_build(_args(rollback=True))
        assert code == 2
        assert "no earlier build" in capsys.readouterr().err


class TestFindBinaryPrefersTheManagedBuild:
    def test_it_wins_over_path_but_not_over_the_env_var_or_an_explicit_path(
            self, tmp_path, monkeypatch):

        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
        managed = binary_module.managed_current()
        managed.mkdir(parents=True)
        (managed / "llama-server").write_text("#!/bin/sh\nexit 0\n")
        (managed / "llama-server").chmod(0o755)

        on_path = tmp_path / "path-bin" / "llama-server"
        on_path.parent.mkdir()
        on_path.write_text("#!/bin/sh\nexit 0\n")
        on_path.chmod(0o755)
        monkeypatch.setenv("PATH", str(on_path.parent))
        monkeypatch.delenv("LLAMA_CPP_SERVER", raising=False)
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)

        assert binary_module.find_binary("llama-server") == managed / "llama-server"

        explicit = tmp_path / "explicit" / "llama-server"
        explicit.parent.mkdir()
        explicit.write_text("#!/bin/sh\nexit 0\n")
        explicit.chmod(0o755)
        assert binary_module.find_binary("llama-server", explicit=explicit) == explicit

        env_pick = tmp_path / "env" / "llama-server"
        env_pick.parent.mkdir()
        env_pick.write_text("#!/bin/sh\nexit 0\n")
        env_pick.chmod(0o755)
        monkeypatch.setenv("LLAMA_CPP_SERVER", str(env_pick))
        assert binary_module.find_binary("llama-server") == env_pick

    def test_a_named_build_wins_over_current_but_not_over_explicit_or_the_env_var(
            self, tmp_path, monkeypatch):

        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
        current = binary_module.managed_current()
        current.mkdir(parents=True)
        (current / "llama-server").write_text("#!/bin/sh\nexit 0\n")
        (current / "llama-server").chmod(0o755)

        unsloth = binary_module.managed_named() / "unsloth"
        unsloth.mkdir(parents=True)
        (unsloth / "llama-server").write_text("#!/bin/sh\nexit 0\n")
        (unsloth / "llama-server").chmod(0o755)

        monkeypatch.delenv("LLAMA_CPP_SERVER", raising=False)
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.delenv("MLSTACK_LLAMA_BUILD", raising=False)
        monkeypatch.setenv("PATH", "")

        # No build named: current, as always.
        assert binary_module.find_binary("llama-server") == current / "llama-server"

        # build= outranks current.
        assert (binary_module.find_binary("llama-server", build="unsloth")
                == unsloth / "llama-server")

        # $MLSTACK_LLAMA_BUILD does the same for a caller with no build= to pass.
        monkeypatch.setenv("MLSTACK_LLAMA_BUILD", "unsloth")
        assert binary_module.find_binary("llama-server") == unsloth / "llama-server"

        # An explicit path, or $LLAMA_CPP_SERVER, still wins over a named build.
        explicit = tmp_path / "explicit" / "llama-server"
        explicit.parent.mkdir()
        explicit.write_text("#!/bin/sh\nexit 0\n")
        explicit.chmod(0o755)
        assert binary_module.find_binary("llama-server", explicit=explicit,
                                         build="unsloth") == explicit

        env_pick = tmp_path / "env" / "llama-server"
        env_pick.parent.mkdir()
        env_pick.write_text("#!/bin/sh\nexit 0\n")
        env_pick.chmod(0o755)
        monkeypatch.setenv("LLAMA_CPP_SERVER", str(env_pick))
        assert binary_module.find_binary("llama-server", build="unsloth") == env_pick

        # A named build that does not exist is refused, naming it and the command that builds it.
        monkeypatch.delenv("LLAMA_CPP_SERVER", raising=False)
        with pytest.raises(binary_module.BinaryNotFound) as refused:
            binary_module.find_binary("llama-server", build="missing")
        assert "'missing'" in str(refused.value)
        assert "ml-stack-serve build --repo OWNER/REPO" in str(refused.value)
        assert "--name missing" in str(refused.value)

        monkeypatch.setenv("MLSTACK_LLAMA_BUILD", "missing")
        with pytest.raises(binary_module.BinaryNotFound, match="MLSTACK_LLAMA_BUILD"):
            binary_module.find_binary("llama-server")


class TestCheck:
    def test_nothing_built_and_nothing_on_path_says_so(self, monkeypatch):
        monkeypatch.setattr(build_report, "find_binary", lambda *a, **k: None)
        assert build.cmd_build(_args(check=True)) == 1

    def test_a_managed_build_reports_its_commit_and_age(self, monkeypatch, capsys):
        chain = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain.run)
        assert build.cmd_build(_args()) == 0

        assert build.cmd_build(_args(check=True)) == 0
        out = capsys.readouterr().out
        assert "abc1234" in out
        assert "source" in out


class TestAdopt:
    """``--adopt`` registers a flat build directory that already exists -- a hand-built
    binary like ``~/.local/llama-next``, or a release zip someone unpacked by hand -- as a
    managed build, without compiling or downloading anything. Run for real against
    ``~/.local/llama-next`` on 2026-09-01: it correctly refused to switch, because that
    build (commit 62acc89, built Aug 31) reads qwen4exp but not phi4, and the brew build it
    would have replaced reads phi4 -- a real, measured regression, not a false positive in
    the check."""

    def _existing_dir(self, tmp_path, *, version="llama-server 0.3.0-dev (build 1, "
                                                  "commit 62acc89)", arches=None):
        source = tmp_path / "hand-built"
        source.mkdir()
        _fake_server_script(source / build_platform.server_name(), version=version)
        _fake_libllama(source / "libllama.dylib", arches or {"gemma4", "qwen4exp"})
        return source

    def test_a_directory_with_a_superset_is_adopted_and_becomes_current(
            self, tmp_path, monkeypatch):
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        _fake_server_script(baseline_dir / build_platform.server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4"})
        build_verify.relink(build.current_link(), baseline_dir)

        source = self._existing_dir(tmp_path, arches={"gemma4", "qwen4exp"})
        code = build.cmd_build(_args(adopt=str(source)))
        assert code == 0

        dest = build.builds_dir() / "62acc89"
        assert build.current_link().resolve() == dest.resolve()
        manifest = json.loads((dest / "BUILD.json").read_text())
        assert manifest["commit"] == "62acc89"
        assert manifest["source"] == f"adopted from {source.resolve()}"
        assert (dest / build_platform.server_name()).is_file()
        assert (dest / "libllama.dylib").is_file()

    def test_a_directory_missing_an_architecture_the_current_build_has_is_refused(
            self, tmp_path, monkeypatch):
        """The exact case hit for real against ~/.local/llama-next: newer in one
        architecture, older in another -- not a strict superset, so it must not switch."""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        _fake_server_script(baseline_dir / build_platform.server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4", "phi4"})
        build_verify.relink(build.current_link(), baseline_dir)

        source = self._existing_dir(tmp_path, arches={"gemma4", "qwen4exp"})   # no phi4
        code = build.cmd_build(_args(adopt=str(source)))
        assert code == 2
        assert build.current_link().resolve() == baseline_dir.resolve()
        # but it is still registered as a build, ready for --rollback-style bookkeeping
        # once verified -- adoption itself (copying it in) is not what was refused
        assert (build.builds_dir() / "62acc89" / "BUILD.json").is_file()

    def test_the_commit_is_read_from_version_when_it_names_one(self, tmp_path, monkeypatch):
        source = self._existing_dir(
            tmp_path, version="version: 0.3.0 (build 10621, commit c1d0e7a00)")
        assert build.cmd_build(_args(adopt=str(source))) == 0
        assert (build.builds_dir() / "c1d0e7a00").is_dir()

    def test_a_version_with_no_commit_falls_back_to_a_slug_rather_than_failing(
            self, tmp_path, monkeypatch):
        source = self._existing_dir(tmp_path, version="llama-server v9.9.9")
        assert build.cmd_build(_args(adopt=str(source))) == 0
        made = list(build.builds_dir().iterdir())
        assert len(made) == 1 and made[0].name

    def test_a_directory_with_no_server_binary_fails_rather_than_adopting_nothing(
            self, tmp_path, monkeypatch, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        code = build.cmd_build(_args(adopt=str(empty)))
        assert code == 2
        assert "no llama-server" in capsys.readouterr().err

    def test_a_missing_directory_fails_rather_than_adopting_nothing(self, tmp_path):
        code = build.cmd_build(_args(adopt=str(tmp_path / "nowhere")))
        assert code == 2


class TestPersist:
    def test_on_macos_it_writes_a_launchagent_that_reruns_the_build_weekly(
            self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        calls = []
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: calls.append((a, k)) or _ok())
        monkeypatch.setattr(shutil, "which",
                            lambda name: "/usr/local/bin/ml-stack-serve"
                            if name == "ml-stack-serve" else None)

        assert build.cmd_build(_args(persist=True)) == 0
        assert build_persist.PERSIST_PLIST.is_file()

        import plistlib
        plist = plistlib.loads(build_persist.PERSIST_PLIST.read_bytes())
        assert plist["ProgramArguments"] == ["/usr/local/bin/ml-stack-serve", "build"]
        assert plist["StartInterval"] == build_persist.WEEK_SECONDS
        assert plist["Label"] == build_persist.PERSIST_LABEL

        launchctl_calls = [c[0][0] for c in calls if Path(c[0][0][0]).name == "launchctl"]
        assert any("load" in c for c in launchctl_calls)

    def test_on_windows_it_registers_a_weekly_scheduled_task(self, monkeypatch):
        """Untested against a real Windows machine -- this only proves schtasks is called
        with the right arguments against a faked subprocess.run."""
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return _ok()

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(shutil, "which",
                            lambda name: "C:\\ml-stack\\ml-stack-serve.exe"
                            if name == "ml-stack-serve" else None)

        assert build.cmd_build(_args(persist=True)) == 0
        assert len(calls) == 1
        argv = calls[0]
        assert argv[0] == "schtasks"
        assert "/Create" in argv and "/SC" in argv
        assert argv[argv.index("/SC") + 1] == "WEEKLY"
        assert argv[argv.index("/TN") + 1] == build_persist.PERSIST_TASK
        tr = argv[argv.index("/TR") + 1]
        assert "ml-stack-serve" in tr and "build" in tr

    def test_a_refused_launchctl_job_fails_rather_than_pretending_to_be_installed(
            self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(subprocess, "run",
                            lambda *a, **k: _ok(returncode=1, stderr="nope"))
        monkeypatch.setattr(shutil, "which", lambda name: None)

        assert build.cmd_build(_args(persist=True)) == 2


class TestPlatformSelection:
    """``--from`` defaults to source only when a compiler is actually on PATH."""

    def test_defaults_to_release_with_no_compiler(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        assert build_platform.can_build_from_source() is False

    def test_defaults_to_source_when_cmake_and_a_compiler_are_present(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(shutil, "which",
                            lambda name: f"/usr/bin/{name}" if name in ("cmake", "cc") else None)
        assert build_platform.can_build_from_source() is True

    def test_windows_asset_glob_prefers_cuda_then_vulkan_then_cpu(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "machine", lambda: "AMD64")

        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.delenv("VULKAN_SDK", raising=False)
        assert build_release._release_asset_globs() == ["llama-*-bin-win-cpu-x64.zip"]

        monkeypatch.setenv("VULKAN_SDK", "C:\\VulkanSDK")
        assert build_release._release_asset_globs() == [
            "llama-*-bin-win-vulkan-x64.zip", "llama-*-bin-win-cpu-x64.zip"]

        monkeypatch.setattr(shutil, "which", lambda name: "nvcc.exe" if name == "nvcc" else None)
        assert build_release._release_asset_globs()[0] == "llama-*-bin-win-cuda-12.4-x64.zip"

    def test_macos_and_linux_asset_globs_are_tar_gz_not_zip(self, monkeypatch):
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert build_release._release_asset_globs() == ["llama-*-bin-macos-arm64.tar.gz"]

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert build_release._release_asset_globs() == ["llama-*-bin-ubuntu-arm64.tar.gz"]

    def test_cudart_companion_is_matched_by_the_same_cuda_version(self):
        assets = {
            "llama-b1-bin-win-cuda-12.4-x64.zip": {"name": "llama-b1-bin-win-cuda-12.4-x64.zip"},
            "cudart-llama-bin-win-cuda-12.4-x64.zip":
                {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip"},
            "cudart-llama-bin-win-cuda-13.3-x64.zip":
                {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip"},
        }
        found = build_release._cudart_companion("llama-b1-bin-win-cuda-12.4-x64.zip", assets)
        assert found is not None and found["name"] == "cudart-llama-bin-win-cuda-12.4-x64.zip"
        assert build_release._cudart_companion("llama-b1-bin-win-vulkan-x64.zip", assets) is None


class TestNamedBuild:
    """``--repo OWNER/REPO --name NAME`` builds a fork and keeps it beside ``current``
    instead of replacing it -- e.g. unslothai/llama.cpp, whose MTP heads mainline cannot
    load yet."""

    def test_a_named_source_build_is_linked_and_current_is_untouched(self, monkeypatch):
        chain = FakeToolchain(commit="86bd2d3", arches={"gemma4", "qwen4exp"})
        monkeypatch.setattr(subprocess, "run", chain.run)

        code = build.cmd_build(
            _args(repo="unslothai/llama.cpp", ref="b10715-mix-86bd2d3", name="unsloth"))
        assert code == 0

        dest = build.builds_dir() / "unsloth-86bd2d3"
        assert (dest / build_platform.server_name()).is_file()
        manifest = json.loads((dest / "BUILD.json").read_text())
        assert manifest["repo"] == "unslothai/llama.cpp"
        assert manifest["ref"] == "b10715-mix-86bd2d3"
        assert manifest["name"] == "unsloth"

        assert (build.named_dir() / "unsloth").resolve() == dest.resolve()
        assert not build.current_link().exists() and not build.current_link().is_symlink(), \
            "a named build must never become 'current'"

    def test_a_named_build_missing_an_architecture_the_baseline_has_is_linked_anyway(
            self, monkeypatch, capsys):
        """The whole point of --name: a fork is not required to be a superset of master's
        own build, unlike the default build a bare 'ml-stack-serve build' would replace."""
        baseline_dir = build.root().parent / "baseline"
        baseline_dir.mkdir(parents=True)
        _fake_server_script(baseline_dir / build_platform.server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4", "qwen4exp"})
        build_verify.relink(build.current_link(), baseline_dir)

        chain = FakeToolchain(arches={"gemma4"})   # missing qwen4exp -- fine for --name
        monkeypatch.setattr(subprocess, "run", chain.run)

        code = build.cmd_build(_args(repo="unslothai/llama.cpp", name="unsloth"))
        assert code == 0
        out = capsys.readouterr().out
        assert "missing qwen4exp" in out
        assert (build.named_dir() / "unsloth").is_symlink()
        assert build.current_link().resolve() == baseline_dir.resolve()

    def test_name_without_repo_is_refused(self, capsys):
        code = build.cmd_build(_args(name="unsloth"))
        assert code == 2
        assert "--repo" in capsys.readouterr().err

    def test_a_named_release_build_downloads_the_forks_own_asset_naming(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")

        releases = [{"tag_name": "b10715-mix-86bd2d3", "assets": [
            {"name": "app-b10715-mix-86bd2d3-windows-x64-cpu.zip"},   # not this machine
            {"name": "llama-b10715-mix-86bd2d3-bin-macos-arm64.tar.gz", "size": 0},
        ]}]
        monkeypatch.setattr(build_release, "_releases_for", lambda repo, **kw: releases)

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        payload = archive_dir / "payload"
        payload.mkdir()
        _fake_server_script(payload / build_platform.server_name())
        _fake_libllama(payload / "libllama.dylib", {"gemma4"})
        archive = archive_dir / "llama-b10715-mix-86bd2d3-bin-macos-arm64.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(payload, arcname=payload.name)

        downloaded = []

        def fake_download(asset, into, **kw):
            downloaded.append(asset["name"])
            return archive

        import ml_stack.fleet.updates as gh_updates
        monkeypatch.setattr(gh_updates, "download", fake_download)

        code = build.cmd_build(_args(repo="unslothai/llama.cpp", name="unsloth",
                                     source_kind="release"))
        assert code == 0
        assert downloaded == ["llama-b10715-mix-86bd2d3-bin-macos-arm64.tar.gz"]
        manifest = json.loads(
            (build.named_dir() / "unsloth" / "BUILD.json").read_text())
        assert manifest["commit"] == "b10715-mix-86bd2d3"
        assert manifest["source"] == "release"
        assert manifest["repo"] == "unslothai/llama.cpp"

    def test_list_shows_current_and_every_named_build(self, monkeypatch, capsys):
        chain = FakeToolchain(commit="abc1234")
        monkeypatch.setattr(subprocess, "run", chain.run)
        assert build.cmd_build(_args()) == 0

        chain2 = FakeToolchain(commit="86bd2d3")
        monkeypatch.setattr(subprocess, "run", chain2.run)
        assert build.cmd_build(
            _args(repo="unslothai/llama.cpp", ref="b10715-mix-86bd2d3", name="unsloth")) == 0

        code = build.cmd_build(_args(list=True))
        assert code == 0
        out = capsys.readouterr().out
        assert "current" in out and "abc1234" in out
        assert "unsloth" in out and "86bd2d3" in out and "unslothai/llama.cpp" in out

    def test_list_with_nothing_built_says_so(self, capsys):
        code = build.cmd_build(_args(list=True))
        assert code == 0
        out = capsys.readouterr().out
        assert "not built yet" in out
        assert "no named builds" in out


class TestNamedReleaseAssetGlobs:
    """Encoding unslothai/llama.cpp's own release asset naming, read for real on
    2026-09-01 -- a different shape from ggml-org/llama.cpp's own releases."""

    def test_macos_matches_the_same_shape_mainline_uses(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        assert build_release._release_asset_globs(fork=True) == ["llama-*-bin-macos-arm64.tar.gz"]

    def test_windows_uses_the_forks_app_bundle_naming_not_llama_bin(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "machine", lambda: "AMD64")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.delenv("VULKAN_SDK", raising=False)
        assert build_release._release_asset_globs(fork=True) == ["app-*-windows-x64-cpu.zip"]


def _serve_a_release(monkeypatch, tmp_path, tag: str, arches: set[str],
                     defined: set[str] | None = None) -> list[str]:
    """Fake GitHub releases holding one real macOS arm64 archive; returns the downloads made."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    name = f"llama-{tag}-bin-macos-arm64"
    releases = [
        {"tag_name": "b999999", "assets": [{"name": "llama-b999999-ui.tar.gz"}]},  # no match
        {"tag_name": tag, "assets": [{"name": f"{name}.tar.gz", "size": 0}]},
    ]
    monkeypatch.setattr(build_release, "_llama_releases", lambda *a, **k: releases)
    if defined is not None:
        monkeypatch.setattr(build_release, "arches_at",
                            lambda ref, **k: set(defined) if ref == tag else set())

    archive_dir = tmp_path / "archive"
    payload = archive_dir / name
    payload.mkdir(parents=True)
    _fake_server_script(payload / build_platform.server_name())
    _fake_libllama(payload / "libllama.dylib", arches)
    archive = archive_dir / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload, arcname=payload.name)

    downloaded: list[str] = []

    def fake_download(asset, into, **kw):
        downloaded.append(asset["name"])
        return archive

    import ml_stack.fleet.updates as gh_updates
    monkeypatch.setattr(gh_updates, "download", fake_download)
    return downloaded


def _current_build(tag: str, arches: set[str], defined: set[str]) -> Path:
    """A managed build ``current`` points at, recording the architectures its source defines."""
    dest = build.builds_dir() / tag
    dest.mkdir(parents=True)
    _fake_server_script(dest / build_platform.server_name())
    _fake_libllama(dest / "libllama.dylib", arches)
    (dest / "BUILD.json").write_text(json.dumps(
        {"commit": tag, "source": "release", "arches": sorted(defined)}))
    build_verify.relink(build.current_link(), dest)
    return dest


def test_the_names_llama_cpp_defines_are_read_whatever_family_they_belong_to(tmp_path):
    import ml_stack.setup as setup_module

    _fake_libllama(tmp_path / "libllama.dylib", {"t5", "bert", "gemma3n_e", "gemma4", "phi4", "chatml"})
    assert setup_module._arches(tmp_path, known={"t5", "bert", "gemma3n_e", "gemma4", "mamba"}) == {
        "t5", "bert", "gemma3n_e", "gemma4"}
    assert setup_module._arches(tmp_path) == {"gemma4", "phi4"}


class TestReleaseInstall:
    def test_the_newest_release_with_a_matching_asset_is_downloaded_and_installed(
            self, monkeypatch, tmp_path):
        downloaded = _serve_a_release(monkeypatch, tmp_path, "b998", {"gemma4"},
                                      defined={"gemma4", "qwen4exp"})

        code = build.cmd_build(_args(source_kind="release"))
        assert code == 0
        assert downloaded == ["llama-b998-bin-macos-arm64.tar.gz"]
        dest = build.builds_dir() / "b998"
        assert (dest / build_platform.server_name()).is_file()
        manifest = json.loads((dest / "BUILD.json").read_text())
        assert manifest["commit"] == "b998"
        assert manifest["source"] == "release"
        assert manifest["arches"] == ["gemma4", "qwen4exp"]

    def test_a_first_build_becomes_current_whatever_llama_server_is_on_path(
            self, monkeypatch, tmp_path):
        on_path = tmp_path / "homebrew" / "bin"
        on_path.mkdir(parents=True)
        _fake_server_script(on_path / build_platform.server_name())
        _fake_libllama(on_path / "libllama.dylib", {"gemma4", "phi4", "olmo9"})
        for name in ("LLAMA_CPP_SERVER", "LLAMA_CPP_DIR", "MLSTACK_LLAMA_BUILD"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("PATH", f"{on_path}{os.pathsep}/usr/bin{os.pathsep}/bin")
        monkeypatch.setattr(binary_module, "_LOGIN_SHELL_DIRS", ())
        assert binary_module.find_binary("llama-server") == on_path / "llama-server"
        _serve_a_release(monkeypatch, tmp_path, "b11147", {"gemma4", "qwen4exp"})

        assert build.cmd_build(_args(source_kind="release")) == 0
        assert build.current_link().resolve() == (build.builds_dir() / "b11147").resolve()

    def test_a_chat_template_name_in_the_current_build_is_not_an_architecture_lost(
            self, monkeypatch, tmp_path):
        _current_build("b10621", {"gemma4", "phi4"}, defined={"gemma4"})
        _serve_a_release(monkeypatch, tmp_path, "b11147", {"gemma4", "qwen4exp"},
                         defined={"gemma4", "qwen4exp"})

        assert build.cmd_build(_args(source_kind="release")) == 0
        assert build.current_link().resolve() == (build.builds_dir() / "b11147").resolve()

    def test_an_architecture_the_current_source_defines_and_the_new_build_lacks_is_refused(
            self, monkeypatch, tmp_path, capsys):
        old = _current_build("b10621", {"gemma4", "phi4", "olmo9"}, defined={"gemma4", "olmo9"})
        _serve_a_release(monkeypatch, tmp_path, "b11147", {"gemma4"}, defined={"gemma4"})

        assert build.cmd_build(_args(source_kind="release")) == 2
        err = capsys.readouterr().err
        assert "missing olmo9," in err and "phi4" not in err
        assert build.current_link().resolve() == old.resolve()

    def test_no_matching_asset_in_recent_releases_fails_rather_than_silently_picking_one(
            self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        monkeypatch.setattr(build_release, "_llama_releases",
                            lambda *a, **k: [{"tag_name": "b1", "assets": []}])
        code = build.cmd_build(_args(source_kind="release"))
        assert code == 2


class TestPatches:
    """The patches a source build carries on top of upstream."""

    @staticmethod
    def _repo(tmp_path: Path) -> Path:
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.txt").write_text("one\ntwo\nthree\n")
        for args in (("init", "-q"), ("add", "a.txt"),
                     ("-c", "user.email=t@t", "-c", "user.name=t",
                      "commit", "-qm", "first")):
            _REAL_RUN(["git", *args], cwd=source, check=True, capture_output=True)
        return source

    @staticmethod
    def _patch(tmp_path: Path, body: str) -> Path:
        where = tmp_path / "patches"
        where.mkdir(exist_ok=True)
        (where / "0001-x.patch").write_text(body)
        return where

    BODY = ("diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n"
            "@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n")

    def test_the_shipped_patch_set_is_found_and_stamped(self, monkeypatch):
        monkeypatch.delenv("MLSTACK_LLAMA_PATCHES")
        names = [f.name for f in build.patch_files()]
        assert "0001-speculative-per-request.patch" in names
        assert build.patch_stamp().startswith("p")

    def test_no_patches_means_no_stamp(self):
        assert build.patch_stamp() == ""
        assert build.patch_files() == []

    def test_a_patch_is_applied_to_the_checkout(self, tmp_path, monkeypatch):
        source = self._repo(tmp_path)
        monkeypatch.setenv("MLSTACK_LLAMA_PATCHES", str(self._patch(tmp_path, self.BODY)))
        assert build_source._apply_patches(source) == ["0001-x.patch"]
        assert "TWO" in (source / "a.txt").read_text()

    def test_applying_twice_is_not_an_error(self, tmp_path, monkeypatch):
        source = self._repo(tmp_path)
        monkeypatch.setenv("MLSTACK_LLAMA_PATCHES", str(self._patch(tmp_path, self.BODY)))
        build_source._apply_patches(source)
        assert build_source._apply_patches(source) == ["0001-x.patch"]
        assert (source / "a.txt").read_text().count("TWO") == 1

    def test_a_changed_patch_set_does_not_land_on_the_last_one(self, tmp_path, monkeypatch):
        """A checkout the previous build patched is reset before this one is applied; two
        patch sets stacking is what makes a rebuild produce something neither describes."""
        source = self._repo(tmp_path)
        monkeypatch.setenv("MLSTACK_LLAMA_PATCHES", str(self._patch(tmp_path, self.BODY)))
        build_source._apply_patches(source)
        assert "TWO" in (source / "a.txt").read_text()

        other = self.BODY.replace("+TWO", "+DEUX")
        monkeypatch.setenv("MLSTACK_LLAMA_PATCHES", str(self._patch(tmp_path, other)))
        build_source._apply_patches(source)
        text = (source / "a.txt").read_text()
        assert "DEUX" in text and "TWO" not in text

    def test_a_patch_that_does_not_apply_fails_the_build(self, tmp_path, monkeypatch):
        source = self._repo(tmp_path)
        body = self.BODY.replace(" one\n-two", " ONE\n-two")
        monkeypatch.setenv("MLSTACK_LLAMA_PATCHES", str(self._patch(tmp_path, body)))
        with pytest.raises(build.BuildFailed, match="does not apply"):
            build_source._apply_patches(source)

    def test_the_stamp_names_the_build_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLSTACK_LLAMA_PATCHES", str(self._patch(tmp_path, self.BODY)))
        chain = FakeToolchain()
        monkeypatch.setattr(subprocess, "run", chain.run)
        monkeypatch.setattr(build_source, "_apply_patches", lambda source: ["0001-x.patch"])
        dest, tag = build_source.build_from_source(_args())
        assert tag.endswith("-" + build.patch_stamp())
        assert dest.name == tag
        assert json.loads((dest / "BUILD.json").read_text())["patches"] == ["0001-x.patch"]
