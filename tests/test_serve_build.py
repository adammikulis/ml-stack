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
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from ml_stack.serve import build

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
        _fake_server_script(bin_dir / build._server_name())
        _fake_libllama(bin_dir / "libllama.0.3.0.dylib", self.arches)
        return _ok()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Every managed path lives under tmp_path, never a real ~/.ml-stack."""
    root = tmp_path / "ml-stack" / "llama.cpp"
    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "SRC_DIR", root / "src")
    monkeypatch.setattr(build, "BUILDS_DIR", root / "builds")
    monkeypatch.setattr(build, "CURRENT_LINK", root / "current")
    monkeypatch.setattr(build, "PERSIST_PLIST",
                        tmp_path / "Library" / "LaunchAgents" / f"{build.PERSIST_LABEL}.plist")
    # No earlier build to compare against, unless a test says otherwise.
    monkeypatch.setattr(build, "find_binary", lambda *a, **k: None)
    yield


def _args(**over):
    base = dict(commit="", jobs=0, source="", force=False, check=False, rollback=False,
               persist=False, source_kind="source", adopt="")
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

        dest = build.BUILDS_DIR / "abc1234"
        assert (dest / build._server_name()).is_file()
        assert (dest / "libllama.0.3.0.dylib").is_file()
        manifest = json.loads((dest / "BUILD.json").read_text())
        assert manifest["commit"] == "abc1234"
        assert manifest["source"] == "source"
        assert "built_at" in manifest and manifest["version"]

    def test_an_existing_checkout_is_fetched_and_fast_forwarded_not_cloned(
            self, monkeypatch):
        (build.SRC_DIR / ".git").mkdir(parents=True)
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
        assert (build.BUILDS_DIR / "deadbee").is_dir()

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
        baseline_dir = build.ROOT.parent / "baseline"
        baseline_dir.mkdir(parents=True)
        _fake_server_script(baseline_dir / build._server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4", "qwen4exp", "phi9"})
        monkeypatch.setattr(build, "find_binary",
                            lambda *a, **k: baseline_dir / build._server_name())

        chain = FakeToolchain(arches={"gemma4", "qwen4exp"})   # missing phi9
        monkeypatch.setattr(subprocess, "run", chain.run)

        code = build.cmd_build(_args())
        assert code == 2
        err = capsys.readouterr().err
        assert "phi9" in err
        assert not build.CURRENT_LINK.exists() and not build.CURRENT_LINK.is_symlink(), \
            "a build that lost an architecture must not become current"

    def test_a_build_that_answers_no_help_is_refused(self, monkeypatch, capsys):
        chain = FakeToolchain()

        def silent_cmake(args, cwd):
            bin_dir = Path(cwd) / "build" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            if args and args[0] == "-B":
                return _ok()
            path = bin_dir / build._server_name()
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
            return _ok()

        monkeypatch.setattr(chain, "_cmake", silent_cmake)
        monkeypatch.setattr(subprocess, "run", chain.run)

        code = build.cmd_build(_args())
        assert code == 2
        assert "did not answer --help" in capsys.readouterr().err
        assert not build.CURRENT_LINK.exists() and not build.CURRENT_LINK.is_symlink()

    def test_a_build_that_reads_a_superset_becomes_current(self, monkeypatch, capsys):
        baseline_dir = build.ROOT.parent / "baseline"
        baseline_dir.mkdir(parents=True)
        _fake_server_script(baseline_dir / build._server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4"})
        monkeypatch.setattr(build, "find_binary",
                            lambda *a, **k: baseline_dir / build._server_name())

        chain = FakeToolchain(arches={"gemma4", "qwen4exp"})
        monkeypatch.setattr(subprocess, "run", chain.run)

        assert build.cmd_build(_args()) == 0
        assert build.CURRENT_LINK.is_symlink()
        assert build.CURRENT_LINK.resolve() == (build.BUILDS_DIR / "abc1234").resolve()
        out = capsys.readouterr().out
        assert "superset of the current 1" in out

    def test_a_dylib_string_that_only_shares_a_family_prefix_does_not_block_the_switch(
            self, monkeypatch):
        """Measured for real: `phi4` turns up in libllama and reads exactly like an
        architecture, but master's own llama-arch.cpp defines no LLM_ARCH_PHI4 -- it names
        a chat template, not a model architecture. A source checkout is what tells the
        difference; without one this heuristic cannot, so this is exactly what the source
        checkout `_arches_from_source` reads is for."""
        (build.SRC_DIR / "src").mkdir(parents=True)
        (build.SRC_DIR / "src" / "llama-arch.cpp").write_text(
            '{ LLM_ARCH_GEMMA4,   "gemma4"   },\n'
            '{ LLM_ARCH_QWEN4EXP, "qwen4exp" },\n')

        baseline_dir = build.ROOT.parent / "baseline"
        baseline_dir.mkdir(parents=True)
        _fake_server_script(baseline_dir / build._server_name())
        # The baseline has the false-positive string too -- exactly the real machine.
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4", "phi4"})
        monkeypatch.setattr(build, "find_binary",
                            lambda *a, **k: baseline_dir / build._server_name())

        chain = FakeToolchain(arches={"gemma4", "qwen4exp"})   # no "phi4" -- correctly so
        monkeypatch.setattr(subprocess, "run", chain.run)

        assert build.cmd_build(_args()) == 0
        assert build.CURRENT_LINK.resolve() == (build.BUILDS_DIR / "abc1234").resolve()


class TestRollback:
    def test_rolling_back_points_current_at_the_previous_build(self, monkeypatch):
        for commit in ("first01", "second2"):
            chain = FakeToolchain(commit=commit)
            monkeypatch.setattr(subprocess, "run", chain.run)
            assert build.cmd_build(_args()) == 0

        assert build.CURRENT_LINK.resolve() == (build.BUILDS_DIR / "second2").resolve()
        assert build.cmd_build(_args(rollback=True)) == 0
        assert build.CURRENT_LINK.resolve() == (build.BUILDS_DIR / "first01").resolve()

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
        import ml_stack.serve.binary as binary_module

        managed = tmp_path / "current"
        managed.mkdir()
        (managed / "llama-server").write_text("#!/bin/sh\nexit 0\n")
        (managed / "llama-server").chmod(0o755)
        monkeypatch.setattr(binary_module, "MANAGED_CURRENT", managed)

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


class TestCheck:
    def test_nothing_built_and_nothing_on_path_says_so(self, monkeypatch):
        monkeypatch.setattr(build, "find_binary", lambda *a, **k: None)
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
        _fake_server_script(source / build._server_name(), version=version)
        _fake_libllama(source / "libllama.dylib", arches or {"gemma4", "qwen4exp"})
        return source

    def test_a_directory_with_a_superset_is_adopted_and_becomes_current(
            self, tmp_path, monkeypatch):
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        _fake_server_script(baseline_dir / build._server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4"})
        monkeypatch.setattr(build, "find_binary",
                            lambda *a, **k: baseline_dir / build._server_name())

        source = self._existing_dir(tmp_path, arches={"gemma4", "qwen4exp"})
        code = build.cmd_build(_args(adopt=str(source)))
        assert code == 0

        dest = build.BUILDS_DIR / "62acc89"
        assert build.CURRENT_LINK.resolve() == dest.resolve()
        manifest = json.loads((dest / "BUILD.json").read_text())
        assert manifest["commit"] == "62acc89"
        assert manifest["source"] == f"adopted from {source.resolve()}"
        assert (dest / build._server_name()).is_file()
        assert (dest / "libllama.dylib").is_file()

    def test_a_directory_missing_an_architecture_the_current_build_has_is_refused(
            self, tmp_path, monkeypatch):
        """The exact case hit for real against ~/.local/llama-next: newer in one
        architecture, older in another -- not a strict superset, so it must not switch."""
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        _fake_server_script(baseline_dir / build._server_name())
        _fake_libllama(baseline_dir / "libllama-old.dylib", {"gemma4", "phi4"})
        monkeypatch.setattr(build, "find_binary",
                            lambda *a, **k: baseline_dir / build._server_name())

        source = self._existing_dir(tmp_path, arches={"gemma4", "qwen4exp"})   # no phi4
        code = build.cmd_build(_args(adopt=str(source)))
        assert code == 2
        assert not build.CURRENT_LINK.exists() and not build.CURRENT_LINK.is_symlink()
        # but it is still registered as a build, ready for --rollback-style bookkeeping
        # once verified -- adoption itself (copying it in) is not what was refused
        assert (build.BUILDS_DIR / "62acc89" / "BUILD.json").is_file()

    def test_the_commit_is_read_from_version_when_it_names_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build, "find_binary", lambda *a, **k: None)
        source = self._existing_dir(
            tmp_path, version="version: 0.3.0 (build 10621, commit c1d0e7a00)")
        assert build.cmd_build(_args(adopt=str(source))) == 0
        assert (build.BUILDS_DIR / "c1d0e7a00").is_dir()

    def test_a_version_with_no_commit_falls_back_to_a_slug_rather_than_failing(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(build, "find_binary", lambda *a, **k: None)
        source = self._existing_dir(tmp_path, version="llama-server v9.9.9")
        assert build.cmd_build(_args(adopt=str(source))) == 0
        made = list(build.BUILDS_DIR.iterdir())
        assert len(made) == 1 and made[0].name

    def test_a_directory_with_no_server_binary_fails_rather_than_adopting_nothing(
            self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(build, "find_binary", lambda *a, **k: None)
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
        assert build.PERSIST_PLIST.is_file()

        import plistlib
        plist = plistlib.loads(build.PERSIST_PLIST.read_bytes())
        assert plist["ProgramArguments"] == ["/usr/local/bin/ml-stack-serve", "build"]
        assert plist["StartInterval"] == build.WEEK_SECONDS
        assert plist["Label"] == build.PERSIST_LABEL

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
        assert argv[argv.index("/TN") + 1] == build.PERSIST_TASK
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
        assert build._can_build_from_source() is False

    def test_defaults_to_source_when_cmake_and_a_compiler_are_present(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(shutil, "which",
                            lambda name: f"/usr/bin/{name}" if name in ("cmake", "cc") else None)
        assert build._can_build_from_source() is True

    def test_windows_asset_glob_prefers_cuda_then_vulkan_then_cpu(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "machine", lambda: "AMD64")

        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.delenv("VULKAN_SDK", raising=False)
        assert build._platform_asset_globs() == ["llama-*-bin-win-cpu-x64.zip"]

        monkeypatch.setenv("VULKAN_SDK", "C:\\VulkanSDK")
        assert build._platform_asset_globs() == [
            "llama-*-bin-win-vulkan-x64.zip", "llama-*-bin-win-cpu-x64.zip"]

        monkeypatch.setattr(shutil, "which", lambda name: "nvcc.exe" if name == "nvcc" else None)
        assert build._platform_asset_globs()[0] == "llama-*-bin-win-cuda-12.4-x64.zip"

    def test_macos_and_linux_asset_globs_are_tar_gz_not_zip(self, monkeypatch):
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        assert build._platform_asset_globs() == ["llama-*-bin-macos-arm64.tar.gz"]

        monkeypatch.setattr(platform, "system", lambda: "Linux")
        assert build._platform_asset_globs() == ["llama-*-bin-ubuntu-arm64.tar.gz"]

    def test_cudart_companion_is_matched_by_the_same_cuda_version(self):
        assets = {
            "llama-b1-bin-win-cuda-12.4-x64.zip": {"name": "llama-b1-bin-win-cuda-12.4-x64.zip"},
            "cudart-llama-bin-win-cuda-12.4-x64.zip":
                {"name": "cudart-llama-bin-win-cuda-12.4-x64.zip"},
            "cudart-llama-bin-win-cuda-13.3-x64.zip":
                {"name": "cudart-llama-bin-win-cuda-13.3-x64.zip"},
        }
        found = build._cudart_companion("llama-b1-bin-win-cuda-12.4-x64.zip", assets)
        assert found is not None and found["name"] == "cudart-llama-bin-win-cuda-12.4-x64.zip"
        assert build._cudart_companion("llama-b1-bin-win-vulkan-x64.zip", assets) is None


class TestReleaseInstall:
    def test_the_newest_release_with_a_matching_asset_is_downloaded_and_installed(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")

        releases = [
            {"tag_name": "b999", "assets": [{"name": "llama-b999-ui.tar.gz"}]},  # no match
            {"tag_name": "b998", "assets": [
                {"name": "llama-b998-bin-macos-arm64.tar.gz", "size": 0}]},
        ]
        monkeypatch.setattr(build, "_llama_releases", lambda *a, **k: releases)

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        payload = archive_dir / "llama-b998-bin-macos-arm64"
        payload.mkdir()
        _fake_server_script(payload / build._server_name())
        _fake_libllama(payload / "libllama.dylib", {"gemma4"})
        archive = archive_dir / "llama-b998-bin-macos-arm64.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(payload, arcname=payload.name)

        downloaded = []

        def fake_download(asset, into, **kw):
            downloaded.append(asset["name"])
            return archive

        import ml_stack.fleet.updates as gh_updates
        monkeypatch.setattr(gh_updates, "download", fake_download)
        monkeypatch.setattr(build, "find_binary", lambda *a, **k: None)

        code = build.cmd_build(_args(source_kind="release"))
        assert code == 0
        assert downloaded == ["llama-b998-bin-macos-arm64.tar.gz"]
        dest = build.BUILDS_DIR / "b998"
        assert (dest / build._server_name()).is_file()
        manifest = json.loads((dest / "BUILD.json").read_text())
        assert manifest["commit"] == "b998"
        assert manifest["source"] == "release"

    def test_no_matching_asset_in_recent_releases_fails_rather_than_silently_picking_one(
            self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        monkeypatch.setattr(build, "_llama_releases",
                            lambda *a, **k: [{"tag_name": "b1", "assets": []}])
        code = build.cmd_build(_args(source_kind="release"))
        assert code == 2
