"""Package source admission and offline command construction."""

import subprocess
from pathlib import Path

import pytest

from ml_stack.httpguard import Refused
from ml_stack.net import packages, policy

pytestmark = pytest.mark.redteam


def capture(monkeypatch, tmp_path, config=""):
    calls = []
    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        output = config if "config" in argv else "installed"
        return subprocess.CompletedProcess(argv, 0, output, "")
    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setattr(policy, "default", lambda: policy.Policy(["packages.example"], path=tmp_path / "approvals"))
    return calls


@pytest.mark.parametrize("source", ["https://evil.example/simple", "http://packages.example/simple"])
def test_unapproved_or_plaintext_index_never_runs_install(monkeypatch, tmp_path, source):
    calls = capture(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        packages.run("python", ["install", "--index-url", source, "tensor"], timeout=5, env={})
    assert len(calls) == 1 and calls[0][0][3:] == ["config", "list"]


@pytest.mark.parametrize("source", ["tensor @ git+https://evil.example/repo", "git+ssh://user@packages.example/repo", "git+file:///tmp/repo"])
def test_unapproved_or_unsupported_vcs_never_runs_install(monkeypatch, tmp_path, source):
    calls = capture(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        packages.run("python", ["install", "--no-index", source], timeout=5, env={})
    assert len(calls) == 1


def test_trusted_index_find_links_extras_and_git_environment_are_preserved(monkeypatch, tmp_path):
    calls = capture(monkeypatch, tmp_path, "global.index-url='https://packages.example/simple'\nglobal.find-links='/bundle'\n")
    wheel = tmp_path / "runtime.whl"
    packages.run("python", ["install", "ml-stack[train] @ " + wheel.as_uri(), "tensor @ git+https://packages.example/repo"],
                 timeout=7, env={"GIT_SSH_COMMAND": "unsafe", "SDKROOT": "/sdk"})
    argv, options = calls[-1]
    assert argv == ["python", "-m", "pip", "install", "--index-url", "https://packages.example/simple",
                    "--find-links", "/bundle", "ml-stack[train] @ " + wheel.as_uri(), "tensor @ git+https://packages.example/repo"]
    assert options["timeout"] == 7
    env = options["env"]
    assert env["SDKROOT"] == "/sdk" and env["GIT_ALLOW_PROTOCOL"] == "https"
    assert "GIT_SSH_COMMAND" not in env
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"


@pytest.mark.parametrize("offline", ["argument", "environment", "config"])
def test_offline_installs_ignore_network_indexes_and_preserve_local_wheels(monkeypatch, tmp_path, offline):
    config = "global.index-url='https://evil.example/simple'\n"
    if offline == "config":
        config += "global.no-index='true'\n"
    if offline == "environment":
        config += ":env:.no-index='1'\n"
    calls = capture(monkeypatch, tmp_path, config)
    args = ["install", "--find-links", str(tmp_path), "ml-stack[train] @ " + (tmp_path / "runtime.whl").as_uri()]
    if offline == "argument":
        args.insert(1, "--no-index")
    packages.run(Path("python"), args, timeout=5, env={"PIP_INDEX_URL": "https://evil.example/simple"})
    argv, options = calls[-1]
    assert "--no-index" in argv and "--index-url" not in argv
    assert argv[-1] == args[-1] and str(tmp_path) in argv
    assert "PIP_INDEX_URL" not in options["env"]


@pytest.mark.parametrize("option", ["--trusted-host", "--proxy", "-r", "--constraint"])
def test_unchecked_source_configuration_is_refused(monkeypatch, tmp_path, option):
    calls = capture(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        packages.run("python", ["install", "--no-index", option, "unchecked"], timeout=5, env={})
    assert len(calls) == 1


def test_standard_package_provider_is_scoped_and_keeps_distrust_refusals(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from ml_stack.net.policy import Distrusted
    from ml_stack.sentinel import observers

    calls = capture(monkeypatch, tmp_path)
    packages.run("python", ["install", "tensor"], timeout=5, env={})
    assert calls[-1][0][4:6] == ["--index-url", "https://pypi.org/simple"]
    assert not policy.default().listed("pypi.org")
    monkeypatch.setattr(observers, "gate", lambda kind, key: SimpleNamespace(since=1, reason="distrusted"))
    calls.clear()
    with pytest.raises(Distrusted):
        packages.run("python", ["install", "tensor"], timeout=5, env={})
    assert len(calls) == 1


@pytest.mark.parametrize("source", ["file://foreign/share/wheel.whl", "//foreign/share", "\\\\foreign\\share"])
def test_remote_filesystems_do_not_bypass_package_source_admission(monkeypatch, tmp_path, source):
    calls = capture(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        packages.run("python", ["install", "--no-index", "--find-links", source, "tensor"], timeout=5, env={})
    assert len(calls) == 1


@pytest.mark.parametrize("source", [r"C:\wheelhouse", "D:/wheelhouse", "/bundle", "./wheelhouse"])
@pytest.mark.parametrize("configured", [False, True])
def test_local_find_links_paths_are_preserved(monkeypatch, tmp_path, source, configured):
    config = f":env:.find-links={source!r}\n" if configured else ""
    calls = capture(monkeypatch, tmp_path, config)
    args = ["install", "--no-index", "tensor"]
    if not configured:
        args.extend(["--find-links", source])
    packages.run("python", args, timeout=5, env={})
    argv = calls[-1][0]
    assert argv[argv.index("--find-links") + 1] == source


@pytest.mark.parametrize("source", [r"\\foreign\share", "//foreign/share", "c:relative", "c://foreign/share",
                                         "ftp://packages.example/wheels", "http://packages.example/wheels",
                                         "https://evil.example/wheels"])
@pytest.mark.parametrize("configured", [False, True])
def test_find_links_sources_require_admission(monkeypatch, tmp_path, source, configured):
    config = f":env:.find-links={source!r}\n" if configured else ""
    calls = capture(monkeypatch, tmp_path, config)
    args = ["install", "--no-index", "tensor"]
    if not configured:
        args.extend(["--find-links", source])
    with pytest.raises(Refused):
        packages.run("python", args, timeout=5, env={})
    assert len(calls) == 1


def test_configured_find_links_splits_whitespace_and_preserves_each_path(monkeypatch, tmp_path):
    sources = [r"C:\wheelhouse", r"D:\packages"]
    calls = capture(monkeypatch, tmp_path, f"global.find-links={chr(10).join(sources)!r}\n")
    packages.run("python", ["install", "--no-index", "tensor"], timeout=5, env={})
    assert calls[-1][0][4:] == ["--no-index", "--find-links", sources[0], "--find-links", sources[1], "tensor"]


@pytest.mark.parametrize("args", [["index", "versions", "tensor"], ["list", "--outdated"], ["list", "--uptodate"]])
def test_other_network_commands_cannot_bypass_admission(monkeypatch, tmp_path, args):
    calls = capture(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        packages.run("python", args, timeout=5, env={})
    assert calls == []


@pytest.mark.parametrize("name", ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"])
def test_unapproved_proxy_environment_never_runs_install(monkeypatch, tmp_path, name):
    calls = capture(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        packages.run("python", ["install", "tensor"], timeout=5, env={name: "https://evil.example"})
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["requirement", "constraint", "build-constraint", "editable", "target", "prefix", "root", "user"])
def test_configured_source_files_and_install_targets_are_refused(monkeypatch, tmp_path, name):
    calls = capture(monkeypatch, tmp_path, f":env:.{name}='1'\n")
    with pytest.raises(Refused):
        packages.run("python", ["install", "--no-index", "tensor"], timeout=5, env={})
    assert len(calls) == 1


@pytest.mark.parametrize("option", ["-echeckout", "--editable", "--build-constraint", "--target", "--prefix", "--root", "--user", "--isolated"])
def test_source_and_target_options_cannot_escape_allocated_interpreter(monkeypatch, tmp_path, option):
    calls = capture(monkeypatch, tmp_path)
    with pytest.raises(Refused):
        packages.run("python", ["install", "--no-index", option, "unchecked"], timeout=5, env={})
    assert len(calls) == 1
