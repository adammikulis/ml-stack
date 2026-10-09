"""Native Windows sandbox grant path validation."""

import sys

import pytest

from ml_stack.sandbox.policy import Policy, PolicyError, checked_path


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows paths")
def test_windows_local_paths_validate_in_every_grant(tmp_path):
    path = str(tmp_path)
    policy = Policy(read=(path,), write=(path,), exec=(sys.executable,), cache=path).validated()
    assert policy.read == (path,)
    assert policy.write == (path,)
    assert policy.exec == (sys.executable,)
    assert policy.cache == path


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows paths")
@pytest.mark.parametrize("suffix,reason", [
    ("\\..\\other", "climbs"), ("\\.\\other", r"\. segment"),
    ("\\\\other", "empty segment"), ("\\", "ends with a slash"),
    (":secret", "alternate data stream"), ("\\missing", "does not exist"),
    ("\x00", "control character"),
])
def test_windows_paths_reject_ambiguous_segments(tmp_path, suffix, reason):
    with pytest.raises(PolicyError, match=reason):
        checked_path(str(tmp_path) + suffix)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows paths")
@pytest.mark.parametrize("path", [
    "relative", r"C:relative", r"\relative", r"\\server\share\file",
    r"\\?\C:\file", r"\\.\pipe\test",
])
def test_windows_paths_reject_nonlocal_or_drive_relative_names(path):
    with pytest.raises(PolicyError, match="absolute local drive"):
        checked_path(path)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows paths")
def test_windows_paths_reject_junction_and_allow_resolved_target(tmp_path):
    import subprocess

    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(junction), str(target)],
                   check=True, capture_output=True)
    try:
        with pytest.raises(PolicyError, match="symlink"):
            checked_path(str(junction))
        assert checked_path(str(target)) == str(target)
    finally:
        junction.rmdir()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows paths")
def test_mcp_policy_grants_native_system_program_directory(tmp_path):
    import os

    import win32api

    from ml_stack.sandbox import policies

    policy = policies.mcp_server(sys.executable, tmp_path, tmp_path).validated()
    system = os.path.realpath(win32api.GetSystemDirectory())
    assert (system,) == policies.SYSTEM_EXEC
    assert policy.exec == (os.path.realpath(sys.executable), system)
    assert policy.env["PATH"] == system
