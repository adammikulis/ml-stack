"""Windows token files enforce native account ownership and private ACLs."""

import os

import pytest

from ml_stack.workspace import tokens
from ml_stack.workspace.identity import PREFIX


def test_workspace_token_directory_rejects_windows_mount(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    monkeypatch.setattr(tokens, "_windows_mount", lambda path: True)
    base.mkdir()
    assert "Windows-mounted filesystem" in tokens.problem(base)
    with pytest.raises(ValueError, match="Windows-mounted filesystem"):
        tokens.prepare(base)
    assert not (base / "tokens").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows access control")
def test_created_token_and_directory_are_private(tmp_path):
    held = f"{PREFIX}fixture-agent.private-token"
    target = tokens.store(tmp_path, "fixture-agent", held)
    assert tokens.problem(target.parent) == ""
    assert tokens.problem(target) == ""
    assert tokens.load(tmp_path, "fixture-agent") == held


@pytest.mark.skipif(os.name != "nt", reason="native Windows access control")
def test_another_account_grant_refuses_token_reads(tmp_path):
    import ntsecuritycon
    import win32security

    from ml_stack.windows_private import restrict
    from ml_stack.workspace.identity import Denied

    target = tokens.store(tmp_path, "fixture-agent", f"{PREFIX}fixture-agent.private-token")
    descriptor = win32security.GetNamedSecurityInfo(
        str(target), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION)
    acl = descriptor.GetSecurityDescriptorDacl()
    everyone = win32security.CreateWellKnownSid(win32security.WinWorldSid)
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, ntsecuritycon.FILE_GENERIC_READ, everyone)
    try:
        win32security.SetNamedSecurityInfo(
            str(target), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION,
            None, None, acl, None)
        with pytest.raises(Denied, match="another account"):
            tokens.read_file(target)
    finally:
        restrict(target)


@pytest.mark.skipif(os.name != "nt", reason="native Windows junctions")
def test_junction_token_directory_is_refused_without_changing_target_acl(tmp_path):
    import subprocess

    import win32security

    target = tmp_path / "other-private-directory"
    target.mkdir()
    marker = target / "kept.txt"
    marker.write_text("preserved", encoding="utf-8")
    security = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    before = win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(
        win32security.GetFileSecurity(str(target), security), win32security.SDDL_REVISION_1, security)
    base = tmp_path / "workspace"
    base.mkdir()
    junction = base / "tokens"
    created = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                             capture_output=True, check=False)
    assert created.returncode == 0, created.stderr
    try:
        assert "reparse point" in tokens.problem(junction)
        with pytest.raises(ValueError, match="reparse point"):
            tokens.prepare(base)
        after = win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(
            win32security.GetFileSecurity(str(target), security), win32security.SDDL_REVISION_1, security)
        assert after == before and marker.read_text(encoding="utf-8") == "preserved"
    finally:
        junction.rmdir()
