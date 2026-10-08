"""Windows token files enforce native account ownership and private ACLs."""

import os
import subprocess

import pytest

from ml_stack import private_path, windows_private
from ml_stack.workspace import tokens
from ml_stack.workspace.identity import PREFIX, Denied

if os.name == "nt":
    import ntsecuritycon
    import win32security


def test_workspace_token_directory_rejects_windows_mount(tmp_path, monkeypatch):
    base = tmp_path / "workspace"
    monkeypatch.setattr(private_path, "windows_mount", lambda path: True)
    base.mkdir()
    assert "Windows-mounted filesystem" in private_path.problem(base)
    with pytest.raises(ValueError, match="Windows-mounted filesystem"):
        tokens.prepare(base)
    assert not (base / "tokens").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows access control")
def test_created_token_and_directory_are_private(tmp_path):
    held = f"{PREFIX}fixture-agent.private-token"
    target = tokens.store(tmp_path, "fixture-agent", held)
    assert private_path.problem(target.parent) == ""
    assert private_path.problem(target) == ""
    assert tokens.load(tmp_path, "fixture-agent") == held


@pytest.mark.skipif(os.name != "nt", reason="native Windows access control")
def test_another_account_grant_refuses_token_reads(tmp_path):


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
        windows_private.restrict(target)


@pytest.mark.skipif(os.name != "nt", reason="native Windows junctions")
def test_junction_token_directory_is_refused_without_changing_target_acl(tmp_path):


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
        assert "reparse point" in private_path.problem(junction)
        with pytest.raises(ValueError, match="reparse point"):
            tokens.prepare(base)
        after = win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(
            win32security.GetFileSecurity(str(target), security), win32security.SDDL_REVISION_1, security)
        assert after == before and marker.read_text(encoding="utf-8") == "preserved"
    finally:
        junction.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="native Windows access control")
def test_restrict_refuses_foreign_owner_before_any_security_mutation(tmp_path, monkeypatch):


    path = tmp_path / "foreign-storage"
    path.mkdir()
    foreign = win32security.CreateWellKnownSid(win32security.WinWorldSid)
    descriptor = win32security.SECURITY_DESCRIPTOR()
    descriptor.SetSecurityDescriptorOwner(foreign, False)
    read = win32security.GetNamedSecurityInfo
    monkeypatch.setattr(win32security, "GetNamedSecurityInfo", lambda name, *args:
                        descriptor if name == str(path) else read(name, *args))
    monkeypatch.setattr(win32security, "SetNamedSecurityInfo",
                        lambda *args: pytest.fail("foreign ownership was changed"))
    with pytest.raises(ValueError, match="belongs to another user"):
        windows_private.restrict(path)
    with pytest.raises(ValueError, match="belongs to another user"):
        tokens.prepare(path)
    assert not (path / "tokens").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows access control")
def test_owned_broad_permissions_are_repaired_without_changing_owner(tmp_path):


    path = tmp_path / "own-storage"
    path.mkdir()
    security = win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
    descriptor = win32security.GetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT, security)
    owner = descriptor.GetSecurityDescriptorOwner()
    acl = descriptor.GetSecurityDescriptorDacl()
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, ntsecuritycon.FILE_GENERIC_READ,
                           win32security.CreateWellKnownSid(win32security.WinWorldSid))
    win32security.SetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT,
                                     win32security.DACL_SECURITY_INFORMATION, None, None, acl, None)
    assert windows_private.problem(path)
    tokens.prepare(path)
    assert windows_private.problem(path) == ""
    assert win32security.GetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION).GetSecurityDescriptorOwner() == owner


@pytest.mark.skipif(os.name != "nt", reason="native Windows junctions")
def test_junction_ancestor_refused_before_creating_workspace(tmp_path):


    target = tmp_path / "unchanged"
    target.mkdir()
    junction = tmp_path / "redirect"
    made = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                          capture_output=True, check=False)
    assert made.returncode == 0, made.stderr
    try:
        base = junction / "new-workspace"
        with pytest.raises(ValueError, match="reparse point"):
            windows_private.validate(base)
        with pytest.raises(ValueError, match="reparse point"):
            tokens.prepare(base)
        assert list(target.iterdir()) == []
    finally:
        junction.rmdir()
