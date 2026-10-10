"""A launcher other accounts can read and run is accepted on Windows; one they can write is not."""

import sys

import pytest


def _grant(path, mask):
    import ntsecuritycon
    import win32security
    everyone = win32security.ConvertStringSidToSid("S-1-1-0")
    descriptor = win32security.GetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION)
    acl = descriptor.GetSecurityDescriptorDacl()
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, mask, everyone)
    win32security.SetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION, None, None, acl, None)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_a_launcher_others_can_only_read_and_run_is_owned(tmp_path):
    import ntsecuritycon

    from poolhouse import windows_private
    target = tmp_path / "poolhouse-peers.exe"
    target.write_text("x")
    windows_private.restrict(target)
    _grant(target, ntsecuritycon.FILE_GENERIC_READ | ntsecuritycon.FILE_GENERIC_EXECUTE)
    assert windows_private.launcher_problem(target) == ""


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_a_launcher_another_account_can_write_is_refused(tmp_path):
    import ntsecuritycon

    from poolhouse import windows_private
    target = tmp_path / "poolhouse-peers.exe"
    target.write_text("x")
    windows_private.restrict(target)
    _grant(target, ntsecuritycon.FILE_GENERIC_WRITE)
    assert windows_private.launcher_problem(target)
