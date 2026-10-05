"""Native Windows ownership and access checks for workspace token files."""

import sys
from pathlib import Path

if sys.platform == "win32":
    import ntsecuritycon
    import pywintypes
    import win32api
    import win32con
    import win32security

def _user():
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        return win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        token.Close()


def restrict(path: Path) -> None:
    """Grant this process's account exclusive access to a token file or directory."""
    user = _user()
    acl = win32security.ACL()
    flags = (win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE
             if path.is_dir() else 0)
    acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, flags,
                            ntsecuritycon.FILE_ALL_ACCESS, user)
    win32security.SetNamedSecurityInfo(
        str(path), win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION, user, None, acl, None)


def problem(path: Path) -> str:
    """Describe token ownership or access grants that permit another account."""
    try:
        descriptor = win32security.GetNamedSecurityInfo(
            str(path), win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION)
        user = _user()
        if descriptor.GetSecurityDescriptorOwner() != user:
            return "belongs to another user"
        acl = descriptor.GetSecurityDescriptorDacl()
        if acl is None:
            return "has unrestricted Windows access"
        for index in range(acl.GetAceCount()):
            ace = acl.GetAce(index)
            if ace[0][0] != win32security.ACCESS_DENIED_ACE_TYPE and ace[-1] != user:
                return "Windows permissions grant access to another account"
        return ""
    except pywintypes.error:
        return "Windows ownership or permissions could not be verified"
