"""Native Windows account-private file and directory permissions."""

import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING or sys.platform == "win32":
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


def validate(path: Path) -> None:
    """Require a plain path owned by this account before creating or restricting it."""
    path = Path.cwd() / path
    existing = None
    for candidate in (path, *path.parents):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(info.st_mode)
                or info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f"private storage {candidate} is a Windows reparse point")
        if existing is None:
            existing = candidate
    if existing is None:
        raise ValueError("private storage ownership could not be verified")
    try:
        descriptor = win32security.GetNamedSecurityInfo(
            str(existing), win32security.SE_FILE_OBJECT, win32security.OWNER_SECURITY_INFORMATION)
    except pywintypes.error as error:
        raise ValueError("private storage ownership could not be verified") from error
    if descriptor.GetSecurityDescriptorOwner() != _user():
        raise ValueError(f"private storage {existing} belongs to another user")


def restrict(path: Path) -> None:
    """Grant this process's account exclusive access to its own file or directory."""
    validate(path)
    user = _user()
    acl = win32security.ACL()
    flags = (win32security.OBJECT_INHERIT_ACE | win32security.CONTAINER_INHERIT_ACE
             if path.is_dir() else 0)
    acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, flags,
                            ntsecuritycon.FILE_ALL_ACCESS, user)
    win32security.SetNamedSecurityInfo(
        str(path), win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION, None, None, acl, None)


def problem(path: Path) -> str:
    """Describe token ownership or access grants that permit another account."""
    try:
        try:
            validate(path)
        except ValueError as error:
            return str(error)
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


_WRITE = (ntsecuritycon.FILE_WRITE_DATA | ntsecuritycon.FILE_APPEND_DATA | ntsecuritycon.FILE_WRITE_EA
          | ntsecuritycon.FILE_WRITE_ATTRIBUTES | ntsecuritycon.DELETE | ntsecuritycon.WRITE_DAC
          | ntsecuritycon.WRITE_OWNER | ntsecuritycon.GENERIC_WRITE | ntsecuritycon.GENERIC_ALL) if sys.platform == "win32" else 0
_SYSTEM_SIDS = ("S-1-5-18", "S-1-5-32-544", "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464")


def launcher_problem(path: Path) -> str:
    """Describe a launcher this account does not own or that another account could change.

    The Windows form of the Unix rule (owned by the user, no group or other write): other accounts may read and run it,
    and the system and administrator accounts keep the access they always have, but no one else may write it."""
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
        trusted = [user, *(win32security.ConvertStringSidToSid(sid) for sid in _SYSTEM_SIDS)]
        for index in range(acl.GetAceCount()):
            ace = acl.GetAce(index)
            if ace[0][0] == win32security.ACCESS_ALLOWED_ACE_TYPE and ace[1] & _WRITE and ace[-1] not in trusted:
                return "Windows permissions let another account change it"
        return ""
    except pywintypes.error:
        return "Windows ownership or permissions could not be verified"
