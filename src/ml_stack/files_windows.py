"""Windows read handles that permit atomic file replacement."""

import ctypes
import msvcrt
import os
from ctypes import wintypes
from pathlib import Path

_kernel = ctypes.WinDLL("kernel32", use_last_error=True)
_create = _kernel.CreateFileW
_create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                   wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
_create.restype = wintypes.HANDLE
_close = _kernel.CloseHandle
_close.argtypes = [wintypes.HANDLE]
_close.restype = wintypes.BOOL
_set = _kernel.SetFileInformationByHandle
_set.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
_set.restype = wintypes.BOOL

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
FILE_SHARE_DELETE = 4
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
DELETE = 0x00010000
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_RENAME_INFO_EX = 22
FILE_RENAME_REPLACE_IF_EXISTS = 1
FILE_RENAME_POSIX_SEMANTICS = 2


class _Rename(ctypes.Structure):
    _fields_ = [("flags", wintypes.DWORD), ("root", wintypes.HANDLE),
                ("length", wintypes.DWORD), ("name", wintypes.WCHAR * 1)]


def open_read(path: str, flags: int) -> int:
    """Return a read descriptor whose handle permits path replacement."""
    handle = _create(os.fsdecode(path), GENERIC_READ,
                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, None,
                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, flags | os.O_BINARY)
    except OSError:
        _close(handle)
        raise


def replace(source: Path, target: Path) -> None:
    """Replace a path while existing readers retain their open file snapshots."""
    handle = _create(str(source.absolute()), DELETE,
                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, None,
                     OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, None)
    if handle == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        name = str(target.absolute()).encode("utf-16-le")
        buffer = ctypes.create_string_buffer(ctypes.sizeof(_Rename) + len(name))
        info = _Rename.from_buffer(buffer)
        info.flags = FILE_RENAME_REPLACE_IF_EXISTS | FILE_RENAME_POSIX_SEMANTICS
        info.length = len(name)
        ctypes.memmove(ctypes.addressof(buffer) + _Rename.name.offset, name, len(name))
        if not _set(handle, FILE_RENAME_INFO_EX, buffer, len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _close(handle)
