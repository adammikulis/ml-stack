"""The few Win32 calls the node's launcher needs, through ctypes, bound when first used so the module imports anywhere.

A process by id (`os.kill(pid, 0)` on Windows is not a probe: signal 0 is CTRL_C_EVENT), a named event to set or wait on
(the stop signal of a detached process, which has no console to send a signal to), and a client handle on a named pipe
opened so that its server can not act as the caller.
"""

from __future__ import annotations

import functools
import os
from typing import Any

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
EVENT_MODIFY_STATE = 0x0002
GENERIC_READ_WRITE = 0x80000000 | 0x40000000
OPEN_EXISTING = 3
SECURITY_SQOS_PRESENT = 0x00100000
SECURITY_IDENTIFICATION = 0x00010000
STILL_ACTIVE = 259
ERROR_ACCESS_DENIED = 5
ERROR_PIPE_BUSY = 231
WAIT_OBJECT_0 = 0


@functools.cache
def _k32() -> Any:
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    k.OpenProcess.restype, k.OpenProcess.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.GetExitCodeProcess.restype, k.GetExitCodeProcess.argtypes = wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.restype, k.CloseHandle.argtypes = wintypes.BOOL, [wintypes.HANDLE]
    k.CreateEventW.restype, k.CreateEventW.argtypes = wintypes.HANDLE, [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    k.OpenEventW.restype, k.OpenEventW.argtypes = wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k.SetEvent.restype, k.SetEvent.argtypes = wintypes.BOOL, [wintypes.HANDLE]
    k.WaitForSingleObject.restype, k.WaitForSingleObject.argtypes = wintypes.DWORD, [wintypes.HANDLE, wintypes.DWORD]
    k.CreateFileW.restype = wintypes.HANDLE
    k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    return k


def _last_error() -> int:
    import ctypes

    return ctypes.get_last_error()  # type: ignore[attr-defined]


def process_alive(pid: int) -> bool:
    """Whether a process with this id is running. One this user may not inspect exists."""
    if pid <= 0:
        return False
    k = _k32()
    handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return _last_error() == ERROR_ACCESS_DENIED
    try:
        import ctypes
        from ctypes import wintypes

        code = wintypes.DWORD()
        return bool(k.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        k.CloseHandle(handle)


class Event:
    """A named manual-reset event, created by the process that waits for it."""

    def __init__(self, name: str) -> None:
        self._handle = _k32().CreateEventW(None, True, False, name)
        if not self._handle:
            raise OSError(_last_error(), f"cannot create the event {name}")

    def is_set(self) -> bool:
        return _k32().WaitForSingleObject(self._handle, 0) == WAIT_OBJECT_0

    def close(self) -> None:
        if self._handle:
            _k32().CloseHandle(self._handle)
            self._handle = None


def signal_event(name: str) -> bool:
    """Set the event called ``name``; False when no process holds it."""
    k = _k32()
    handle = k.OpenEventW(EVENT_MODIFY_STATE, False, name)
    if not handle:
        return False
    try:
        return bool(k.SetEvent(handle))
    finally:
        k.CloseHandle(handle)


def open_pipe(name: str) -> Any:
    """A binary read-write file on the named pipe, opened at identification level; OSError when nothing serves it.

    Raises `OSError` with ``winerror`` 231 when every instance is busy, which a caller may retry.
    """
    import msvcrt

    handle = _k32().CreateFileW(name, GENERIC_READ_WRITE, 0, None, OPEN_EXISTING, SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, None)
    if handle in (None, -1, 0xFFFFFFFFFFFFFFFF):
        code = _last_error()
        raise OSError(0, f"cannot open {name}", None, code)
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY), "r+b", buffering=0)  # type: ignore[attr-defined]
