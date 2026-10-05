"""Native terminal identity and explicit human grants."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_stack import person
from ml_stack.person import HumanRequired, is_terminal, require_person


@pytest.mark.skipif(sys.platform != "win32", reason="Windows NUL device")
def test_windows_null_device_is_not_a_human_terminal():
    with Path(os.devnull).open() as stream:
        assert not is_terminal(stream)


def test_explicit_terminal_tuple_preserves_authority_injection():
    require_person("test", terminal=(True, True), env={})
    with pytest.raises(HumanRequired):
        require_person("test", terminal=(False, True), env={})


def test_windows_tty_requires_a_valid_console_handle(monkeypatch):
    monkeypatch.setattr(person.sys, "platform", "win32")
    monkeypatch.setattr(person, "_MSVCRT", SimpleNamespace(get_osfhandle=lambda _fd: 888))
    mode = SimpleNamespace(GetConsoleMode=lambda handle, _mode: handle.value == 888)
    monkeypatch.setattr(person.ctypes, "windll", SimpleNamespace(kernel32=mode), raising=False)
    stream = SimpleNamespace(isatty=lambda: True, fileno=lambda: 17)
    assert is_terminal(stream)
    mode.GetConsoleMode = lambda *_args: 0
    assert not is_terminal(stream)
