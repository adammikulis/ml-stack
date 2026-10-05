"""Native terminal identity and explicit human grants."""

import os
import sys

import pytest

from ml_stack.person import HumanRequired, is_terminal, require_person


@pytest.mark.skipif(sys.platform != "win32", reason="Windows NUL device")
def test_windows_null_device_is_not_a_human_terminal():
    with open(os.devnull) as stream:
        assert not is_terminal(stream)


def test_explicit_terminal_tuple_preserves_authority_injection():
    require_person("test", terminal=(True, True), env={})
    with pytest.raises(HumanRequired):
        require_person("test", terminal=(False, True), env={})
