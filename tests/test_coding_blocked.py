"""Native approval denial terminates a turn without retrying or weakening policy."""
import pytest

from poolhouse.workspace import coding_events, coding_turns


def test_native_denial_stops_before_another_tool_call():
    row = {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True,
        "content": "PreToolUse:Bash hook error: poolhouse: the person did not allow this call (KeystoreUnavailable)"}]}}
    turn = coding_turns.Turn("test")
    with pytest.raises(RuntimeError, match="blocked by approval policy"):
        coding_turns.Manager(None)._event(turn, row, "claude")
    assert not turn.cancelled.is_set()
    assert coding_events.event({"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True,
        "content": "ordinary command failure"}]}}, "claude") == {}
