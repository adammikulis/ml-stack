"""A reply the server cut off is reported as cut off, not as a model that cannot write JSON."""

import pytest

from ml_stack.client import Client
from ml_stack.client.chat import Reply, ServerError


def test_a_length_stop_names_the_knob(monkeypatch):
    client = Client("http://127.0.0.1:1")
    half = '{"concepts": [{"name": "Vault Currents", "kind": "concept", "definition": "flows tha'

    def cut(*a, **k):
        return Reply(content=half, finish_reason="length")

    monkeypatch.setattr(client, "chat", cut)
    with pytest.raises(ServerError) as err:
        client.extract("some text", {"type": "object"}, tries=1)
    said = str(err.value)
    assert "cut off" in said and "finish_reason=length" in said and "context" in said
    assert err.value.body == half, "the whole reply rides on the error"


def test_a_reply_that_stopped_on_its_own_is_still_not_json(monkeypatch):
    client = Client("http://127.0.0.1:1")

    def wrong(*a, **k):
        return Reply(content="not json at all", finish_reason="stop")

    monkeypatch.setattr(client, "chat", wrong)
    with pytest.raises(ServerError, match="not JSON"):
        client.extract("some text", {"type": "object"}, tries=1)
