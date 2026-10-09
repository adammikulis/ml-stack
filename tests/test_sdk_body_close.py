"""A streamed SDK body whose socket an earlier close already took is closed without an error."""

import asyncio
import types

from poolhouse.client import sdk


class _TakenSocket:
    """A socket that still shuts down but has no descriptor left to hand over."""

    def shutdown(self, how):
        pass

    def detach(self):
        return -1


def test_closing_a_body_whose_descriptor_is_already_taken_is_harmless():
    closed = []
    response = types.SimpleNamespace(fp=types.SimpleNamespace(raw=types.SimpleNamespace(_sock=_TakenSocket())),
                                     close=lambda: closed.append(True))
    asyncio.run(sdk._Body(response).aclose())
    assert closed == [True]
