"""The decoy endpoint: a loopback listener that serves fake cloud credentials at a path nothing
legitimate asks for.

Its address is written only to a decoy file under the state root (``credentials.endpoint``) and
is never advertised, so a request to it means something read that file and followed it. The hit
is a high-confidence finding; every session with a tool call running at that moment is frozen
(a request on loopback carries nothing else to attribute it by), and a hit with no session in
flight is recorded against no one. It answers with an obviously fake document and holds nothing.
The handler is `ReplyHandler`, the one the onboarding servers share.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from poolhouse.fleet.onboard.web import Call, Listener, Reply, json_reply
from poolhouse.sentinel import Mode, Sentinel
from poolhouse.sentinel.events import Event, Severity
from poolhouse.sentinel.watch import opt_out

__all__ = ["BECAUSE", "ENV", "DecoyListener", "arm"]

logger = logging.getLogger("poolhouse.sentinel")

ENV = "POOLHOUSE_SENTINEL_DECOY"
BECAUSE = "POOLHOUSE_SENTINEL_DECOY_BECAUSE"

PATH = "/latest/meta-data/iam/security-credentials/poolhouse"
FAKE = {"Code": "Success", "Type": "AWS-HMAC", "AccessKeyId": "ASIAXXXXXXXXXXXXXXXX",
        "SecretAccessKey": "decoy-not-a-secret", "Token": "decoy", "Expiration": "2000-01-01T00:00:00Z"}


class DecoyListener:
    """The decoy HTTP listener of one sentinel, on 127.0.0.1 and an ephemeral port."""

    def __init__(self, node: Sentinel) -> None:
        self.node = node
        self.hits = 0
        self._listener = Listener(self._dispatch, ("127.0.0.1", 0))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._listener.port}{PATH}"

    def _dispatch(self, call: Call) -> Reply:
        self.hits += 1
        self.node.bus.emit(Event("honey.endpoint_request", Severity.WARNING, "honey",
                                 "decoy:endpoint", {"method": call.method, "path": call.path[:80]},
                                 self.node.clock()))
        self.node.decoy_hit(call.path)
        return json_reply(200, FAKE)

    def start(self) -> DecoyListener:
        """Listen, and write the decoy file that names the address."""
        self._listener.start()
        self.node.honey.plant_endpoint(self.url)
        return self

    def stop(self) -> None:
        self._listener.stop()


def arm(node: Sentinel, env: Mapping[str, str] | None = None) -> DecoyListener | None:
    """Start the decoy listener for a long-running process. None when the mode is off, or when
    it was switched off with ``POOLHOUSE_SENTINEL_DECOY=off`` and a reason in
    ``POOLHOUSE_SENTINEL_DECOY_BECAUSE`` (without one the switch is ignored and logged)."""
    if node.mode == Mode.OFF:
        return None
    env = os.environ if env is None else env
    if env.get(ENV, "").strip().lower() == "off":
        because = env.get(BECAUSE, "").strip()
        if because:
            opt_out(node, "the decoy endpoint", because)
            return None
        logger.warning("sentinel: %s=off needs %s; the decoy endpoint stays armed", ENV, BECAUSE)
        node.bus.emit(Event("sentinel.decoy_off_refused", Severity.WARNING, "watch", "",
                            {"why": f"{ENV}=off needs {BECAUSE}"}, node.clock()))
    return DecoyListener(node).start()

