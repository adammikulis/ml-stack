"""Where a conversation can be sent, and sending it."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

__all__ = ["ChatError", "Target", "find", "reply_text", "stream", "targets"]

CHUNK = 1 << 12
CHAT_PATH = "/v1/chat/completions"


class ChatError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Target:
    """One model a conversation can be sent to."""

    model: str
    url: str
    token: str = ""
    peer: str = ""

    @property
    def local(self) -> bool:
        return not self.peer

    def public(self) -> dict[str, Any]:
        return {"model": self.model, "peer": self.peer, "local": self.local}


def targets(peers: list[dict[str, Any]], serving: Any = None,
            token: str = "") -> list[Target]:
    """Every model this machine can reach, its own and the ones on the network."""
    out: list[Target] = []
    for served in (serving.live() if serving is not None else []):
        for model in served.models:
            out.append(Target(model=model,
                              url=f"http://127.0.0.1:{served.port}{CHAT_PATH}"))
    here = {t.model for t in out}
    for beacon in peers:
        if beacon.get("is_self"):
            continue
        base = str(beacon.get("base_url") or "").rstrip("/")
        if not base:
            continue
        for served in (beacon.get("device", {}).get("serving") or []):
            for model in (served.get("models") or []):
                if model in here:
                    continue
                out.append(Target(model=str(model),
                                  url=f"{base}/infer{CHAT_PATH}",
                                  token=token,
                                  peer=str(beacon.get("name") or "")))
    out.sort(key=lambda t: (not t.local, t.peer, t.model))
    return out


def find(available: list[Target], model: str) -> Target | None:
    """The target for ``model``, preferring one on this machine."""
    if not model:
        return available[0] if available else None
    for target in available:
        if target.model == model:
            return target
    for target in available:
        if model.lower() in target.model.lower():
            return target
    return None


def stream(target: Target, payload: dict[str, Any], *,
           timeout: float = 600.0) -> Iterator[bytes]:
    """The model server's reply, in the pieces it arrives in."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        target.url, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream" if payload.get("stream") else "*/*"})
    if target.token:
        req.add_header("Authorization", f"Bearer {target.token}")
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise ChatError(
            f"{target.peer or 'this machine'} answered {exc.code}: "
            f"{exc.read().decode(errors='replace')[:400]}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise ChatError(
            f"{target.peer or 'this machine'} did not answer: {exc}") from None
    with response:
        while True:
            # read1, not read: read(n) waits for n bytes and delivers a whole
            # completion at once.
            block = response.read1(CHUNK)
            if not block:
                break
            yield block


def reply_text(raw: bytes) -> str:
    """The assistant's words out of a stream of server-sent events."""
    out = []
    for line in raw.decode(errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        frame = line[5:].strip()
        if not frame or frame == "[DONE]":
            continue
        try:
            parsed = json.loads(frame)
        except ValueError:
            continue
        for choice in parsed.get("choices") or []:
            piece = ((choice.get("delta") or {}).get("content")
                     or (choice.get("message") or {}).get("content"))
            if piece:
                out.append(str(piece))
    return "".join(out)
