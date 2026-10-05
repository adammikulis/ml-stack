"""Explicit local SDK models over the maintained fleet HTTP transport."""

from __future__ import annotations

import asyncio
import socket
import threading
from contextlib import suppress
from typing import Any

import httpx
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from ml_stack.http import ServerError, open_stream


class _Body(httpx.AsyncByteStream):
    def __init__(self, response: Any) -> None:
        self.response = response

    async def __aiter__(self):
        try:
            while block := await asyncio.to_thread(self.response.read1, 4096):
                yield block
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        fp = getattr(self.response, "fp", None)
        raw = getattr(fp, "raw", None)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        self.response.close()


def _open_response(url: str, token: str, data: bytes, cancelled: threading.Event) -> Any:
    response = open_stream(url, data=data, method="POST", token=token,
                           headers={"Content-Type": "application/json",
                                    "Accept": "text/event-stream"}, timeout=600.0)
    if cancelled.is_set():
        response.close()
    return response


class FleetTransport(httpx.AsyncBaseTransport):
    """Send SDK requests with fleet signatures, server keys and pinned TLS."""

    def __init__(self, url: str, token: str) -> None:
        self.url, self.token = url, token
        self.bodies: list[_Body] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) != self.url or request.method != "POST":
            raise ValueError("SDK requests must use the selected model endpoint")
        data = await request.aread()
        cancelled = threading.Event()
        opening = asyncio.create_task(asyncio.to_thread(
            _open_response, self.url, self.token, data, cancelled))
        try:
            response = await asyncio.shield(opening)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except ServerError as exc:
            return httpx.Response(exc.status or 502, content=exc.body or str(exc), request=request)
        body = _Body(response)
        self.bodies.append(body)
        return httpx.Response(response.status, headers=dict(response.headers), stream=body,
                              request=request)

    async def aclose(self) -> None:
        for body in self.bodies:
            await body.aclose()
        self.bodies.clear()


def local_client(target: Any, model: str, *, token: str = "") -> AsyncOpenAI:
    """Create an explicit client for the selected local or peer chat endpoint."""
    suffix = "/chat/completions"
    url = str(target.url)
    if not url.endswith(suffix) or not model:
        raise ValueError("a model and complete chat endpoint are required")
    transport = FleetTransport(url, token or target.token)
    return AsyncOpenAI(base_url=url[:-len(suffix)], api_key="local-transport",
                       max_retries=0, http_client=httpx.AsyncClient(transport=transport,
                                                                  trust_env=False))


def local_model(target: Any, model: str, *, token: str = "") -> OpenAIChatCompletionsModel:
    """Create an SDK model backed by the selected maintained HTTP transport."""
    return OpenAIChatCompletionsModel(model, local_client(target, model, token=token))
