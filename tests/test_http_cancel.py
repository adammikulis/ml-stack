"""Cancellation and explicit timeouts before HTTP response headers."""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import httpx
import pytest

from ml_stack.client.sdk import FleetTransport
from ml_stack.http import Server


@contextmanager
def pending_headers():
    received, closed = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            received.set()
            self.connection.settimeout(3)
            try:
                assert self.connection.recv(1) == b''
                closed.set()
            except OSError:
                pass
        def log_message(self, *_args):
            pass
    server = Server(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/chat/completions', received, closed
    finally:
        server.shutdown()
        server.server_close()
        worker.join(1)


def test_unlimited_sdk_opening_can_be_cancelled_before_headers():
    with pending_headers() as (url, received, closed):
        async def run():
            transport = FleetTransport(url, '')
            request = httpx.Request('POST', url, json={})
            task = asyncio.create_task(transport.handle_async_request(request))
            assert await asyncio.to_thread(received.wait, 2)
            began = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert time.monotonic() - began < 1
            assert await asyncio.to_thread(closed.wait, 1)
            await transport.aclose()
        asyncio.run(run())


def test_explicit_sdk_opening_timeout_remains_effective():
    with pending_headers() as (url, received, closed):
        async def run():
            transport = FleetTransport(url, '', timeout=0.1)
            began = time.monotonic()
            response = await transport.handle_async_request(httpx.Request('POST', url, json={}))
            assert response.status_code == 502
            assert 0.09 <= time.monotonic() - began < 1
            assert received.is_set() and await asyncio.to_thread(closed.wait, 1)
            await transport.aclose()
        asyncio.run(run())
