"""Put ``src`` on the path, and provide a real HTTP server to test against.

No mocking of the transport. Every client test below runs a real ``http.server`` in a
thread and does a real socket round trip, because the bugs these modules exist to prevent
are transport-shaped: a server that answers /health while still loading, one that ignores
a Range header, one that returns 500 on a concurrent request. A mocked ``urlopen`` cannot
reproduce any of them.
"""

from __future__ import annotations

import contextlib
import json
import struct
import sys
import threading
import types
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


Handler = Callable[[str, str, bytes], tuple[int, bytes]]
"""``(method, path, body) -> (status, response_body)``"""


class _Server:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[tuple[str, str, bytes]] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _dispatch(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append((self.command, self.path, body))
                try:
                    status, payload = outer.handler(self.command, self.path, body)
                except Exception as exc:  # surface handler bugs as 500s, not hangs
                    status, payload = 500, str(exc).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _dispatch
            do_POST = _dispatch

            def log_message(self, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Callable[[Handler], _Server]:
    """Start a real HTTP server with a caller-supplied handler."""
    started: list[_Server] = []

    def start(handler: Handler) -> _Server:
        instance = _Server(handler).__enter__()
        started.append(instance)
        return instance

    yield start

    for instance in started:
        instance.__exit__()


def json_reply(payload: object, status: int = 200) -> tuple[int, bytes]:
    """The body a fake HTTP handler answers with, JSON-encoded, with a status."""
    return status, json.dumps(payload).encode()


# -- nothing here reads the machine it runs on ------------------------------------------
#
# Every one of these seams was found the same way: a test passed on this laptop and would
# have read a real runs store, a real fit record or a real server on another. The guard is
# autouse and suite-wide so a test that forgets cannot reach any of them; a test that means
# to exercise one overrides it with its own `monkeypatch.setattr`, which runs after this.

_HOMED = (
    # module path, attribute, the name it gets under the fixture's scratch home
    ("ml_stack.graph.bench", "HOME", "bench"),
    ("ml_stack.graph.bench.extract", "HOME", "bench"),   # bound at import, not looked up
    ("ml_stack.mcp", "MCP_HOME", "mcp"),
)

#: Environment a developer's shell may carry that would otherwise steer a test.
_STEERING = ("MLSTACK_BENCH_CEILING", "MLSTACK_BENCH_TRACE", "MLSTACK_LLAMA_BUILD",
             "MLSTACK_SEARCH", "MLSTACK_TRAIN_CEILING", "MLSTACK_WEB_PROFILE")


@pytest.fixture(autouse=True)
def _no_machine_state(monkeypatch, tmp_path):
    """Point every home, record file and machine reading at an empty temporary directory.

    ``bench.HOME`` is the runs store a whole evening of measuring sits in; ``serving_lines``
    and ``results_since`` read what is serving on this machine right now and what the last
    job kept. A test that saw either would pass or fail on what the laptop happened to be
    doing. The `MLSTACK_*` variables are deleted rather than set, so a shell that exports
    one cannot change a result either.
    """
    home = tmp_path / "machine-state"
    monkeypatch.setenv("MLSTACK_BENCH_HOME", str(home / "bench"))
    monkeypatch.setenv("MLSTACK_INGEST_HOME", str(home / "ingest"))
    monkeypatch.setenv("MLSTACK_FIT_FILE", str(home / "fit.json"))
    monkeypatch.setenv("MLSTACK_PROFILES_FILE", str(home / "profiles.json"))
    for name in _STEERING:
        monkeypatch.delenv(name, raising=False)

    import importlib

    for path, attr, leaf in _HOMED:
        module = sys.modules.get(path) or importlib.import_module(path)
        monkeypatch.setattr(module, attr, home / leaf, raising=False)

    running = sys.modules.get("ml_stack.graph.bench.run") or importlib.import_module(
        "ml_stack.graph.bench.run")
    monkeypatch.setattr(running, "serving_lines", lambda: [])
    monkeypatch.setattr(running, "results_since", lambda started, kept=None: "")


# -- awaiting a coroutine without depending on who ran first ----------------------------

def on_a_fresh_loop(coro):
    """Await ``coro`` on an event loop of its own, in a thread of its own.

    ``asyncio.run`` refuses to run inside a live loop, so a test that calls it fails when a
    neighbour in the same xdist worker left one running -- which made a green test red on
    ordering alone. A private thread has no ambient loop, so this never depends on that.
    """
    import asyncio

    done: list[object] = []
    raised: list[BaseException] = []

    def go() -> None:
        loop = asyncio.new_event_loop()
        try:
            done.append(loop.run_until_complete(coro))
        except BaseException as exc:  # re-raised in the calling thread below
            raised.append(exc)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    worker = threading.Thread(target=go, name="fresh-event-loop")
    worker.start()
    worker.join()
    if raised:
        raise raised[0]
    return done[0]


# -- a real server on a real socket, in a thread ----------------------------------------

@contextlib.contextmanager
def threaded_server(handler_class, *, port: int = 0):
    """Serve ``handler_class`` on loopback in a daemon thread; yield its url, then stop.

    Fourteen test modules had written this same eight lines. Getting it wrong leaks a
    thread and a port into every test that follows, which is how one file's failure
    started showing up in another's.
    """
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_class)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# -- a model file, without a model ------------------------------------------------------

#: Enough of a real ``--help`` to answer every flag ``command()`` emits for a bare
#: ServerSpec, so the flag check (which nothing using this is testing) does not refuse.
LLAMA_SERVER_HELP = (
    "-m,    --model FNAME                    model path\n"
    "-c,    --ctx-size N                     size of the prompt context\n"
    "-ngl,  --gpu-layers, --n-gpu-layers N   number of layers to store in VRAM\n"
    "-fa,   --flash-attn [on|off|auto]       set Flash Attention use\n"
    "       --host HOST                      ip address to listen on\n"
    "       --port PORT                      port to listen on\n"
    "       --jinja                          use jinja template for chat\n"
)


def write_gguf(path: Path, metadata: dict, *, tensor_count: int = 0) -> Path:
    """A real, minimal GGUF v3 file: magic, version, counts, one key/value pair per
    metadata item -- ints as uint32, floats as float32, strings as strings -- and no
    tensors, because nothing under test reads them.

    It refuses a type it cannot write rather than coercing one: a copy that silently
    stringified everything let a test assert on a field the real reader would never
    have parsed.
    """

    def kv(name: str, value: object) -> bytes:
        head = struct.pack("<Q", len(name.encode())) + name.encode()
        if isinstance(value, bool):
            return head + struct.pack("<I", 7) + struct.pack("<?", value)
        if isinstance(value, int):
            return head + struct.pack("<I", 4) + struct.pack("<I", value)
        if isinstance(value, float):
            return head + struct.pack("<I", 6) + struct.pack("<f", value)
        if isinstance(value, str):
            encoded = value.encode()
            return head + struct.pack("<I", 8) + struct.pack("<Q", len(encoded)) + encoded
        raise TypeError(f"unsupported metadata type: {type(value)}")

    body = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count)
            + struct.pack("<Q", len(metadata)))
    for name, value in metadata.items():
        body += kv(name, value)
    path.write_bytes(body)
    return path


def fake_binary(tmp_path: Path, *, help_text: str = "-m, --model FNAME  model path\n") -> Path:
    """An executable that answers ``--help`` with ``help_text`` and exits 0 for anything else."""
    path = tmp_path / "llama-server"
    path.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then cat <<'HELP'\n"
                    + help_text + "HELP\nexit 0\nfi\nexit 0\n")
    path.chmod(0o755)
    return path


# -- psutil, as far as anything here reads it -------------------------------------------

def fake_process(argv: list[str], rss: int, *, pid: int = 0, state: str = "running"):
    """One row of ``psutil.process_iter(attrs=...)``: its argv, its resident set, its state."""
    info = {"name": Path(argv[0]).name, "cmdline": argv,
            "memory_info": types.SimpleNamespace(rss=rss)}
    if pid:
        info["pid"] = pid
    return types.SimpleNamespace(info=info, pid=pid, status=lambda: state)


def fake_memory(*, total: int, available: int, wired: int):
    """What ``psutil.virtual_memory()`` answers, with the fields the readers here use."""
    return types.SimpleNamespace(total=total, available=available, wired=wired)


# -- a bench run, without measuring one --------------------------------------------------

def a_row(question: str, *, expected: list[str], shown: list[str], calls: int = 3,
          chars: int = 200, error: str = "", label: str = "tried"):
    """One measured question: what was wanted, what the answer showed, what it cost."""
    from ml_stack.graph.bench import Row

    return Row(label=label, question=question, expected=expected, shown=shown,
               calls=calls, answer_chars=chars, error=error)


def scored_rows(label: str, *, questions: int, hits: int, seconds: float,
                expected: str = "person:iris", miss: list[str] | None = None,
                question: str = "q{n}?", tokens: tuple[int, int] = (0, 0),
                draft: tuple[int, int] = (0, 0)) -> list:
    """``hits`` of ``questions`` answered in full, the rest showing ``miss``, over
    ``seconds`` altogether -- so the F1 of the run is ``hits / questions`` exactly.

    ``tokens`` is (processed, completion) per row and ``draft`` is (guessed, taken), both
    of which the report and the rates read; left at zero they are simply not measured.
    """
    out = []
    for n in range(questions):
        row = a_row(question.format(n=n), expected=[expected],
                    shown=[expected] if n < hits else list(miss or []), label=label)
        row.seconds = seconds / questions
        row.processed_tokens, row.completion_tokens = tokens
        row.draft_tokens, row.draft_taken = draft
        out.append(row)
    return out


# -- the fit records, pointed away from the ones this repository ships -------------------

@pytest.fixture
def fit_files(tmp_path, monkeypatch):
    """Point both halves of the fit source of truth at ``tmp_path``, and fill them.

    ``package_file`` is a function for exactly this reason; the machine's own half moves
    with ``$MLSTACK_FIT_FILE``. Without both, a test would read the measurements this
    repository ships and a ``--measure`` test would write into it. Call it with the records
    the shipped half should hold, ``mine=`` for the machine's own half, and ``room=`` to fix
    what ``hub.room`` answers so no test depends on the machine it runs on.
    """
    from ml_stack.serve import fit as fit_mod

    def point(records=(), *, mine=(), room: int | None = None):
        shipped = tmp_path / "ssot" / "fit.json"
        shipped.parent.mkdir(parents=True, exist_ok=True)
        shipped.write_text(json.dumps([f.as_dict() for f in records], indent=2),
                           encoding="utf-8")
        local = tmp_path / "local" / "fit.json"
        if mine:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps([f.as_dict() for f in mine]), encoding="utf-8")
        monkeypatch.setattr(fit_mod, "package_file", lambda: shipped)
        monkeypatch.setattr(fit_mod, "writable_file", lambda: shipped)
        monkeypatch.setenv("MLSTACK_FIT_FILE", str(local))
        if room is not None:
            monkeypatch.setattr("ml_stack.hub.room", lambda: room)
        return types.SimpleNamespace(shipped=shipped, mine=local)

    return point
