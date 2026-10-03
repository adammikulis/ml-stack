"""The smoke test a new llama-server build passes before it is trusted."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack import home
from ml_stack.http import request_json
from ml_stack.hub import header
from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port, slotdump

__all__ = ["Result", "run", "smallest_model"]

LIMIT = 4 * 1024 ** 3
SKIP = ("vocab", "mmproj", "-of-", "mtp")


@dataclass(slots=True)
class Result:
    """The checks a build ran: ``checks`` is ``(name, ok, detail)`` in order."""

    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        return ok

    def as_dict(self) -> dict:
        return {"passed": self.passed, "checks": [{"name": n, "ok": ok, "detail": d}
                                                  for n, ok, d in self.checks]}


def _generates(path: Path) -> bool:
    architecture = str((header.meta(path) or {}).get("general.architecture", ""))
    return bool(architecture) and "bert" not in architecture and "assistant" not in architecture


def smallest_model() -> Path | None:
    """The smallest text-generating GGUF in the Hugging Face cache or ``~/.cache``, or the
    file ``$ML_STACK_SMOKE_GGUF`` names."""
    named = os.environ.get("ML_STACK_SMOKE_GGUF")
    if named:
        return Path(named).expanduser()
    user = home.user_home() / ".cache"
    found = [p for root in (user / "huggingface" / "hub", user) if root.is_dir()
             for p in root.rglob("*.gguf")
             if not any(word in p.name.lower() for word in SKIP)
             and 0 < p.stat().st_size < LIMIT and _generates(p)]
    return min(found, key=lambda p: p.stat().st_size) if found else None


def run(binary: Path, model: Path | None = None, *, timeout: float = 300.0,
        say: Callable[[str], None] = lambda _text: None) -> Result:
    """Serve ``model`` (default: the smallest local GGUF) with ``binary`` through a manager
    lease, then check /health, a chat completion, ``top_logprobs`` and a slot save and restore."""
    result = Result()
    model = model or smallest_model()
    if model is None:
        result.add("model", False, "no local GGUF to load; set ML_STACK_SMOKE_GGUF")
        return result
    manager = ServerManager(LlamaServerBackend(binary=binary))
    with tempfile.TemporaryDirectory(prefix="ml-stack-smoke-") as made:
        slots = Path(made)
        try:
            info = manager.lease(ServerSpec(model=model, port=free_port(), context=512,
                                            slot_save_path=slots), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - any failure to start is a failed smoke test
            result.add("lease", False, f"{type(exc).__name__}: {exc}"[:600])
            return result
        try:
            result.add("lease", True, info.base_url)
            _checks(result, info.base_url, slots, say)
        finally:
            manager.release(info)
    return result


def _checks(result: Result, base: str, slots: Path, say: Callable[[str], None]) -> None:
    def attempt(name: str, check: Callable[[], str]) -> bool:
        say(f"  smoke: {name}")
        try:
            return result.add(name, True, check())
        except Exception as exc:  # noqa: BLE001 - the failure is the result
            return result.add(name, False, f"{type(exc).__name__}: {exc}"[:600])

    def health() -> str:
        reply = request_json(f"{base}/health", method="GET", timeout=30)
        if (reply or {}).get("status") != "ok":
            raise ValueError(f"/health said {reply!r}")
        return "ok"

    def chat(extra: dict) -> dict:
        reply = request_json(f"{base}/v1/chat/completions", timeout=120, payload={
            "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
            "temperature": 0, **extra})
        if not reply or not reply.get("choices"):
            raise ValueError(f"no choices in {reply!r}"[:200])
        return reply["choices"][0]

    def completion() -> str:
        choice = chat({})
        message = choice.get("message") or {}
        if not (message.get("content") or message.get("reasoning_content")):
            raise ValueError("the completion was empty")
        return f"{choice.get('finish_reason')}"

    def logprobs() -> str:
        content = (chat({"logprobs": True, "top_logprobs": 3}).get("logprobs") or {}).get("content")
        if not content or not content[0].get("top_logprobs"):
            raise ValueError("top_logprobs came back empty")
        return f"{len(content[0]['top_logprobs'])} alternatives"

    def slot_cycle() -> str:
        saved = slotdump.save_slot(base, 0, "smoke.bin", directory=slots)
        restored = slotdump.restore_slot(base, 0, "smoke.bin", directory=slots)
        return f"saved {saved.tokens} tokens, restored {restored.tokens}"

    if not attempt("health", health):
        return
    if attempt("chat completion", completion):
        attempt("top_logprobs", logprobs)
        attempt("slot save and restore", slot_cycle)
