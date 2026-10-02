"""``POST /decide``: a decision made on this machine, over a model server it is serving or a
backend it has configured."""

from __future__ import annotations

import threading
from typing import Any

from ml_stack.decide import router
from ml_stack.decide.types import BackendUnavailable, DecideError, options_of
from ml_stack.fleet.serving import Serving

MAX_REQUEST = 256 * 1024
"""Bytes of request body `/decide` reads; a larger one is refused before it is read."""


class Deciding:
    """Answers decision requests; the deciders it builds stay warm between requests."""

    def __init__(self, serving: Serving | None = None, config: router.Config | None = None
                 ) -> None:
        self.serving = serving
        self.config = config or router.Config()
        self._configs: dict[tuple[str, str], router.Config] = {}
        self._lock = threading.Lock()

    def _for(self, backend: str, model: str) -> router.Config:
        url = self.config.url
        if not url and self.serving is not None:
            port = self.serving.port_for(model)
            url = f"http://127.0.0.1:{port}" if port else ""
        key = (backend, url)
        with self._lock:
            held = self._configs.get(key)
            if held is None:
                held = router.Config(backend=backend, url=url, token=self.config.token,
                                     embed_url=self.config.embed_url,
                                     embed_head=self.config.embed_head,
                                     calibration=self.config.calibration,
                                     rules=self.config.rules, order=self.config.order)
                self._configs[key] = held
            return held

    def answer(self, body: Any) -> tuple[int, dict[str, Any]]:
        """``(status, payload)`` for a decoded request body."""
        if not isinstance(body, dict):
            return 400, {"error": "the body must be a JSON object"}
        try:
            question, options = str(body["question"]), body["options"]
            if not isinstance(options, (list, dict)):
                raise TypeError("options must be a list of names or an object of descriptions")
            abstain = body.get("abstain_below")
            options = options_of(options)
            config = self._for(str(body.get("backend") or "auto"), str(body.get("model") or ""))
            got = router.decide(question, body.get("state", ""), options,
                                abstain_below=None if abstain is None else float(abstain),
                                config=config)
        except KeyError as exc:
            return 400, {"error": f"missing {exc.args[0]!r}"}
        except (TypeError, ValueError) as exc:
            return 400, {"error": str(exc)}
        except BackendUnavailable as exc:
            return 503, {"error": str(exc)}
        except DecideError as exc:
            return 502, {"error": str(exc)}
        return 200, {"decision": got.public()}
