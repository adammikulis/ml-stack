"""The decision model's second opinion on a tool call: one typed question over a sanitised
rendering of the call, answered with probabilities, cached and bounded in time."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping

from ml_stack.decide import questions, router
from ml_stack.decide.guards.states import QUESTIONS, call_text
from ml_stack.guard.harm import Verdict
from ml_stack.interventions import Call

__all__ = ["ModelLayer", "from_environment"]

QUESTION = "is_destructive"
MAX_STATE = 6000


class ModelLayer:
    """Asks the decider whether a call is safe, reversible or destructive.

    An answer below ``floor`` is ``unsure``. A decider that is unreachable, errors or takes longer
    than ``timeout`` seconds gives None and the deterministic verdict stands. Answers are cached
    per tool and exact arguments, ``cache`` of them.
    """

    def __init__(self, config: router.Config | None = None, *, floor: float = 0.8,
                 timeout: float = 3.0, cache: int = 256) -> None:
        text, options = QUESTIONS["destructive"]
        self.question = questions.choice(text, {o.name: o.description for o in options})
        self.config, self.floor, self.timeout, self.size = config, floor, timeout, cache
        self.held: OrderedDict[str, Verdict] = OrderedDict()
        self.lock = threading.Lock()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                          thread_name_prefix="destructive-model")

    @staticmethod
    def key(call: Call) -> str:
        """The cache key: the tool and its exact arguments."""
        raw = json.dumps([call.name, call.arguments], sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()

    def _ask(self, call: Call) -> Verdict | None:
        state = f"Tool call: {call_text(call.name, call.arguments or {})}"[:MAX_STATE]
        got = questions.decide(state, {QUESTION: self.question}, config=self.config,
                               abstain_below=self.floor)[QUESTION]
        if got.error:
            return None
        if got.abstained or got.choice is None:
            return Verdict("unsure", [f"the decision model is unsure ({got.certainty:.2f})"],
                           "model", got.certainty)
        reasons = [f"the decision model says this is {got.choice} ({got.certainty:.2f})"]
        return Verdict(got.choice, reasons, "model", got.certainty)

    def classify(self, call: Call) -> Verdict | None:
        """The model's verdict for ``call``, or None when it could not answer in time."""
        key = self.key(call)
        with self.lock:
            if key in self.held:
                self.held.move_to_end(key)
                return self.held[key]
        future = self.pool.submit(self._ask, call)
        try:
            verdict = future.result(timeout=self.timeout)
        except (concurrent.futures.TimeoutError, *questions.FAILURES):
            return None
        if verdict is not None:
            with self.lock:
                self.held[key] = verdict
                while len(self.held) > self.size:
                    self.held.popitem(last=False)
        return verdict


def from_environment(env: Mapping[str, str]) -> ModelLayer | None:
    """A `ModelLayer` when the person turned it on with ``ML_STACK_DESTRUCTIVE_MODEL=1`` (and
    optionally ``ML_STACK_DESTRUCTIVE_FLOOR`` between 0.5 and 1); only a person's environment
    does."""
    if env.get("ML_STACK_DESTRUCTIVE_MODEL") != "1":
        return None
    try:
        floor = min(1.0, max(0.5, float(env.get("ML_STACK_DESTRUCTIVE_FLOOR", "0.8"))))
    except ValueError:
        floor = 0.8
    return ModelLayer(floor=floor)

