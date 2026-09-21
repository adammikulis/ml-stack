"""The steps of an escalation: what is live, what a bigger slot count costs, saving
every conversation before the relaunch and putting it back afterwards."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.client.chat import Client
from ml_stack.client.settings import Request, Transport
from ml_stack.http import ServerError, request_json
from ml_stack.serve.backend import ServerFailed, ServerSpec
from ml_stack.serve.events import Event, emit
from ml_stack.serve.fit import records
from ml_stack.serve.matching import model_matches
from ml_stack.units import human_bytes

__all__ = ["SUMMARY_PROMPT", "SUMMARY_SUFFIX", "Escalating", "EscalationRefused", "Plan",
           "plan_for", "restore", "save_live", "slots_on", "summarise"]

# What a slot's cache is asked to become when a live conversation cannot be summarised.
SUMMARY_PROMPT = (
    "Summarise this conversation so far in under 200 words. Keep every fact, decision "
    "and open question a continuation of it would need.")

# Appended to a slot's own cached prompt text for a raw /completion continuation, so the
# shared prefix is a cache hit and only this tail and the generation are new work.
SUMMARY_SUFFIX = (
    "\n\nSummarise the conversation above in under 200 words. Keep every fact, decision "
    "and open question a continuation of it would need.\n\nSummary:")


class EscalationRefused(ServerFailed):
    """Growing or splitting a server's slots would drop a live conversation's cache, and
    summarising it did not rescue that. The saved cache named in the message is kept."""


@dataclass(frozen=True)
class Escalating:
    """The server an escalation is working on, and how it reports its steps."""

    base_url: str
    port: int
    timeout: float | None = None
    on_event: Event | None = None

    @property
    def wait(self) -> float:
        """The timeout a slot call gets."""
        return self.timeout or 120.0


@dataclass(frozen=True)
class Plan:
    """How the extra slots are found: by growing the cache, or by splitting what is there."""

    mode: str
    reason: str
    new_context: int
    too_long: list[tuple[int, int]]


def _fit_for(model: str | Path, *, room: int) -> Any:
    """The measured :class:`~ml_stack.serve.fit.Fit` for ``model`` at ``room``, or ``None``
    when nothing has measured it."""
    for one in records(room=room):
        if model_matches(one.model, model):
            return one
    return None


def slots_on(run: Escalating) -> tuple[list[tuple[int, int]], dict[int, str]]:
    """The slots holding a conversation, as ``(id, tokens)``, and every slot's prompt text."""
    try:
        slots = request_json(f"{run.base_url}/slots", method="GET",
                             timeout=run.timeout or 30.0)
    except ServerError as exc:
        raise ServerFailed(
            f"port {run.port} has no /slots endpoint to escalate from: {exc}") from exc
    if not isinstance(slots, list):
        raise ServerFailed(
            f"port {run.port}: /slots answered with {type(slots).__name__}, not a list")

    live = sorted((int(s["id"]), int(s.get("n_prompt_tokens") or 0)) for s in slots
                 if isinstance(s, dict) and int(s.get("n_prompt_tokens") or 0) > 0)
    # Only present with LLAMA_SERVER_SLOTS_DEBUG=1 (backend.py sets it whenever
    # slot_save_path is), which is what a summary is asked to read rather than the
    # bare instruction a slot's cache cannot answer on its own.
    prompts = {int(s["id"]): str(s.get("prompt") or "")
              for s in slots if isinstance(s, dict) and s.get("id") is not None}
    return live, prompts


def plan_for(spec: ServerSpec, *, new_slots: int, per_slot: int,
             live: list[tuple[int, int]], room_bytes: int) -> Plan:
    """Whether the extra slots are grown, split, or split after a summary."""
    fit = _fit_for(spec.model, room=room_bytes)
    if fit is not None:
        loaded, each = fit.line(per_slot)
        need = loaded + new_slots * each
        if need <= room_bytes:
            return Plan(
                "grow",
                f"{new_slots} slots of {per_slot:,} tokens need {human_bytes(need)}, "
                f"which fits in {human_bytes(room_bytes)} of room",
                per_slot * new_slots, [])

    new_context = int(spec.context)
    split_per_slot = max(1, new_context // new_slots)
    too_long = [(sid, tok) for sid, tok in live if tok > split_per_slot]
    if too_long:
        return Plan(
            "summarize",
            f"{len(too_long)} live conversation(s) do not fit the "
            f"{split_per_slot:,}-token slot a split leaves them and will be summarised",
            new_context, too_long)
    return Plan(
        "split",
        f"the existing {new_context:,}-token cache split across "
        f"{new_slots} slots is {split_per_slot:,} each",
        new_context, [])


def save_live(run: Escalating, live: list[tuple[int, int]]) -> dict[int, str]:
    """Save every live slot's cache, returning the file each one went to."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    saved: dict[int, str] = {}
    for sid, tok in live:
        filename = f"escalate-{run.port}-{sid}-{stamp}.bin"
        emit(run.on_event, "saving", port=run.port, slot=sid, tokens=tok,
             filename=filename)
        try:
            request_json(f"{run.base_url}/slots/{sid}?action=save",
                        payload={"filename": filename}, timeout=run.wait)
        except ServerError as exc:
            raise ServerFailed(
                f"could not save slot {sid} on port {run.port}: {exc}") from exc
        saved[sid] = filename
    return saved


def _on_slot(run: Escalating, sid: int) -> Client:
    """A client pinned to slot ``sid`` of ``run``'s server."""
    return Client(run.base_url, request=Request(slot=sid), transport=Transport(timeout=run.wait))


def _one_summary(run: Escalating, sid: int, prior: str) -> str:
    """The model's own summary of what slot ``sid`` holds."""
    if prior:
        # A raw continuation of the slot's own cached prompt: the shared prefix is a
        # cache hit, so this costs the generation and nothing about the reprocessing
        # the coordinator's cheap-summary case rests on.
        return _on_slot(run, sid).complete(
            prior + SUMMARY_SUFFIX, n_predict=512)
    # No prompt text to read (LLAMA_SERVER_SLOTS_DEBUG was not on for this server) --
    # the model is asked cold and told nothing.
    reply = _on_slot(run, sid).chat(
        [{"role": "user", "content": SUMMARY_PROMPT}], n_predict=512)
    return (getattr(reply, "content", "") or "").strip()


def summarise(run: Escalating, too_long: list[tuple[int, int]], *,
              prompts: dict[int, str], saved: dict[int, str]) -> dict[int, str]:
    """A summary per slot too long for the slot a split leaves it."""
    summaries: dict[int, str] = {}
    for sid, tok in too_long:
        emit(run.on_event, "summarizing", port=run.port, slot=sid, tokens=tok)
        try:
            summary = _one_summary(run, sid, prompts.get(sid, ""))
        except Exception as exc:
            raise EscalationRefused(
                f"slot {sid} on port {run.port} holds {tok:,} tokens, too long for "
                f"the slot a split leaves it, and summarising it failed: {exc}. Its "
                f"cache is kept at {saved[sid]}."
            ) from exc
        if not summary:
            raise EscalationRefused(
                f"slot {sid} on port {run.port} holds {tok:,} tokens, too long for "
                f"the slot a split leaves it, and summarising it returned nothing. "
                f"Its cache is kept at {saved[sid]}."
            )
        summaries[sid] = summary
        # The text itself, not just its length: llama-server's chat API is stateless
        # per request, so a slot's cache being re-seeded with this summary carries no
        # memory a later /v1/chat/completions call will see on its own -- the caller
        # holding the conversation has to fold the summary into its own transcript to
        # actually continue from it, and can only do that if this event carries it.
        emit(run.on_event, "summarized", port=run.port, slot=sid, summary=summary,
             tokens=len(summary.split()))
    return summaries


def restore(run: Escalating, live: list[tuple[int, int]], *, summaries: dict[int, str],
            saved: dict[int, str]) -> None:
    """Put every live conversation back: its summary where there is one, else its cache."""
    for sid, _tok in live:
        if sid in summaries:
            emit(run.on_event, "restoring", port=run.port, slot=sid, mode="summary")
            try:
                _on_slot(run, sid).complete(
                    summaries[sid], n_predict=1)
            except Exception as exc:
                raise ServerFailed(
                    f"could not re-seed the summary for slot {sid} on port "
                    f"{run.port}: {exc}. Its full cache is kept at {saved[sid]}."
                ) from exc
        else:
            emit(run.on_event, "restoring", port=run.port, slot=sid, mode="cache")
            try:
                request_json(f"{run.base_url}/slots/{sid}?action=restore",
                            payload={"filename": saved[sid]}, timeout=run.wait)
            except ServerError as exc:
                raise ServerFailed(
                    f"could not restore slot {sid} on port {run.port} from "
                    f"{saved[sid]}: {exc}"
                ) from exc
