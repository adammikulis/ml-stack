"""``ml-stack-requests``: list, show, watch and answer the requests that wait for a person.
Every command is for a person: a process an agent started is refused."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

from ml_stack import activity, person, requests
from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn
from ml_stack.requests.model import PENDING, STATES, Request

__all__ = ["COMMANDS", "line", "main", "render"]

COMMANDS = Group("ml-stack-requests",
                 "Every request that needs a person: a tool call, a remembered fact, something "
                 "sentinel holds. Answering needs a person at a terminal; an agent's process is refused.")
JSON = option("json")
REFUSED, FAILED = 2, 1
FILTERS = (flag("--agent", default="", help="only this agent's requests"),
           flag("--project", default="", help="only this project's requests"),
           flag("--kind", default="", choices=["", *requests.KINDS], help="only this kind"))


def _person(action: str) -> bool:
    found = person.marked()
    if found:
        warn(f"ml-stack-requests: {action} is for a person; this process was started by an agent "
             f"({found} is set)")
    return not found


def line(request: Request, now: float) -> str:
    """One row: id, state, kind, who, what, and how long it has left."""
    who = "/".join(x for x in (request.raised_by.agent, request.raised_by.project) if x) or "-"
    left = f"{max(0, int(request.expires - now))}s left" if request.state == PENDING else request.state
    return f"{request.id}  {left:<9} {request.kind:<22} {who:<24} {request.subject[:70]}"


def render(request: Request) -> list[str]:
    """Every word of a request, as a person is shown it before answering."""
    rows = [f"{request.id}  {request.state}  {request.kind}",
            f"from:    {request.raised_by.who or '-'}  project {request.raised_by.project or '-'}",
            f"about:   {request.subject}", f"because: {request.reason}"]
    rows += [f"  {c.id:<14} {c.label}: {c.effect}" for c in request.choices]
    if request.human_only:
        rows.append("  (the action itself asks again before it runs)")
    if request.answer:
        rows.append(f"answered: {request.answer} via {request.answered_by}")
    rows.append(f"fingerprint: {request.fingerprint[:16]}")
    return rows


def _dump(rows: list[Request]) -> list[dict[str, Any]]:
    return [{**r.to_json(), "fingerprint": r.fingerprint} for r in rows]


@COMMANDS.command("list", help="the requests, waiting ones first", options=(
    JSON, *FILTERS, flag("--state", default="", choices=["", *STATES]),
    flag("-n", "--limit", type=int, default=50)))
def list_(args: argparse.Namespace) -> int:
    """Print the requests."""
    if not _person("listing requests"):
        return REFUSED
    rows = requests.list_requests(state=args.state, agent=args.agent, project=args.project,
                                  kind=args.kind, limit=args.limit)
    now = time.time()
    say(json.dumps(_dump(rows), indent=2) if args.json else
        "\n".join(line(r, now) for r in rows) or "(no requests)")
    return 0


@COMMANDS.command("show", help="every word of one request", options=(JSON, flag("id")))
def show(args: argparse.Namespace) -> int:
    """Print one request."""
    if not _person("showing a request"):
        return REFUSED
    found = requests.get(args.id)
    if found is None:
        warn(f"ml-stack-requests: no request {args.id}")
        return FAILED
    say(json.dumps(_dump([found])[0], indent=2) if args.json else "\n".join(render(found)))
    return 0


def _confirmed(found: Request, choice: str, asked: Callable[[str], str]) -> bool:
    say("\n".join(render(found)))
    return asked(f"{choice}? [y/N] ").strip().lower() in ("y", "yes")


@COMMANDS.command("answer", help="answer a request (a person at a terminal)", options=(
    flag("id"), flag("choice"),
    flag("--fingerprint", default="", help="the fingerprint `show` printed; without it the request is "
         "shown and you are asked to confirm")))
def answer(args: argparse.Namespace, asked: Callable[[str], str] | None = None) -> int:
    """Answer one request; refused when it changed since it was shown or was already answered."""
    activity.mirror_requests()
    try:
        person.require_person("answering a request")
    except person.HumanRequired as exc:
        warn(f"ml-stack-requests: {exc}")
        return REFUSED
    found = requests.get(args.id)
    if found is None:
        warn(f"ml-stack-requests: no request {args.id}")
        return FAILED
    if args.fingerprint:
        if len(args.fingerprint) < 8 or not found.fingerprint.startswith(args.fingerprint):
            warn("ml-stack-requests: the request is not what that fingerprint was taken from; show it again")
            return FAILED
    elif not _confirmed(found, args.choice, asked or input):
        say("not answered")
        return FAILED
    try:
        done = requests.answer(found.id, args.choice, found.fingerprint, "terminal")
    except (requests.Refused, requests.Unavailable) as exc:
        warn(f"ml-stack-requests: {exc}")
        return FAILED
    except person.HumanRequired as exc:
        warn(f"ml-stack-requests: {exc}")
        return REFUSED
    say(f"{done.id}: {done.state} ({done.answer})")
    return 0


@COMMANDS.command("watch", help="a quiet live view: a line when a request arrives or is answered", options=(
    flag("--interval", type=float, default=1.0), flag("--once", action="store_true")))
def watch(args: argparse.Namespace) -> int:
    """Print the waiting requests, then only what changes, until interrupted."""
    if not _person("watching requests"):
        return REFUSED
    seen: dict[str, str] = {}
    try:
        while True:
            now = time.time()
            for r in reversed(requests.list_requests(limit=100)):
                if seen.get(r.id) != r.state and (r.state == PENDING or r.id in seen):
                    say(line(r, now))
                seen[r.id] = r.state
            if args.once:
                return 0
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        return 0


main = COMMANDS.run
