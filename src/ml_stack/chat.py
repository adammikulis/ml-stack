"""``ml-stack-chat`` -- a conversation with a served model that operates ml-stack.

One conversation across turns: the history is kept (and compacted when it fills the
context), saved after every turn and resumable by id. The model has the status, serving,
download, benchmark and security-view tools of `ml_stack.do`, under `ml_stack.chatpolicy`:
reads run, anything that changes or costs something waits for the person's yes, and what
only a person at a terminal may do is refused with the command to run.
"""

from __future__ import annotations

import argparse
import inspect
import re
import secrets
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from ml_stack import chatpolicy as policy, do, files, guard as rails, home, mcp
from ml_stack.agent import Compacted, Compacting, Compaction
from ml_stack.command import Group, flag, option
from ml_stack.guard import NOTICE
from ml_stack.guard.native import screen as native_screen
from ml_stack.guard.policy import ToolPolicyRail
from ml_stack.guard.secrets import SecretRail
from ml_stack.guard.untrusted import UntrustedRail
from ml_stack.interventions import Run
from ml_stack.serve import suggest
from ml_stack.taint import TaintRail

__all__ = ["COMMAND", "Chat", "Session", "default_model", "main", "serve", "tools_for_chat"]

SCHEMA_VERSION = 1
ROUNDS = 12
"""Tool-calling rounds one message may spend."""

SYSTEM = (
    "You are the ml-stack assistant: a person talks to you at a terminal and you operate "
    "ml-stack for them with the tools you have been given -- what is serving, models on this "
    "machine and on the Hub, downloads, benchmarks, jobs, and read-only views of the security "
    "review. You can run nothing except by calling a tool.\n\n"
    "Look before you act: check serve_status and bench_status before starting anything, because "
    "one thing runs on the GPU at a time. A tool that starts, stops, downloads or measures asks "
    "the person yes or no itself when you call it; do not ask them in words first, and never "
    "treat a refusal as something to route around. Say plainly what failed and what the error "
    "said. Never claim a result a tool did not return.\n\n"
    "Some things only the person can do, in their own terminal: releasing or purging what is "
    "held in quarantine (`ml-stack-security review`), approving a host "
    "(`ml-stack-security approve-host HOST`), minting a grant, changing the security mode or "
    "policy (`ml-stack-security mode`, `ml-stack-security scan-policy`). You have no tool for "
    "these. When asked, say so and give that command; show what is held with review_view.\n\n"
    "Keep answers short. Ask one question when a choice is open.")

NOTE = ("Note from ml-stack: the message above asks you to {what}. You cannot: only the person "
        "can, in their own terminal. Tell them to run: {command}")


def _id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
_ROLES = {"user", "assistant", "tool"}


@dataclass
class Session:
    """The saved conversation: ``messages`` without the system prompt, under an id."""

    id: str = field(default_factory=_id)
    model: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def folder() -> Path:
        return home.state("chat")

    @property
    def path(self) -> Path:
        return self.folder() / f"{self.id}.json"

    def save(self, messages: Sequence[dict[str, Any]]) -> Path:
        """Write the conversation, its system message left out."""
        self.messages = [m for m in messages if m.get("role") != "system"]
        files.write_json(self.path, {"schema_version": SCHEMA_VERSION, "id": self.id,
                                     "model": self.model, "saved": time.strftime("%FT%T"),
                                     "messages": self.messages})
        return self.path

    @classmethod
    def load(cls, which: str = "last") -> Session:
        """The session ``which`` names, or the newest. Raises ``LookupError`` when none."""
        if which in ("", "last"):
            found = sorted(cls.folder().glob("*.json"), key=lambda p: p.stat().st_mtime) \
                if cls.folder().is_dir() else []
            if not found:
                raise LookupError("no saved chat to resume")
            which = found[-1].stem
        if not _ID.match(which):
            raise LookupError(f"{which!r} is not a chat id")
        row = files.read_json(cls.folder() / f"{which}.json", None)
        if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
            raise LookupError(f"no chat {which} (or it is from another version)")
        return cls(which, str(row.get("model") or ""), _clean(row.get("messages")))


def _clean(rows: Any) -> list[dict[str, Any]]:
    """The saved messages that are the shape the loop writes: user, assistant and tool
    messages only, ending where every tool call has its result."""
    kept = [dict(m) for m in rows or [] if isinstance(m, dict) and m.get("role") in _ROLES
            and isinstance(m.get("content", ""), (str, type(None)))]
    while kept and (kept[-1]["role"] == "tool" or kept[-1].get("tool_calls")):
        kept.pop()
        while kept and kept[-1]["role"] == "tool":
            kept.pop()
    return kept


def tools_for_chat(*, person: do.Person, registry: Sequence[mcp.Tool] | None = None,
                   files: Sequence[Path] | None = None,
                   fetch: Any = None) -> list[tuple[dict[str, Any], Callable[..., Any]]]:
    """The tools the chat offers: the reads and the asking-first ones of `ml_stack.do`, the
    security views, and ``ask_user`` and ``plan``."""
    wanted = policy.READ | frozenset(policy.CONFIRM)
    tools = [(s, fn) for s, fn in do.command_tools(registry, files=files, fetch=fetch)
             if s["function"]["name"] in wanted - {"review_view"}]
    view = mcp.Tool("review_view", "Read-only views of the security review.", policy.review_view)
    tools.append((do._schema(view.name, view.description, view.fn,
                             inspect.getdoc(view.fn) or ""), view.fn))
    return [*tools, *(t for t in person.tools() if t[0]["function"]["name"] != "done")]


def default_model() -> tuple[str, str]:
    """``(reference, why)`` of the model to chat with when none is named: the best ranked
    for an agent among those already downloaded, a mixture-of-experts first. The
    reference is empty, and ``why`` says what to pull, when nothing suitable is here."""
    pull = ("no model is downloaded for this: pull the mixture-of-experts Flash-Next with "
            "`ml-stack-models find Qwen3.8 Flash Next` then `ml-stack-models fetch <hf: reference>`")
    try:
        have = [r for r in suggest.recommend(goal="agent") if r.installed]
    except OSError:
        have = []
    if not have:
        return "", pull
    moe = [r for r in have if re.search(r"flash-?next|-a\d+b|moe", r.choice.candidate.name + r.ref, re.I)]
    best = (moe or have)[0]
    return best.ref, ("a mixture-of-experts model, ranked best for an agent here" if moe else
                      "the best ranked for an agent here (not a mixture-of-experts)")


class Chat:
    """One conversation: a client, a person, the tools, the rails and the history."""

    def __init__(self, client: Any, person: do.Person, *,
                 tools: Sequence[tuple[dict[str, Any], Callable[..., Any]]] | None = None,
                 session: Session | None = None, screen: Sequence[Any] = ()) -> None:
        self.client, self.person, self.rounds = client, person, ROUNDS
        self.offered = list(tools if tools is not None else tools_for_chat(person=person))
        self.session = session or Session()
        self.extra = list(screen)
        self.begin(self.session.messages)

    def begin(self, history: Sequence[dict[str, Any]] = ()) -> None:
        """Start the conversation over ``history`` with a fresh set of rails."""
        schemas = [s for s, _ in self.offered]
        names = {s["function"]["name"] for s in schemas}
        self.limits = ToolPolicyRail()
        mine = [policy.HumanOnlyRail(), self.limits, policy.ConfirmRail(lambda: names),
                UntrustedRail(external=policy.FENCED), SecretRail(),
                TaintRail(registries={"models": do.on_disk_ids}), *self.extra]
        self.watch: Run = rails.start(mine, offered=schemas, task="", confirm=self.person.confirm)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM + "\n\n" + NOTICE}, *history]
        self.watch.context.messages = self.messages

    def new(self) -> None:
        """Forget the conversation and start another under a new id."""
        self.session = Session(model=self.session.model)
        self.begin()

    def _only(self, names: set[str]) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
        keep = [(s, fn) for s, fn in self.offered if s["function"]["name"] in names]
        return [s for s, _ in keep], {s["function"]["name"]: fn for s, fn in keep}

    def _say(self, schemas: list[dict[str, Any]]) -> tuple[str, list[Any]]:
        """One model call, its words streamed to the person; the words after the guard's
        screen and the calls."""
        streamed: list[str] = []

        def delta(kind: str, text: str) -> None:
            if kind == "content":
                streamed.append(text)
                self.person.stdout.write(text)
                self.person.stdout.flush()

        try:
            reply = self.client.chat(self.messages, think=False, tools=schemas, on_delta=delta)
        except NotImplementedError:
            reply = self.client.chat(self.messages, think=False, tools=schemas)
        said = getattr(reply, "content", "") or ""
        shown = self.watch.screen_model(said).text
        if streamed:
            self.person.say("")
            if shown != said:
                self.person.say("(the guard changed that reply; what is kept:)")
                self.person.say(shown)
        elif shown.strip():
            self.person.say(shown.strip())
        return shown, list(getattr(reply, "tool_calls", None) or [])

    def turn(self, text: str, *, only: set[str] | None = None, prefix: str = "") -> do.Outcome:
        """One message from the person: the model answers, calling tools until it has said
        its piece. ``only`` limits the tools for this message."""
        names = only or {s["function"]["name"] for s, _ in self.offered}
        schemas, run_by = self._only(names)
        refusal = policy.refusal_for(text)
        body = prefix + text
        if refusal:
            self.person.say(f"\n{policy.refused(*refusal)}")
            body += "\n\n" + NOTE.format(what=refusal[0], command=refusal[1])
        self.watch.context.task = text
        self.limits.__post_init__()
        self.messages.append({"role": "user", "content": body})
        out = do.Outcome(messages=self.messages)
        for _ in range(self.rounds):
            said, calls = self._say(schemas)
            if not calls:
                self.messages.append({"role": "assistant", "content": said})
                break
            out.rounds += 1
            self.messages.append({"role": "assistant", "content": said, "tool_calls": calls})
            for call in calls:
                do.answer_call(call, run_by, self.watch, self.person, out)
                if self.person.left:
                    break
            if self.person.left:
                break
        else:
            self.person.say(f"\n(stopped after {self.rounds} rounds; say go on to continue)")
        self.session.save(self.messages)
        return out


PLAN = ("Plan only. Do not start, stop, download or measure anything this turn. Look with the "
        "read tools if you need to, then call plan with the steps, each naming the tool and its "
        "arguments, and wait for the person.\n\nRequest: ")

HELP = """/help          this list
/new           forget this conversation and start another
/tools         the tools, and which ask you first
/plan TEXT     have the model plan TEXT and show the steps, doing nothing
/model [REF]   the model in use; with REF, switch to it
/quit          leave (the conversation is saved: ml-stack-chat --resume)
anything else  is said to the model"""


def _tools_text(chat: Chat) -> str:
    lines = []
    for schema, _ in chat.offered:
        name = schema["function"]["name"]
        mark = "asks first" if name in policy.CONFIRM else "reads"
        lines.append(f"  {name:<16} {mark}")
    return "tools:\n" + "\n".join(lines)


def repl(chat: Chat, stdin: TextIO, stdout: TextIO, *,
         connect: Callable[[str], Any] | None = None) -> int:
    """Read messages and slash commands from ``stdin`` until /quit or EOF."""
    stdout.write(f"chat {chat.session.id} -- /help for commands, /quit to leave\n")
    while True:
        stdout.write("\nyou> ")
        stdout.flush()
        line = stdin.readline()
        if line == "":
            break
        text = line.strip()
        if not text:
            continue
        if not text.startswith("/"):
            chat.turn(text)
        else:
            word, _, rest = text.partition(" ")
            if word in ("/quit", "/exit"):
                break
            if word == "/help":
                stdout.write(HELP + "\n")
            elif word == "/new":
                chat.new()
                stdout.write(f"new chat {chat.session.id}\n")
            elif word == "/tools":
                stdout.write(_tools_text(chat) + "\n")
            elif word == "/plan":
                if rest.strip():
                    chat.turn(rest.strip(), only=policy.READ | {"plan", "ask_user"}, prefix=PLAN)
                else:
                    stdout.write("/plan TEXT: what to plan\n")
            elif word == "/model":
                if not rest.strip():
                    stdout.write(f"model: {chat.session.model or 'a server given with --url'}\n")
                elif connect is None:
                    stdout.write("this chat cannot switch models\n")
                else:
                    chat.client = connect(rest.strip())
                    chat.session.model = rest.strip()
                    stdout.write(f"model: {rest.strip()}\n")
            else:
                stdout.write(f"{word} is not a command; /help lists them\n")
        stdout.flush()
    return 0


OPTIONS = (
    flag("--model", default="", help="a model to lease (default: the best downloaded "
         "mixture-of-experts model ranked for an agent)"),
    flag("--url", default="", help="a server already up, e.g. http://127.0.0.1:8080"),
    flag("--resume", nargs="?", const="last", default="", metavar="ID",
         help="continue a saved chat: the newest, or the id given"),
    option("port", default=8080, help="where --model is served"),
    flag("--draft", default="auto", metavar="HEAD",
         help="the draft head: 'auto', 'none' or a named head (default: %(default)s)"),
    flag("--rounds", type=int, default=ROUNDS,
         help="tool-calling rounds one message may spend (default: %(default)s)"),
    flag("--n-predict", type=int, default=do.N_PREDICT),
    option("timeout", default=900.0, help="seconds to wait for one reply (default: %(default)s)"),
    flag("--context-size", type=int, default=0, metavar="TOKENS",
         help="the context the history is compacted against (default: ask the server)"),
    flag("--no-compact", action="store_true", help="never summarise the history"),
    option("dry-run", help="print the system prompt and the tools as the model sees them"),
)


def serve(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    """Run the chat the parsed ``args`` describe over ``stdin`` and ``stdout``."""
    if args.model and args.url:
        stdout.write("--model and --url name two servers: give one\n")
        return 2
    person = do.Person(stdin, stdout)
    if args.dry_run:
        stdout.write(SYSTEM + "\n\n" + NOTICE + "\n\n")
        do._print_offer(tools_for_chat(person=person), stdout)
        return 0
    try:
        session = Session.load(args.resume) if args.resume else Session()
    except LookupError as exc:
        stdout.write(f"{exc}\n")
        return 1
    if not args.model and not args.url:
        args.model, why = (session.model, "the model this chat used") if session.model \
            else default_model()
        if not args.model:
            stdout.write(why + "\n")
            return 1
        stdout.write(f"model: {args.model} ({why})\n")
    session.model = args.model or session.model

    def connect(ref: str) -> Any:
        client = do.client_for(argparse.Namespace(**{**vars(args), "model": ref, "url": ""}))
        return _compacting(client, args, stdout)

    client = connect(args.model) if not args.url else _compacting(do.client_for(args), args, stdout)
    screen = native_screen()
    chat = Chat(client, person, session=session, screen=screen)
    chat.rounds = args.rounds
    if session.messages:
        stdout.write(f"resumed {len(session.messages)} messages\n")
    try:
        return repl(chat, stdin, stdout, connect=connect)
    finally:
        for one in screen:
            close = getattr(one, "close", None)
            if close is not None:
                close()


COMMAND = Group(
    "ml-stack-chat",
    "Talk to a served model that operates ml-stack: what is serving, models, downloads, "
    "benchmarks, jobs, read-only security views. Anything that starts, stops, downloads or "
    "measures asks you yes or no first; releasing quarantine, approving a host and changing "
    "the security policy are yours alone, in your own terminal.",
    options=OPTIONS, run=lambda args: serve(args, sys.stdin, sys.stdout), allow_abbrev=False)
main = COMMAND.run


def _compacting(client: Any, args: argparse.Namespace, stdout: TextIO) -> Any:
    if args.no_compact:
        return client

    def shown(event: Any) -> None:
        if isinstance(event, Compacted):
            stdout.write(f"(compacted {event.dropped} messages: {event.before:.0%} -> "
                         f"{event.after:.0%} of the context)\n")

    return Compacting(client, Compaction(context_size=args.context_size or None), on_event=shown)


if __name__ == "__main__":  # pragma: no cover - the entry point is `ml-stack-chat`
    raise SystemExit(main())
