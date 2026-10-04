"""``ml-stack-chat`` -- the one agent command: a conversation or a task, under a role.

With no task it is a conversation: the history is kept (and compacted when it fills the
context), saved after every turn and resumable by id. With a task in words it shows its plan,
asks go, acts and ends on ``done``. Either way the model has the tools of `ml_stack.do` held
to a role (`ml_stack.roles`) and the rails of `ml_stack.chatpolicy`: reads run, an acting call
asks unless the role's approved plan names it, and what only a person at a terminal may do is
refused with the command to run.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import secrets
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TextIO

from ml_stack import (
    chatpolicy as policy,
    do,
    files,
    guard as rails,
    home,
    mcp,
    memory,
    roles,
    rules as saved,
)
from ml_stack.agent import Compacted, Compacting, Compaction
from ml_stack.command import Group, flag, option
from ml_stack.guard import NOTICE, parse_call
from ml_stack.guard.destructive_model import from_environment
from ml_stack.guard.destructive_rail import DestructiveRail, default_roots
from ml_stack.guard.native import screen as native_screen
from ml_stack.guard.policy import Limits, ToolPolicyRail
from ml_stack.guard.secrets import SecretRail
from ml_stack.guard.untrusted import UntrustedRail
from ml_stack.interventions import Deny, Run, Verdict
from ml_stack.memory import Memory
from ml_stack.serve import suggest
from ml_stack.taint import TaintRail

__all__ = ["COMMAND", "Chat", "Outcome", "Session", "default_model", "main", "run_task", "serve",
           "tools_for_chat"]

SCHEMA_VERSION = 1
ROUNDS = 12
"""Tool-calling rounds one message may spend."""
TASK_ROUNDS = 40
"""Tool-calling rounds one task may spend."""
CUT = 6000
"""Characters of one tool result the model is shown."""

SYSTEM = (
    "You are the ml-stack assistant: a person talks to you at a terminal and you operate "
    "ml-stack for them with the tools you have been given -- what is serving, models on this "
    "machine and on the Hub, downloads, benchmarks, jobs, and read-only views of the security "
    "review. You can run nothing except by calling a tool.\n\n"
    "Look before you act: check serve_status and bench_status before starting anything, because "
    "one thing runs on the GPU at a time. A tool that starts, stops, downloads or measures may "
    "ask the person yes or no itself when you call it (your role, below, says when); do not ask "
    "them in words first, and never treat a refusal as something to route around. Say plainly "
    "what failed and what the error said. Never claim a result a tool did not return.\n\n"
    "Some things only the person can do, in their own terminal: releasing or purging what is "
    "held in quarantine (`ml-stack-security review`), approving a host "
    "(`ml-stack-security approve-host HOST`), minting a grant, changing the security mode or "
    "policy (`ml-stack-security mode`, `ml-stack-security scan-policy`), and changing your own "
    "role or the rules for what runs unasked (`/role`, `/rules`, typed by the person). You have "
    "no tool for these. When asked, say so and give that command; show what is held with "
    "review_view.\n\n"
    "Keep answers short. Ask one question when a choice is open.")

TASK = (
    "You were given a task in words. Ask before assuming. When the task leaves a choice open "
    "-- which model or which file, how many questions, a sample or the full set, with or "
    "without a draft head, where to write -- call ask_user rather than guessing, one question "
    "per call, and wait for the answer before asking the next. Look things up first so the "
    "question names what was found: a task that names a model is answered with models_on_disk "
    "and, when it names Ollama or two backends, ollama_models too, and the person is asked to "
    "confirm the exact files before anything starts. Do not start a measurement the person has "
    "not confirmed.\n\n"
    "Then call plan with the steps in order, each naming the tool first and every argument it "
    "will be called with; it asks the person \"go?\", and the go covers those values and no "
    "others: a call the plan does not name asks the person again. Then act: call the tools in "
    "the order planned. A long command detaches and returns a log and a pid; call jobs_wait to "
    "wait for it rather than calling status again and again.\n\n"
    "Last, call done with what was measured and where it is -- the labels, the numbers if "
    "any came back, the files written. Say plainly when something failed and what the "
    "error said. Never claim a result a tool did not return.")

NUDGE = ("You replied in words and called nothing. If the task is finished, call done with "
         "a summary of what was measured and where it is; if you need something from the "
         "person, call ask_user; otherwise call the next tool.")

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
    """Every tool any role may use: the reads and the asking-first ones of `ml_stack.do`, the
    security views, and ``ask_user``, ``plan`` and ``done``. A role shows the model a subset."""
    wanted = policy.READ | frozenset(policy.CONFIRM)
    tools = [(s, fn) for s, fn in do.command_tools(registry, files=files, fetch=fetch)
             if s["function"]["name"] in wanted - {"review_view"}]
    view = mcp.Tool("review_view", "Read-only views of the security review.", policy.review_view)
    tools.append((do._schema(view.name, view.description, view.fn,
                             inspect.getdoc(view.fn) or ""), view.fn))
    return [*tools, *person.tools()]


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


@dataclass
class Outcome:
    """What one message or task did."""

    done: bool = False
    summary: str = ""
    rounds: int = 0
    seconds: float = 0.0
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[Verdict] = field(default_factory=list)
    screened: int = 0
    withheld: int = 0
    asked: int = 0

    def cost(self) -> str:
        """The line printed when a task ends."""
        return (f"cost: {self.rounds} rounds, {len(self.calls)} calls, {self.asked} asked, "
                f"{len(self.blocked)} blocked, {self.withheld} withheld, {self.seconds:.1f} s")


def _compact(args: dict[str, Any], most: int = 160) -> str:
    text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return text if len(text) <= most else text[: most - 3] + "..."


def transcript(messages: Iterable[dict[str, Any]]) -> str:
    """The conversation as lines a person reads: role, then what was said or called."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            lines.append(f"{role}: -> {fn.get('name')}({fn.get('arguments', '')})")
        text = re.sub(r"\s+", " ", str(m.get("content") or "").strip())
        if text:
            name = f" {m['name']}" if role == "tool" and m.get("name") else ""
            lines.append(f"{role}{name}: {text[:300]}")
    return "\n".join(lines)


class Chat:
    """One session: a client, a person, the tools, a role, the rails and the history. With
    ``task`` the session ends on ``done`` and nudges a model that answers in words; setting
    ``guard`` and calling ``begin`` replaces the rails and the role (for code that supplies
    its own)."""

    def __init__(self, client: Any, person: do.Person, *,  # noqa: PLR0913
                 tools: Sequence[tuple[dict[str, Any], Callable[..., Any]]] | None = None,
                 session: Session | None = None, screen: Sequence[Any] = (),
                 role: str = roles.DEFAULT, task: bool = False,
                 extension: roles.Extension | None = None) -> None:
        self.client, self.person, self.task = client, person, task
        self.guard: Sequence[Any] | None = None
        self.rounds = TASK_ROUNDS if task else ROUNDS
        self.extension = extension or roles.Extension()
        self.offered = list(tools if tools is not None else tools_for_chat(person=person))
        have = {s["function"]["name"] for s, _ in self.offered}
        self.offered += [(s, fn) for s, fn in self.extension.tools() if s["function"]["name"] not in have]
        self.session = session or Session()
        self.extra = list(screen)
        self.role = roles.get(role)
        self.plan = roles.PlanLedger()
        self.begin(self.session.messages)

    def names(self) -> set[str]:
        """The tool names the model is shown now."""
        every = {s["function"]["name"] for s, _ in self.offered}
        if self.guard is not None:
            return every
        own = set(roles.OWN) - (set() if self.task else {"done"})
        return every & (set(self.role.tools) | self.extension.allowed(self.role) | own)

    def system(self) -> str:
        role = f"Your role is {self.role.name}: {self.role.summary}."
        return "\n\n".join(p for p in (SYSTEM, TASK if self.task else "", role,
                                        self.extension.context(), NOTICE) if p)

    def begin(self, history: Sequence[dict[str, Any]] = ()) -> None:
        """Start over ``history`` with a fresh set of rails."""
        schemas = [s for s, _ in self.offered]
        self.plan.clear()
        self.rules = saved.Rules()
        self.gate = roles.RoleRail(self.role, self.names, self.plan, extension=self.extension,
                                   rules=self.rules)
        self.person.rules = self.rules
        floors = {**dict.fromkeys(self.extension.reads, "safe"),
                  **dict.fromkeys(self.extension.asks, "reversible")}
        mine = list(self.guard) if self.guard is not None else [
            policy.HumanOnlyRail(),
            ToolPolicyRail(limits=Limits(calls=self.role.max_calls)), self.gate,
            DestructiveRail(roots=default_roots(), model=from_environment(os.environ),
                            catalog=policy.catalog(), floors=floors,
                            skip={*roles.OWN, *self.extension.asks_itself}),
            UntrustedRail(external=policy.FENCED), SecretRail(),
            TaintRail(registries={"models": do.on_disk_ids}), *self.extra]
        self.limits = next((r for r in mine if isinstance(r, ToolPolicyRail)), ToolPolicyRail())
        self.watch: Run = rails.start(mine, offered=schemas, task="", confirm=self.person.confirm)
        self.started = False
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system()},
                                               *history]
        self.watch.context.messages = self.messages

    def use_role(self, name: str) -> None:
        """Run under another role from now on; only the person's typed ``/role`` calls this."""
        self.role = roles.get(name)
        self.gate.set(self.role)
        self.limits.limits = replace(self.limits.limits, calls=self.role.max_calls)
        self.messages[0] = {"role": "system", "content": self.system()}

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

    def turn(self, text: str, *, only: set[str] | None = None, prefix: str = "") -> Outcome:
        """One message from the person (or the task): the model answers, calling tools until
        it has said its piece or, for a task, called ``done``. ``only`` limits the tools for
        this message."""
        names = (only or set(self.names())) & self.names() if self.guard is None \
            else (only or self.names())
        schemas, run_by = self._only(names)
        refusal = policy.refusal_for(text)
        if not self.started:
            self.started = True
            first = self.extension.start(text)
            prefix = first + "\n\n" + prefix if first else prefix
        body = prefix + text
        if refusal:
            self.person.say(f"\n{policy.refused(*refusal)}")
            body += "\n\n" + NOTE.format(what=refusal[0], command=refusal[1])
        self.watch.context.task = text
        self.limits.__post_init__()
        self.messages.append({"role": "user", "content": body})
        out = Outcome(messages=self.messages)
        asked0, began = self.person.asked, time.monotonic()
        exhausted, nudged = True, False
        for _ in range(self.rounds):
            said, calls = self._say(schemas)
            if not calls:
                self.messages.append({"role": "assistant", "content": said})
                if not self.task or nudged:
                    exhausted = False
                    break
                nudged = True
                self.messages.append({"role": "user", "content": NUDGE})
                continue
            out.rounds += 1
            self.messages.append({"role": "assistant", "content": said, "tool_calls": calls})
            for call in calls:
                self._answer(call, run_by, out)
                if self.person.finished or self.person.left:
                    break
            if self.person.finished or self.person.left:
                exhausted = False
                break
        out.seconds = round(time.monotonic() - began, 2)
        out.asked = self.person.asked - asked0
        out.done, out.summary = self.person.finished, self.person.summary
        if exhausted and self.task:
            self.person.say(f"\nran out of {self.rounds} rounds without done; the transcript:")
            self.person.say(transcript(self.messages))
        elif exhausted:
            self.person.say(f"\n(stopped after {self.rounds} rounds; say go on to continue)")
        if self.task:
            self.person.say(out.cost())
        self.session.save(self.messages)
        return out

    def _answer(self, call: dict[str, Any], run_by: dict[str, Callable[..., Any]],
                out: Outcome) -> None:
        """Run one call the model made, if the rails let it, and append what came back."""
        asked = parse_call(call)
        args = asked.arguments or {}
        person = self.person
        if asked.name not in roles.OWN:
            person.say(self.watch.screen_model(f"-> {asked.name}({_compact(args)})").text)
        gate = self.watch.check_call(asked)
        tool = run_by.get(asked.name)
        if not gate.allowed:
            by = getattr(gate.verdict, "by", "") or "guard"
            why = gate.verdict.reason if isinstance(gate.verdict, Deny) else gate.text
            result: Any = {"error": f"blocked by the {by} rail: {why}"}
            out.blocked.append(gate.verdict)
        elif tool is None:
            result = {"error": f"no such tool: {asked.name}"}
        else:
            began = time.monotonic()
            try:
                result = tool(**args)
            except Exception as exc:  # noqa: BLE001 - the error is the answer
                result = {"error": f"{type(exc).__name__}: {exc}"}
            self.gate.spent(asked.name, time.monotonic() - began)
        out.calls.append((asked.name, args))
        if asked.name == "plan" and gate.allowed and result.get("go"):
            steps = [str(step) for step in args.get("steps") or []]
            self.watch.approve(" ".join(steps))
            self.plan.approve(steps)
        text = json.dumps(mcp._plain(result), ensure_ascii=False, default=str)[:CUT]
        answer = text
        if asked.name not in roles.OWN and gate.allowed:
            shown = self.watch.screen_result(asked, text)
            answer = shown.text
            out.screened += 1
            out.withheld += shown.withheld
        if self.watch.guides:
            answer = f"{answer}\n\n{' '.join(self.watch.guides)}"
            self.watch.guides = []
        if asked.name not in roles.OWN:
            person.say("   " + self.watch.screen_model(text).text[:300])
        out.messages.append({"role": "tool", "tool_call_id": call.get("id") or asked.name,
                             "name": asked.name, "content": answer})


def run_task(task: str, client: Any, *,  # noqa: PLR0913
             tools: Sequence[tuple[dict[str, Any], Callable[..., Any]]] | None = None,
             person: do.Person | None = None, rounds: int = TASK_ROUNDS,
             guard: Sequence[Any] | None = None, role: str = roles.TASK_DEFAULT) -> Outcome:
    """One task through the loop. ``tools`` (default: all of them) are offered with the
    person's own three; ``guard`` replaces the rails and the role, and without it the role's
    rails and the model-based screen are on."""
    person = person or do.Person(sys.stdin, sys.stdout)
    screen = native_screen() if guard is None else []
    offered = tools_for_chat(person=person) if tools is None else [*tools, *person.tools()]
    try:
        chat = Chat(client, person, tools=offered, screen=screen, role=role, task=True)
        chat.guard = guard
        chat.begin()
        chat.rounds = rounds
        return chat.turn(task)
    finally:
        _close(screen)


def _close(screen: Iterable[Any]) -> None:
    for one in screen:
        close = getattr(one, "close", None)
        if close is not None:
            close()


PLAN = ("Plan only. Do not start, stop, download or measure anything this turn. Look with the "
        "read tools if you need to, then call plan with the steps, each naming the tool and its "
        "arguments, and wait for the person.\n\nRequest: ")

HELP = f"""/help          this list
/new           forget this conversation and start another
/tools         the tools in this role, and which ask you first
/role [NAME]   the roles; with NAME, run under that role (only you can: the model has no way to)
/plan TEXT     have the model plan TEXT and show the steps, doing nothing
/model [REF]   the model in use; with REF, switch to it
/memory [TEXT] what would be recalled for TEXT (default: your last message), with the scope
               of each fact (user: you, all projects; project: this one)
/quit          leave (the conversation is saved: ml-stack-chat --resume)
anything else  is said to the model
{saved.RULES_HELP}"""


def _tools_text(chat: Chat) -> str:
    lines = [f"role: {chat.role.name} -- {chat.role.summary}"]
    shown = {s["function"]["name"] for s, _ in chat.offered} & chat.names()
    for name in sorted(shown):
        mark = ("asks first" if chat.role.asks == "each" else "asks unless the plan names it") \
            if name in policy.CONFIRM else "asks you itself, every time" if name in chat.extension.asks_itself \
            else "reads"
        lines.append(f"  {name:<16} {mark}")
    return "tools:\n" + "\n".join(lines)


def _roles_text(chat: Chat) -> str:
    return "\n".join(f"{'*' if r is chat.role else ' '} {r.name:<9} {r.summary}"
                      for r in roles.ROLES.values())


def _slash(chat: Chat, word: str, rest: str, stdout: TextIO,
           connect: Callable[[str], Any] | None) -> None:
    """Do the slash command ``word`` the person typed."""
    if word == "/help":
        stdout.write(HELP + "\n")
    elif word == "/new":
        chat.new()
        stdout.write(f"new chat {chat.session.id}\n")
    elif word == "/tools":
        stdout.write(_tools_text(chat) + "\n")
    elif word == "/role":
        try:
            if rest.strip():
                chat.use_role(rest.strip())
                stdout.write(f"role: {chat.role.name}\n")
            else:
                stdout.write(_roles_text(chat) + "\n")
        except ValueError as exc:
            stdout.write(f"{exc}\n")
    elif word == "/rules":
        stdout.write(saved.run_command(chat.rules, rest.split()) + "\n")
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
    elif word[1:] in chat.extension.commands:
        stdout.write(chat.extension.commands[word[1:]](rest, chat.messages) + "\n")
    else:
        stdout.write(f"{word} is not a command; /help lists them\n")


def repl(chat: Chat, stdin: TextIO, stdout: TextIO, *,
         connect: Callable[[str], Any] | None = None) -> int:
    """Read messages and slash commands from ``stdin`` until /quit or EOF."""
    stdout.write(f"chat {chat.session.id} ({chat.role.name}) -- /help for commands, "
                 f"/quit to leave\n")
    while True:
        stdout.write("\nyou> ")
        stdout.flush()
        line = stdin.readline()
        if line == "":
            break
        text = line.strip()
        if not text:
            continue
        word, _, rest = text.partition(" ")
        if not text.startswith("/"):
            chat.turn(text)
        elif word in ("/quit", "/exit"):
            break
        else:
            _slash(chat, word, rest, stdout, connect)
        stdout.flush()
    return 0


OPTIONS = (
    flag("words", nargs="*", metavar="TASK",
         help="a task in words: the agent plans, asks go, acts and ends on done (default: a "
              "conversation); `rules ...` lists and edits the saved always/never rules"),
    flag("--task", default="", metavar="TEXT", help="the task, as an option instead of words"),
    flag("--role", default="", choices=["", *roles.ROLES],
         help="reader (reads only), operator (acting calls ask; the conversation default) or "
              "runner (the approved plan runs unasked; the task default)"),
    flag("--model", default="", help="a model to lease (default: the best downloaded "
         "mixture-of-experts model ranked for an agent)"),
    flag("--url", default="", help="a server already up, e.g. http://127.0.0.1:8080"),
    flag("--resume", nargs="?", const="last", default="", metavar="ID",
         help="continue a saved chat: the newest, or the id given"),
    option("port", default=8080, help="where --model is served"),
    flag("--draft", default="auto", metavar="HEAD",
         help="the draft head: 'auto', 'none' or a named head (default: %(default)s)"),
    flag("--rounds", type=int, default=0,
         help=f"tool-calling rounds one message may spend (default: {ROUNDS}, a task {TASK_ROUNDS})"),
    flag("--project", default="", metavar="PATH",
         help="the project whose memory is used with yours (default: the git repository or "
              "directory the chat was started in)"),
    flag("--n-predict", type=int, default=do.N_PREDICT),
    option("timeout", default=900.0, help="seconds to wait for one reply (default: %(default)s)"),
    flag("--context-size", type=int, default=0, metavar="TOKENS",
         help="the context the history is compacted against (default: ask the server)"),
    flag("--no-compact", action="store_true", help="never summarise the history"),
    option("dry-run", help="print the system prompt and the tools as the model sees them"),
)


def extensions(person: do.Person, mem: Memory | None = None) -> roles.Extension:
    """The tools, context and commands features add to every session.

    EXTENSION POINT: wire a feature here with its `roles.Extension` (``tools``, ``context``,
    ``start``, ``commands``, ``reads``, ``asks``); its tools are offered, held to the roles and
    shown to the model with its context in the system message, ``start`` text goes in front of
    the first message of a session, and ``commands`` are slash commands for the person. A tool
    that asks the person itself is named in ``asks_itself`` so the role does not ask twice.

    Memory: the person's memory and the open project's (``mem``), searched together."""
    mem = mem or Memory.open()
    merged = mem.merged()

    def shown(rest: str, messages: Sequence[dict[str, Any]]) -> str:
        asked = rest.strip() or next((str(m["content"]) for m in reversed(messages)
                                      if m.get("role") == "user"), "")
        return memory.describe(merged, asked[:300])

    return roles.Extension(
        tools=lambda: memory.tools(confirm=person.choose, store=mem.user, project=mem.project),
        context=lambda: memory.guidance(mem.project.project.name if mem.project and mem.project.project else ""),
        start=lambda task: memory.session_context(task, store=merged),
        commands={"memory": shown}, reads=memory.READ, asks_itself=memory.ACTING)


def _task_of(args: argparse.Namespace) -> str:
    return args.task or " ".join(args.words)


def serve(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    """Run the session the parsed ``args`` describe over ``stdin`` and ``stdout``."""
    if args.words and args.words[0] == "rules" and (
            len(args.words) == 1 or args.words[1] in ("list", "remove", "flip", "tainted", "clear")):
        stdout.write(saved.run_command(saved.Rules(), args.words[1:]) + "\n")
        return 0
    if args.model and args.url:
        stdout.write("--model and --url name two servers: give one\n")
        return 2
    task = _task_of(args)
    role = args.role or (roles.TASK_DEFAULT if task else roles.DEFAULT)
    person = do.Person(stdin, stdout)
    try:
        mem = Memory.open(explicit=Path(args.project) if args.project else None)
    except ValueError as exc:
        stdout.write(f"{exc}\n")
        return 2
    if args.dry_run:
        chat = Chat(None, person, role=role, task=bool(task), extension=extensions(person, mem))
        stdout.write(chat.system() + "\n\n")
        do._print_offer([(s, fn) for s, fn in chat.offered if s["function"]["name"] in chat.names()],
                        stdout)
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
    chat = Chat(client, person, session=session, screen=screen, role=role, task=bool(task),
                extension=extensions(person, mem))
    chat.rounds = args.rounds or chat.rounds
    if session.messages:
        stdout.write(f"resumed {len(session.messages)} messages\n")
    try:
        if task:
            return 0 if chat.turn(task).done else 1
        return repl(chat, stdin, stdout, connect=connect)
    finally:
        _close(screen)


COMMAND = Group(
    "ml-stack-chat",
    "An agent that operates ml-stack, in a conversation or on a task in words, under a role: "
    "what is serving, models, downloads, benchmarks, jobs, read-only security views. Anything "
    "that starts, stops, downloads or measures asks you first (Allow this time / Always allow / "
    "Never allow; `rules` edits those), except calls the plan you approved names in the "
    "runner role; releasing quarantine, approving a host, changing the security policy, the "
    "role and the rules are yours alone.",
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
