"""What `ml-stack-claude` and `ml-stack-codex` share: the served model, the session's files and the workspace join.

A coding session is served at ``DEFAULT_CTX`` tokens with a q8_0 KV cache on one slot, so the
harness's long prompt prefix is reused by every turn. The session's settings live in a directory
under the state root, outside the working tree the model can edit, and are read-only once written.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import shlex
import shutil
import stat
import sys
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from ml_stack import harnessid, home, hub
from ml_stack.chatpolicy import APPROVE_FIRST, ROLE_NAMES
from ml_stack.harnesshook import WAIT_S
from ml_stack.serve import chat_template, leases, profile, wired
from ml_stack.serve.recent import note
from ml_stack.serve.serving import Config, Serving, drafted, served, serving_params

__all__ = [
    "DEFAULT_CTX",
    "DEFAULT_MODEL",
    "DEFAULT_ROLE",
    "PARENT",
    "WAIT_S",
    "Session", "SessionFiles",
    "Want",
    "admitted",
    "check_role",
    "config_for",
    "hook_command",
    "label_for",
    "parser",
    "protected_paths",
    "serving",
    "session_files",
    "window_of",
]

DEFAULT_CTX = 262144
DEFAULT_MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
DEFAULT_ROLE = APPROVE_FIRST
PARENT = "claude-code"
KV = "q8_0"


def parser(name: str, what: str, port: int, slots: int) -> argparse.ArgumentParser:
    """The command line ``ml-stack-<name>`` takes: the model, how it is served, the role and
    the workspace identity; everything after ``--`` is the harness's."""
    ap = argparse.ArgumentParser(
        prog=f"ml-stack-{name}", allow_abbrev=False,
        description=f"{what} on a model this machine serves. Everything after `--` goes to {name}.",
        usage=f"ml-stack-{name} {{MODEL | --on URL}} [--port N] [--slots N] [--ctx N] [--role R] [--name L] "
              f"[--as AGENT] [--project DIR] [--orders-from AGENT] [--no-profile] [--draft HEAD] [--{name} PATH] [-- {name} arguments]")
    ap.add_argument("model", nargs="?", default="",
                    help=f"the model file or a name ml-stack-models finds (default: {DEFAULT_MODEL})")
    ap.add_argument("--on", metavar="URL", default="",
                    help="a server already running, e.g. http://127.0.0.1:8080; it is left running "
                         "and no model is named")
    ap.add_argument("--port", type=int, default=port)
    ap.add_argument("--slots", type=int, default=slots,
                    help="conversations the server holds at once (default: %(default)s)")
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX,
                    help="tokens served in all, split across the slots (default: %(default)s)")
    ap.add_argument("--role", default=DEFAULT_ROLE,
                    help="read-only, approve-first or plan-and-go: what the hooks let a call do "
                         "(default: %(default)s)")
    ap.add_argument("--name", default="", help="the workspace label (default: local-<model>)")
    ap.add_argument("--as", dest="parent", default=PARENT,
                    help="the joined workspace agent this session acts for (default: %(default)s)")
    ap.add_argument("--project", default="", metavar="DIR",
                    help="the project directory the session works in (default: the current directory)")
    ap.add_argument("--orders-from", action="append", default=[], metavar="AGENT",
                    help="a workspace identity the session obeys besides the person and the lead")
    ap.add_argument("--for", dest="lease_for", default="", metavar="TEXT",
                    help="why the model is leased, one line, shown by `ml-stack-serve status|leases|history`")
    ap.add_argument("--no-profile", action="store_true", help="serve the model bare")
    ap.add_argument("--draft", default="auto", metavar="HEAD",
                    help="the draft head that guesses tokens ahead for the model to check: 'auto' takes "
                         "the smallest one on this machine, 'none' serves without one, or name one "
                         "(default: %(default)s)")
    ap.add_argument(f"--{name}", default="", metavar="PATH",
                    help=f"the {name} binary (default: the one on PATH)")
    return ap


def label_for(name: str, alias: str) -> str:
    """The workspace label: the person's ``--name``, else ``local-<model>``."""
    return name or f"local-{alias}"


def window_of(base_url: str) -> int:
    """The tokens one conversation gets on the server at ``base_url``; 0 when it will not say."""
    params = serving_params(base_url, timeout=5.0)
    if params is None or not params.n_ctx:
        return 0
    return int(params.n_ctx) // max(1, int(params.total_slots or 1))


def admitted(found: str, ctx: int, say: Callable[[str], None], *, plan: Callable[..., object] | None = None) -> bool:
    """Whether the wired-memory limit now holds ``found`` at ``ctx``; when it does not, the
    command a person runs to raise it is printed and nothing is raised here."""
    try:
        got = (plan or wired.plan)(found, wired.Ask(context=ctx, kv=KV))
    except (OSError, ValueError):
        return True
    if getattr(got, "enough_now", True):
        return True
    say(f"error: {Path(found).name} at {ctx:,} tokens needs a wiring limit of {got.needed_mb:,} MB "
        f"and this machine's is {got.current_mb or got.default_mb:,} MB.\n"
        f"  Raising it is for a person at their own terminal: "
        f"ml-stack-serve memory --for {Path(found).name} --ctx {ctx} --kv {KV} --apply")
    return False


@dataclasses.dataclass(frozen=True, slots=True)
class Want:
    """How a session wants its model served."""

    port: int = 8080
    slots: int = 1
    ctx: int = DEFAULT_CTX
    no_profile: bool = False
    draft: str = "auto"


def config_for(found: str, want: Want, say: Callable[[str], None]):
    """The `Config` a session serves: the model's measured settings (or bare), ``ctx`` tokens
    over ``slots`` slots, q8_0 KV cache."""
    port, slots, draft = want.port, want.slots, want.draft
    whole = chat_template.trained_context(found)
    each = (min(want.ctx, whole) if whole else want.ctx) // max(1, slots)
    measured = None if want.no_profile else profile.profile_for(found)
    if measured is not None:
        config = measured.config(port=port, slots=slots, model=found)
        say(f"serving in the settings it scored best with: {profile.said(measured)}")
        config = drafted(config, "none", say=say)
    else:
        config = Config(serving=Serving(model=found, port=port, slots=slots, slot_context=each))
        say(f"serving bare: nothing measured for this model, {each:,} tokens a slot")
        config = drafted(config, draft, say=say)
    say(f"  {each:,} tokens a slot, {KV} KV cache")
    return dataclasses.replace(config, serving=dataclasses.replace(config.serving, slot_context=each,
                                                                   cache_type=KV))


@contextlib.contextmanager
def serving(model: str, want: Want, say: Callable[[str], None], by: str) -> Iterator[tuple[str, object, str]]:
    """``(base_url, config, found)`` for the model served for the block; the server goes when
    the block ends unless one was already up."""
    found = str(hub.located(model, loose=True) or model)
    note(found, by=by)
    leasing = say if leases.already_up(found, want.port) is None else (lambda _line: None)
    config = config_for(found, want, leasing)
    if not admitted(found, config.serving.context, say):
        raise SystemExit(2)
    patched = chat_template.written_beside(found)
    if patched is not None:
        leasing("this model's template refuses a system message after the first; serving "
                f"with one that renders it instead ({patched.name})")
    with served(config, say=say, timeout=900.0, cache_reuse=256, warmup=False, escalate=True,
                chat_template_file=patched, reason=f"{by} session on {Path(found).name}",
                on_event=lambda e: say(f"  {e.get('event')}: " + ", ".join(
                    f"{k}={v}" for k, v in e.items() if k != "event"))) as base_url:
        yield base_url, config, found


@dataclasses.dataclass(slots=True)
class SessionFiles:
    """The directory a session's settings are written to, and its release."""

    path: Path

    def write(self, name: str, text: str) -> Path:
        """Write ``name`` read-only into the session directory."""
        target = self.path / name
        target.write_text(text, encoding="utf-8")
        target.chmod(stat.S_IRUSR)
        return target

    def lock(self) -> None:
        """Take write access away from the directory itself."""
        self.path.chmod(stat.S_IRUSR | stat.S_IXUSR)

    def release(self) -> None:
        self.path.chmod(stat.S_IRWXU)
        shutil.rmtree(self.path, ignore_errors=True)


def session_files(cwd: Path) -> SessionFiles:
    """A new private directory under the state root; refused when ``cwd`` is inside it."""
    base = home.state("harness")
    base.mkdir(parents=True, exist_ok=True)
    base.chmod(stat.S_IRWXU)
    path = base / uuid.uuid4().hex[:12]
    path.mkdir(mode=stat.S_IRWXU)
    if path in (cwd, *cwd.parents) or cwd in path.parents:
        shutil.rmtree(path, ignore_errors=True)
        raise ValueError("the working directory and the session's settings directory overlap")
    return SessionFiles(path)


def hook_command(event: str, *, role: str, label: str, root: Path, protect: list[str]) -> str:
    """The shell line a harness runs for a hook: absolute interpreter, role and paths fixed."""
    words = [sys.executable, "-m", "ml_stack.harnesshook", event, "--role", role, "--label", label,
             "--root", str(root), "--wait", str(int(WAIT_S))]
    for each in protect:
        words += ["--protect", each]
    return shlex.join(words)


def protected_paths(files: SessionFiles) -> list[str]:
    """Paths no tool call may name: the session directory and the state root."""
    return [str(files.path), str(home.home())]


def check_role(role: str) -> str:
    """``role`` when it is one of the three, else ``ValueError``."""
    if role not in ROLE_NAMES:
        raise ValueError(f"no role {role!r}: the roles are {', '.join(ROLE_NAMES)}")
    return role


@dataclasses.dataclass(frozen=True, slots=True)
class Session:
    """What a harness run needs from the shared setup: its files, workspace seat, working
    directory, the two hook command lines and the workspace brief."""

    files: SessionFiles
    seat: harnessid.Seat
    cwd: Path
    pre: str
    post: str
    brief: str


@contextlib.contextmanager
def opened(args: argparse.Namespace, harness: str, served: tuple[str, str, int],
           say: Callable[[str], None]) -> Iterator[Session]:
    """The session for one run of ``harness``: files outside the working tree, a workspace seat
    announced as joined, the hook commands. The seat is revoked and the files removed on exit."""
    _base_url, alias, _window = served
    cwd = Path(args.project or Path.cwd()).resolve()
    files = session_files(cwd)
    seat = None
    try:
        invite = getattr(args, "seat_factory", None) or harnessid.invite
        seat = invite(harnessid.agent_name(alias, harness, args.name), cwd, args.parent, say)
        if not seat.record_model(alias, harness):
            say(f"the model of {seat.name} ({alias}, {harness}) is not recorded: a person-started launcher records it")
        pre = hook_command("pre", role=args.role, label=seat.name, root=cwd, protect=protected_paths(files))
        post = hook_command("post", role=args.role, label=seat.name, root=cwd, protect=[])
        harnessid.announce(seat, f"{harness} on {alias} ({args.role}), project {cwd.name}", say)
        yield Session(files, seat, cwd, pre, post,
                      harnessid.brief(seat.name, alias, harness, args.parent, args.orders_from))
    finally:
        if seat is not None:
            seat.revoke()
        files.release()
