"""release-main: the facts about a push of main that the hook computes from git, the approval question rendered from them, and the check of a push against an approval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ml_stack.sentinel.events import EventLog
from ml_stack.workspace import person_ancestry, person_auth, person_store, person_targets

__all__ = ["KIND", "Facts", "approved", "check_push", "echo_for", "facts", "options", "question", "run", "target"]

KIND = "release-main"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
ECHO = re.compile(
    r"authorize release-main (?P<target>\S+) (?P<minutes>\d+)m x(?P<uses>\d+) #(?P<tag>[0-9a-f]{12})")
TARGET = re.compile(r"(?P<project>\S+)@(?P<remote>[^:\s]+):(?P<sha>[0-9a-f]{40}):(?P<tree>[0-9a-f]{40})")
ZERO = "0" * 40


class Refused(RuntimeError):
    """A push that a release-main approval cannot cover, with the reason."""


@dataclass(frozen=True, slots=True)
class Facts:
    """What git says about releasing ``sha``: its tree, subject, commits ahead of the remote's main and diffstat."""

    project: str
    remote: str
    sha: str
    tree: str
    subject: str
    ahead: int
    stat: str


def facts(checkout: person_targets.Checkout, sha: str = "", remote: str = "") -> Facts | None:
    """The `Facts` for ``sha`` (default the local ``main``) against the remote's main, or None when git cannot say."""
    remote = remote or checkout.remote
    where = checkout.project
    full = person_targets.run_git(where, "rev-parse", "--verify", f"{sha or 'main'}^{{commit}}")
    base = f"refs/remotes/{remote}/main"
    if not FULL_SHA.fullmatch(full) or any(c.isspace() for c in where) or not remote or ":" in remote:
        return None
    tree = person_targets.run_git(where, "rev-parse", f"{full}^{{tree}}")
    subject = person_targets.run_git(where, "log", "-1", "--format=%s", full)
    ahead = person_targets.run_git(where, "rev-list", "--count", f"{base}..{full}")
    stat = person_targets.run_git(where, "diff", "--shortstat", base, full) or "no file changes"
    if not FULL_SHA.fullmatch(tree) or not ahead.isdigit():
        return None
    return Facts(where, remote, full, tree, " ".join(subject.split())[:200], int(ahead), stat)


def target(found: Facts) -> str:
    """The target an approval is bound to: repository, remote, commit and tree."""
    return f"{found.project}@{found.remote}:{found.sha}:{found.tree}"


def question(found: Facts) -> str:
    """The approval question's text, made only of what git reported."""
    return (f"Release main to {found.sha}?\nSubject: {found.subject}\n"
            f"Commits ahead of {found.remote}/main: {found.ahead}\nChanges: {found.stat}\nRemote: {found.remote}")


def options(found: Facts) -> tuple[str, str]:
    """The two answers: the approval is the first."""
    return f"Approve release of {found.sha[:7]}", "Do not release"


def echo_for(log: EventLog, found: Facts, minutes: int = person_store.DEFAULT_MINUTES, uses: int = 1) -> str:
    """The question's final line: kind, target, life and uses, with a tag made under the store's key."""
    tag = log.mac(f"{KIND}|{target(found)}|{minutes}|{uses}".encode())[:12]
    return f"authorize {KIND} {target(found)} {minutes}m x{uses} #{tag}"


def approved(log: EventLog, text: str, label: str, checkout: person_targets.Checkout) -> Facts | None:
    """The `Facts` an answered question approves: its final line is a valid echo, the rest is exactly the
    question git now renders for that commit, and ``label`` is the approving answer."""
    *body, last = text.rstrip().splitlines() or [""]
    echoed, parts = ECHO.fullmatch(last.strip()), None
    if echoed is None or int(echoed["minutes"]) != person_store.DEFAULT_MINUTES or int(echoed["uses"]) != 1:
        return None
    parts = TARGET.fullmatch(echoed["target"])
    if parts is None or parts["project"] != checkout.project:
        return None
    found = facts(checkout, parts["sha"], parts["remote"])
    if found is None or target(found) != echoed["target"] or "\n".join(body).strip() != question(found):
        return None
    if label != options(found)[0] or echo_for(log, found) != last.strip():
        return None
    return found


def check_push(checkout: person_targets.Checkout, remote: str, lines: list[str]) -> str:
    """The target of the single fast-forward of ``refs/heads/main`` that ``lines`` (a pre-push input) describe;
    raises `Refused` for any other push."""
    fields = [line.split() for line in lines if line.strip()]
    if len(fields) != 1 or len(fields[0]) != 4:
        raise Refused("a release is one push of main and nothing else")
    _, local_sha, remote_ref, remote_sha = fields[0]
    if remote_ref != "refs/heads/main":
        raise Refused(f"{remote_ref} is not refs/heads/main")
    if not FULL_SHA.fullmatch(local_sha) or not FULL_SHA.fullmatch(remote_sha) or ZERO in (local_sha, remote_sha):
        raise Refused("a deletion, a new main or an unresolved commit")
    if not person_targets.git_ok(checkout.project, "merge-base", "--is-ancestor", remote_sha, local_sha):
        raise Refused("not a fast-forward from the remote's main")
    found = facts(checkout, local_sha, remote)
    if found is None:
        raise Refused("git cannot show the facts of that commit against the remote's main")
    return target(found)


def _ancestry_bound(log_rows: list[dict]) -> set[tuple[int, float]]:
    return set(person_store.bound_sessions(log_rows))


def run(argv: list[str], stdin: str, cwd: str) -> tuple[int, str]:
    """The `person-consume` commands: propose, peek, consume REMOTE (pre-push input on stdin) and harness."""
    verb = argv[0] if argv else ""
    checkout = person_targets.inspect(cwd)
    try:
        log = person_store.open_log()
        if verb == "harness":
            rows = person_store.records(log) if log.path.exists() else []
            return (0 if person_ancestry.under_harness(_ancestry_bound(rows)) else 1), ""
        if verb == "propose":
            found = facts(checkout, argv[1] if len(argv) > 1 else "")
            if found is None:
                return 1, "git cannot show the facts of that commit against the remote's main"
            return 0, f"{question(found)}\n{echo_for(log, found)}\n\noptions: {options(found)[0]} | {options(found)[1]}"
        identity = person_auth.Identity(person_auth.session_of_ancestry(log))
        if verb == "peek":
            held = [a for a in person_auth.live(KIND, None, identity, log=log) if a.project == checkout.project]
            return (0 if held else 1), ""
        if verb == "consume" and len(argv) == 2:
            return 0, person_auth.consume(KIND, check_push(checkout, argv[1], stdin.splitlines()), identity, log=log)
    except (Refused, person_auth.NotAuthorized, person_store.Unreadable, OSError, ValueError) as error:
        return 1, str(error)
    return 2, "usage: person-consume propose [SHA] | peek | consume REMOTE | harness"
