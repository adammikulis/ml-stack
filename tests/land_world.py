"""A real node with a board, a landing runner, requesters and a reviewer, over a real git repository and a bare origin."""

from __future__ import annotations

import sys
from pathlib import Path

import keyring
from land_support import DEV, Project, git
from onboard_support import FileKeyring

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import land_board

from poolhouse.board import session
from poolhouse.board.session import Native, Session
from poolhouse.workspace import landing

SLOW_TEST = "import time\ntime.sleep(60)\n"
MODELS = (("alice", "claude-sonnet-5-5"), ("bob", "claude-sonnet-5-5"), ("carol", "claude-opus-5-5"),
          ("haiku", "claude-haiku-5-5"))


class World:
    """One node, a runner session, four sessions of known models and a repository with a bare origin."""

    def __init__(self, monkeypatch, tmp_path: Path, node) -> None:
        monkeypatch.setenv("POOLHOUSE_DEV_BRANCH", DEV)
        self.node = node
        (tmp_path / "git").mkdir()
        self.proj = Project(tmp_path / "git", node.state.parent)
        self.proj.env["POOLHOUSE_BOARD"] = node.board
        # this process and the runner's commands share the one keystore the project provisioned
        monkeypatch.setenv("POOLHOUSE_TEST_KEYRING", self.proj.env["POOLHOUSE_TEST_KEYRING"])
        keyring.set_keyring(FileKeyring())
        self.origin = tmp_path / "origin.git"
        git(tmp_path, "init", "-q", "--bare", str(self.origin))
        git(self.proj.root, "remote", "add", "origin", str(self.origin))
        git(self.proj.root, "push", "-q", "origin", DEV)
        self.members = {name: node.member(name, model=model) for name, model in MODELS}
        self.who = {name: node.session(m) for name, m in self.members.items()}
        self.lead = self.runner_session("lander")
        self.runner = self.make_runner()

    def runner_session(self, native_id: str) -> Session:
        """A session that is not an agent of any model, as the landing runner is."""
        made = session.register(self.node.client, self.node.board, Native("", "land-runner", native_id))
        return Session(self.node.client, self.node.board, made.token, made.name)

    def make_runner(self, **options) -> land_board.Runner:
        return land_board.Runner(self.lead, self.proj.root, env=self.proj.env, poll_s=0.1, **options)

    def branch(self, name: str, files: dict[str, str]) -> str:
        self.proj.branch(name, files)
        return git(self.proj.root, "rev-parse", name)

    def ask(self, who: str, name: str, sha: str) -> str:
        return landing.request(self.who[who], {"branch": name, "sha": sha, "selectors": ["tests/test_mod.py"]})["id"]

    def ready(self, name: str, files: dict[str, str], who: str = "alice") -> tuple[str, str]:
        """A branch requested by ``who`` and accepted by carol; its request id and sha."""
        sha = self.branch(name, files)
        rid = self.ask(who, name, sha)
        landing.review(self.who["carol"], rid, sha, "accept")
        return rid, sha

    def requests(self) -> dict[str, dict]:
        return landing.fold(self.lead).requests

    def status(self, rid: str) -> str:
        return self.requests()[rid]["status"]

    def detail(self, rid: str) -> str:
        return self.requests()[rid]["detail"]

    def texts(self, who: str) -> list[str]:
        """The direct messages ``who`` holds, oldest first."""
        return [e.text for e in self.who[who].read(inbox=True).entries]

    def announced(self, kind: str) -> list[str]:
        """The text of every announcement of ``kind``."""
        return [e.text for e in self.lead.read(channel="#announcements").entries if e.fields.get("type") == kind]

    def origin_head(self) -> str:
        return git(self.origin, "rev-parse", DEV)

    def local_head(self) -> str:
        return git(self.proj.root, "rev-parse", DEV)


def mod(n: int, extra: str = "") -> dict[str, str]:
    return {f"src/poolhouse/m{n}.py": f"VALUE = {n}\n", f"tests/test_m{n}.py": f"import poolhouse.m{n}\n{extra}"}
