"""The board-fed landing runner over real git repositories, a bare origin and the real board."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from land_support import Project, git
from workspace_kit import Kit, clean_env, cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import land_board

from ml_stack.workspace import landing
from ml_stack.workspace.identity import Denied

PERSON = {"terminal": (True, True), "env": {}}
SLOW_TEST = "import time\ntime.sleep(60)\n"


class World:
    """A board with a lead runner, a requester, a reviewer and a repository with a bare origin."""

    def __init__(self, monkeypatch, tmp_path: Path) -> None:
        self.kit = Kit(clean_env(monkeypatch, tmp_path))
        self.ws = self.kit.ws
        (tmp_path / "git").mkdir()
        self.proj = Project(tmp_path / "git")
        self.origin = tmp_path / "origin.git"
        git(tmp_path, "init", "-q", "--bare", str(self.origin))
        git(self.proj.root, "remote", "add", "origin", str(self.origin))
        git(self.proj.root, "push", "-q", "origin", "0.2dev")
        self.tokens = {name: self.kit.agent(name) for name in ("alice", "bob", "carol", "haiku")}
        for name, model in (("alice", "claude-sonnet-5-5"), ("bob", "claude-sonnet-5-5"),
                            ("carol", "claude-opus-5-5"), ("haiku", "claude-haiku-5-5")):
            self.ws.set_model(name, model, **PERSON)
        self.lead = self.kit.agent("lander", "lead")
        self.runner = self.make_runner()

    def make_runner(self, **options) -> land_board.Runner:
        return land_board.Runner(self.ws, self.lead, self.proj.root, env=self.proj.env, poll_s=0.1, **options)

    def branch(self, name: str, files: dict[str, str]) -> str:
        self.proj.branch(name, files)
        return git(self.proj.root, "rev-parse", name)

    def ask(self, who: str, name: str, sha: str) -> str:
        return landing.request(self.ws, self.tokens[who], {"branch": name, "sha": sha,
                                                           "selectors": ["tests/test_mod.py"]})["id"]

    def ready(self, name: str, files: dict[str, str], who: str = "alice") -> tuple[str, str]:
        """A branch requested by ``who`` and accepted by carol; its request id and sha."""
        sha = self.branch(name, files)
        rid = self.ask(who, name, sha)
        landing.review(self.ws, self.tokens["carol"], rid, sha, "accept")
        return rid, sha

    def status(self, rid: str) -> str:
        return landing.fold(self.ws).requests[rid]["status"]

    def origin_head(self) -> str:
        return git(self.origin, "rev-parse", "0.2dev")

    def local_head(self) -> str:
        return git(self.proj.root, "rev-parse", "0.2dev")


@pytest.fixture
def world(monkeypatch, tmp_path):
    return World(monkeypatch, tmp_path)


def mod(n: int, extra: str = "") -> dict[str, str]:
    return {f"src/ml_stack/m{n}.py": f"VALUE = {n}\n", f"tests/test_m{n}.py": f"import ml_stack.m{n}\n{extra}"}


def test_two_requests_land_in_queue_order_as_one_batch_and_push_only_origin_development(world):
    first, _ = world.ready("a", mod(1))
    second, _ = world.ready("b", mod(2), who="bob")
    before = world.origin_head()
    result = world.runner.once()
    assert result["status"] == "landed" and sorted(result["merged"]) == ["a", "b"]
    assert world.status(first) == world.status(second) == "landed"
    assert world.origin_head() == world.local_head() != before
    assert git(world.origin, "branch", "--list").split() == ["0.2dev"]
    assert [c for c in world.proj.calls() if c.startswith("gate")] == ["gate"]
    assert not (world.proj.base / "a").exists() and not (world.proj.base / "b").exists()
    inbox = world.ws.inbox(world.tokens["alice"])
    assert any("landed" in m["text"] for m in inbox)


def test_conflicting_pair_reports_needs_human_and_lands_the_other(world):
    one, _ = world.ready("x", {"src/ml_stack/mod.py": "VALUE = 2\n"})
    two, _ = world.ready("y", {"src/ml_stack/mod.py": "VALUE = 3\n"}, who="bob")
    result = world.runner.once()
    states = {world.status(one), world.status(two)}
    assert states == {"landed", "needs-human"}, result
    stuck = next(r for r in landing.fold(world.ws).requests.values() if r["status"] == "needs-human")
    assert "merge conflict" in stuck["detail"]
    assert world.origin_head() == world.local_head()
    mine = world.ws.inbox(world.tokens[stuck["by"]])
    assert any("needs-human" in m["text"] for m in mine)


def test_red_gate_does_not_push_and_names_the_failing_test(world):
    rid, _ = world.ready("bad", mod(3, "# FAILME\n"))
    before_origin, before_local = world.origin_head(), world.local_head()
    result = world.runner.once()
    assert result["status"] != "landed"
    assert world.status(rid) == "failed"
    detail = landing.fold(world.ws).requests[rid]["detail"]
    assert "tests/test_m3.py" in detail and "clean 0.2dev" in detail
    assert world.origin_head() == before_origin and world.local_head() == before_local


def test_moved_branch_tip_is_refused_not_landed(world):
    rid, _ = world.ready("moving", mod(4))
    Project.write(world.proj.base / "moving", "src/ml_stack/extra.py", "X = 1\n")
    Project.commit(world.proj.base / "moving", "feat: more")
    before = world.origin_head()
    world.runner.once()
    assert world.status(rid) == "refused"
    assert "request it again" in landing.fold(world.ws).requests[rid]["detail"]
    assert world.origin_head() == before and world.local_head() == before


def test_unreviewed_request_is_marked_needs_review_and_never_lands(world):
    sha = world.branch("raw", mod(5))
    rid = world.ask("alice", "raw", sha)
    before = world.origin_head()
    assert world.runner.once()["status"] == "idle"
    req = landing.fold(world.ws).requests[rid]
    assert req["status"] == "needs-review" and "review" in req["detail"]
    assert world.origin_head() == before
    with pytest.raises(Denied):
        landing.review(world.ws, world.tokens["alice"], rid, sha, "accept")
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "needs-review"


def test_review_must_name_the_exact_sha_and_a_rejection_blocks(world):
    sha = world.branch("rev", mod(6))
    rid = world.ask("alice", "rev", sha)
    with pytest.raises(ValueError, match="exact commit"):
        landing.review(world.ws, world.tokens["carol"], rid, "0" * 40, "accept")
    landing.review(world.ws, world.tokens["carol"], rid, sha, "accept")
    landing.review(world.ws, world.tokens["bob"], rid, sha, "reject")
    world.runner.once()
    assert world.status(rid) == "needs-review"


def test_only_landing_level_identities_may_request(world):
    sha = world.branch("low", mod(7))
    with pytest.raises(Denied):
        world.ask("haiku", "low", sha)
    child = world.ws.delegate(world.tokens["alice"], "helper")
    with pytest.raises(Denied):
        landing.request(world.ws, child, {"branch": "low", "sha": sha, "selectors": ["t"]})
    for bad in ("main", "master"):
        with pytest.raises(ValueError):
            landing.request(world.ws, world.tokens["alice"], {"branch": bad, "sha": sha, "selectors": ["t"]})
    with pytest.raises(ValueError):
        landing.request(world.ws, world.tokens["alice"], {"branch": "low", "sha": sha, "selectors": ["t"],
                                                          "target": "main"})
    with pytest.raises(ValueError):
        landing.request(world.ws, world.tokens["alice"], {"branch": "low", "sha": "abc", "selectors": ["t"]})


def test_pause_and_resume_gate_the_runner_and_only_controllers_may(world):
    rid, _ = world.ready("p", mod(8))
    with pytest.raises(Denied):
        landing.brake(world.ws, world.tokens["bob"], True)
    landing.brake(world.ws, world.lead, True, "checking something")
    assert world.runner.once()["status"] == "paused"
    assert world.status(rid) == "queued"
    assert "PAUSED" in "\n".join(landing.status_lines(world.ws))
    landing.brake(world.ws, world.kit.owner, False)
    assert world.runner.once()["status"] == "landed"
    assert world.status(rid) == "landed"


def test_cancel_by_requester_or_controller_only(world):
    rid, _ = world.ready("c", mod(9))
    with pytest.raises(Denied):
        landing.cancel(world.ws, world.tokens["bob"], rid)
    landing.cancel(world.ws, world.tokens["alice"], rid)
    before = world.origin_head()
    assert world.runner.once()["status"] == "idle"
    assert world.status(rid) == "cancelled" and world.origin_head() == before


def test_new_request_for_a_branch_supersedes_the_old_one(world):
    old, _ = world.ready("s", mod(10))
    Project.write(world.proj.base / "s", "src/ml_stack/more.py", "Y = 1\n")
    new_sha = Project.commit(world.proj.base / "s", "feat: more")
    new = world.ask("alice", "s", new_sha)
    assert world.status(old) == "superseded"
    assert world.status(new) == "needs-review"


def test_runner_never_pushes_main_or_a_missing_remote(world):
    with pytest.raises(ValueError, match="never touches main"):
        world.make_runner(target="main").push()
    git(world.proj.root, "checkout", "-q", "-b", "side")
    assert "not 0.2dev" in world.runner.push()
    git(world.proj.root, "checkout", "-q", "0.2dev")
    code, out, _ = world.proj.land("serve", "--once", "--target", "main")
    assert code == 4 and "never touches main" in out


def test_second_runner_is_refused_while_the_claim_is_held(world):
    world.runner.once()
    other = world.kit.agent("other-lander", "lead")
    rival = land_board.Runner(world.ws, other, world.proj.root, env=world.proj.env)
    assert rival.once()["status"] == "runner-held"


def test_stuck_gate_is_aborted_reported_and_not_pushed(world):
    Project.write(world.proj.root, "scripts/test", SLOW_TEST)
    Project.commit(world.proj.root, "chore: slow gate")
    git(world.proj.root, "push", "-q", "origin", "0.2dev")
    rid, _ = world.ready("slow", mod(11))
    before = world.origin_head()
    runner = world.make_runner(stall_s=1.0)
    result = runner.once()
    assert result["status"] == "stuck"
    req = landing.fold(world.ws).requests[rid]
    assert req["status"] == "needs-human" and "no progress" in req["detail"]
    assert world.origin_head() == before
    announced = [r for r in world.ws.bus.log.rows() if r.get("to") == "#announcements" and r.get("type") == "blocked"]
    assert any("stuck" in r["body"] for r in announced)


def test_cli_request_and_queue(world):
    sha = world.branch("cli", mod(12))
    token = world.tokens["alice"]
    done = cli(world.kit.base, token, "land-request", "cli", sha, "--test", "tests/test_m12.py",
               "--replaces", "nothing", "--json")
    assert done.returncode == 0, done.stderr
    shown = cli(world.kit.base, token, "land-queue", "--json")
    assert "cli@" in shown.stdout and "needs-review" in shown.stdout
    refused = cli(world.kit.base, world.tokens["haiku"], "land-request", "cli", sha, "--test", "t")
    assert refused.returncode != 0
