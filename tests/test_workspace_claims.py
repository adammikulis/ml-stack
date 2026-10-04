"""Scratch folders and the ownership registry, with real files and real processes."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import Conflict, Denied, Workspace


@pytest.fixture
def kit(monkeypatch, tmp_path):
    return Kit(clean_env(monkeypatch, tmp_path))


def test_a_scratch_folder_is_private_and_namespaced_by_agent(kit):
    a, b = kit.agent("alpha"), kit.agent("beta")
    path = kit.ws.scratch_new(a, "build")
    assert path == str(kit.base / "scratch" / "alpha" / "build")
    assert oct(Path(path).stat().st_mode & 0o777) == "0o700"
    assert oct((kit.base / "scratch" / "alpha").stat().st_mode & 0o777) == "0o700"
    assert kit.ws.scratch_new(b, "build") != path
    assert [f["name"] for f in kit.ws.scratch_ls(a)] == ["build"]


@pytest.mark.parametrize("relative", ["../beta/build/x", "/etc/passwd", "a/../../other",
                                      "..", "sub/../../../.."])
def test_paths_outside_the_folder_are_refused(kit, relative):
    a = kit.agent("alpha")
    kit.ws.scratch_new(a, "build")
    with pytest.raises(Denied, match="outside"):
        kit.ws.scratch_path(a, "build", relative)
    assert any(r["event"] == "scratch.refused" for r in kit.ws.audit_log.rows())


def test_a_symlink_inside_a_folder_cannot_lead_out(kit, tmp_path):
    a = kit.agent("alpha")
    folder = kit.ws.scratch_new(a, "build")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "key.txt").write_text("x")
    (Path(folder) / "door").symlink_to(secret)
    with pytest.raises(Denied):
        kit.ws.scratch_path(a, "build", "door/key.txt")
    assert kit.ws.scratch_path(a, "build", "plain.txt").endswith("build/plain.txt")
    kit.ws.scratch_rm(a, "build")
    assert (secret / "key.txt").exists()


def test_a_symlinked_folder_or_agent_directory_is_not_followed(kit, tmp_path):
    a = kit.agent("alpha")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep").write_text("x")
    kit.ws.scratch_new(a, "first")
    (kit.base / "scratch" / "alpha" / "linked").symlink_to(elsewhere)
    with pytest.raises(Denied):
        kit.ws.scratch_path(a, "linked", "keep")
    with pytest.raises(Denied):
        kit.ws.scratch_rm(a, "linked")
    assert (elsewhere / "keep").exists()
    with pytest.raises(Denied):
        kit.ws.scratch_new(a, "linked")
    b = kit.agent("beta")
    (kit.base / "scratch" / "beta").symlink_to(elsewhere)
    with pytest.raises(Denied):
        kit.ws.scratch_new(b, "x")
    assert not (elsewhere / "x").exists()


def test_one_agent_cannot_reach_anothers_folders_but_a_lead_can(kit):
    a, b = kit.agent("alpha"), kit.agent("beta")
    lead = kit.agent("lead-1", "lead")
    kit.ws.scratch_new(a, "mine")
    with pytest.raises(Denied):
        kit.ws.scratch_path(b, "mine", "", owner="alpha")
    with pytest.raises(Denied):
        kit.ws.scratch_ls(b, "alpha")
    with pytest.raises(Denied):
        kit.ws.scratch_rm(b, "mine", owner="alpha")
    assert kit.ws.scratch_path(b, "mine") != kit.ws.scratch_path(a, "mine")
    assert [f["name"] for f in kit.ws.scratch_ls(lead, "alpha")] == ["mine"]
    assert kit.ws.scratch_rm(lead, "mine", owner="alpha") is True


@pytest.mark.parametrize("name", ["../x", "a/b", ".hidden", "", "x" * 80, "UP", "a b"])
def test_folder_names_are_plain(kit, name):
    with pytest.raises(ValueError, match="usable folder name"):
        kit.ws.scratch_new(kit.agent("alpha"), name)


def test_folders_have_a_count_limit_a_size_limit_and_an_expiry(kit):
    now = [1000.0]
    kit.limits(scratch_folders=2, scratch_bytes=1000, scratch_ttl_s=100.0)
    ws = Workspace(kit.base, lambda: now[0])
    a = ws.mint(kit.owner, "alpha", "agent")
    first = ws.scratch_new(a, "one")
    ws.scratch_new(a, "two")
    with pytest.raises(ValueError, match="already has 2"):
        ws.scratch_new(a, "three")
    (Path(first) / "big.bin").write_bytes(b"x" * 2000)
    listing = {f["name"]: f for f in ws.scratch_ls(a)}
    assert listing["one"]["over_limit"] is True and listing["two"]["over_limit"] is False
    ws.scratch_rm(a, "two")
    with pytest.raises(ValueError, match="over the limit"):
        ws.scratch_new(a, "three")
    now[0] += 101
    lead = ws.mint(kit.owner, "lead-1", "lead")
    assert ws.gc(lead)["scratch"] == [first]
    assert not Path(first).exists()


def test_a_claim_conflicts_until_released(kit):
    a, b = kit.agent("alpha"), kit.agent("beta")
    kit.ws.claim(a, "port", "8081", note="api")
    with pytest.raises(Conflict) as caught:
        kit.ws.claim(b, "port", "8081")
    assert caught.value.owner["owner"] == "alpha"
    assert kit.ws.who_owns("port", "8081")["owner"] == "alpha"
    assert kit.ws.claim(a, "port", "8081")["owner"] == "alpha"
    with pytest.raises(Denied):
        kit.ws.release(b, "port", "8081")
    kit.ws.release(a, "port", "8081")
    assert kit.ws.who_owns("port", "8081") is None
    assert kit.ws.claim(b, "port", "8081")["owner"] == "beta"
    assert any(r["event"] == "claim.conflict" for r in kit.ws.audit_log.rows())


def test_path_claims_conflict_when_one_contains_the_other(kit, tmp_path):
    a, b = kit.agent("alpha"), kit.agent("beta")
    tree = tmp_path / "wt"
    (tree / "src").mkdir(parents=True)
    kit.ws.claim(a, "worktree", str(tree))
    with pytest.raises(Conflict):
        kit.ws.claim(b, "file", str(tree / "src" / "x.py"))
    with pytest.raises(Conflict):
        kit.ws.claim(b, "worktree", str(tmp_path))
    assert kit.ws.who_owns("file", str(tree / "src" / "y.py"))["owner"] == "alpha"
    link = tmp_path / "alias"
    link.symlink_to(tree)
    with pytest.raises(Conflict):
        kit.ws.claim(b, "worktree", str(link))
    kit.ws.claim(b, "file", str(tmp_path / "other.txt"))
    with pytest.raises(ValueError, match="absolute"):
        kit.ws.claim(b, "file", "relative.txt")


@pytest.mark.parametrize("kind,key", [("port", "0"), ("port", "70000"), ("port", "80x"),
                                      ("branch", "../x"), ("branch", "a b"), ("server", ""),
                                      ("planet", "mars")])
def test_bad_claim_keys_are_refused(kit, kind, key):
    with pytest.raises(ValueError):
        kit.ws.claim(kit.agent("alpha"), kind, key)


def test_claims_expire_and_a_heartbeat_keeps_them(kit):
    now = [1000.0]
    ws = Workspace(kit.base, lambda: now[0])
    a, b = ws.mint(kit.owner, "alpha", "agent"), ws.mint(kit.owner, "beta", "agent")
    ws.claim(a, "branch", "agent/x", ttl_s=100)
    ws.claim(a, "server", "api", ttl_s=100)
    now[0] += 80
    assert ws.heartbeat(a, 100) == 2
    now[0] += 80
    with pytest.raises(Conflict):
        ws.claim(b, "branch", "agent/x")
    now[0] += 30
    assert ws.who_owns("branch", "agent/x") is None
    assert ws.claim(b, "branch", "agent/x")["owner"] == "beta"
    assert sum(1 for r in ws.audit_log.rows() if r["event"] == "claim.expired") == 2


def test_a_claim_is_released_when_its_process_dies(kit):
    a, b = kit.agent("alpha"), kit.agent("beta")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        kit.ws.claim(a, "port", "9100", pid=child.pid)
        with pytest.raises(Conflict):
            kit.ws.claim(b, "port", "9100")
        child.kill()
        child.wait(timeout=30)
        deadline = time.monotonic() + 10
        while kit.ws.who_owns("port", "9100") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert kit.ws.who_owns("port", "9100") is None
        assert kit.ws.claim(b, "port", "9100")["owner"] == "beta"
        assert any(r["event"] == "claim.dead-pid" and r["who"] == "alpha"
                   for r in kit.ws.audit_log.rows())
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_a_lead_can_release_an_agents_claim_and_claims_are_listed(kit):
    a = kit.agent("alpha")
    lead = kit.agent("lead-1", "lead")
    kit.ws.claim(a, "server", "api")
    kit.ws.claim(a, "branch", "agent/y")
    assert [c["key"] for c in kit.ws.claims.listing(kind="branch")] == ["agent/y"]
    assert len(kit.ws.claims.listing(owner="alpha")) == 2
    kit.ws.release(lead, "server", "api")
    assert kit.ws.who_owns("server", "api") is None


def test_two_processes_racing_for_a_port_have_exactly_one_winner(kit):
    tokens = [kit.agent(f"racer{i}") for i in range(4)]
    code = ("import os,sys\nfrom ml_stack.workspace import Workspace, Conflict\n"
            "try:\n Workspace().claim(os.environ['ML_STACK_WORKSPACE_TOKEN'],'port','9200')\n"
            " print('won')\nexcept Conflict:\n print('lost')\n")
    from workspace_kit import SRC, STRIPPED

    procs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                              env={**{k: v for k, v in os.environ.items() if k not in STRIPPED},
                                   "ML_STACK_WORKSPACE_HOME": str(kit.base), "PYTHONPATH": SRC,
                                   "ML_STACK_WORKSPACE_TOKEN": t}) for t in tokens]
    results = sorted(p.communicate(timeout=120)[0].strip() for p in procs)
    assert results == ["lost", "lost", "lost", "won"]


def test_a_claim_taken_over_an_expired_one_is_audited_as_stolen(monkeypatch, tmp_path):
    now = [1000.0]
    kit = Kit(clean_env(monkeypatch, tmp_path), clock=lambda: now[0])
    a, b = kit.agent("alpha"), kit.agent("beta")
    kit.ws.claim(a, "port", "9300", ttl_s=10)
    now[0] += 60
    kit.ws.claim(b, "port", "9300")
    events = {r["event"]: r for r in kit.ws.audit_log.rows()}
    assert events["claim.expired"]["who"] == "alpha"
    assert events["claim.stolen"]["who"] == "beta" and events["claim.stolen"]["previous"] == "alpha"
    assert events["claim.stolen"]["key"] == "9300"


def test_renewing_says_which_claims_the_lifetime_cap_held_back(monkeypatch, tmp_path):
    now = [1000.0]
    kit = Kit(clean_env(monkeypatch, tmp_path), clock=lambda: now[0])
    a = kit.agent("alpha")
    kit.ws.claim(a, "port", "9301", ttl_s=100)
    for _ in range(9):
        now[0] += 3000
        renewed = kit.ws.renew(a, 3600)
    assert [c["capped"] for c in renewed] == [True]
    from ml_stack.workspace import tools
    monkeypatch.setenv("ML_STACK_WORKSPACE_TOKEN", a)
    out = tools.workspace_heartbeat(3600)
    assert out["renewed"] == 1 and out["capped"] == ["port:9301"]
    listing = kit.ws.claims.listing()[0]
    assert "expires_in_s" in listing and "expiring_soon" in listing
