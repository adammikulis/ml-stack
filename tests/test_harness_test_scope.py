"""Routine scoped testing and inventory use the existing reversible policy."""
from poolhouse import roles
from poolhouse.harnesspolicy import decide


def test_queue_is_reversible_only_in_authorized_isolated_worktree(tmp_path):
    root = tmp_path / "worktree"
    (root / "scripts").mkdir(parents=True)
    (root / ".git").write_text("gitdir: elsewhere")
    (root / "scripts/test").write_text("# maintained runner")
    call = {"command": "python scripts/test fast tests/test_example.py -n 1"}
    assert decide(roles.PLAN_AND_GO, "Bash", call, roots=[str(root)]).action == "allow"
    assert decide(roles.READ_ONLY, "Bash", call, roots=[str(root)]).action == "deny"
    assert decide(roles.PLAN_AND_GO, "Bash", {"command": "python /outside/scripts/test fast"}, roots=[str(root)]).action == "ask"
    (root / "scripts/test").unlink()
    (root / "scripts/test").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_text("other program")
    assert decide(roles.PLAN_AND_GO, "Bash", call, roots=[str(root)]).action == "ask"


def test_inventory_and_policy_source_reads_do_not_request_mutation_authority():
    assert decide(roles.READ_ONLY, "Bash", {"command": "pyenv versions 2>/dev/null"}).action == "allow"
    assert decide(roles.PLAN_AND_GO, "Bash", {"command": "pyenv exec arbitrary-program"}).action == "ask"
    assert decide(roles.READ_ONLY, "Bash", {"command": "git log -- src/poolhouse/sentinel/policy.py"}).action == "allow"
    assert decide(roles.PLAN_AND_GO, "Bash", {"command": "poolhouse-security mode off"}).action != "allow"


def test_native_testing_cannot_bypass_queue_in_reversible_role():
    for command in ("pytest tests/test_one.py", "python -m pytest tests/test_one.py",
                    "python3.13 -m pytest", "uv run pytest", "uv run --project . pytest",
                    "poetry run pytest", "python -m unittest", "pytest --collect-only"):
        decision = decide(roles.PLAN_AND_GO, "Bash", {"command": command})
        assert decision.action == "deny", command
        assert "scripts/test" in decision.reason
