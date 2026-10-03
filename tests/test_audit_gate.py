"""The audit and red-team workflows must fail on findings, and accepted findings must expire.

GitHub Actions cannot run here, so this parses the workflow files and runs the gate script they call.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def steps(workflow: str, job: str) -> list[dict]:
    return yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))["jobs"][job]["steps"]


def step_with(workflow: str, job: str, needle: str) -> dict:
    found = [s for s in steps(workflow, job) if needle in str(s.get("run", ""))]
    assert len(found) == 1, f"{workflow}/{job}: expected one step running {needle!r}"
    return found[0]


def gate_module():
    spec = importlib.util.spec_from_file_location("audit_gate", ROOT / "scripts" / "audit_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pip_audit_step_runs_the_gate_and_cannot_be_swallowed():
    step = step_with("audit.yml", "pip-audit", "pip-audit")
    assert "scripts/audit_gate.py" in step["run"]
    assert step.get("continue-on-error") in (None, False)
    last = [line for line in step["run"].splitlines() if line.strip()][-1]
    assert last.strip().startswith("python scripts/audit_gate.py") and "||" not in last


def test_the_redteam_attack_step_compares_with_the_committed_baseline_and_cannot_be_swallowed():
    step = step_with("redteam.yml", "redteam", "ml_stack.redteam run")
    assert step.get("continue-on-error") in (None, False)
    run = step["run"]
    assert "--against" in run and "||" not in run
    baseline = run.split("--against")[1].split()[0]
    assert (ROOT / baseline).is_file()
    assert "chat" not in run.split("--scenarios")[1].split()[0], "a gullible stub always loses the chat scenario"


def test_every_workflow_is_still_scheduled():
    for name in ("audit.yml", "redteam.yml"):
        doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
        assert "schedule" in (doc.get(True) or doc.get("on"))


def test_every_accepted_finding_has_a_reason_and_an_expiry_in_the_future():
    allow = gate_module().load_allow()
    for entry in allow:
        assert entry["reason"].strip()
        assert date.fromisoformat(entry["expires"]) > date.today(), f"{entry['id']} expired: renew or remove it"


REPORT = {"dependencies": [{"name": "Foo", "version": "1.0", "vulns": [{"id": "PYSEC-1"}]},
                           {"name": "bar", "version": "2.0", "vulns": []}]}


def test_the_gate_fails_on_a_vulnerability_nobody_accepted():
    assert gate_module().gate(REPORT, [], date.today()) == ["Foo 1.0: PYSEC-1"]


def test_the_gate_passes_a_finding_with_a_live_acceptance_and_fails_it_once_lapsed():
    entry = {"id": "PYSEC-1", "package": "foo", "reason": "r", "expires": (date.today() + timedelta(days=5)).isoformat()}
    gate = gate_module().gate
    assert gate(REPORT, [entry], date.today()) == []
    problems = gate(REPORT, [entry], date.today() + timedelta(days=6))
    assert problems and "expired" in problems[0]


def test_the_script_exits_one_on_a_finding_and_zero_on_a_clean_report(tmp_path):
    bad, good = tmp_path / "bad.json", tmp_path / "good.json"
    bad.write_text(json.dumps(REPORT), encoding="utf-8")
    good.write_text(json.dumps({"dependencies": []}), encoding="utf-8")
    run = lambda p: subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_gate.py"), str(p)],  # noqa: E731
                                   capture_output=True, text=True, check=False)
    assert run(bad).returncode == 1 and "PYSEC-1" in run(bad).stdout
    assert run(good).returncode == 0
