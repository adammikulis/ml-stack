"""Dependency auditing fails (pip, npm, cargo) and the release ships an SBOM.

GitHub Actions cannot run here, so the workflow files are parsed and the scripts they call are run on
recorded reports in the formats of ``pip-audit --format json``, ``npm audit --json`` (npm 7+) and
``cargo audit --json``. The SBOM is generated from this interpreter's real environment and the real
``Cargo.lock``, and validated against the CycloneDX 1.5 rules the generator relies on."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
TODAY = date.today()


def module(name: str):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", ROOT / "scripts" / f"{name}.py")
    made = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = made
    spec.loader.exec_module(made)
    return made


gate = module("audit_gate")
sbom = module("sbom")

NPM = {"vulnerabilities": {
    "left-pad": {"name": "left-pad", "range": "<1.3.1", "via": [
        {"source": 1, "name": "left-pad", "title": "bad", "url": "https://github.com/advisories/GHSA-aaaa-bbbb-cccc",
         "severity": "high"}]},
    "wrapper": {"name": "wrapper", "range": "*", "via": ["left-pad"]},     # only because of left-pad
}, "metadata": {}}
CARGO = {"vulnerabilities": {"found": True, "count": 1, "list": [
    {"advisory": {"id": "RUSTSEC-2099-0001", "package": "tiny", "aliases": ["CVE-2099-1"]},
     "package": {"name": "tiny", "version": "0.3.0"}}]}, "warnings": {"unmaintained": [
         {"advisory": {"id": "RUSTSEC-2099-0002"}, "package": {"name": "old", "version": "1"}}]}}
PIP = {"dependencies": [{"name": "Foo", "version": "1.0", "vulns": [
    {"id": "PYSEC-1", "aliases": ["CVE-2099-5"]}]}]}


def entry(package, vid, days=5, ecosystem=None):
    row = {"id": vid, "package": package, "reason": "r", "expires": (TODAY + timedelta(days=days)).isoformat()}
    return {**row, "ecosystem": ecosystem} if ecosystem else row


def test_an_npm_advisory_fails_the_gate_and_a_derived_entry_is_not_a_second_finding():
    assert gate.gate(NPM, [], TODAY, "npm") == ["left-pad <1.3.1: GHSA-aaaa-bbbb-cccc"]


def test_a_cargo_vulnerability_fails_the_gate_but_an_unmaintained_warning_does_not():
    assert gate.gate(CARGO, [], TODAY, "cargo") == ["tiny 0.3.0: RUSTSEC-2099-0001"]


def test_an_acceptance_matches_by_id_or_alias_only_in_its_own_ecosystem_and_lapses():
    assert gate.gate(CARGO, [entry("tiny", "CVE-2099-1", ecosystem="cargo")], TODAY, "cargo") == []
    assert gate.gate(NPM, [entry("left-pad", "GHSA-aaaa-bbbb-cccc", ecosystem="npm")], TODAY, "npm") == []
    assert gate.gate(PIP, [entry("foo", "CVE-2099-5")], TODAY, "pip") == []
    # a pip acceptance does not accept an npm finding of the same name and id
    assert gate.gate(NPM, [entry("left-pad", "GHSA-aaaa-bbbb-cccc")], TODAY, "npm")
    late = gate.gate(CARGO, [entry("tiny", "RUSTSEC-2099-0001", ecosystem="cargo")], TODAY + timedelta(days=6), "cargo")
    assert late and "expired" in late[0]


@pytest.mark.parametrize("kind,text", [("npm", "{}"), ("cargo", '{"vulnerabilities": {}}'), ("pip", "[]"),
                                       ("pip", ""), ("npm", "<html>")])
def test_a_report_the_tool_did_not_produce_fails_the_gate_instead_of_passing_it(tmp_path, kind, text):
    report = tmp_path / "report.json"
    report.write_text(text, encoding="utf-8")
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_gate.py"), "--kind", kind, str(report)],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 1 and "unusable" in done.stdout


def test_the_script_exit_codes_for_npm_and_cargo_reports(tmp_path):
    run = lambda kind, doc: subprocess.run(  # noqa: E731
        [sys.executable, str(ROOT / "scripts" / "audit_gate.py"), "--kind", kind, str(doc)],
        capture_output=True, text=True, check=False)
    for kind, bad, clean in (("npm", NPM, {"vulnerabilities": {}}),
                             ("cargo", CARGO, {"vulnerabilities": {"found": False, "count": 0, "list": []}})):
        (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        (tmp_path / "ok.json").write_text(json.dumps(clean), encoding="utf-8")
        assert run(kind, tmp_path / "bad.json").returncode == 1
        assert run(kind, tmp_path / "ok.json").returncode == 0


def jobs(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))["jobs"]


def test_every_audit_job_ends_in_the_gate_and_nothing_in_the_audit_workflow_is_swallowed():
    text = (WORKFLOWS / "audit.yml").read_text(encoding="utf-8")
    assert "continue-on-error" not in text
    for job, kind in (("pip-audit", None), ("npm-audit", "npm"), ("cargo-audit", "cargo")):
        runs = [s["run"] for s in jobs("audit.yml")[job]["steps"] if "audit_gate.py" in str(s.get("run", ""))]
        assert len(runs) == 1, job
        last = [ln for ln in runs[0].splitlines() if ln.strip()][-1].strip()
        want = "python scripts/audit_gate.py " + (f"--kind {kind} " if kind else "")
        assert last.startswith(want) and "||" not in last, (job, last)


def test_the_audit_and_release_workflows_use_pinned_actions_only():
    for name in ("audit.yml", "release.yml", "release-build.yml"):
        for ref in re.findall(r"uses:\s*(\S+)", (WORKFLOWS / name).read_text(encoding="utf-8")):
            if ref.startswith("./.github/workflows/"):
                continue
            assert re.fullmatch(r".+@[0-9a-f]{40}", ref), f"{name}: {ref} is not pinned to a commit"


def test_the_allow_list_has_only_known_ecosystems_and_live_acceptances():
    for e in gate.load_allow():
        assert e["reason"].strip() and date.fromisoformat(e["expires"]) > TODAY


# -- SBOM ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bom():
    return sbom.build()


def test_the_sbom_is_valid_cyclonedx_for_the_rules_the_generator_relies_on(bom):
    assert sbom.problems(bom) == []
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(bom, SCHEMA_SUBSET)


def test_the_sbom_lists_every_package_notices_lists(bom):
    notices = module("notices")
    listed = {c["purl"] for c in bom["components"]}
    for name, version, _licence in notices.python_closure():
        assert f"pkg:pypi/{name.lower()}@{version}" in listed, name
    lock = sbom.tomllib.loads((ROOT / "app" / "Cargo.lock").read_text(encoding="utf-8"))["package"]
    crates = [p for p in lock if p.get("source")]
    assert crates and all(f"pkg:cargo/{p['name']}@{p['version']}" in listed for p in crates)
    assert all(c["hashes"] for c in bom["components"] if c["purl"].startswith("pkg:cargo/"))


def test_the_sbom_is_the_same_bytes_every_time(bom):
    assert json.dumps(sbom.build(), sort_keys=True) == json.dumps(bom, sort_keys=True)


def test_a_broken_sbom_is_caught(bom):
    broken = json.loads(json.dumps(bom))
    broken["components"][0]["purl"] = "not a purl"
    broken["components"].append(dict(broken["components"][1]))              # a repeated bom-ref
    broken["components"][2]["hashes"] = [{"alg": "SHA-256", "content": "xyz"}]
    broken["specVersion"] = "1.2"
    found = " ".join(sbom.problems(broken))
    assert "malformed purl" in found and "used twice" in found and "malformed hash" in found and "specVersion" in found


def test_the_release_workflow_builds_checks_and_uploads_the_sbom():
    steps = jobs("release-build.yml")["wheels"]["steps"]
    (make,) = [s for s in steps if s.get("name") == "SBOM"]
    assert "scripts/sbom.py --out sbom.cdx.json" in make["run"] and "scripts/sbom.py --check" in make["run"]
    assert "curl" not in make["run"]
    uploads = [s for s in steps if "upload-artifact" in str(s.get("uses", "")) and s["with"]["name"] == "sbom"]
    assert uploads and uploads[0]["with"]["path"] == "sbom.cdx.json"


SCHEMA_SUBSET = {
    "type": "object", "required": ["bomFormat", "specVersion", "version", "components"],
    "properties": {
        "bomFormat": {"const": "CycloneDX"}, "specVersion": {"const": "1.5"},
        "serialNumber": {"type": "string", "pattern": "^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"},
        "version": {"type": "integer", "minimum": 1},
        "metadata": {"type": "object", "properties": {"component": {"$ref": "#/$defs/component"},
                                                      "timestamp": {"type": "string", "format": "date-time"}}},
        "components": {"type": "array", "items": {"$ref": "#/$defs/component"}},
    },
    "$defs": {"component": {
        "type": "object", "required": ["type", "name"], "additionalProperties": True,
        "properties": {
            "type": {"enum": ["application", "framework", "library", "container", "platform", "file"]},
            "name": {"type": "string", "minLength": 1}, "version": {"type": "string"},
            "purl": {"type": "string"}, "bom-ref": {"type": "string"},
            "hashes": {"type": "array", "items": {"type": "object", "required": ["alg", "content"],
                                                  "properties": {"alg": {"enum": ["MD5", "SHA-1", "SHA-256", "SHA-384", "SHA-512"]},
                                                                 "content": {"type": "string", "pattern": "^[a-fA-F0-9]{64,128}$"}}}},
            "licenses": {"type": "array", "items": {"type": "object", "minProperties": 1}},
        }}},
}
