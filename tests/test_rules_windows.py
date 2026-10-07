"""Native Windows access checks for saved tool rules."""

import json
import sys

import pytest

from ml_stack.rules import Rule, Rules


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_rules_use_private_windows_acl_after_atomic_replacement(tmp_path):
    from ml_stack import windows_private

    path = tmp_path / "rules.json"
    rules = Rules(path)
    rules.rules = [Rule("serve_up", (("model", "quince.gguf"),), "never")]
    rules._save()
    assert windows_private.problem(path) == ""
    assert Rules(path).rules == rules.rules
    rules.fire(rules.rules[0])
    assert windows_private.problem(path) == ""
    assert Rules(path).rules[0].fired == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_rules_reject_windows_acl_grant_to_everyone(tmp_path):
    import ntsecuritycon
    import win32security

    from ml_stack import windows_private

    path = tmp_path / "rules.json"
    rules = Rules(path)
    rules.rules = [Rule("serve_up", (("model", "quince.gguf"),), "never")]
    rules._save()
    descriptor = win32security.GetNamedSecurityInfo(
        str(path), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION)
    acl = descriptor.GetSecurityDescriptorDacl()
    everyone = win32security.CreateWellKnownSid(win32security.WinWorldSid, None)
    acl.AddAccessAllowedAce(win32security.ACL_REVISION, ntsecuritycon.FILE_GENERIC_READ,
                           everyone)
    win32security.SetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT,
                                     win32security.DACL_SECURITY_INFORMATION,
                                     None, None, acl, None)
    denied = Rules(path)
    assert denied.rules == []
    assert "another account" in denied.broken
    assert denied.covers("serve_up", {"model": "quince.gguf"}, "approve-first", False) is None
    windows_private.restrict(path)
    assert Rules(path).rules == rules.rules


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_rules_do_not_promote_when_windows_acl_restriction_fails(tmp_path, monkeypatch):
    from ml_stack import windows_private

    path = tmp_path / "rules.json"
    rules = Rules(path)
    rules._save()
    original = path.read_bytes()

    def fail(_path):
        raise OSError("ACL update refused")

    monkeypatch.setattr(windows_private, "restrict", fail)
    rules.rules = [Rule("serve_up", (("model", "quince.gguf"),), "never")]
    with pytest.raises(OSError, match="ACL update refused"):
        rules._save()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_rules_missing_file_has_no_grants(tmp_path):
    rules = Rules(tmp_path / "missing.json")
    assert rules.rules == []
    assert rules.broken == ""


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_rules_reject_junction_parent(tmp_path):
    import subprocess

    target = tmp_path / "target"
    target.mkdir()
    rules = Rules(target / "rules.json")
    rules._save()
    junction = tmp_path / "junction"
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(junction), str(target)],
                   check=True, capture_output=True)
    try:
        denied = Rules(junction / "rules.json")
        assert "reparse point" in denied.broken
        with pytest.raises(ValueError, match="reparse point"):
            denied._save()
    finally:
        junction.rmdir()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows file permissions")
def test_rules_read_saved_schema_version_and_save_canonical_version(tmp_path):
    from ml_stack import windows_private

    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"schema_version": 2, "rules": [{
        "tool": "serve_up", "match": {"model": "quince.gguf"}, "verdict": "never",
    }]}), encoding="utf-8")
    windows_private.restrict(path)
    rules = Rules(path)
    assert len(rules.rules) == 1 and rules.broken == ""
    rules.fire(rules.rules[0])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["version"] == 2 and "schema_version" not in saved
    assert Rules(path).rules[0].fired == 1
    saved["version"] = 999
    path.write_text(json.dumps(saved), encoding="utf-8")
    rejected = Rules(path)
    assert rejected.rules == [] and "unknown schema_version" in rejected.broken
