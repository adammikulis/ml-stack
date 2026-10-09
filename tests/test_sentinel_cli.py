"""``poolhouse security``: every listing is JSON on request and nothing prints a secret."""

from __future__ import annotations

import json
import os

from poolhouse import home, sentinel
from poolhouse.sentinel import Mode, State
from poolhouse.sentinel.cli import command
from poolhouse.sentinel.store import Holding


def run(capsys, *argv):
    code = command(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_status_json_reports_the_mode_and_a_verified_log(capsys):
    node = sentinel.default()
    node.store.watch("session", "s1", "x")
    code, out, _ = run(capsys, "status", "--json")
    data = json.loads(out)
    assert code == 0 and data["mode"] == "guarded" and data["log_ok"] is True
    assert data["subjects"]["watch"] == 1


def test_baseline_pins_then_scan_finds_a_change(capsys):
    model = home.home() / "m.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF" * 1000)
    code, out, _ = run(capsys, "baseline", "--pin", str(model), "--json", "--honey")
    assert code == 0 and json.loads(out)[0]["path"] == str(model)
    assert run(capsys, "scan", "--deep")[0] == 0
    original = model.stat()
    model.write_bytes(b"GGUF" * 999 + b"evil")
    os.utime(model, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert model.stat().st_size == original.st_size
    assert model.stat().st_mtime_ns == original.st_mtime_ns
    node = sentinel.default()
    assert str(model) in node.manifest.pins()
    assert node.store.state_of("model", str(model)) == State.CLEAR
    code, out, err = run(capsys, "scan", "--deep", "--json")
    assert code == 1 and json.loads(out)[0]["kind"] == "integrity.content_changed", (
        out, err, node.manifest.pins(), node.store.state_of("model", str(model)))
    code, out, _ = run(capsys, "quarantine", "list", "--json")
    rows = json.loads(out)
    assert rows[0]["state"] == "quarantined" and rows[0]["moved"] is True
    assert not model.exists()


def test_events_filter_by_kind_and_severity(capsys):
    node = sentinel.default()
    node.store.quarantine(("peer", "1.2.3.4"), "forged", {"token": "hf_" + "z" * 30})
    code, out, _ = run(capsys, "events", "--json", "--severity", "critical")
    rows = json.loads(out)
    assert code == 0 and rows and all(r["severity"] == "critical" for r in rows)
    assert "hf_zzz" not in out
    assert json.loads(run(capsys, "events", "--json", "--kind", "nothing.here")[1]) == []


def test_verify_passes_then_fails_after_an_edit(capsys):
    node = sentinel.default()
    node.store.watch("session", "s1", "x")
    node.store.watch("session", "s2", "x")
    assert run(capsys, "verify")[0] == 0
    log = node.bus.log.path
    log.write_text(log.read_text().replace("quarantine.watch", "quarantine.clear", 1))
    code, out, _ = run(capsys, "verify", "--json")
    assert code == 1 and json.loads(out)["ok"] is False


def test_show_prints_metadata_but_never_the_held_text_without_a_person(capsys):
    node = sentinel.default()
    held = node.store.quarantine(("message", "s:1"), "denied", None,
                                 Holding(text="SECRET INSTRUCTION TEXT"))
    code, out, _ = run(capsys, "quarantine", "show", held.id, "--json")
    assert code == 0 and "SECRET INSTRUCTION" not in out and held.id in out
    code, out, err = run(capsys, "quarantine", "show", held.id, "--text")
    assert code == 2 and "SECRET INSTRUCTION" not in out + err


def test_add_quarantines_a_peer_by_hand(capsys):
    code, out, _ = run(capsys, "quarantine", "add", "peer:10.5.5.5", "--reason", "seen scanning")
    assert code == 0 and "quarantined peer:10.5.5.5" in out
    assert sentinel.default().peer_blocked("10.5.5.5")


def test_honey_plant_status_remove(capsys):
    assert run(capsys, "honey", "plant")[0] == 0
    rows = json.loads(run(capsys, "honey", "status", "--json")[1])
    assert len(rows) == 3
    assert run(capsys, "honey", "remove")[0] == 0
    assert json.loads(run(capsys, "honey", "status", "--json")[1]) == []


def test_mode_shows_and_refuses_a_change_without_a_person(capsys):
    assert run(capsys, "mode")[1].strip() == Mode.GUARDED.value
    assert run(capsys, "mode", "off")[0] == 2
    assert sentinel.default().store.state_of("peer", "x") == State.CLEAR
