"""Every finding kind the sentinel can raise has a plain-language sentence, and nothing a
held subject says reaches a person's screen as anything but visible, bounded text."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ml_stack.sentinel import explain
from ml_stack.sentinel.policy import NEVER_ACT
from ml_stack.sentinel.store import KINDS, Record, State

SRC = Path(__file__).resolve().parents[1] / "src" / "ml_stack"
DYNAMIC = {"_found": "honey", "_changed": "integrity"}
"""Helpers that pass a kind to ``finding`` as a variable, and the family of kinds they build."""


def _calls(name: str):
    for path in sorted(SRC.rglob("*.py")):
        if path.name != "findings.py" and "sentinel.findings" not in path.read_text():
            continue                       # another module's own ``finding`` (bench, setup)
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Call):
                func = node.func
                if (func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")) == name:
                    yield path, node


def finding_kinds() -> set[str]:
    """Every kind some detector can pass to ``finding``, read from the source."""
    found: set[str] = set()
    for path, call in _calls("finding"):
        first = call.args[0] if call.args else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
            continue
        rel = path.relative_to(SRC).as_posix()
        assert rel in ("sentinel/honey.py", "sentinel/integrity.py", "sentinel/rates.py"), (
            f"{rel}:{call.lineno} builds a finding kind at run time; teach this test where its "
            "kinds come from and add a sentence for each to explain.WHY")
    for helper, family in DYNAMIC.items():
        for path, call in _calls(helper):
            literal = [a.value for a in call.args
                       if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if path.name in ("honey.py", "integrity.py") and literal:
                found.add(literal[0] if literal[0].startswith(family) else f"{family}.{literal[0]}")
    found |= {"peer.version_mismatch", "peer.binary_mismatch"}     # PeerWatch.mismatch(what)
    return found


def test_the_scan_finds_the_kinds_it_should():
    kinds = finding_kinds()
    assert {"peer.forged_traffic", "server.unmanaged", "tools.mix_shift", "honey.token_seen",
            "honey.path_named", "integrity.content_changed", "integrity.link_retargeted",
            "integrity.missing", "integrity.unreadable", "canary.hard_drift",
            "guard.tainted", "score.quarantine", "abuse.resource"} <= kinds


def test_every_finding_kind_has_a_sentence():
    missing = sorted(k for k in finding_kinds() | set(NEVER_ACT) if k not in explain.WHY)
    assert not missing, f"add a plain-language sentence to explain.WHY for: {missing}"
    assert all(len(s) > 20 and s.endswith(".") for s in explain.WHY.values())


def test_every_subject_kind_says_what_a_quarantine_blocks():
    assert set(explain.BLOCKS) == set(KINDS)


def record(reason: str, kind: str = "peer", key: str = "1.2.3.4", state=State.QUARANTINED):
    return Record("q-1", kind, key, state, reason, {}, 100.0, 100.0)


def test_the_reason_is_found_from_the_finding_code_or_the_caller_s_words():
    assert explain.describe(record("peer.forged_traffic: x=1"), 1, 160.0).why == \
        explain.WHY["peer.forged_traffic"]
    assert "reply" in explain.describe(record("model output carries a decoy value"), 1, 0).why
    assert explain.describe(record("something new"), 1, 0).why == explain.UNKNOWN


def test_what_it_blocks_and_what_to_do_depend_on_the_state():
    held = explain.describe(record("peer.forged_traffic: x"), 1, 160.0)
    assert held.blocks == explain.BLOCKS["peer"] and "false alarm" in held.advice
    watch = explain.describe(record("server.unmanaged: x", "server", "port:51089", State.WATCH),
                             1, 0)
    assert watch.name == "server :51089" and watch.blocks.startswith("Nothing is blocked")
    assert "stop watching" in watch.advice and "nothing changes" in watch.advice


@pytest.mark.parametrize(("seconds", "text"), [(5, "just now"), (300, "5 min ago"),
                                               (7200, "2 h ago"), (3 * 86400, "3 d ago")])
def test_age_in_words(seconds, text):
    assert explain.age(seconds) == text


def test_show_escapes_control_bidi_and_invisible_characters_and_bounds_length():
    nasty = "a\x1b[31mb‮c​d\ne\x00f\x9bg﻿h"
    shown = explain.show(nasty, 200)
    assert shown == "a\\x1b[31mb\\u202ec\\u200bd\\ne\\x00f\\x9bg\\ufeffh"
    assert explain.show("x" * 10_000, 40) == "x" * 39 + "…"
    assert len(explain.show("\x1b" * 10_000, 50)) <= 50
    assert explain.show("plain words ok") == "plain words ok"


def test_show_masks_secrets_before_anything_else():
    assert "hf_" + "a" * 30 not in explain.show("token hf_" + "a" * 30)


def test_a_name_is_short_and_never_the_raw_path():
    assert explain.name_of("model", "/home/me/models/qwen.gguf") == "model qwen.gguf"
    assert explain.name_of("peer", "10.0.0.1") == "peer 10.0.0.1"
    assert explain.name_of("server", "port:51089") == "server :51089"
    assert len(explain.name_of("session", "\x1b" * 5000)) <= 60
