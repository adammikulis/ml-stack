"""The memory store is encrypted at rest, per user, under a key that never sits beside it."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import keyring
import pytest
from keyring.backends import null

import ml_stack
from ml_stack import home, keystore, memory
from ml_stack.memory import vault
from ml_stack.memory.store import Setup, Tampered
from tests import memory_keys

ring = memory_keys.ring

CANARY = "zephyrquokka-canary-7731"
ENTITY = "quillfeather-entity-4412"
SCOPE = {"machine": "box", "build": "b7"}


def make(path: Path, **kw) -> memory.Store:
    return memory.Store(path / "graph.enc", scope=lambda: SCOPE, **kw)


def every_file(*roots: Path):
    for root in roots:
        for here, _, names in os.walk(root):
            for name in names:
                yield Path(here, name)


def assert_no_plaintext(*roots: Path, needles=(CANARY, ENTITY)) -> int:
    seen = 0
    for each in every_file(*roots):
        data = each.read_bytes()
        seen += 1
        for needle in needles:
            assert needle.encode() not in data, f"{needle} in {each}"
            assert needle.lower().encode() not in data.lower(), f"{needle} in {each}"
    return seen


def fill(store: memory.Store) -> None:
    store.add(f"{CANARY} serves 64 slots", "result", entities=[f"topic:{ENTITY}", "model:Qwen3.8-Flash-Next"],
)
    store.add(f"second {CANARY} note", "note", entities=[f"topic:{ENTITY}"])


def test_no_fact_or_entity_text_is_in_any_file_after_writes_close_and_backups(tmp_path):
    store = make(tmp_path)
    fill(store)
    fill_more = store.add(f"third {CANARY}", "note")
    store.confirm(fill_more.id)
    assert assert_no_plaintext(tmp_path, home.state()) >= 3
    store.close()
    assert store.prev.exists()
    assert_no_plaintext(tmp_path, home.state())


def test_the_default_store_in_the_state_directory_holds_no_plaintext():
    store = memory.Store()
    fill(store)
    store.forget(store.facts()[0].id)
    assert_no_plaintext(home.state())
    assert "ml-stack-memory" not in "".join(p.read_text(errors="ignore") for p in every_file(home.state()))


def test_a_write_that_dies_midway_leaves_only_ciphertext(tmp_path, monkeypatch):
    store = make(tmp_path)
    fill(store)
    real, calls = os.replace, []

    def dying(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("power cut")
        real(src, dst)

    monkeypatch.setattr(os, "replace", dying)
    with pytest.raises(OSError, match="power cut"):
        store.add(f"{CANARY} one more", "note")
    monkeypatch.setattr(os, "replace", real)
    assert_no_plaintext(tmp_path)
    survivor = make(tmp_path)
    assert survivor.status == "recovered" and len(survivor.facts()) == 2


def test_a_killed_process_leaves_only_ciphertext(tmp_path):
    code = ("import os\nfrom ml_stack import memory\n"
            f"s = memory.Store({str(tmp_path / 'graph.enc')!r}, scope=lambda: {{}})\n"
            f"s.add({CANARY!r} + ' dies', 'note', entities=['topic:{ENTITY}'])\n"
            f"s.add({CANARY!r} + ' dies again', 'note')\nos._exit(9)\n")
    env = {**os.environ, vault.KEYS_ENV: "passphrase", vault.PASSPHRASE_ENV: "correct horse",
           "PYTHONPATH": str(Path(ml_stack.__file__).parent.parent), "ML_STACK_HOME": str(tmp_path / "h")}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          cwd=tmp_path, check=False)
    assert done.returncode == 9, done.stderr
    assert_no_plaintext(tmp_path)


def test_the_key_is_never_written_beside_the_data(tmp_path, ring):
    store = make(tmp_path)
    fill(store)
    held = next(iter(ring.held.values()))
    raw = base64.b64decode(held[3:])
    assert len(raw) == 32 and len(ring.held) == 1
    for each in every_file(tmp_path, home.state()):
        data = each.read_bytes()
        assert raw not in data and held.encode() not in data, each
    assert {p.name for p in tmp_path.iterdir()} <= {"graph.enc", "graph.enc.prev", "store.lock", "machine-state"}


def test_the_file_is_not_readable_without_the_key(tmp_path, ring):
    store = make(tmp_path)
    fill(store)
    ring.held.clear()
    keystore.default().lock()
    other = make(tmp_path)
    assert other.status == "locked" and other.facts() == []
    assert "holds no key" in other.why
    with pytest.raises(memory.KeyUnavailable):
        other.add("a fact")
    assert "locked" in memory.session_context("anything", store=other)


def test_another_key_cannot_open_it(tmp_path, ring):
    store = make(tmp_path)
    fill(store)
    account = next(iter(ring.held))
    ring.held[account] = "v1:" + base64.b64encode(os.urandom(32)).decode()
    keystore.default().lock()
    assert make(tmp_path).status == "tampered"


def test_flipping_any_byte_of_the_ciphertext_or_header_is_detected(tmp_path):
    store = make(tmp_path)
    fill(store)
    store.prev.unlink(missing_ok=True)
    good = store.path.read_bytes()
    for at in (0, 4, 5, 10, 22, 30, 45, len(good) // 2, len(good) - 1):
        raw = bytearray(good)
        raw[at] ^= 0x80
        store.path.write_bytes(bytes(raw))
        assert make(tmp_path).status == "tampered", at
    store.path.write_bytes(good[:-1])
    assert make(tmp_path).status == "tampered"
    store.path.write_bytes(good)
    assert make(tmp_path).status == "ok"


def test_a_file_moved_to_another_user_or_profile_does_not_authenticate(tmp_path, ring):
    mine = make(tmp_path / "a", setup=Setup(user="501:alice"))
    fill(mine)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "graph.enc").write_bytes(mine.path.read_bytes())
    stolen = make(tmp_path / "b", setup=Setup(user="502:bob"))
    assert stolen.facts() == [] and stolen.status in ("locked", "tampered")
    assert stolen.status == "tampered"
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "graph.enc").write_bytes(mine.path.read_bytes())
    assert make(tmp_path / "c", setup=Setup(user="501:alice", profile="work")).status == "tampered"


def test_a_second_user_on_the_same_file_has_no_key_for_it(tmp_path):
    alice = make(tmp_path, setup=Setup(user="501:alice"))
    fill(alice)
    bob = make(tmp_path, setup=Setup(user="502:bob"))
    assert bob.status in ("locked", "tampered") and bob.facts() == []
    with pytest.raises((memory.KeyUnavailable, Tampered)):
        bob.add("bob writes")
    assert len(make(tmp_path, setup=Setup(user="501:alice")).facts()) == 2


def test_two_users_and_two_profiles_on_one_home_stay_separate():
    alice, bob = memory.Store(setup=Setup(user="501:alice")), memory.Store(setup=Setup(user="502:bob"))
    work = memory.Store(setup=Setup(user="501:alice", profile="work"))
    assert len({alice.path, bob.path, work.path}) == 3
    alice.add("alice only fact")
    bob.add("bob only fact")
    work.add("work only fact")
    assert [f.text for f in alice.facts()] == ["alice only fact"]
    assert [f.text for f in bob.facts()] == ["bob only fact"]
    assert [f.text for f in work.facts()] == ["work only fact"]
    assert "bob" not in memory.session_context("fact", store=alice)
    assert memory.Store(setup=Setup(user="502:bob")).path == bob.path


def test_the_default_user_is_the_process_identity_not_anything_passed_in(monkeypatch):
    assert memory.Store().user == vault.identity()
    monkeypatch.setenv("USER", "someone-else")
    monkeypatch.setenv("LOGNAME", "someone-else")
    assert memory.Store().user == vault.identity()
    with pytest.raises(ValueError, match="profile"):
        memory.Store(setup=Setup(profile="../escape"))
    assert memory.Store(setup=Setup(profile="work")).path != memory.Store().path


def test_no_keystore_fails_closed_without_writing_a_key(tmp_path, monkeypatch):
    keyring.set_keyring(null.Keyring())
    monkeypatch.delenv(vault.KEYS_ENV, raising=False)
    store = make(tmp_path)
    assert store.status == "fresh"
    with pytest.raises(memory.KeyUnavailable, match=vault.KEYS_ENV):
        store.add("a fact")
    assert not any(every_file(tmp_path)) or [p.name for p in every_file(tmp_path)] == ["store.lock"]
    assert not store.path.exists()


def test_a_passphrase_is_an_explicit_fallback_and_never_a_key_file(tmp_path, monkeypatch):
    keyring.set_keyring(null.Keyring())
    monkeypatch.setenv(vault.KEYS_ENV, "passphrase")
    monkeypatch.setenv(vault.PASSPHRASE_ENV, "correct horse battery")
    store = make(tmp_path)
    fill(store)
    assert_no_plaintext(tmp_path)
    assert b"correct horse" not in store.path.read_bytes()
    assert make(tmp_path).status == "ok" and len(make(tmp_path).facts()) == 2
    monkeypatch.setenv(vault.PASSPHRASE_ENV, "wrong")
    assert make(tmp_path).status == "tampered"
    monkeypatch.setenv(vault.KEYS_ENV, "keystore")
    assert make(tmp_path).status == "locked"


def test_a_passphrase_store_without_a_terminal_or_variable_is_locked(tmp_path, monkeypatch):
    monkeypatch.setenv(vault.KEYS_ENV, "passphrase")
    monkeypatch.setenv(vault.PASSPHRASE_ENV, "pw")
    fill(make(tmp_path))
    monkeypatch.delenv(vault.PASSPHRASE_ENV)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    locked = make(tmp_path)
    assert locked.status == "locked" and vault.PASSPHRASE_ENV in locked.why


def test_rekey_moves_everything_to_a_new_salt_and_so_a_new_key(tmp_path, ring):
    store = make(tmp_path)
    fill(store)
    before = store.path.read_bytes()
    store.rekey()
    after = store.path.read_bytes()
    assert vault.header(before)[1] != vault.header(after)[1]
    assert vault.header(store.prev.read_bytes())[1] == vault.header(after)[1]
    again = make(tmp_path)
    assert again.status == "ok" and len(again.facts()) == 2
    assert len(ring.held) == 1
    store.path.write_bytes(after[:-1] + bytes([after[-1] ^ 1]))
    assert make(tmp_path).status == "recovered"
    assert_no_plaintext(tmp_path)


def test_rekey_interrupted_before_the_write_still_opens_under_the_old_key(tmp_path, ring):
    store = make(tmp_path)
    fill(store)
    store.keys.rotate(os.urandom(vault.SALT))
    assert make(tmp_path).status == "ok"


def test_the_cli_rekey_is_for_a_person_only(monkeypatch, capsys):
    from ml_stack.memory import cli

    monkeypatch.setenv("CLAUDECODE", "1")
    assert cli.main(["rekey"]) == cli.DENIED
