"""The keystore: one item per user, subkeys per purpose, and a gate that keeps the operating
system's prompts few. Every test installs a fake keyring backend; the real ones refuse."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from keyring.errors import KeyringError

from ml_stack import home, keystore, memory, sentinel
from ml_stack.fleet.onboard import manifest as mf
from ml_stack.fleet.onboard.signing import SigningKeys
from ml_stack.keystore import Keystore, Wires
from ml_stack.memory import vault
from ml_stack.sentinel import human
from ml_stack.sentinel.events import Bus, Event, Severity
from tests import keystore_support
from tests.onboard_support import Clock

counting = keystore_support.counting
REAL_INTERACTIVE = keystore.interactive
CANARY = b"canary-secret-91ab33f0-do-not-store"


class Said:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        self.lines.append(text)


def make(tmp_path: Path, clock: Clock | None = None, *, person: bool = True,
         say: Said | None = None, bus: Bus | None = None) -> Keystore:
    wires = Wires(clock=clock or Clock(), say=say or Said(), bus=bus or Bus(), interactive=lambda: person)
    return Keystore(directory=tmp_path / "ks", wires=wires)


def master_of(ring) -> bytes:
    (held,) = ring.held.values()
    return base64.b64decode(held[3:])


def every_file(*roots: Path):
    for root in roots:
        for here, _, names in os.walk(root):
            for name in names:
                yield Path(here, name)


def holds(needle: bytes, *roots: Path) -> list[Path]:
    forms = (needle, base64.b64encode(needle), base64.urlsafe_b64encode(needle), needle.hex().encode())
    return [p for p in every_file(*roots) if any(f in p.read_bytes() for f in forms)]


# -- lazy -------------------------------------------------------------------------------------


def test_nothing_touches_the_keystore_until_a_key_is_needed(tmp_path, counting):
    ks = make(tmp_path)
    ks.status(), ks.pending_notice(), ks.available(), ks.lock()
    store = memory.Store(tmp_path / "m" / "graph.enc")
    assert store.facts() == [] and store.status == "fresh"
    SigningKeys(tmp_path / "fleet").entry(Path(__file__))
    from ml_stack import credentials
    credentials.describe(), credentials.status("NOPE_KEY")
    assert counting.calls == []


def test_a_missing_master_is_not_made_by_a_read(tmp_path, counting):
    ks = make(tmp_path)
    with pytest.raises(keystore.KeystoreMissing):
        ks.subkey("memory", "a", create=False)
    assert counting.calls == ["get"] and counting.held == {}


def test_the_first_write_makes_the_master_with_one_read_and_one_create(tmp_path, counting):
    ks = make(tmp_path)
    ks.wrap("credentials", "HF_TOKEN", CANARY)
    assert counting.calls == ["get", "set"]


# -- one item per user, subkeys per purpose ----------------------------------------------------


def test_every_purpose_shares_one_item(tmp_path, counting):
    ks = make(tmp_path)
    ks.subkey("memory", "a")
    ks.subkey("reputation", "u")
    ks.wrap("fleet-signing", "d", b"x")
    ks.subkey("credentials", "HF")
    assert list(counting.held) == [("ml-stack", ks.account)]
    assert ks.account == "master/" + keystore.os_user()
    assert counting.held[("ml-stack", ks.account)].startswith("v1:") and len(master_of(counting)) == 32
    assert counting.calls == ["get", "set"]


def test_subkeys_differ_by_purpose_owner_and_context_and_repeat_exactly(tmp_path, counting):
    ks = make(tmp_path)
    keys = [ks.subkey("memory", "a"), ks.subkey("reputation", "a"), ks.subkey("memory", "b"),
            ks.subkey("memory", "a", context=b"salt-1"), ks.subkey("memory", "a", context=b"salt-2"),
            ks.subkey("credentials", "a")]
    assert len(set(keys)) == len(keys) and all(len(k) == 32 for k in keys)
    assert ks.subkey("memory", "a") == keys[0]
    assert keystore.hkdf(master_of(counting), keystore._fields(b"memory", b"a", b"")) == keys[0]
    assert master_of(counting) not in keys


def test_hkdf_is_rfc_5869_test_case_1():
    okm = keystore.hkdf(bytes.fromhex("0b" * 22), bytes.fromhex("f0f1f2f3f4f5f6f7f8f9"),
                        salt=bytes.fromhex("000102030405060708090a0b0c"), length=42)
    assert okm.hex() == ("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
                         "34007208d5b887185865")


def test_a_wrapped_value_opens_only_for_its_own_purpose_and_owner(tmp_path, counting):
    ks = make(tmp_path)
    blob = ks.wrap("fleet-signing", "dir-1", CANARY)
    assert CANARY not in blob and ks.unwrap("fleet-signing", "dir-1", blob) == CANARY
    for purpose, owner in (("credentials", "dir-1"), ("fleet-signing", "dir-2"), ("memory", "")):
        with pytest.raises(keystore.KeystoreError):
            ks.unwrap(purpose, owner, blob)
    for at in (0, 5, 20, len(blob) - 1):
        bad = bytearray(blob)
        bad[at] ^= 1
        with pytest.raises(keystore.KeystoreError):
            ks.unwrap("fleet-signing", "dir-1", bytes(bad))
    key = ks.subkey("fleet-signing", "dir-1")
    with pytest.raises(InvalidTag):
        AESGCM(key).decrypt(blob[4:16], blob[16:], b"")
    assert AESGCM(key).decrypt(blob[4:16], blob[16:], keystore._aad("fleet-signing", "dir-1")) == CANARY


def test_a_label_cannot_be_split_to_collide_with_another_purpose(tmp_path, counting):
    ks = make(tmp_path)
    assert ks.subkey("memory", "a\0b") != ks.subkey("memory\0a", "b")
    assert ks.subkey("memory", "a", context=b"b") != ks.subkey("memory", "ab")
    assert keystore._aad("memory", "a\0b") != keystore._aad("memory\0a", "b")


# -- cache --------------------------------------------------------------------------------------


def test_a_second_use_in_a_process_costs_no_backend_call(tmp_path, counting):
    ks = make(tmp_path)
    first = ks.subkey("memory", "a")
    counting.calls.clear()
    assert ks.subkey("memory", "a") == first and ks.subkey("reputation", "b") != first
    ks.unwrap("memory", "a", ks.wrap("memory", "a", b"x"))
    assert counting.calls == []
    ks.lock()
    assert ks.subkey("memory", "a") == first and counting.calls == ["get"]


# -- single flight across processes ------------------------------------------------------------

CHILD = """
import os, sys, time
from pathlib import Path
import keyring
if type(keyring.get_keyring()).__name__ != "CountingFileRing":
    sys.exit(3)
while not Path(os.environ["GO"]).exists():
    time.sleep(0.005)
from ml_stack import keystore
ks = keystore.Keystore(wires=keystore.Wires(interactive=lambda: True, say=lambda _m: None))
print(ks.subkey("test", "owner").hex())
"""


def child_env(tmp_path: Path) -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "ML_STACK_HOME": str(tmp_path / "home"),
            "PYTHON_KEYRING_BACKEND": "tests.keystore_support.CountingFileRing",
            "ML_STACK_TEST_KEYRING": str(tmp_path / "ring.json"), "GO": str(tmp_path / "go"),
            "ML_STACK_NOTIFY": "off", "ML_STACK_TEST_KEYRING_DELAY": "0.3"}


def test_six_processes_starting_together_create_one_master(tmp_path):
    env = child_env(tmp_path)
    kids = [subprocess.Popen([sys.executable, "-c", CHILD], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True) for _ in range(6)]
    (tmp_path / "go").write_text("go")
    results = [(k.communicate(timeout=120), k.returncode) for k in kids]
    assert [code for _, code in results] == [0] * 6, [err for (_, err), _ in results]
    assert len({out.strip() for (out, _), _ in results}) == 1
    calls = Path(f"{tmp_path / 'ring.json'}.calls").read_text().split()
    assert calls.count("set") == 1 and calls.count("get") <= 6 and "delete" not in calls
    assert list(json.loads((tmp_path / "ring.json").read_text())) == [f"ml-stack/master/{keystore.os_user()}"]


def test_processes_behind_a_refusal_do_not_ask_again(tmp_path):
    env = child_env(tmp_path)
    state = tmp_path / "home" / "keystore"
    state.mkdir(parents=True)
    (state / "denied.json").write_text(json.dumps({"until": 4e12, "cause": "KeyringError"}))
    kids = [subprocess.Popen([sys.executable, "-c", CHILD], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True) for _ in range(3)]
    (tmp_path / "go").write_text("go")
    for kid in kids:
        _, err = kid.communicate(timeout=120)
        assert kid.returncode != 0 and "declined" in err
    assert not Path(f"{tmp_path / 'ring.json'}.calls").exists()


# -- denial -------------------------------------------------------------------------------------


@pytest.mark.parametrize("error", [KeyringError, OSError])
def test_a_refusal_is_remembered_and_never_retried_in_a_loop(tmp_path, counting, error):
    clock, bus = Clock(), Bus()
    counting.refuse = error
    first = make(tmp_path, clock, bus=bus)
    with pytest.raises(keystore.KeystoreDenied, match="ml-stack-security unlock"):
        first.subkey("memory", "a")
    assert counting.calls == ["get"]
    second = make(tmp_path, clock)
    for _ in range(50):
        with pytest.raises(keystore.KeystoreDenied):
            first.subkey("memory", "a")
        with pytest.raises(keystore.KeystoreDenied):
            second.subkey("memory", "a")
    assert counting.calls == ["get"]
    clock.advance(keystore.COOLDOWN_S + 1)
    with pytest.raises(keystore.KeystoreDenied):
        first.subkey("memory", "a")
    assert counting.calls == ["get"]
    with pytest.raises(keystore.KeystoreDenied):
        second.subkey("memory", "a")
    assert counting.calls == ["get", "get"]
    denied = [e for e in bus.recent(kind="keystore") if e.kind == "keystore.denied"]
    assert len(denied) == 1 and denied[0].severity.name == "WARNING"


def test_unlock_by_a_person_clears_the_refusal(tmp_path, counting):
    counting.refuse = KeyringError
    ks = make(tmp_path)
    with pytest.raises(keystore.KeystoreDenied):
        ks.subkey("memory", "a")
    counting.refuse = None
    with pytest.raises(keystore.KeystoreDenied):
        ks.subkey("memory", "a")
    assert ks.provision() is True
    assert ks.subkey("memory", "a") and ks.status()["denied_for_s"] == 0


# -- rate limit -----------------------------------------------------------------------------------


def test_a_burst_of_500_calls_reaches_the_backend_at_most_the_ceiling(tmp_path, counting):
    clock, bus = Clock(), Bus()
    make(tmp_path, clock, bus=bus).subkey("memory", "a")
    busy = ok = 0
    for _ in range(500):
        try:
            make(tmp_path, clock, bus=bus).subkey("memory", "a")
            ok += 1
        except keystore.KeystoreBusy:
            busy += 1
    assert len(counting.calls) == keystore.RATE_CEILING == 20
    assert ok == 18 and busy == 482
    assert any(e.kind == "keystore.refused" and e.evidence["outcome"] == "busy" for e in bus.recent())
    clock.advance(keystore.RATE_WINDOW_S + 1)
    assert make(tmp_path, clock).subkey("memory", "a")
    assert len(counting.calls) == 21


def test_one_instance_locking_and_reading_in_a_loop_is_capped_too(tmp_path, counting):
    ks = make(tmp_path)
    failures = 0
    for _ in range(100):
        ks.lock()
        try:
            ks.subkey("memory", "a")
        except keystore.KeystoreBusy:
            failures += 1
    assert len(counting.calls) == 20 and failures == 100 - 19


# -- background processes ----------------------------------------------------------------------


def test_a_background_process_never_prompts_or_creates(tmp_path, counting):
    ks = make(tmp_path, person=False)
    with pytest.raises(keystore.KeystoreLocked, match="ml-stack-security unlock"):
        ks.subkey("memory", "a")
    assert counting.calls == [] and counting.held == {}


def test_after_unlock_a_background_process_reads_but_still_does_not_create(tmp_path, counting):
    person = make(tmp_path)
    key = person.subkey("memory", "a")
    background = make(tmp_path, person=False)
    assert background.subkey("memory", "a") == key
    counting.held.clear()
    background.lock()
    with pytest.raises(keystore.KeystoreLocked):
        background.subkey("memory", "a")
    assert counting.held == {}


def test_what_counts_as_interactive(monkeypatch):
    class Tty(io.StringIO):
        def __init__(self, tty: bool) -> None:
            super().__init__()
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    monkeypatch.delenv(keystore.ENV_NONINTERACTIVE, raising=False)
    monkeypatch.setattr(sys, "stdin", Tty(True))
    assert REAL_INTERACTIVE() is True
    monkeypatch.setenv(keystore.ENV_NONINTERACTIVE, "1")
    assert REAL_INTERACTIVE() is False
    monkeypatch.delenv(keystore.ENV_NONINTERACTIVE)
    monkeypatch.setattr(sys, "stdin", Tty(False))
    monkeypatch.setattr(keystore, "_desktop", lambda: False)
    assert REAL_INTERACTIVE() is False
    monkeypatch.setattr(keystore, "_desktop", lambda: True)
    assert REAL_INTERACTIVE() is True


# -- first-use notice ----------------------------------------------------------------------------


def test_the_person_is_told_once_before_the_first_prompt(tmp_path, counting):
    said = Said()
    ks = make(tmp_path, say=said)
    assert ks.pending_notice() == keystore.NOTICE and "one Keychain prompt" in keystore.NOTICE
    ks.subkey("memory", "a")
    ks.lock()
    ks.subkey("memory", "a")
    assert said.lines == [keystore.NOTICE]
    again = Said()
    assert make(tmp_path, say=again).pending_notice() == ""
    make(tmp_path, say=again).subkey("memory", "a")
    assert again.lines == []


def test_a_refused_background_process_says_nothing_about_a_prompt(tmp_path, counting):
    said = Said()
    with pytest.raises(keystore.KeystoreLocked):
        make(tmp_path, person=False, say=said).subkey("memory", "a")
    assert said.lines == []


# -- the unlock command -----------------------------------------------------------------------


@pytest.fixture
def at_a_terminal(monkeypatch):
    """Stdin and stdout count as terminals; the agent-marker check stays real."""
    for name in human.AGENT_MARKERS:
        monkeypatch.delenv(name, raising=False)
    real = human.require_person
    monkeypatch.setattr(human, "require_person", lambda action, terminal=None, env=None: real(action, (True, True), env))


def test_unlock_is_for_a_person_at_a_terminal(counting, monkeypatch, capsys):
    from ml_stack.net import cli

    for marker in human.AGENT_MARKERS:
        monkeypatch.setenv(marker, "1")
        assert cli.command(["unlock"]) == 2
        monkeypatch.delenv(marker)
    assert cli.command(["unlock"]) == 2
    assert counting.calls == [] and counting.held == {}


def test_unlock_makes_the_key_once_and_lets_background_processes_read_it(counting, at_a_terminal, capsys):
    from ml_stack.net import cli

    assert cli.command(["unlock", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is True
    assert cli.command(["unlock", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["created"] is False
    assert len(counting.held) == 1
    background = Keystore(wires=Wires(interactive=lambda: False, say=Said()))
    assert background.subkey("memory", "a")


def test_the_status_command_makes_no_backend_call(counting, capsys):
    from ml_stack.net import cli

    assert cli.command(["keystore", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["provisioned"] is False
    assert counting.calls == []


def test_reset_deletes_the_key_after_the_person_types_the_account(counting, at_a_terminal, monkeypatch):
    from ml_stack.net import cli

    ks = keystore.default()
    ks.subkey("memory", "a")
    real = human.mint
    typed = ["wrong"]
    monkeypatch.setattr(human, "mint", lambda a, s, **kw: real(a, s, typed=lambda _p: typed[0], **kw))
    assert cli.command(["keystore-reset"]) == 2 and counting.held
    typed[0] = ks.user
    assert cli.command(["keystore-reset"]) == 0 and counting.held == {}
    assert ks.status()["provisioned"] is False


# -- migration ------------------------------------------------------------------------------------


def fresh_store(tmp_path) -> memory.Store:
    store = memory.Store(tmp_path / "m" / "graph.enc")
    store.add("prefers short answers", "preference")
    store.add("serves 64 slots", "result")
    return store


def as_an_older_version_wrote_it(store: memory.Store, ring, *, files=None) -> bytes:
    """Put the store under a random key kept in a per-feature item, the way it used to be."""
    old = os.urandom(32)
    for each in files or (store.path, store.prev):
        if each.exists():
            blob = each.read_bytes()
            _, salt = vault.header(blob)
            plain, _ = vault.open_blob(blob, store.keys.keys(salt), owner=store.owner)
            each.write_bytes(vault.seal_blob(plain, old, mode="keystore", salt=salt, owner=store.owner))
    ring.set_password(vault.SERVICE, store.keys.account, json.dumps({"current": base64.b64encode(old).decode()}))
    (home.state("keystore") / "legacy.json").unlink(missing_ok=True)
    keystore._DEFAULTS.clear()
    return old


def test_a_memory_store_under_an_older_key_moves_to_the_master_and_the_old_item_goes(tmp_path, counting):
    store = fresh_store(tmp_path)
    store.add("third", "note")
    old = as_an_older_version_wrote_it(store, counting)
    assert store.prev.exists()
    again = memory.Store(tmp_path / "m" / "graph.enc")
    assert len(again.facts()) == 3 and again.status == "ok"
    assert (vault.SERVICE, store.keys.account) not in counting.held
    for each in (store.path, store.prev):
        blob = each.read_bytes()
        with pytest.raises(vault.BadSeal):
            vault.open_blob(blob, [old], owner=store.owner)
        vault.open_blob(blob, store.keys.keys(vault.header(blob)[1]), owner=store.owner)
    assert len(memory.Store(tmp_path / "m" / "graph.enc").facts()) == 3


def test_the_old_item_is_kept_when_a_file_does_not_verify(tmp_path, counting):
    store = fresh_store(tmp_path)
    as_an_older_version_wrote_it(store, counting)
    store.path.write_bytes(store.path.read_bytes()[:-1])
    memory.Store(tmp_path / "m" / "graph.enc").facts()
    assert (vault.SERVICE, store.keys.account) in counting.held


def test_a_migration_cut_off_between_the_two_files_finishes_on_the_next_run(tmp_path, counting):
    store = fresh_store(tmp_path)
    store.add("third", "note")
    as_an_older_version_wrote_it(store, counting, files=(store.prev,))
    old = json.loads(counting.held[(vault.SERVICE, store.keys.account)])
    assert store.path.exists() and store.prev.exists() and old
    again = memory.Store(tmp_path / "m" / "graph.enc")
    assert again.status == "ok" and len(again.facts()) == 3
    assert (vault.SERVICE, store.keys.account) not in counting.held
    assert memory.Store(tmp_path / "m" / "graph.enc").status == "ok"


def test_a_refused_delete_leaves_verified_data_readable_and_the_item_for_later(tmp_path, counting):
    clock = Clock()
    store = fresh_store(tmp_path)
    as_an_older_version_wrote_it(store, counting)
    counting.refuse, counting.refuse_ops = KeyringError, {"delete"}
    again = memory.Store(tmp_path / "m" / "graph.enc")
    assert len(again.facts()) == 2
    assert (vault.SERVICE, store.keys.account) in counting.held
    counting.refuse = None
    keystore.default()._denied = ""
    (home.state("keystore") / "denied.json").unlink()
    assert len(memory.Store(tmp_path / "m" / "graph.enc").facts()) == 2
    assert (vault.SERVICE, store.keys.account) not in counting.held
    assert clock()


def test_the_fleet_signing_key_moves_into_a_wrapped_file_and_the_old_item_goes(tmp_path, counting):
    signer = mf.Signer.generate()
    keys = SigningKeys(tmp_path / "fleet", say=lambda _m: None)
    keys._write({"store": "keyring", "key_id": signer.key_id, "public": base64.b64encode(signer.public).decode(),
                 "created": 1.0, "rotations": [], "revoked": [], "confirm": False})
    counting.set_password("ml-stack", keys.account, base64.b64encode(signer.private_raw()).decode())
    mf.verify(keys.sign([], serial=1), signer.public)
    assert ("ml-stack", keys.account) not in counting.held
    assert json.loads(keys.meta_path.read_text())["store"] == "keystore" and keys.wrapped_path.exists()
    assert holds(signer.private_raw(), tmp_path, home.state()) == []
    mf.verify(keys.sign([], serial=2), signer.public)


def test_a_signing_key_move_cut_off_after_the_wrap_is_finished_by_the_next_call(tmp_path, counting):
    signer = mf.Signer.generate()
    keys = SigningKeys(tmp_path / "fleet", say=lambda _m: None)
    keys._write({"store": "keyring", "key_id": signer.key_id, "public": base64.b64encode(signer.public).decode(),
                 "created": 1.0, "rotations": [], "revoked": [], "confirm": False})
    counting.set_password("ml-stack", keys.account, base64.b64encode(signer.private_raw()).decode())
    counting.refuse, counting.refuse_ops = KeyringError, {"delete"}
    with pytest.raises(Exception, match="declined"):
        keys.sign([], serial=1)
    assert keys.wrapped_path.exists() and ("ml-stack", keys.account) in counting.held
    assert json.loads(keys.meta_path.read_text())["store"] == "keyring"
    counting.refuse = None
    keystore.default()._denied = ""
    (home.state("keystore") / "denied.json").unlink()
    mf.verify(keys.sign([], serial=2), signer.public)
    assert ("ml-stack", keys.account) not in counting.held
    assert json.loads(keys.meta_path.read_text())["store"] == "keystore"


def test_a_signing_key_move_cut_off_after_the_old_item_went_still_signs(tmp_path, counting):
    signer = mf.Signer.generate()
    keys = SigningKeys(tmp_path / "fleet", say=lambda _m: None)
    keys._write({"store": "keyring", "key_id": signer.key_id, "public": base64.b64encode(signer.public).decode(),
                 "created": 1.0, "rotations": [], "revoked": [], "confirm": False})
    keys._wrap(signer.private_raw())
    mf.verify(keys.sign([], serial=1), signer.public)
    assert json.loads(keys.meta_path.read_text())["store"] == "keystore"


# -- no secret anywhere ---------------------------------------------------------------------------


def test_no_file_or_log_holds_the_master_a_subkey_or_a_wrapped_secret(tmp_path, counting, caplog):
    caplog.set_level("DEBUG")
    ks = Keystore(wires=Wires(interactive=lambda: True, say=Said()))
    sub = ks.subkey("memory", "a")
    blob = ks.wrap("credentials", "HF_TOKEN", CANARY)
    raw = master_of(counting)
    keys = SigningKeys(tmp_path / "fleet", say=lambda _m: None)
    keys.sign([], serial=1)
    seed = keys._get("keystore").private_raw()
    fresh_store(tmp_path)
    ks.unwrap("credentials", "HF_TOKEN", blob)
    sentinel.default().bus.emit(Event("keystore.probe", Severity.INFO, "test", "", {}))
    roots = (tmp_path, home.home())
    for secret in (raw, sub, CANARY, seed, base64.b64encode(raw)):
        assert holds(secret, *roots) == [], secret
    assert not any(raw.hex() in r.getMessage() or CANARY.decode() in r.getMessage() for r in caplog.records)
    assert any(every_file(home.state("sentinel")))


# -- audit ------------------------------------------------------------------------------------------


def test_every_backend_operation_is_a_sentinel_event_with_a_purpose_and_an_outcome(tmp_path, counting):
    bus = Bus()
    ks = make(tmp_path, bus=bus)
    ks.subkey("memory", "a")
    ks.legacy_get("ml-stack-memory", "someone", "memory")
    ks.legacy_drop("ml-stack-memory", "someone", "memory")
    seen = [(e.kind, e.subject, e.evidence["outcome"]) for e in bus.recent(kind="keystore")]
    assert seen == [("keystore.read", "purpose:memory", "absent"), ("keystore.create", "purpose:memory", "ok"),
                    ("keystore.read", "purpose:memory", "absent"), ("keystore.delete", "purpose:memory", "absent")]
    counting.refuse = KeyringError
    ks2 = make(tmp_path, bus=bus)
    with pytest.raises(keystore.KeystoreDenied):
        ks2.subkey("memory", "a")
    assert bus.recent(kind="keystore.denied")[0].severity.name == "WARNING"
    blob = json.dumps([e.to_record() for e in bus.recent()])
    assert base64.b64encode(master_of(counting)).decode() not in blob


def test_the_events_reach_the_sentinel_log(counting):
    Keystore(wires=Wires(interactive=lambda: True, say=Said())).subkey("memory", "a")
    kinds = [e.kind for e in sentinel.default().bus.recent(kind="keystore")]
    assert kinds == ["keystore.read", "keystore.create"]
