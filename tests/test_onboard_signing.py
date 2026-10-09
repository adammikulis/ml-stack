"""The signing key: in the OS keystore by default, encrypted at rest when there is none, never
plaintext on disk; export, rotation and revocation only with a person."""

import base64
import io
import json

import keyring
import pytest
from onboard_support import FileKeyring, Recorder

from poolhouse import keystore as keystore_module
from poolhouse.fleet.onboard import manifest as mf, signing
from poolhouse.fleet.onboard.human import HumanRequired, mint
from poolhouse.fleet.onboard.signing import KeyStoreError, SigningKeys


@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


@pytest.fixture
def keystore(tmp_path, monkeypatch):
    """The OS keystore, stood in for by a real keyring backend over a file."""
    monkeypatch.setenv("POOLHOUSE_TEST_KEYRING", str(tmp_path / "keystore.json"))
    before = keyring.get_keyring()
    keyring.set_keyring(FileKeyring())
    yield tmp_path / "keystore.json"
    keyring.set_keyring(before)


@pytest.fixture
def no_keystore(monkeypatch):
    from keyring.backends import fail
    before = keyring.get_keyring()
    keyring.set_keyring(fail.Keyring())
    yield
    keyring.set_keyring(before)


def person(action, subject):
    return mint(action, subject, typed=lambda _p: subject, terminal=(True, True), env={})


def keys_in(tmp_path, **more):
    rec = Recorder()
    said = []
    return SigningKeys(tmp_path / "state", bus=rec.bus, say=said.append, **more), rec, said


def files_under(root):
    return [p for p in root.rglob("*") if p.is_file()]


def holds_secret(tmp_path, signer_raw):
    needles = (signer_raw, base64.b64encode(signer_raw), base64.urlsafe_b64encode(signer_raw),
               signer_raw.hex().encode())
    return [p for p in files_under(tmp_path / "state")
            if any(n in p.read_bytes() for n in needles)]


def test_the_default_home_of_the_key_is_the_keystore_and_no_file_holds_it(tmp_path, keystore):
    keys, rec, said = keys_in(tmp_path)
    doc = keys.meta()
    assert doc["store"] == "keystore" and said == []
    raw = keys._get("keystore").private_raw()
    assert holds_secret(tmp_path, raw) == []
    assert keys.wrapped_path.exists() and not keys.file_path.exists()
    held = json.loads(keystore.read_text())
    assert list(held) == [f"{keystore_module.SERVICE}/{keystore_module.default().account}"]
    assert all(raw not in base64.b64decode(v[3:]) for v in held.values())
    assert (tmp_path / "state" / "signing.json").stat().st_mode & 0o077 == 0
    assert base64.b64decode(doc["public"]) == keys.public and doc["key_id"] == keys.key_id
    assert rec.of("onboard.signing.created")[0].evidence == {"store": "keystore"}


def test_signing_is_automatic_and_the_manifest_is_short_lived(tmp_path, keystore):
    keys, rec, _ = keys_in(tmp_path)
    f = tmp_path / "a-0.1-py3-none-any.whl"
    f.write_bytes(b"w" * 1000)
    raw = keys.sign([mf.Signer.entry_for(f, kind="wheel")], serial=5)     # no prompt, no grant
    m = mf.verify(raw, keys.public)
    assert m.serial == 5 and 0 < m.expires - m.issued <= 7 * 86400
    assert rec.of("onboard.signing.signed")


def test_without_a_keystore_the_key_is_encrypted_at_rest_and_the_owner_is_warned(
        tmp_path, no_keystore):
    keys, rec, said = keys_in(tmp_path, passphrase=lambda _p: "a long passphrase")
    doc = keys.meta()
    assert doc["store"] == "file"
    assert said and "no OS keystore" in said[0] and str(keys.file_path) in said[0]
    assert rec.of("onboard.signing.file_fallback")[0].severity == "warning"
    assert keys.file_path.stat().st_mode & 0o077 == 0
    raw = signing.open_file(keys.file_path, "a long passphrase")
    assert len(raw) == 32 and holds_secret(tmp_path, raw) == []
    assert json.loads(keys.file_path.read_text())["kdf"] == "scrypt"
    f = tmp_path / "p.whl"
    f.write_bytes(b"x" * 100)
    mf.verify(keys.sign([mf.Signer.entry_for(f, kind="wheel")], serial=1), keys.public)


def test_the_wrong_passphrase_does_not_unlock_the_file(tmp_path, no_keystore):
    keys, _, _ = keys_in(tmp_path, passphrase=lambda _p: "right passphrase")
    keys.meta()
    wrong, _, _ = keys_in(tmp_path, passphrase=lambda _p: "wrong passphrase")
    with pytest.raises(KeyStoreError, match="wrong passphrase"):
        wrong.sign([], serial=1)


def test_no_keystore_and_no_passphrase_is_a_clear_refusal_not_a_plaintext_key(
        tmp_path, no_keystore, monkeypatch):
    monkeypatch.delenv(signing.UNLOCK_ENV, raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO())       # not a terminal
    keys = SigningKeys(tmp_path / "state", bus=Recorder().bus, say=lambda _m: None)
    with pytest.raises(KeyStoreError, match="passphrase"):
        keys.meta()
    assert files_under(tmp_path / "state") == []


def test_exporting_needs_a_person_and_writes_only_an_encrypted_copy(tmp_path, keystore):
    keys, rec, _ = keys_in(tmp_path)
    keys.meta()
    out = tmp_path / "backup.enc"
    with pytest.raises(HumanRequired):
        keys.export(person("rotate", keys.key_id), out, "pw pw pw pw")     # a grant for another act
    assert not out.exists()
    keys.export(person("export", keys.key_id), out, "pw pw pw pw")
    raw = signing.open_file(out, "pw pw pw pw")
    assert mf.Signer.from_raw(raw).key_id == keys.key_id
    assert raw not in out.read_bytes() and rec.of("onboard.signing.exported")


def test_rotating_needs_a_person_and_the_old_key_announces_the_new_one(tmp_path, keystore):
    keys, rec, _ = keys_in(tmp_path)
    old_public, old_id = keys.public, keys.key_id
    f = tmp_path / "p.whl"
    f.write_bytes(b"x" * 100)
    with pytest.raises(HumanRequired):
        keys.rotate(person("export", old_id))
    assert keys.key_id == old_id
    fresh = keys.rotate(person("rotate", old_id))
    assert fresh["key_id"] != old_id and len(fresh["rotations"]) == 1
    raw = keys.sign([mf.Signer.entry_for(f, kind="wheel")], serial=2)
    with pytest.raises(mf.RotationAnnounced) as moved:            # a member still on the old key
        mf.verify(raw, old_public)
    assert moved.value.new_public == keys.public
    assert mf.verify(raw, keys.public).key_id == keys.key_id      # once a person has pinned it
    held = json.loads(keystore.read_text())
    assert list(held) == [f"{keystore_module.SERVICE}/{keystore_module.default().account}"]   # only the master is in the keystore
    assert keys.wrapped_path.exists()
    assert rec.of("onboard.signing.rotated")[0].severity == "warning"


def test_revoking_needs_a_person_and_travels_in_every_manifest_after(tmp_path, keystore):
    keys, _, _ = keys_in(tmp_path)
    with pytest.raises(HumanRequired):
        keys.revoke(person("rotate", "ab" * 32), "ab" * 32)
    keys.revoke(person("revoke", "ab" * 32), "ab" * 32)
    assert mf.verify(keys.sign([], serial=1), keys.public).revoked_keys == ("ab" * 32,)


def test_confirm_before_signing_asks_every_time_and_turning_it_on_needs_a_person(
        tmp_path, keystore):
    keys, _, _ = keys_in(tmp_path)
    with pytest.raises(HumanRequired):
        keys.require_confirmation(person("export", keys.key_id), True)
    keys.require_confirmation(person("confirm-signing", keys.key_id), True)
    asked = []
    with pytest.raises(KeyStoreError, match="confirmation"):
        keys.sign([], serial=1)
    with pytest.raises(KeyStoreError):
        keys.sign([], serial=1, confirm=lambda s: asked.append(s) or False)
    mf.verify(keys.sign([], serial=1, confirm=lambda s: asked.append(s) or True), keys.public)
    assert len(asked) == 2 and "0 files" in asked[0]
    keys.require_confirmation(person("confirm-signing", keys.key_id), False)
    keys.sign([], serial=2)


def test_a_key_that_was_in_the_keystore_does_not_silently_become_a_file(tmp_path, keystore):
    keys, _, _ = keys_in(tmp_path)
    keys.meta()
    keyring.set_keyring(__import__("keyring.backends.fail", fromlist=["x"]).Keyring())
    with pytest.raises(KeyStoreError, match="keystore"):
        keys.sign([], serial=1)
    assert not keys.file_path.exists()


def test_an_agent_has_no_path_to_the_key_commands(tmp_path, keystore, monkeypatch):
    """The commands that need a person refuse a process an agent started, at a terminal or not."""
    keys, _, _ = keys_in(tmp_path)
    keys.meta()
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(HumanRequired, match="agent"):
        mint("rotate", keys.key_id, typed=lambda _p: keys.key_id, terminal=(True, True))


def test_rotating_a_key_held_in_an_encrypted_file_replaces_the_file(tmp_path, no_keystore):
    keys, _, _ = keys_in(tmp_path, passphrase=lambda _p: "a long passphrase")
    old = keys.key_id
    fresh = keys.rotate(person("rotate", old))
    assert fresh["key_id"] != old and fresh["store"] == "file"
    assert signing.open_file(keys.file_path, "a long passphrase")
    mf.verify(keys.sign([], serial=1), keys.public)
