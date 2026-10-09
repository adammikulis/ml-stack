"""The signed manifest: accepted only from the pinned key, fresh, and well formed."""

import base64
import json

import pytest

from ml_stack.fleet.onboard import manifest as mf


@pytest.fixture(autouse=True)
def needs_cryptography():
    pytest.importorskip("cryptography")


@pytest.fixture
def signer():
    return mf.Signer.generate()


def entries(tmp_path, signer, **more):
    f = tmp_path / "ml_stack-0.2-py3-none-any.whl"
    f.write_bytes(b"w" * 200_000)
    return [signer.entry(f, kind="wheel", chunk_size=65536, **more)]


def test_a_manifest_signed_by_the_pinned_key_verifies_and_lists_chunks(tmp_path, signer):
    raw = signer.sign(entries(tmp_path, signer), serial=7)
    m = mf.verify(raw, signer.public)
    assert m.serial == 7 and m.key_id == signer.key_id
    e = m.entry("ml_stack-0.2-py3-none-any.whl")
    assert e.size == 200_000 and len(e.chunks) == 4 and e.sharing == "open"


def test_another_key_signing_is_refused(tmp_path, signer):
    raw = mf.Signer.generate().sign(entries(tmp_path, signer), serial=1)
    with pytest.raises(mf.ManifestError, match="not by the pinned key"):
        mf.verify(raw, signer.public)


def test_any_change_to_the_signed_body_is_refused(tmp_path, signer):
    outer = json.loads(signer.sign(entries(tmp_path, signer), serial=1))
    outer["manifest"]["entries"][0]["sha256"] = "0" * 64
    with pytest.raises(mf.ManifestError, match="not by the pinned key"):
        mf.verify(json.dumps(outer).encode(), signer.public)
    outer = json.loads(signer.sign(entries(tmp_path, signer), serial=1))
    outer["manifest"]["entries"][0]["sharing"] = "open"
    outer["manifest"]["serial"] = 99
    with pytest.raises(mf.ManifestError):
        mf.verify(json.dumps(outer).encode(), signer.public)


def test_an_expired_manifest_and_an_older_serial_are_refused(tmp_path, signer):
    raw = signer.sign(entries(tmp_path, signer), serial=5, valid_s=100, now=1000.0)
    assert mf.verify(raw, signer.public, now=1050.0)
    with pytest.raises(mf.ManifestError, match="expired"):
        mf.verify(raw, signer.public, now=1101.0)
    with pytest.raises(mf.ManifestError, match="older"):
        mf.verify(raw, signer.public, now=1050.0, min_serial=6)


def test_a_revoked_signing_key_is_refused(tmp_path, signer):
    raw = signer.sign(entries(tmp_path, signer), serial=1)
    with pytest.raises(mf.ManifestError, match="revoked"):
        mf.verify(raw, signer.public, revoked_keys=[signer.key_id])


@pytest.mark.parametrize("change", [
    {"name": "../evil.whl"}, {"name": "a/b.whl"}, {"name": "nul.whl"}, {"size": -1},
    {"size": True}, {"chunk_size": 10}, {"chunk_size": 1 << 40}, {"sha256": "xyz"},
    {"chunks": ["0" * 64]}, {"kind": "exe"}, {"sharing": "everyone"}])
def test_a_malformed_or_hostile_entry_is_refused_even_when_signed(tmp_path, signer, change):
    """The signer is trusted to vouch, not to be well-formed: a name that climbs out of the
    staging directory is refused however it was signed."""
    row = entries(tmp_path, signer)[0].to_json() | change
    bad = mf.Entry.from_json
    with pytest.raises(mf.ManifestError):
        bad(row)


def test_duplicate_names_and_oversize_lists_are_refused(tmp_path, signer):
    one = entries(tmp_path, signer)[0]
    with pytest.raises(mf.ManifestError, match="one name"):
        mf.verify(signer.sign([one, one], serial=1), signer.public)
    with pytest.raises(mf.ManifestError):
        mf.verify(b"x" * (9 * 1024 * 1024), signer.public)
    with pytest.raises(mf.ManifestError, match="not a signed manifest"):
        mf.verify(b"{}", signer.public)


def test_sharing_level_licence_and_source_survive_signing(tmp_path, signer):
    raw = signer.sign(entries(tmp_path, signer, sharing="owner", licence="gated",
                              licence_url="https://x/licence", source="hf://org/model"),
                      serial=1)
    e = mf.verify(raw, signer.public).entries[0]
    assert (e.sharing, e.licence, e.licence_url, e.source) == (
        "owner", "gated", "https://x/licence", "hf://org/model")


def test_an_entry_that_does_not_say_is_treated_as_restricted(tmp_path, signer):
    row = entries(tmp_path, signer)[0].to_json()
    del row["sharing"]
    assert mf.Entry.from_json(row).sharing == "owner"


def test_changing_the_sharing_field_of_a_signed_manifest_is_caught(tmp_path, signer):
    outer = json.loads(signer.sign(entries(tmp_path, signer, sharing="never"), serial=1))
    outer["manifest"]["entries"][0]["sharing"] = "open"
    with pytest.raises(mf.ManifestError, match="not by the pinned key"):
        mf.verify(json.dumps(outer).encode(), signer.public)


def test_a_rotation_signed_by_the_old_key_is_announced_not_trusted(tmp_path, signer):
    new = mf.Signer.generate()
    announcement = signer.announce_rotation(new)
    raw = new.sign(entries(tmp_path, new), serial=2, carried=mf.Carried((announcement,)))
    with pytest.raises(mf.RotationAnnounced) as moved:
        mf.verify(raw, signer.public)                # still pinned to the old key
    assert moved.value.new_public == new.public
    assert mf.verify(raw, new.public).key_id == new.key_id      # after a person pins the new one


def test_a_rotation_not_signed_by_the_pinned_key_is_just_a_bad_signature(tmp_path, signer):
    thief, new = mf.Signer.generate(), mf.Signer.generate()
    raw = new.sign(entries(tmp_path, new), serial=2,
                   carried=mf.Carried((thief.announce_rotation(new),)))
    with pytest.raises(mf.ManifestError, match="not by the pinned key") as err:
        mf.verify(raw, signer.public)
    assert not isinstance(err.value, mf.RotationAnnounced)


def test_revoked_keys_travel_in_the_manifest_and_manifests_last_days_not_months(tmp_path, signer):
    raw = signer.sign(entries(tmp_path, signer), serial=1, now=1000.0,
                      carried=mf.Carried(revoked=("ab" * 32,)))
    m = mf.verify(raw, signer.public, now=1001.0)
    assert m.revoked_keys == ("ab" * 32,)
    assert m.expires - m.issued == mf.VALID_S <= 7 * 86400


def test_a_manifest_naming_another_key_than_the_one_that_signed_it_is_refused(tmp_path, signer):
    outer = json.loads(signer.sign(entries(tmp_path, signer), serial=1))
    outer["manifest"]["key_id"] = "0" * 64
    outer["signature"] = base64.b64encode(
        signer._private.sign(mf._encoded(outer["manifest"]))).decode()
    with pytest.raises(mf.ManifestError, match="different key"):
        mf.verify(json.dumps(outer).encode(), signer.public)


def test_chunk_sizes_outside_the_range_are_refused_even_when_the_digests_add_up(tmp_path, signer):
    row = entries(tmp_path, signer)[0].to_json()
    row.update(size=100, chunk_size=10, chunks=["0" * 64] * 10)
    with pytest.raises(mf.ManifestError, match="chunk size"):
        mf.Entry.from_json(row)
