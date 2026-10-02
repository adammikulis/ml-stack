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
    assert e.size == 200_000 and len(e.chunks) == 4 and e.shareable


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
    outer["manifest"]["entries"][0]["shareable"] = True
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
    {"chunks": ["0" * 64]}, {"kind": "exe"}, {"shareable": "yes"}])
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


def test_shareable_and_source_survive_signing(tmp_path, signer):
    raw = signer.sign(entries(tmp_path, signer, shareable=False, licence="gated",
                              source="hf://org/model"), serial=1)
    e = mf.verify(raw, signer.public).entries[0]
    assert (e.shareable, e.licence, e.source) == (False, "gated", "hf://org/model")


def test_the_private_key_is_written_private_and_loads_back(tmp_path, signer):
    path = tmp_path / "sub" / "signing.key"
    signer.save(path)
    assert path.stat().st_mode & 0o077 == 0
    assert mf.load_signer(path).public == signer.public
    with pytest.raises(FileExistsError):
        signer.save(path)                       # never overwritten silently
    assert len(base64.b64decode(path.read_bytes())) == 32


def test_a_manifest_naming_another_key_than_the_one_that_signed_it_is_refused(tmp_path, signer):
    outer = json.loads(signer.sign(entries(tmp_path, signer), serial=1))
    outer["manifest"]["key_id"] = "0" * 64
    outer["signature"] = base64.b64encode(
        signer._private.sign(mf._canonical(outer["manifest"]))).decode()
    with pytest.raises(mf.ManifestError, match="different key"):
        mf.verify(json.dumps(outer).encode(), signer.public)


def test_chunk_sizes_outside_the_range_are_refused_even_when_the_digests_add_up(tmp_path, signer):
    row = entries(tmp_path, signer)[0].to_json()
    row.update(size=100, chunk_size=10, chunks=["0" * 64] * 10)
    with pytest.raises(mf.ManifestError, match="chunk size"):
        mf.Entry.from_json(row)
