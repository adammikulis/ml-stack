"""install.sh checks the release zip against the sha256 GitHub reports before it unpacks it."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "packaging" / "install.sh"
SH = shutil.which("sh")

RELEASE = (
    '{"tag_name": "v1", "assets": ['
    '{"name": "ml-stack-macos-arm64.zip", "size": 10, "digest": "sha256:' + "a" * 64 + '", '
    '"download_count": 0, "browser_download_url": "https://example.invalid/ml-stack-macos-arm64.zip"},'
    '{"name": "ml-stack-linux-x86_64.zip", "size": 10, "digest": "sha256:' + "b" * 64 + '", '
    '"download_count": 0, "browser_download_url": "https://example.invalid/ml-stack-linux-x86_64.zip"},'
    '{"name": "ml-stack-windows-x64.zip", "size": 10, '
    '"browser_download_url": "https://example.invalid/ml-stack-windows-x64.zip"}]}')


def _functions() -> str:
    text = SCRIPT.read_text()
    start = text.index("release_digest() {")
    end = text.index("# -- python ----")
    return ("have() { command -v \"$1\" >/dev/null 2>&1; }\n"
            "die() { printf 'error: %s\\n' \"$*\" >&2; exit 1; }\n" + text[start:end])


def _sh(program: str, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run([SH, "-c", _functions() + program], input=stdin, capture_output=True,
                          text=True)


@pytest.mark.skipif(SH is None, reason="needs a POSIX shell")
def test_the_digest_belongs_to_the_asset_the_key_names():
    assert _sh('release_digest ml-stack-macos-arm64', RELEASE).stdout.strip() == "a" * 64
    assert _sh('release_digest ml-stack-linux-x86_64', RELEASE).stdout.strip() == "b" * 64


@pytest.mark.skipif(SH is None, reason="needs a POSIX shell")
def test_an_asset_with_no_digest_gets_none_and_does_not_borrow_the_one_before_it():
    got = _sh('release_digest ml-stack-windows-x64', RELEASE).stdout.strip()
    assert got == "", "a digest from the neighbouring asset would pass the wrong file"


@pytest.mark.skipif(SH is None, reason="needs a POSIX shell")
def test_a_file_is_checked_against_its_digest(tmp_path):
    blob = tmp_path / "pkg.zip"
    blob.write_bytes(b"the release")
    good = hashlib.sha256(b"the release").hexdigest()
    assert _sh(f'verify_sha256 "{blob}" {good}').returncode == 0
    assert _sh(f'verify_sha256 "{blob}" {"0" * 64}').returncode == 1
    blob.write_bytes(b"the release, altered")
    assert _sh(f'verify_sha256 "{blob}" {good}').returncode == 1


@pytest.mark.skipif(SH is None, reason="needs a POSIX shell")
def test_the_installer_refuses_a_release_that_reports_no_digest():
    text = SCRIPT.read_text()
    assert "reports no sha256" in text and "does not match the sha256" in text
    assert text.index("verify_sha256 \"$TMP/pkg.zip\"") < text.index("unzip -q \"$TMP/pkg.zip\"")
