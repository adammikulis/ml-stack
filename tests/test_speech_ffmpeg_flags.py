"""The ffmpeg command that decodes uploaded audio is restricted to local protocols and no extra files."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from ml_stack.speech import asr


def test_ffmpeg_is_limited_to_pipe_file_and_crypto_with_no_playlist_files(monkeypatch):
    seen: list[list[str]] = []

    def run(cmd, **kw):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout=b"RIFF", stderr=b"")

    monkeypatch.setattr(asr.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(asr.subprocess, "run", run)
    asr.to_wav_16k(b"\x00\x00")
    cmd = seen[0]
    assert cmd[cmd.index("-protocol_whitelist") + 1] == "pipe,file,crypto"
    assert cmd[cmd.index("-allowed_extensions") + 1] == "NONE"
    assert cmd.index("-protocol_whitelist") < cmd.index("-i")
    assert cmd.index("-allowed_extensions") < cmd.index("-i")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_playlist_cannot_pull_in_a_network_resource():
    playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://127.0.0.1:9/x.ts\n#EXT-X-ENDLIST\n"
    with pytest.raises(asr.AudioConversionError):
        asr.to_wav_16k(playlist)
