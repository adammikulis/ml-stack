"""WAV containers, image format sniffing, and asset download."""

from __future__ import annotations

import hashlib
import struct
import zlib

import pytest
from ml_stack.media import (
    DownloadError,
    ImageError,
    WavError,
    decode,
    duration_s,
    encode,
    fetch,
    from_data_url,
    kind,
    mime,
    probe_png,
    streaming_header,
    to_data_url,
)


class TestWav:
    def test_roundtrip_preserves_pcm_and_format(self):
        pcm = b"\x01\x02" * 800
        data, info = decode(encode(pcm, sample_rate=22050, channels=2))
        assert data == pcm
        assert (info.sample_rate, info.channels, info.sample_width) == (22050, 2, 2)

    def test_duration_is_frames_over_rate(self):
        one_second = b"\x00\x00" * 16000
        assert duration_s(encode(one_second, sample_rate=16000)) == pytest.approx(1.0)

    def test_decode_walks_past_a_LIST_chunk(self):
        """Real encoders emit LIST/fact chunks between `fmt ` and `data`. Skipping a
        fixed 44 bytes turns those chunk bytes into a burst of noise at the start of the
        audio -- audible, but easy to blame on the microphone."""
        pcm = b"\xAB\xCD" * 100
        canonical = encode(pcm, sample_rate=16000)
        head, tail = canonical[:36], canonical[36:]  # split at the `data` chunk

        software = b"ml_stack\x00\x00"  # NUL-terminated, word-aligned
        info_body = b"INFOISFT" + struct.pack("<I", len(software)) + software
        listed = head + b"LIST" + struct.pack("<I", len(info_body)) + info_body + tail

        recovered, meta = decode(listed)
        assert recovered == pcm, "decode did not skip the LIST chunk"
        assert meta.sample_rate == 16000

    def test_decode_tolerates_a_truncated_download(self):
        """A short file must report the bytes it has, not raise a struct error far from
        the cause."""
        full = encode(b"\x11\x22" * 500)
        recovered, _ = decode(full[: len(full) // 2])
        assert 0 < len(recovered) < 1000

    def test_streaming_header_is_44_bytes_and_claims_a_huge_length(self):
        """The whole point is starting playback before synthesis finishes; a real length
        would mean buffering the utterance first."""
        head = streaming_header()
        assert len(head) == 44
        (declared,) = struct.unpack("<I", head[40:44])
        assert declared > 10**9

    def test_rejects_non_riff(self):
        with pytest.raises(WavError, match="RIFF"):
            decode(b"not audio at all, really")

    def test_rejects_compressed_wav(self):
        """Format 2 is ADPCM; handing its bytes to a model as PCM produces static."""
        data = bytearray(encode(b"\x00" * 64))
        data[20:22] = struct.pack("<H", 2)
        with pytest.raises(WavError, match="PCM"):
            decode(bytes(data))


class TestImage:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "jpeg"),
            (b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, "png"),
            (b"GIF89a" + b"\x00" * 16, "gif"),
            (b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8, "webp"),
            (b"\x00\x00\x00\x18ftypheic" + b"\x00" * 8, "heic"),
        ],
    )
    def test_sniffs_by_magic_bytes(self, raw: bytes, expected: str):
        """Not by extension and not by Content-Type; both arrive wrong routinely."""
        assert kind(raw) == expected

    def test_unknown_format_is_none_not_a_guess(self):
        assert kind(b"<html><body>404 not found</body></html>") is None
        with pytest.raises(ImageError):
            mime(b"<html>nope</html>")

    def test_data_url_roundtrip(self):
        raw = probe_png([(255, 0, 0)], size=8)
        recovered, mime_type = from_data_url(to_data_url(raw))
        assert recovered == raw
        assert mime_type == "image/png"

    def test_bad_base64_raises_rather_than_yielding_empty_bytes(self):
        """An empty image reaches the model as 'no image' and comes back with a
        confidently hallucinated description. Failing loudly is the whole point."""
        with pytest.raises(ImageError, match="base64"):
            from_data_url("data:image/png;base64,!!!!not base64!!!!")

    def test_probe_png_is_a_real_decodable_png(self):
        """It is hand-rolled from zlib+struct so a missing Pillow cannot make a vision
        self-test fail open. That only helps if the bytes are actually valid."""
        raw = probe_png([(10, 20, 30), (200, 100, 50)], size=16)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

        width, height, depth, colour = struct.unpack(">IIBB", raw[16:26])
        assert (width, height, depth, colour) == (16, 16, 8, 2)

        idat = raw[raw.index(b"IDAT") + 4 : raw.index(b"IEND") - 8]
        rows = zlib.decompress(idat)
        assert len(rows) == 16 * (1 + 16 * 3)
        assert rows[0] == 0                       # per-row filter byte
        assert tuple(rows[1:4]) == (10, 20, 30)   # first band
        assert tuple(rows[-3:]) == (200, 100, 50) # last band

    def test_probe_png_needs_a_colour(self):
        with pytest.raises(ImageError):
            probe_png([])


class TestDownload:
    def _serve(self, server, body: bytes, *, honour_range: bool = True):
        def handler(method: str, path: str, _body: bytes):
            return 200, body

        instance = server(handler)
        return instance

    def test_fetches_and_verifies(self, server, tmp_path):
        body = b"weights" * 1000
        instance = self._serve(server, body)
        target = tmp_path / "model.gguf"

        result = fetch(
            f"{instance.base_url}/model.gguf",
            target,
            expect_sha256=hashlib.sha256(body).hexdigest(),
            expect_bytes=len(body),
        )
        assert result.read_bytes() == body

    def test_is_idempotent_and_does_not_refetch(self, server, tmp_path):
        """This is called on every startup, so a second call must not touch the
        network."""
        body = b"x" * 256
        instance = self._serve(server, body)
        target = tmp_path / "asset.bin"

        fetch(f"{instance.base_url}/a", target)
        first = len(instance.requests)
        fetch(f"{instance.base_url}/a", target)
        assert len(instance.requests) == first, "refetched an asset already on disk"

    def test_corrupt_download_is_removed_not_left_behind(self, server, tmp_path):
        """The failure this prevents: a truncated file that a later run mistakes for a
        complete one, and that only fails at model-load time."""
        instance = self._serve(server, b"truncated")
        target = tmp_path / "model.gguf"

        with pytest.raises(DownloadError, match="sha256"):
            fetch(f"{instance.base_url}/m", target, expect_sha256="00" * 32)

        assert not target.exists()
        assert not target.with_suffix(".gguf.part").exists()

    def test_wrong_size_is_rejected(self, server, tmp_path):
        instance = self._serve(server, b"12345")
        with pytest.raises(DownloadError, match="bytes"):
            fetch(f"{instance.base_url}/m", tmp_path / "m.bin", expect_bytes=999)

    def test_nothing_is_left_in_place_until_verified(self, server, tmp_path):
        """The atomic .part -> replace: a killed process never leaves a half-file at the
        real path."""
        instance = self._serve(server, b"partial content here")
        target = tmp_path / "sub" / "dir" / "m.bin"

        with pytest.raises(DownloadError):
            fetch(f"{instance.base_url}/m", target, expect_bytes=99999)
        assert not target.exists()

    def test_progress_reports_reach_completion(self, server, tmp_path):
        body = b"y" * 300_000
        instance = self._serve(server, body)
        seen: list = []

        fetch(f"{instance.base_url}/big", tmp_path / "big.bin", on_progress=seen.append)

        assert seen, "no progress was reported"
        assert seen[-1].downloaded == len(body)
        assert seen[-1].fraction == 1.0

    def test_a_server_ignoring_range_restarts_instead_of_appending(self, server, tmp_path):
        """A server that answers 200 to a Range request sends the whole body. Appending
        that to an existing .part produces a corrupt archive that only fails much later,
        at decompression. Detect the missing 206 and start over."""
        body = b"complete-body-" * 100
        instance = self._serve(server, body)

        target = tmp_path / "m.bin"
        partial = target.with_suffix(".bin.part")
        partial.write_bytes(b"STALE-PREFIX")

        result = fetch(f"{instance.base_url}/m", target, resume=True)
        assert result.read_bytes() == body, "appended to a stale partial file"

    def test_unreachable_host_raises_download_error(self, tmp_path):
        with pytest.raises(DownloadError, match="cannot fetch"):
            fetch("http://127.0.0.1:1/nope", tmp_path / "x.bin", timeout=1.0)
