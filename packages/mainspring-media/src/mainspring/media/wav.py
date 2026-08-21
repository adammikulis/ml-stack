"""PCM <-> WAV, with no dependencies.

The header is packed by hand rather than through ``wave`` so that ``encode`` can write a
streaming placeholder length: ``wave`` insists on seeking back to patch the size, which a
pipe does not support.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

DEFAULT_SAMPLE_RATE = 16000
"""16 kHz mono is what every ASR model here wants. Resample once, at the edge."""

_HEADER_BYTES = 44
_PCM_FORMAT = 1


class WavError(ValueError):
    """Malformed WAV data."""


@dataclass(frozen=True, slots=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width: int
    frames: int

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frames / self.sample_rate


def header(
    data_len: int,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """A 44-byte canonical PCM RIFF header for ``data_len`` bytes of samples.

    Pass ``data_len=0xFFFFFFFF - _HEADER_BYTES`` (see ``streaming_header``) when the
    length is not yet known.
    """
    if channels < 1:
        raise WavError(f"channels must be >= 1, got {channels}")
    if sample_width not in (1, 2, 3, 4):
        raise WavError(f"unsupported sample width {sample_width}")

    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", _HEADER_BYTES - 8 + data_len),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, _PCM_FORMAT, channels, sample_rate,
                        byte_rate, block_align, sample_width * 8),
            b"data",
            struct.pack("<I", data_len),
        )
    )


def streaming_header(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """A header claiming an effectively-unbounded length, for a pipe.

    Players read until the stream ends. This is what lets TTS start playing before
    synthesis finishes -- the alternative is buffering the whole utterance to learn its
    length, which is exactly the latency the streaming path exists to avoid.
    """
    return header(
        0xFFFFFFFF - _HEADER_BYTES,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def encode(
    pcm: bytes,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw little-endian PCM in a WAV container."""
    return header(
        len(pcm),
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    ) + pcm


def decode(data: bytes) -> tuple[bytes, WavInfo]:
    """Pull the PCM payload and format out of a WAV file.

    Walks the chunk list rather than assuming a 44-byte header: real encoders emit
    ``LIST``/``fact`` chunks between ``fmt `` and ``data``, and skipping a fixed 44 bytes
    turns those into a burst of noise at the start of the audio.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file")

    fmt: tuple[int, int, int] | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        (chunk_len,) = struct.unpack("<I", data[offset + 4:offset + 8])
        body = offset + 8

        if chunk_id == b"fmt " and chunk_len >= 16:
            audio_format, channels, sample_rate = struct.unpack("<HHI", data[body:body + 8])
            (bits,) = struct.unpack("<H", data[body + 14:body + 16])
            if audio_format != _PCM_FORMAT:
                raise WavError(f"only uncompressed PCM is supported, got format {audio_format}")
            fmt = (sample_rate, channels, bits // 8)

        elif chunk_id == b"data":
            if fmt is None:
                raise WavError("data chunk precedes fmt chunk")
            sample_rate, channels, sample_width = fmt
            # Trust the file's length only as far as the buffer actually goes: a
            # truncated download otherwise raises a struct error far from the cause.
            pcm = data[body:body + min(chunk_len, len(data) - body)]
            frame_size = max(channels * sample_width, 1)
            return pcm, WavInfo(sample_rate, channels, sample_width, len(pcm) // frame_size)

        offset = body + chunk_len + (chunk_len & 1)  # chunks are word-aligned

    raise WavError("no data chunk found")


def duration_s(data: bytes) -> float:
    """Duration of a WAV file in seconds."""
    return decode(data)[1].duration_s
