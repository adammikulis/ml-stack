"""Media bytes: WAV containers, image formats, asset downloads.

Device tier: importable with nothing but the standard library, so it runs on embedded
targets and inside host applications that cannot add wheels. It must not grow a
dependency on Pillow, soundfile, httpx or numpy; the parts that genuinely need those --
resizing, resampling -- live in the host-tier packages.
"""

from __future__ import annotations

from mainspring.media.download import (
    DownloadError,
    Progress,
    ProgressFn,
    bar,
    fetch,
)
from mainspring.media.image import (
    ImageError,
    from_data_url,
    kind,
    mime,
    probe_png,
    to_data_url,
)
from mainspring.media.wav import (
    DEFAULT_SAMPLE_RATE,
    WavError,
    WavInfo,
    decode,
    duration_s,
    encode,
    header,
    streaming_header,
)

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "DownloadError",
    "ImageError",
    "Progress",
    "ProgressFn",
    "WavError",
    "WavInfo",
    "bar",
    "decode",
    "duration_s",
    "encode",
    "fetch",
    "from_data_url",
    "header",
    "kind",
    "mime",
    "probe_png",
    "streaming_header",
    "to_data_url",
]
