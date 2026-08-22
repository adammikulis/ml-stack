"""What a speech provider has to be, whichever engine is behind it.

Three protocols -- recognition, synthesis, voice activity -- each with the same lifecycle:
``probe`` to ask whether it could work, ``start`` to load the model, ``stop`` to release
it. Keeping load separate from construction matters because loading is the expensive part
and the part that fails; a provider that constructs fine and then cannot load its weights
is the normal case, not an edge one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

DEFAULT_SAMPLE_RATE = 16000
"""What speech models want. Resample once, at the edge."""


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Whether a provider could work here, and why not if not."""

    available: bool
    detail: str = ""
    model: str | None = None

    def __bool__(self) -> bool:
        return self.available

    @classmethod
    def ok(cls, model: str | None = None, detail: str = "") -> "ProviderHealth":
        return cls(True, detail, model)

    @classmethod
    def missing(cls, detail: str) -> "ProviderHealth":
        return cls(False, detail)


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    language: str | None = None
    duration_s: float = 0.0
    segments: tuple[Segment, ...] = ()
    model: str | None = None

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Speech:
    """Synthesised audio as raw PCM plus the rate it was produced at.

    PCM rather than a WAV blob: a caller streaming to a speaker does not want a container,
    and one that needs a file gets it from ``ml_stack.media.wav.encode``.
    """

    pcm: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    sample_width: int = 2
    voice: str | None = None

    @property
    def duration_s(self) -> float:
        frame = max(self.channels * self.sample_width, 1)
        return len(self.pcm) / frame / self.sample_rate if self.sample_rate else 0.0

    def to_wav(self) -> bytes:
        from ml_stack.media import wav

        return wav.encode(
            self.pcm,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class VoiceActivity:
    """Where speech was detected, and how confident that is."""

    speech: bool
    confidence: float = 0.0
    regions: tuple[SpeechRegion, ...] = field(default=())

    def __bool__(self) -> bool:
        return self.speech


@runtime_checkable
class ASRProvider(Protocol):
    """Speech to text."""

    name: str

    def probe(self) -> ProviderHealth: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def transcribe(self, audio: Path | str | bytes, *, language: str | None = None) -> Transcript: ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text to speech."""

    name: str

    def probe(self) -> ProviderHealth: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def synthesize(self, text: str, *, voice: str | None = None) -> Speech: ...


@runtime_checkable
class VADProvider(Protocol):
    """Is anyone speaking?"""

    name: str

    def probe(self) -> ProviderHealth: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def detect(self, pcm: bytes, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> VoiceActivity: ...


class StreamingASR(Protocol):
    """An ASR provider that can transcribe while audio is still arriving.

    Separate from ``ASRProvider`` because most cannot, and a ``stream`` method that
    secretly buffers the whole utterance to a temporary file before transcribing is worse
    than not having one: it presents a latency guarantee it does not keep.
    """

    def stream(self, chunks: Iterable[bytes], *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Iterator[Transcript]: ...


class ProviderError(RuntimeError):
    """A provider could not start, or failed on a request."""


class NoProviderAvailable(ProviderError):
    """Nothing on the candidate list could be made to work."""


def as_bytes(audio: Path | str | bytes) -> bytes:
    """Accept a path or raw bytes. Callers pass both, routinely."""
    if isinstance(audio, bytes):
        return audio
    return Path(audio).read_bytes()
