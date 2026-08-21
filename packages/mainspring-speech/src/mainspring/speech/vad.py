"""Voice activity detection.

Two providers with genuinely different characters, not two implementations of one thing:

* **Energy** is a few lines of arithmetic on the PCM. It cannot tell speech from a slammed
  door, but it needs no model, runs in microseconds, and works on any hardware.
* **Silero** is a small neural model that actually distinguishes speech from noise.

The energy detector is the fallback rather than a toy: for a push-to-talk button, "is
there any sound at all" is the whole question, and loading a model to answer it is waste.
"""

from __future__ import annotations

import array
import math
import threading

from mainspring.speech.protocols import (
    DEFAULT_SAMPLE_RATE,
    ProviderError,
    ProviderHealth,
    SpeechRegion,
    VoiceActivity,
)

FRAME_MS = 30


def pcm_to_floats(pcm: bytes) -> list[float]:
    """Little-endian int16 -> floats in [-1, 1]. Trailing odd byte dropped."""
    usable = len(pcm) - (len(pcm) % 2)
    samples = array.array("h")
    samples.frombytes(pcm[:usable])
    if array.array("h").itemsize != 2:  # pragma: no cover - not seen in practice
        raise ProviderError("this platform's short is not 16-bit")
    import sys

    if sys.byteorder == "big":
        samples.byteswap()
    return [s / 32768.0 for s in samples]


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class EnergyVAD:
    """Amplitude threshold over short frames. No model, no dependencies."""

    name = "energy"

    def __init__(
        self,
        *,
        threshold: float = 0.02,
        min_speech_ms: int = 100,
        min_silence_ms: int = 300,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

    def probe(self) -> ProviderHealth:
        return ProviderHealth.ok("energy")

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def detect(self, pcm: bytes, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> VoiceActivity:
        samples = pcm_to_floats(pcm)
        if not samples:
            return VoiceActivity(False)

        frame_size = max(1, int(sample_rate * FRAME_MS / 1000))
        loud = [
            rms(samples[i : i + frame_size]) >= self.threshold
            for i in range(0, len(samples), frame_size)
        ]
        regions = _merge_regions(
            loud,
            frame_s=frame_size / sample_rate,
            min_speech_ms=self.min_speech_ms,
            min_silence_ms=self.min_silence_ms,
        )
        return VoiceActivity(
            speech=bool(regions),
            confidence=rms(samples) / self.threshold if self.threshold else 0.0,
            regions=regions,
        )


class SileroVAD:
    """Silero VAD: small, fast, and actually distinguishes speech from noise."""

    name = "silero"

    def __init__(
        self,
        *,
        threshold: float = 0.3,
        min_speech_ms: int = 100,
        min_silence_ms: int = 300,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self._model = None
        self._lock = threading.Lock()

    def probe(self) -> ProviderHealth:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            return ProviderHealth.missing(f"torch is not installed ({exc})")
        return ProviderHealth.ok("silero_vad")

    def start(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                import torch

                model, _utils = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", trust_repo=True
                )
                self._model = model

    def stop(self) -> None:
        with self._lock:
            self._model = None

    def detect(self, pcm: bytes, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> VoiceActivity:
        import torch

        if self._model is None:
            self.start()

        # Silero is trained at 16 kHz and 8 kHz only. Running it on a 44.1 kHz stream
        # returns confident nonsense rather than an error.
        if sample_rate not in (8000, 16000):
            raise ProviderError(
                f"Silero VAD supports 8 kHz and 16 kHz only, got {sample_rate}. "
                "Resample first -- it will not error on other rates, it will be wrong."
            )

        samples = pcm_to_floats(pcm)
        if not samples:
            return VoiceActivity(False)

        window = 512 if sample_rate == 16000 else 256
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(samples) - window + 1, window):
                chunk = torch.tensor(samples[start : start + window], dtype=torch.float32)
                scores.append(float(self._model(chunk, sample_rate).item()))

        if not scores:
            return VoiceActivity(False)

        loud = [s >= self.threshold for s in scores]
        regions = _merge_regions(
            loud,
            frame_s=window / sample_rate,
            min_speech_ms=self.min_speech_ms,
            min_silence_ms=self.min_silence_ms,
        )
        return VoiceActivity(bool(regions), confidence=max(scores), regions=regions)


def _merge_regions(
    loud: list[bool],
    *,
    frame_s: float,
    min_speech_ms: int,
    min_silence_ms: int,
) -> tuple[SpeechRegion, ...]:
    """Turn per-frame decisions into regions, bridging short gaps.

    The gap bridging is what makes this usable: ordinary speech contains pauses between
    words that are longer than one frame, and without bridging every utterance comes back
    as a dozen fragments.
    """
    if not any(loud):
        return ()

    gap_frames = max(1, int(min_silence_ms / 1000 / frame_s)) if frame_s > 0 else 1
    min_frames = max(1, int(min_speech_ms / 1000 / frame_s)) if frame_s > 0 else 1

    regions: list[tuple[int, int]] = []
    start: int | None = None
    silence = 0

    for index, is_loud in enumerate(loud):
        if is_loud:
            if start is None:
                start = index
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= gap_frames:
                regions.append((start, index - silence + 1))
                start = None
                silence = 0

    if start is not None:
        regions.append((start, len(loud)))

    return tuple(
        SpeechRegion(start_s=a * frame_s, end_s=b * frame_s)
        for a, b in regions
        if (b - a) >= min_frames
    )
