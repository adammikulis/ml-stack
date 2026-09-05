"""What a transcript, a spoken line or a set of speech regions costs a caller: one call.

Everything that drives speech from outside this package -- ``ml-stack-speech``, the MCP
tools, the fleet daemon's ``/speech/*`` routes -- calls these, so a provider is chosen and
cached in one place.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from ml_stack.speech.asr import to_wav_16k
from ml_stack.speech.protocols import (
    Speech,
    Transcript,
    VoiceActivity,
    as_bytes,
)

__all__ = ["as_json", "providers", "regions", "say", "spoken_pcm", "transcribe"]


def _registries() -> dict[str, Any]:
    from ml_stack.speech import ASR, TTS, VAD

    return {"asr": ASR, "tts": TTS, "vad": VAD}


def providers() -> dict[str, dict[str, Any]]:
    """Every provider of every protocol with what its probe said, and which ``auto`` picks."""
    out: dict[str, dict[str, Any]] = {}
    for kind, registry in _registries().items():
        found = [
            {"name": name, "available": health.available, "detail": health.detail,
             "model": health.model}
            for name, health in registry.probe_all().items()
        ]
        chosen = next((one["name"] for one in found if one["available"]), None)
        out[kind] = {"auto": chosen, "providers": found}
    return out


def transcribe(audio: Path | str | bytes, *, provider: str | None = None,
               language: str | None = None) -> Transcript:
    """``audio`` (a path or the bytes of a file) as text, with per-segment times."""
    from ml_stack.speech import ASR

    return ASR.resolve(provider or None).transcribe(audio, language=language)


def say(text: str, *, provider: str | None = None, voice: str | None = None) -> Speech:
    """``text`` spoken, as PCM with the rate it was produced at."""
    from ml_stack.speech import TTS

    return TTS.resolve(provider or None).synthesize(text, voice=voice)


def regions(audio: Path | str | bytes, *, provider: str | None = None) -> VoiceActivity:
    """Where in ``audio`` somebody is speaking."""
    from ml_stack.speech import VAD

    pcm, rate = spoken_pcm(audio)
    return VAD.resolve(provider or None).detect(pcm, sample_rate=rate)


def spoken_pcm(audio: Path | str | bytes) -> tuple[bytes, int]:
    """``audio`` as mono 16-bit PCM and its sample rate, through ffmpeg unless it is
    already a mono 16-bit WAV."""
    from ml_stack.media import wav

    data = as_bytes(audio)
    try:
        pcm, info = wav.decode(data)
        if info.channels == 1 and info.sample_width == 2:
            return pcm, info.sample_rate
    except Exception:  # noqa: BLE001 - anything unreadable goes through ffmpeg
        pass
    pcm, info = wav.decode(to_wav_16k(data))
    return pcm, info.sample_rate


def as_json(value: Any) -> Any:
    """A speech dataclass as JSON-safe fields."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value
