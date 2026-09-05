"""Speech recognition, synthesis and voice activity, behind three protocols."""

from __future__ import annotations

from ml_stack.speech.asr import (
    AudioConversionError,
    FasterWhisperASR,
    TransformersWhisperASR,
    WhisperCppASR,
    to_wav_16k,
)
from ml_stack.speech.protocols import (
    DEFAULT_SAMPLE_RATE,
    ASRProvider,
    NoProviderAvailable,
    ProviderError,
    ProviderHealth,
    Segment,
    Speech,
    SpeechRegion,
    StreamingASR,
    TTSProvider,
    Transcript,
    VADProvider,
    VoiceActivity,
)
from ml_stack.speech.registry import Registry
from ml_stack.speech.tts import KokoroOnnxTTS, PiperTTS, SystemTTS
from ml_stack.speech.vad import EnergyVAD, SileroVAD, pcm_to_floats, rms

ASR: Registry = Registry(kind="asr")
TTS: Registry = Registry(kind="tts")
VAD: Registry = Registry(kind="vad")


def register_defaults() -> None:
    """Put every provider that needs no arguments on its registry, ahead of them the ones
    named by ``MLSTACK_WHISPER_CPP_MODEL``, ``MLSTACK_PIPER_VOICE``, and
    ``MLSTACK_KOKORO_MODEL`` with ``MLSTACK_KOKORO_VOICES``."""
    import os

    ASR.register("faster-whisper", FasterWhisperASR)
    ASR.register("transformers-whisper", TransformersWhisperASR)
    if model := os.environ.get("MLSTACK_WHISPER_CPP_MODEL", ""):
        ASR.register("whisper.cpp", lambda: WhisperCppASR(model), prefer=True)

    TTS.register("system", SystemTTS)
    if voice := os.environ.get("MLSTACK_PIPER_VOICE", ""):
        TTS.register("piper", lambda: PiperTTS(voice), prefer=True)
    kokoro = os.environ.get("MLSTACK_KOKORO_MODEL", "")
    voices = os.environ.get("MLSTACK_KOKORO_VOICES", "")
    if kokoro and voices:
        TTS.register("kokoro-onnx", lambda: KokoroOnnxTTS(kokoro, voices), prefer=True)

    VAD.register("energy", EnergyVAD)
    VAD.register("silero", SileroVAD)


register_defaults()

__all__ = [
    "ASR",
    "DEFAULT_SAMPLE_RATE",
    "TTS",
    "VAD",
    "ASRProvider",
    "AudioConversionError",
    "EnergyVAD",
    "FasterWhisperASR",
    "KokoroOnnxTTS",
    "NoProviderAvailable",
    "PiperTTS",
    "ProviderError",
    "ProviderHealth",
    "Registry",
    "register_defaults",
    "Segment",
    "SileroVAD",
    "Speech",
    "SpeechRegion",
    "StreamingASR",
    "SystemTTS",
    "TTSProvider",
    "Transcript",
    "TransformersWhisperASR",
    "VADProvider",
    "VoiceActivity",
    "WhisperCppASR",
    "pcm_to_floats",
    "rms",
    "to_wav_16k",
]
