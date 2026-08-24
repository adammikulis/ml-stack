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

VAD.register("energy", EnergyVAD)
TTS.register("system", SystemTTS)

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
