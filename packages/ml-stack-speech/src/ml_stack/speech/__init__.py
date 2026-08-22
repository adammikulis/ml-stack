"""Speech recognition, synthesis and voice activity, behind three protocols.

Host tier. The engines are optional extras -- install what you need.

One registry serves all three modalities, and its auto-detection *starts* each candidate
rather than merely constructing it:

    from ml_stack.speech import ASR, TTS, FasterWhisperASR, SystemTTS

    ASR.register("faster-whisper", lambda: FasterWhisperASR("small.en"))
    TTS.register("system", SystemTTS)

    text = ASR.resolve().transcribe("voice-note.ogg").text
    audio = TTS.resolve().synthesize("ready").to_wav()

Constructing a provider proves nothing: the weights load in ``start()``, which is where a
missing model, a blocked download or a wheel built for another architecture shows up.
"""

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

# The two that need no configuration are registered up front, so a fresh install has a
# working fallback before the caller has decided anything. Both are last-resort quality;
# anything registered later goes ahead of them only if the caller says `prefer=True`.
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
