"""Speech synthesis providers."""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from ml_stack.speech.protocols import (
    ProviderError,
    ProviderHealth,
    Speech,
)

KOKORO_SAMPLE_RATE = 24000
PIPER_DEFAULT_RATE = 22050


class KokoroOnnxTTS:
    """Kokoro through ``kokoro-onnx``. No PyTorch, so it installs anywhere."""

    name = "kokoro-onnx"

    def __init__(
        self,
        model_path: Path | str,
        voices_path: Path | str,
        *,
        default_voice: str = "af_heart",
    ) -> None:
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)
        self.default_voice = default_voice
        self._engine = None
        self._lock = threading.Lock()

    def probe(self) -> ProviderHealth:
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError as exc:
            return ProviderHealth.missing(f"kokoro-onnx is not installed ({exc})")
        for path in (self.model_path, self.voices_path):
            if not path.is_file():
                return ProviderHealth.missing(f"missing asset: {path}")
        return ProviderHealth.ok(self.model_path.name)

    def start(self) -> None:
        if self._engine is not None:
            return
        health = self.probe()
        if not health:
            raise ProviderError(health.detail)
        with self._lock:
            if self._engine is None:
                from kokoro_onnx import Kokoro

                self._engine = Kokoro(str(self.model_path), str(self.voices_path))

    def stop(self) -> None:
        with self._lock:
            self._engine = None

    def synthesize(self, text: str, *, voice: str | None = None) -> Speech:
        if self._engine is None:
            self.start()
        chosen = voice or self.default_voice
        samples, rate = self._engine.create(text, voice=chosen, lang=_voice_language(chosen))
        return Speech(pcm=_float_to_pcm16(samples), sample_rate=int(rate), voice=chosen)


class PiperTTS:
    """Piper, through its binary or its Python module."""

    name = "piper"

    def __init__(
        self,
        voice_path: Path | str,
        *,
        binary: str | None = None,
        espeak_data: Path | str | None = None,
    ) -> None:
        self.voice_path = Path(voice_path)
        self.binary = binary
        self.espeak_data = Path(espeak_data) if espeak_data else None

    def _resolve_binary(self) -> str | None:
        for candidate in (self.binary, "piper"):
            if candidate and (found := shutil.which(candidate)):
                return found
        return None

    def probe(self) -> ProviderHealth:
        if not self.voice_path.is_file():
            return ProviderHealth.missing(f"no voice at {self.voice_path}")
        config = self.voice_path.with_suffix(self.voice_path.suffix + ".json")
        if not config.is_file():
            return ProviderHealth.missing(
                f"voice {self.voice_path.name} has no config at {config.name}; "
                "piper needs both files"
            )
        if self._resolve_binary() is None:
            try:
                import piper  # noqa: F401
            except ImportError:
                return ProviderHealth.missing("no piper binary on PATH and no piper module")
        return ProviderHealth.ok(self.voice_path.stem)

    def start(self) -> None:
        health = self.probe()
        if not health:
            raise ProviderError(health.detail)

    def stop(self) -> None:
        return None

    def synthesize(self, text: str, *, voice: str | None = None) -> Speech:
        binary = self._resolve_binary()
        if binary is not None:
            return self._synthesize_binary(binary, text)
        return self._synthesize_module(text)

    def _synthesize_binary(self, binary: str, text: str) -> Speech:
        with tempfile.TemporaryDirectory() as workdir:
            out = Path(workdir) / "speech.wav"
            argv = [binary, "--model", str(self.voice_path), "--output_file", str(out)]
            if self.espeak_data:
                argv += ["--espeak_data", str(self.espeak_data)]

            result = subprocess.run(argv, input=text.encode("utf-8"), capture_output=True)
            if result.returncode != 0 or not out.is_file():
                raise ProviderError(
                    f"piper produced no audio (exit {result.returncode}): "
                    f"{result.stderr.decode('utf-8', 'replace').strip()[:200]}"
                )
            return self._from_wav(out.read_bytes())

    def _synthesize_module(self, text: str) -> Speech:
        import io
        import wave as wave_module

        from piper import PiperVoice

        voice = PiperVoice.load(
            str(self.voice_path), config_path=str(self.voice_path) + ".json"
        )
        buffer = io.BytesIO()
        with wave_module.open(buffer, "wb") as handle:
            voice.synthesize_wav(text, handle)
        return self._from_wav(buffer.getvalue())

    def _from_wav(self, data: bytes) -> Speech:
        from ml_stack.media import wav

        pcm, info = wav.decode(data)
        return Speech(
            pcm=pcm,
            sample_rate=info.sample_rate or PIPER_DEFAULT_RATE,
            channels=info.channels,
            sample_width=info.sample_width,
            voice=self.voice_path.stem,
        )


class SystemTTS:
    """The operating system's own voice. Always last, and always there."""

    name = "system"

    def __init__(self, *, voice: str | None = None) -> None:
        self.voice = voice

    def _command(self) -> list[str] | None:
        if platform.system() == "Darwin" and shutil.which("say"):
            return ["say"]
        if found := shutil.which("espeak-ng") or shutil.which("espeak"):
            return [found]
        return None

    def probe(self) -> ProviderHealth:
        if self._command() is None:
            return ProviderHealth.missing("no system speech binary (say / espeak-ng)")
        return ProviderHealth.ok("system")

    def start(self) -> None:
        health = self.probe()
        if not health:
            raise ProviderError(health.detail)

    def stop(self) -> None:
        return None

    def synthesize(self, text: str, *, voice: str | None = None) -> Speech:
        from ml_stack.media import wav

        command = self._command()
        if command is None:
            raise ProviderError("no system speech binary")

        with tempfile.TemporaryDirectory() as workdir:
            out = Path(workdir) / "speech.wav"
            argv = list(command)
            if command[0].endswith("say"):
                argv += ["-o", str(out), "--data-format=LEI16@22050"]
                if chosen := (voice or self.voice):
                    argv += ["-v", chosen]
                argv.append(text)
            else:
                argv += ["-w", str(out)]
                if chosen := (voice or self.voice):
                    argv += ["-v", chosen]
                argv.append(text)

            result = subprocess.run(argv, capture_output=True)
            if result.returncode != 0 or not out.is_file():
                raise ProviderError(
                    f"system TTS produced no audio (exit {result.returncode})"
                )
            pcm, info = wav.decode(out.read_bytes())

        return Speech(
            pcm=pcm,
            sample_rate=info.sample_rate,
            channels=info.channels,
            sample_width=info.sample_width,
            voice=voice or self.voice or "system",
        )


def _float_to_pcm16(samples) -> bytes:
    """float32 in [-1, 1] -> little-endian int16."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32)
    return (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _voice_language(voice: str) -> str:
    """Kokoro encodes the language in the first character of the voice id."""
    return {
        "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr",
        "h": "hi", "i": "it", "j": "ja", "p": "pt-br", "z": "cmn",
    }.get(voice[:1], "en-us")
