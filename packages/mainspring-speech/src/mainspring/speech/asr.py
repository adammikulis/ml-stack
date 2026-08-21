"""Speech recognition providers.

Every provider here loads its model **once**, behind a lock. Two utterances arriving
together would otherwise each start their own load: several seconds and several gigabytes,
twice, for a model that is about to be identical.

Audio normalisation is ffmpeg's job rather than a format sniffer's. What arrives from a
browser, a phone or a messaging bot is whatever that client encodes -- webm/opus, m4a,
3gp, occasionally a WAV with a surprising sample rate -- and handing all of it to ffmpeg
with an explicit output format is both shorter and more correct than a decode table that
will always be missing a case.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from mainspring.speech.protocols import (
    DEFAULT_SAMPLE_RATE,
    ProviderError,
    ProviderHealth,
    Segment,
    Transcript,
    as_bytes,
)


class AudioConversionError(ProviderError):
    """ffmpeg could not decode the input."""


def to_wav_16k(audio: Path | str | bytes, *, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Anything ffmpeg understands -> mono PCM WAV at ``sample_rate``."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioConversionError(
            "ffmpeg is not on PATH. It is how arbitrary uploaded audio becomes something "
            "a model can read. On macOS: brew install ffmpeg"
        )

    data = as_bytes(audio)
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-ar", str(sample_rate), "-ac", "1", "-f", "wav", "pipe:1"],
        input=data,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:300]
        raise AudioConversionError(f"ffmpeg could not decode this audio: {detail}")
    return result.stdout


class _LoadedOnce:
    """Mixin: load the model once, behind a lock, however many callers arrive."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:  # another caller may have won the lock
                self._model = self._load()

    def stop(self) -> None:
        with self._lock:
            self._model = None

    def _load(self):  # pragma: no cover - subclass responsibility
        raise NotImplementedError

    def _require(self):
        if self._model is None:
            self.start()
        return self._model


class FasterWhisperASR(_LoadedOnce):
    """``faster-whisper``: CTranslate2 Whisper, fast on CPU.

    ``int8`` on CPU by default. The quality loss on speech is small and the speed
    difference is not -- a CPU-only box running fp32 Whisper is slower than real time,
    which makes it useless for anything interactive.
    """

    name = "faster-whisper"

    def __init__(
        self,
        model: str = "small.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size

    def probe(self) -> ProviderHealth:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            return ProviderHealth.missing(f"faster-whisper is not installed ({exc})")
        if shutil.which("ffmpeg") is None:
            return ProviderHealth.missing("ffmpeg is not on PATH")
        return ProviderHealth.ok(self.model)

    def _load(self):
        from faster_whisper import WhisperModel

        return WhisperModel(self.model, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio, *, language: str | None = None) -> Transcript:
        model = self._require()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            handle.write(to_wav_16k(audio))
            handle.flush()
            segments, info = model.transcribe(
                handle.name, beam_size=self.beam_size, language=language
            )
            collected = tuple(
                Segment(text=s.text.strip(), start_s=float(s.start), end_s=float(s.end))
                for s in segments
            )

        return Transcript(
            text=" ".join(s.text for s in collected).strip(),
            language=getattr(info, "language", language),
            duration_s=float(getattr(info, "duration", 0.0)),
            segments=collected,
            model=self.model,
        )


class WhisperCppASR:
    """``whisper.cpp``'s CLI, for machines with no Python ML stack at all.

    Handles one real quirk: some builds write ``<input>.json`` next to the input rather
    than at the requested output path.
    """

    name = "whisper.cpp"

    def __init__(self, model_path: Path | str, *, binary: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.binary = binary

    def _resolve_binary(self) -> str | None:
        for candidate in (self.binary, "whisper-cli", "whisper", "main"):
            if candidate and (found := shutil.which(candidate)):
                return found
        return None

    def probe(self) -> ProviderHealth:
        if self._resolve_binary() is None:
            return ProviderHealth.missing("no whisper-cli/whisper binary on PATH")
        if not self.model_path.is_file():
            return ProviderHealth.missing(f"no model at {self.model_path}")
        return ProviderHealth.ok(self.model_path.name)

    def start(self) -> None:
        health = self.probe()
        if not health:
            raise ProviderError(health.detail)

    def stop(self) -> None:
        return None

    def transcribe(self, audio, *, language: str | None = None) -> Transcript:
        binary = self._resolve_binary()
        if binary is None:
            raise ProviderError("no whisper binary on PATH")

        with tempfile.TemporaryDirectory() as workdir:
            wav_path = Path(workdir) / "input.wav"
            wav_path.write_bytes(to_wav_16k(audio))

            argv = [binary, "--model", str(self.model_path), "--file", str(wav_path),
                    "--output-json", "--no-prints"]
            if language:
                argv += ["--language", language]

            result = subprocess.run(argv, capture_output=True, text=True)
            if result.returncode != 0:
                raise ProviderError(
                    f"whisper exited {result.returncode}: {result.stderr.strip()[:300]}"
                )

            produced = next(Path(workdir).glob("*.json"), None)
            if produced is None:
                raise ProviderError("whisper reported success but wrote no JSON output")

            import json

            payload = json.loads(produced.read_text(encoding="utf-8"))

        segments = tuple(
            Segment(
                text=str(item.get("text", "")).strip(),
                start_s=float(item.get("offsets", {}).get("from", 0)) / 1000.0,
                end_s=float(item.get("offsets", {}).get("to", 0)) / 1000.0,
            )
            for item in payload.get("transcription", [])
        )
        return Transcript(
            text=" ".join(s.text for s in segments).strip(),
            language=language,
            segments=segments,
            model=self.model_path.name,
        )


class TransformersWhisperASR(_LoadedOnce):
    """Whisper through ``transformers``. The widest model selection, the heaviest install."""

    name = "transformers-whisper"

    def __init__(self, model: str = "openai/whisper-small.en", *, device: str | None = None) -> None:
        super().__init__()
        self.model = model
        self.device = device

    def probe(self) -> ProviderHealth:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return ProviderHealth.missing(f"transformers/torch not installed ({exc})")
        return ProviderHealth.ok(self.model)

    def _resolve_dtype(self, torch):
        """float32 on MPS, deliberately.

        In float16 on MPS, Whisper decodes a token or two and stops -- returning a
        *fragment* rather than an error. A short transcript is not distinguishable from a
        short utterance by anything downstream, so this one is worth the memory.
        """
        device = str(self.device or "")
        if device.startswith("mps") or (
            not device and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            return torch.float32
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def _pick_device(self, torch):
        """Pick a torch device without reaching into the lab tier.

        `mainspring.backend.resolve_torch_device` does the same job with more care, but a
        host-tier package must not depend on a lab-tier one -- torch is optional here and
        mandatory there, and inverting that would drag a training dependency onto every
        machine that only wants to transcribe.
        """
        if self.device:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self):
        import torch
        from transformers import pipeline

        device = self._pick_device(torch)
        return pipeline(
            "automatic-speech-recognition",
            model=self.model,
            device=device,
            torch_dtype=self._resolve_dtype(torch),
        )

    def transcribe(self, audio, *, language: str | None = None) -> Transcript:
        model = self._require()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as handle:
            handle.write(to_wav_16k(audio))
            handle.flush()
            kwargs = {"generate_kwargs": {"language": language}} if language else {}
            result = model(handle.name, return_timestamps=True, **kwargs)

        chunks = result.get("chunks") or []
        segments = tuple(
            Segment(
                text=str(c.get("text", "")).strip(),
                start_s=float((c.get("timestamp") or (0, 0))[0] or 0),
                end_s=float((c.get("timestamp") or (0, 0))[1] or 0),
            )
            for c in chunks
        )
        return Transcript(
            text=str(result.get("text", "")).strip(),
            language=language,
            segments=segments,
            model=self.model,
        )
