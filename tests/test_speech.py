"""Protocols, the shared resolver, and the parts of the providers that run without models.

The engines themselves need weights that are not present in CI, so what is tested here is
everything around them: the lifecycle contract, the resolver's fallback behaviour, and the
pure audio arithmetic. A fake provider stands in for an engine — not to mock away the
engine, but because the resolver's job is precisely to be indifferent to which one it has.
"""

from __future__ import annotations

import json
import math
import struct

import pytest
from ml_stack.media import wav
from ml_stack.speech import (
    ASRProvider,
    EnergyVAD,
    NoProviderAvailable,
    ProviderHealth,
    Registry,
    Speech,
    SystemTTS,
    TTSProvider,
    Transcript,
    VADProvider,
    pcm_to_floats,
    rms,
)
from ml_stack.speech.vad import _merge_regions


def tone(seconds: float, *, rate: int = 16000, amplitude: float = 0.5, hz: float = 220.0) -> bytes:
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * hz * i / rate))
        for i in range(int(rate * seconds))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


def silence(seconds: float, *, rate: int = 16000) -> bytes:
    return b"\x00\x00" * int(rate * seconds)


class FakeASR:
    """An engine stand-in. `fail_on` decides whether it constructs or starts cleanly."""

    def __init__(self, name: str = "fake", *, fail_on: str | None = None, text: str = "hello"):
        self.name = name
        self.fail_on = fail_on
        self.text = text
        self.started = False
        self.stopped = False
        if fail_on == "construct":
            raise RuntimeError("construction failed")

    def probe(self):
        if self.fail_on == "probe":
            return ProviderHealth.missing("probe says no")
        return ProviderHealth.ok(self.name)

    def start(self):
        if self.fail_on == "start":
            raise RuntimeError("model weights are missing")
        self.started = True

    def stop(self):
        self.stopped = True

    def transcribe(self, audio, *, language=None):
        return Transcript(text=self.text, model=self.name)


class TestProtocols:
    def test_the_fake_satisfies_the_protocol(self):
        assert isinstance(FakeASR(), ASRProvider)

    def test_the_real_providers_satisfy_their_protocols(self):
        assert isinstance(SystemTTS(), TTSProvider)
        assert isinstance(EnergyVAD(), VADProvider)

    def test_something_missing_a_method_does_not(self):
        class Incomplete:
            name = "incomplete"

            def probe(self):
                return ProviderHealth.ok()

        assert not isinstance(Incomplete(), ASRProvider)

    def test_health_is_truthy_when_available(self):
        assert ProviderHealth.ok("m")
        assert not ProviderHealth.missing("no")

    def test_transcript_stringifies_to_its_text(self):
        assert str(Transcript(text="hello there")) == "hello there"

    def test_speech_duration_is_derived_from_the_pcm(self):
        one_second = Speech(pcm=b"\x00\x00" * 16000, sample_rate=16000)
        assert one_second.duration_s == pytest.approx(1.0)

    def test_speech_converts_to_a_wav_that_decodes_back(self):
        audio = Speech(pcm=tone(0.1), sample_rate=16000)
        pcm, info = wav.decode(audio.to_wav())
        assert pcm == audio.pcm
        assert info.sample_rate == 16000

    def test_zero_rate_does_not_divide_by_zero(self):
        assert Speech(pcm=b"\x00\x00", sample_rate=0).duration_s == 0.0


class TestRegistry:
    def test_it_resolves_a_named_provider(self):
        registry: Registry = Registry(kind="asr")
        registry.register("fake", FakeASR)
        provider = registry.resolve("fake")
        assert provider.name == "fake" and provider.started

    def test_it_caches_so_the_model_loads_once(self):
        registry: Registry = Registry(kind="asr")
        registry.register("fake", FakeASR)
        assert registry.resolve() is registry.resolve()

    def test_asking_for_a_different_name_replaces_the_cache(self):
        """Ignoring the name and returning the cached provider is the alternative, and it
        makes an explicit request silently do nothing."""
        registry: Registry = Registry(kind="asr")
        registry.register("a", lambda: FakeASR("a"))
        registry.register("b", lambda: FakeASR("b"))
        assert registry.resolve("a").name == "a"
        assert registry.resolve("b").name == "b"

    def test_reset_stops_the_cached_provider(self):
        registry: Registry = Registry(kind="asr")
        registry.register("fake", FakeASR)
        provider = registry.resolve()
        registry.reset()
        assert provider.stopped, "the model was dropped without being released"

    def test_auto_falls_through_a_provider_that_will_not_start(self):
        """Constructing proves nothing -- the weights load in start(), which is where a
        missing model or a wheel built for another architecture shows up."""
        registry: Registry = Registry(kind="asr")
        registry.register("broken", lambda: FakeASR("broken", fail_on="start"))
        registry.register("working", lambda: FakeASR("working"))
        assert registry.resolve().name == "working"

    def test_auto_falls_through_a_provider_that_will_not_construct(self):
        registry: Registry = Registry(kind="asr")
        registry.register("broken", lambda: FakeASR("broken", fail_on="construct"))
        registry.register("working", lambda: FakeASR("working"))
        assert registry.resolve().name == "working"

    def test_it_reports_every_failure_together(self):
        """One line saying 'nothing available' sends the reader to check each engine by
        hand; the accumulated reasons usually name the problem outright."""
        registry: Registry = Registry(kind="tts")
        registry.register("one", lambda: FakeASR("one", fail_on="start"))
        registry.register("two", lambda: FakeASR("two", fail_on="construct"))

        with pytest.raises(NoProviderAvailable) as excinfo:
            registry.resolve()
        message = str(excinfo.value)
        assert "one" in message and "two" in message
        assert "model weights are missing" in message

    def test_an_empty_registry_says_so(self):
        with pytest.raises(NoProviderAvailable, match="nothing registered"):
            Registry(kind="asr").resolve()

    def test_an_unknown_name_lists_what_is_registered(self):
        registry: Registry = Registry(kind="asr")
        registry.register("fake", FakeASR)
        with pytest.raises(NoProviderAvailable, match="registered:"):
            registry.resolve("nonexistent")

    def test_prefer_puts_a_provider_first(self):
        registry: Registry = Registry(kind="asr")
        registry.register("default", lambda: FakeASR("default"))
        registry.register("better", lambda: FakeASR("better"), prefer=True)
        assert registry.names()[0] == "better"

    def test_registering_twice_moves_rather_than_duplicates(self):
        registry: Registry = Registry(kind="asr")
        registry.register("a", FakeASR)
        registry.register("b", FakeASR)
        registry.register("a", FakeASR, prefer=True)
        assert registry.names() == ["a", "b"]

    def test_probe_all_reports_without_starting_anything(self):
        registry: Registry = Registry(kind="asr")
        registry.register("good", lambda: FakeASR("good"))
        registry.register("bad", lambda: FakeASR("bad", fail_on="probe"))
        results = registry.probe_all()
        assert results["good"].available
        assert not results["bad"].available

    def test_probe_all_survives_a_provider_that_cannot_construct(self):
        registry: Registry = Registry(kind="asr")
        registry.register("broken", lambda: FakeASR("broken", fail_on="construct"))
        assert not registry.probe_all()["broken"].available

    def test_a_provider_that_fails_to_stop_does_not_wedge_the_registry(self):
        """Refusing to reset guarantees a stuck process; the worst case otherwise is a
        leaked model."""

        class Unstoppable(FakeASR):
            def stop(self):
                raise RuntimeError("cannot release")

        registry: Registry = Registry(kind="asr")
        registry.register("stuck", Unstoppable)
        registry.resolve()
        registry.reset()  # must not raise
        assert registry._cached is None


class TestAudioArithmetic:
    def test_pcm_round_trips_through_floats(self):
        floats = pcm_to_floats(struct.pack("<4h", 0, 16384, -16384, 32767))
        assert floats[0] == 0.0
        assert floats[1] == pytest.approx(0.5, abs=1e-4)
        assert floats[2] == pytest.approx(-0.5, abs=1e-4)

    def test_a_trailing_odd_byte_is_dropped_rather_than_crashing(self):
        """Truncated audio arrives routinely from a cut-off stream."""
        assert len(pcm_to_floats(b"\x00\x00\x00")) == 1

    def test_rms_of_silence_is_zero(self):
        assert rms(pcm_to_floats(silence(0.1))) == 0.0

    def test_rms_of_a_tone_is_amplitude_over_root_two(self):
        assert rms(pcm_to_floats(tone(0.5, amplitude=0.5))) == pytest.approx(0.354, abs=0.02)


class TestEnergyVAD:
    def test_silence_is_not_speech(self):
        assert not EnergyVAD().detect(silence(0.5))

    def test_a_loud_tone_is(self):
        result = EnergyVAD().detect(tone(0.5))
        assert result and result.regions

    def test_empty_audio_is_not_speech(self):
        assert not EnergyVAD().detect(b"")

    def test_the_region_covers_roughly_the_loud_part(self):
        pcm = silence(0.2) + tone(0.5) + silence(0.4)
        regions = EnergyVAD(min_silence_ms=100).detect(pcm).regions
        assert regions
        assert regions[0].start_s == pytest.approx(0.2, abs=0.1)
        assert regions[0].end_s == pytest.approx(0.7, abs=0.15)

    def test_a_short_gap_between_words_does_not_split_the_utterance(self):
        """Ordinary speech has pauses longer than one frame; without bridging, every
        utterance comes back as a dozen fragments."""
        pcm = tone(0.3) + silence(0.05) + tone(0.3)
        assert len(EnergyVAD(min_silence_ms=300).detect(pcm).regions) == 1

    def test_a_long_gap_does_split_it(self):
        pcm = tone(0.3) + silence(0.8) + tone(0.3)
        assert len(EnergyVAD(min_silence_ms=200).detect(pcm).regions) == 2

    def test_a_brief_click_is_below_the_minimum_speech_length(self):
        pcm = silence(0.3) + tone(0.01) + silence(0.3)
        assert not EnergyVAD(min_speech_ms=200).detect(pcm).regions

    def test_a_higher_threshold_rejects_quiet_audio(self):
        quiet = tone(0.5, amplitude=0.01)
        assert not EnergyVAD(threshold=0.1).detect(quiet)
        assert EnergyVAD(threshold=0.001).detect(quiet)

    def test_energy_needs_no_model_and_always_probes_available(self):
        """It is the fallback: for a push-to-talk button, 'is there any sound' is the whole
        question, and loading a model to answer it is waste."""
        assert EnergyVAD().probe().available


class TestRegionMerging:
    def test_no_loud_frames_gives_no_regions(self):
        assert _merge_regions([False] * 10, frame_s=0.03, min_speech_ms=0, min_silence_ms=0) == ()

    def test_a_region_open_at_the_end_is_closed(self):
        regions = _merge_regions(
            [False, True, True], frame_s=0.1, min_speech_ms=0, min_silence_ms=100
        )
        assert regions and regions[-1].end_s == pytest.approx(0.3)

    def test_duration_is_the_difference(self):
        regions = _merge_regions(
            [True] * 5, frame_s=0.1, min_speech_ms=0, min_silence_ms=100
        )
        assert regions[0].duration_s == pytest.approx(0.5)


class TestSystemTTS:
    def test_it_reports_honestly_whether_it_can_run(self):
        health = SystemTTS().probe()
        assert isinstance(health.available, bool)
        if not health.available:
            assert "say" in health.detail or "espeak" in health.detail

    @pytest.mark.skipif(not SystemTTS().probe().available, reason="no system speech binary")
    def test_it_actually_produces_audio(self):
        """The last-resort provider is the one that must work: a device that cannot load
        any real model still needs to say so out loud."""
        speech = SystemTTS().synthesize("testing one two three")
        assert speech.duration_s > 0.2
        assert wav.decode(speech.to_wav())[1].sample_rate > 0


class FakeTTS:
    """A voice stand-in: half a second of a tone, whatever it is asked to say."""

    name = "fake-voice"

    def __init__(self, *, seconds: float = 0.5):
        self.seconds = seconds
        self.said: list[str] = []

    def probe(self):
        return ProviderHealth.ok("fake-voice")

    def start(self):
        return None

    def stop(self):
        return None

    def synthesize(self, text, *, voice=None):
        self.said.append(text)
        return Speech(pcm=tone(self.seconds), sample_rate=16000, voice=voice or "fake-voice")


def a_recording() -> bytes:
    """A WAV holding silence, a burst of noise and silence -- one speech region."""
    return wav.encode(silence(0.3) + tone(0.5) + silence(0.3), sample_rate=16000)


@pytest.fixture
def registered(monkeypatch):
    """A fake on each registry, since no engine is installed where the suite runs."""
    from ml_stack import speech as package
    from ml_stack.speech import Registry

    asr, tts, vad = Registry(kind="asr"), Registry(kind="tts"), Registry(kind="vad")
    asr.register("fake", lambda: FakeASR(text="a machine that can hear"))
    tts.register("fake-voice", FakeTTS)
    vad.register("energy", EnergyVAD)
    for name, registry in (("ASR", asr), ("TTS", tts), ("VAD", vad)):
        monkeypatch.setattr(package, name, registry)
    return asr, tts, vad


class TestTheLibraryFunctions:
    def test_providers_names_every_engine_and_the_one_auto_picks(self, registered):
        from ml_stack.speech.service import providers

        found = providers()
        assert found["asr"]["auto"] == "fake"
        assert [one["name"] for one in found["vad"]["providers"]] == ["energy"]
        assert found["asr"]["providers"][0]["available"]

    def test_providers_says_why_when_nothing_is_installed(self, monkeypatch):
        from ml_stack import speech as package
        from ml_stack.speech import Registry
        from ml_stack.speech.service import providers

        registry: Registry = Registry(kind="asr")
        registry.register("broken", lambda: FakeASR("broken", fail_on="probe"))
        monkeypatch.setattr(package, "ASR", registry)
        asr = providers()["asr"]
        assert asr["auto"] is None
        assert asr["providers"][0]["detail"] == "probe says no"

    def test_transcribe_goes_through_the_registry(self, registered, tmp_path):
        from ml_stack.speech.service import transcribe

        clip = tmp_path / "clip.wav"
        clip.write_bytes(a_recording())
        assert transcribe(clip).text == "a machine that can hear"

    def test_say_returns_audio_that_decodes(self, registered):
        from ml_stack.speech.service import say

        spoken = say("anything")
        assert spoken.duration_s == pytest.approx(0.5, abs=0.01)
        assert wav.decode(spoken.to_wav())[1].sample_rate == 16000

    def test_regions_reads_a_wav_without_ffmpeg(self, registered, tmp_path):
        """A mono 16-bit WAV is already what a VAD wants; converting it would need ffmpeg
        on a machine that may not have it."""
        from ml_stack.speech.service import regions

        clip = tmp_path / "clip.wav"
        clip.write_bytes(a_recording())
        found = regions(clip)
        assert found.speech
        assert found.regions[0].start_s == pytest.approx(0.3, abs=0.1)
        assert found.regions[0].end_s == pytest.approx(0.8, abs=0.15)

    def test_nothing_registered_says_so_rather_than_returning_nothing(self, monkeypatch):
        from ml_stack import speech as package
        from ml_stack.speech import Registry
        from ml_stack.speech.service import say

        monkeypatch.setattr(package, "TTS", Registry(kind="tts"))
        with pytest.raises(NoProviderAvailable):
            say("hello")


class TestTheDefaultRegistrations:
    def test_every_engine_that_needs_no_arguments_is_a_candidate(self, monkeypatch):
        from ml_stack import speech as package
        from ml_stack.speech import Registry, register_defaults

        for attr in ("ASR", "TTS", "VAD"):
            monkeypatch.setattr(package, attr, Registry(kind=attr.lower()))
        register_defaults()
        assert package.ASR.names() == ["faster-whisper", "transformers-whisper"]
        assert package.TTS.names() == ["system"]
        assert package.VAD.names() == ["energy", "silero"]

    def test_a_named_whisper_cpp_model_and_piper_voice_go_in_front(self, monkeypatch, tmp_path):
        """Neither takes a default path, so an engine that needs a file on disk is a
        candidate only once it has been told where the file is."""
        from ml_stack import speech as package
        from ml_stack.speech import Registry, register_defaults

        for attr in ("ASR", "TTS", "VAD"):
            monkeypatch.setattr(package, attr, Registry(kind=attr.lower()))
        monkeypatch.setenv("MLSTACK_WHISPER_CPP_MODEL", str(tmp_path / "ggml-base.bin"))
        monkeypatch.setenv("MLSTACK_PIPER_VOICE", str(tmp_path / "voice.onnx"))
        register_defaults()
        assert package.ASR.names()[0] == "whisper.cpp"
        assert package.TTS.names() == ["piper", "system"]


class TestTheCommand:
    def test_providers_prints_a_line_for_each(self, registered, capsys):
        from ml_stack.speech.cli import main

        assert main(["providers"]) == 0
        printed = capsys.readouterr().out
        assert "asr  (auto: fake)" in printed
        assert "fake-voice" in printed

    def test_providers_as_json(self, registered, capsys):
        from ml_stack.speech.cli import main

        assert main(["providers", "--json"]) == 0
        found = json.loads(capsys.readouterr().out)
        assert found["tts"]["auto"] == "fake-voice"

    def test_transcribe_prints_the_text(self, registered, tmp_path, capsys):
        from ml_stack.speech.cli import main

        clip = tmp_path / "clip.wav"
        clip.write_bytes(a_recording())
        assert main(["transcribe", str(clip)]) == 0
        assert capsys.readouterr().out.strip() == "a machine that can hear"

    def test_transcribe_json_carries_the_segments(self, registered, tmp_path, capsys):
        from ml_stack.speech.cli import main

        clip = tmp_path / "clip.wav"
        clip.write_bytes(a_recording())
        assert main(["transcribe", str(clip), "--json", "--language", "en"]) == 0
        got = json.loads(capsys.readouterr().out)
        assert got["text"] == "a machine that can hear" and got["model"] == "fake"

    def test_say_writes_a_wav_that_decodes(self, registered, tmp_path, capsys):
        from ml_stack.speech.cli import main

        out = tmp_path / "spoken" / "said.wav"
        assert main(["say", "the fleet is up", "--out", str(out)]) == 0
        pcm, info = wav.decode(out.read_bytes())
        assert info.sample_rate == 16000 and len(pcm) > 0
        assert str(out) in capsys.readouterr().out

    def test_regions_prints_start_and_end_seconds(self, registered, tmp_path, capsys):
        from ml_stack.speech.cli import main

        clip = tmp_path / "clip.wav"
        clip.write_bytes(a_recording())
        assert main(["regions", str(clip), "--json"]) == 0
        found = json.loads(capsys.readouterr().out)
        assert found["speech"] and len(found["regions"]) == 1
        assert found["regions"][0]["start_s"] == pytest.approx(0.3, abs=0.1)

    def test_silence_is_no_speech_rather_than_an_error(self, registered, tmp_path, capsys):
        from ml_stack.speech.cli import main

        clip = tmp_path / "quiet.wav"
        clip.write_bytes(wav.encode(silence(0.5), sample_rate=16000))
        assert main(["regions", str(clip)]) == 0
        assert capsys.readouterr().out.strip() == "no speech"

    def test_an_unknown_provider_is_one_line_and_exit_1(self, registered, tmp_path, capsys):
        from ml_stack.speech.cli import main

        clip = tmp_path / "clip.wav"
        clip.write_bytes(a_recording())
        assert main(["transcribe", str(clip), "--provider", "nonexistent"]) == 1
        said = capsys.readouterr()
        assert said.out == ""
        assert said.err.startswith("ml-stack-speech: ") and len(said.err.splitlines()) == 1

    def test_a_file_that_is_not_there_is_one_line_and_exit_1(self, registered, capsys):
        from ml_stack.speech.cli import main

        assert main(["transcribe", "/no/such/clip.wav"]) == 1
        assert len(capsys.readouterr().err.splitlines()) == 1

    def test_nothing_registered_is_one_line_and_exit_1(self, monkeypatch, capsys):
        from ml_stack import speech as package
        from ml_stack.speech import Registry
        from ml_stack.speech.cli import main

        monkeypatch.setattr(package, "TTS", Registry(kind="tts"))
        assert main(["say", "hello", "--out", "/tmp/never-written.wav"]) == 1
        assert "no tts provider" in capsys.readouterr().err

    @pytest.mark.parametrize("command", ["providers", "transcribe", "say", "regions"])
    def test_each_subcommand_has_help(self, command, capsys):
        from ml_stack.speech.cli import main

        with pytest.raises(SystemExit) as left:
            main([command, "--help"])
        assert left.value.code == 0
        assert capsys.readouterr().out.strip()
