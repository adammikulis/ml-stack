"""Protocols, the shared resolver, and the parts of the providers that run without models.

The engines themselves need weights that are not present in CI, so what is tested here is
everything around them: the lifecycle contract, the resolver's fallback behaviour, and the
pure audio arithmetic. A fake provider stands in for an engine — not to mock away the
engine, but because the resolver's job is precisely to be indifferent to which one it has.
"""

from __future__ import annotations

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
