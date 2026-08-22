"""The fabrication gate, and image payload normalisation.

The gate tests are the interesting ones. They stand up three kinds of model — one that
sees, one that is blind but fluent, one that guesses primary colours — and check the gate
tells them apart. A gate that only ever runs against a working model has never been shown
to do anything.
"""

from __future__ import annotations

import io

import pytest
from ml_stack.media import from_data_url, kind, probe_png
from ml_stack.vision import (
    PALETTE,
    GateResult,
    NormalizationReport,
    VisionGate,
    VisionUnverified,
    build_message,
    describe_via_client,
    load_bytes,
    normalize,
    resize_to_fit,
    to_supported_format,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def png_of(size: tuple[int, int], colour=(120, 30, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_of(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 200, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


def sighted(image: bytes, prompt: str) -> str:
    """A model that reads the actual pixels, one band per third."""
    with Image.open(io.BytesIO(image)) as picture:
        width, height = picture.size
        rgb = picture.convert("RGB")
        seen = []
        for band in range(3):
            pixel = rgb.getpixel((int((band + 0.5) * width / 3), height // 2))
            nearest = min(
                PALETTE,
                key=lambda n: sum((a - b) ** 2 for a, b in zip(PALETTE[n][0], pixel)),
            )
            if not seen or seen[-1] != nearest:
                seen.append(nearest)
    return ", ".join(seen)


def blind_but_fluent(image: bytes, prompt: str) -> str:
    """A model served without its projector. It does not error; it describes."""
    return "The image shows three vertical bands of colour, arranged left to right."


def guesses_primaries(image: bytes, prompt: str) -> str:
    """What a blind model reaches for when asked to name colours."""
    return "red, green, blue"


class TestVisionGate:
    def test_a_model_that_sees_passes(self):
        gate = VisionGate(seed=1)
        assert gate.check(sighted, model="sighted")

    def test_a_blind_but_fluent_model_fails(self):
        """The failure this exists to catch: no error, a confident description, and
        nothing in the response to distinguish it from a real one."""
        gate = VisionGate(seed=1)
        result = gate.check(blind_but_fluent, model="blind")
        assert not result
        assert "read []" in result.detail

    def test_guessing_primary_colours_does_not_pass(self):
        """Precisely why the palette is teal/orange/purple rather than red/green/blue."""
        gate = VisionGate(seed=1)
        assert not gate.check(guesses_primaries, model="guesser")

    def test_the_palette_contains_no_primary_colours(self):
        assert not ({"red", "green", "blue", "yellow"} & set(PALETTE))

    def test_a_failing_request_is_a_failed_gate_not_a_crash(self):
        def unreachable(image, prompt):
            raise ConnectionError("server is down")

        result = VisionGate().check(unreachable, model="down")
        assert not result and "probe request failed" in result.detail

    def test_require_raises_on_a_blind_model(self):
        with pytest.raises(VisionUnverified, match="cannot be trusted"):
            VisionGate(seed=1).require(blind_but_fluent, model="blind")

    def test_require_is_silent_on_a_sighted_one(self):
        VisionGate(seed=1).require(sighted, model="sighted")

    def test_the_verdict_is_cached_including_a_failure(self):
        """A model that cannot see will not start seeing, and re-probing spends a full
        inference to learn that again."""
        calls = {"n": 0}

        def counting(image, prompt):
            calls["n"] += 1
            return blind_but_fluent(image, prompt)

        gate = VisionGate(seed=1)
        gate.check(counting, model="m")
        gate.check(counting, model="m")
        assert calls["n"] == 1

    def test_refresh_re_probes(self):
        calls = {"n": 0}

        def counting(image, prompt):
            calls["n"] += 1
            return sighted(image, prompt)

        gate = VisionGate(seed=1)
        gate.check(counting, model="m")
        gate.check(counting, model="m", refresh=True)
        assert calls["n"] == 2

    def test_forgetting_clears_the_verdict_after_a_model_swap(self):
        gate = VisionGate(seed=1)
        gate.check(blind_but_fluent, model="m")
        gate.forget("m")
        assert gate.check(sighted, model="m")

    def test_different_models_get_separate_verdicts(self):
        gate = VisionGate(seed=1)
        assert not gate.check(blind_but_fluent, model="blind")
        assert gate.check(sighted, model="sighted")

    def test_the_probe_image_is_a_real_png_with_the_expected_bands(self):
        image, expected = VisionGate(bands=3, seed=7).build_probe(size=64)
        assert kind(image) == "png"
        assert len(expected) == 3 and len(set(expected)) == 3

    def test_the_probe_varies_between_gates(self):
        """A fixed image is one a model could plausibly have memorised."""
        seen = {VisionGate(seed=s).build_probe()[1] for s in range(8)}
        assert len(seen) > 1

    def test_answer_reading_handles_synonyms(self):
        gate = VisionGate()
        assert gate.read_answer("turquoise, tangerine, violet") == ("teal", "orange", "purple")

    def test_answer_reading_collapses_consecutive_repeats(self):
        """A model that says a colour twice about two bands has named two; holding that
        against it tests its prose, not its eyes."""
        assert VisionGate().read_answer("teal, teal, orange") == ("teal", "orange")

    def test_answer_reading_is_order_of_appearance(self):
        assert VisionGate().read_answer("First olive, then pink.") == ("olive", "pink")

    def test_answer_reading_of_a_colourless_reply(self):
        assert VisionGate().read_answer("I cannot see any image.") == ()

    def test_describe_via_client_builds_a_multimodal_message(self):
        captured = {}

        class FakeClient:
            def chat(self, messages, **kwargs):
                captured["messages"] = messages

                class Reply:
                    content = "teal, orange, purple"

                return Reply()

        describe = describe_via_client(FakeClient())
        assert describe(probe_png([(0, 128, 128)], size=8), "prompt") == "teal, orange, purple"

        content = captured["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


class TestResize:
    def test_a_large_image_shrinks_to_the_limit(self):
        """A phone photo costs thousands of tokens as tiles for a description a 1024 px
        version answers identically."""
        data, resized = resize_to_fit(png_of((4000, 3000)), max_edge=1024)
        assert resized
        with Image.open(io.BytesIO(data)) as image:
            assert max(image.size) == 1024

    def test_a_small_image_is_returned_untouched(self):
        """Re-encoding a JPEG is lossy every time, so doing it for no reason degrades what
        the model sees."""
        original = jpeg_of((320, 240))
        data, resized = resize_to_fit(original, max_edge=1024)
        assert not resized and data is original

    def test_aspect_ratio_survives(self):
        data, _ = resize_to_fit(png_of((2000, 1000)), max_edge=500)
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (500, 250)

    def test_a_png_stays_a_png(self):
        """Usually a screenshot or a diagram, where JPEG artefacts land exactly on the text
        the model is being asked to read."""
        data, _ = resize_to_fit(png_of((3000, 3000)), max_edge=256)
        assert kind(data) == "png"

    def test_a_jpeg_stays_a_jpeg(self):
        data, _ = resize_to_fit(jpeg_of((3000, 3000)), max_edge=256)
        assert kind(data) == "jpeg"


class TestFormatConversion:
    def test_common_formats_pass_through(self):
        for data in (png_of((10, 10)), jpeg_of((10, 10))):
            out, converted = to_supported_format(data)
            assert not converted and out is data

    def test_unrecognised_bytes_raise(self):
        from ml_stack.media import ImageError

        with pytest.raises(ImageError):
            to_supported_format(b"<html>not an image</html>")


class TestNormalize:
    def test_it_produces_data_urls(self):
        urls, report = normalize([png_of((100, 100))])
        assert len(urls) == 1
        assert from_data_url(urls[0])[1] == "image/png"
        assert report.dropped == 0

    def test_an_oversized_image_is_dropped_with_a_reason(self):
        urls, report = normalize([png_of((100, 100))], max_bytes=10)
        assert urls == [] and report.dropped == 1
        assert "exceeds" in report.warnings[0]

    def test_an_unreadable_source_is_dropped_not_fatal(self):
        """One bad attachment must not lose the whole message."""
        urls, report = normalize(["/definitely/not/here.png", png_of((50, 50))])
        assert len(urls) == 1 and report.dropped == 1

    def test_trimming_keeps_the_first_images(self):
        """Later attachments are usually context for earlier ones, so dropping from the
        end loses less than sampling."""
        images = [png_of((20, 20), colour=(i * 40, 0, 0)) for i in range(5)]
        urls, report = normalize(images, max_images=2)
        assert len(urls) == 2
        assert from_data_url(urls[0])[0] == images[0]
        assert report.dropped == 3

    def test_trimming_is_reported_rather_than_silent(self):
        _urls, report = normalize([png_of((10, 10))] * 4, max_images=1)
        assert any("kept the first" in w for w in report.warnings)

    def test_resizes_are_counted(self):
        _urls, report = normalize([png_of((3000, 3000))], max_edge=256)
        assert report.resized == 1

    def test_load_bytes_accepts_a_data_url(self):
        original = png_of((8, 8))
        from ml_stack.media import to_data_url

        assert load_bytes(to_data_url(original)) == original

    def test_load_bytes_accepts_a_path(self, tmp_path):
        path = tmp_path / "image.png"
        path.write_bytes(png_of((8, 8)))
        assert load_bytes(path) == path.read_bytes()

    def test_report_stringifies_usefully(self):
        assert "resized" in str(NormalizationReport(resized=2))


class TestBuildMessage:
    def test_it_builds_an_openai_shaped_message(self):
        message, report = build_message("what is this?", [png_of((100, 100))])
        assert message["role"] == "user"
        assert message["content"][0] == {"type": "text", "text": "what is this?"}
        assert message["content"][1]["type"] == "image_url"
        assert report.dropped == 0

    def test_text_survives_when_every_image_is_dropped(self):
        """A question with an unreadable attachment is still a question."""
        message, report = build_message("what is this?", ["/nope.png"])
        assert message["content"] == [{"type": "text", "text": "what is this?"}]
        assert report.dropped == 1


def test_gate_result_is_falsy_when_it_failed():
    assert not GateResult(False, ("teal",), ())
    assert GateResult(True, ("teal",), ("teal",))


class TestGeometry:
    """Pixels to angles, for a robot that has to drive at what it saw.

    Two things here fail silently and cost a whole behaviour: the SIGN (drive
    away from the target instead of toward it) and the SCALE (a wrong field of
    view, so every turn is proportionally short). Both look like a broken
    detector from the outside.
    """

    @staticmethod
    def _room(w=320, h=240, person=None, wall=0.45, floor=90, wall_v=150, person_v=40):
        """A ROOM, not a void: floor at the bottom and a back wall above it.

        The distinction is the whole reason these tests exist. An earlier
        detector was tested against a uniform floor filling the entire frame
        with a bar drawn on it, passed everything, and returned "dead ahead,
        confidence 1.00" for an EMPTY room -- because a back wall is not floor,
        spans every column, and wins any widest-not-floor search.
        """
        import numpy as np
        f = np.full((h, w, 3), floor, np.uint8)
        f[: int(h * wall)] = wall_v
        if person:
            f[int(h * 0.30):int(h * 0.95), person[0]:person[1]] = person_v
        return f

    def test_the_frame_edges_span_exactly_the_field_of_view(self):
        from ml_stack.vision import column_to_deg
        assert column_to_deg(0, 320, 62.2) == pytest.approx(-31.1, abs=0.01)
        assert column_to_deg(319, 320, 62.2) == pytest.approx(31.1, abs=0.01)

    def test_the_centre_is_dead_ahead(self):
        from ml_stack.vision import column_to_deg
        assert column_to_deg(160, 321, 62.2) == pytest.approx(0.0, abs=0.01)

    def test_the_mapping_is_tangent_not_linear(self):
        """arctan is concave, so mid-frame angles are LARGER than a linear map
        gives. Reversing this makes a robot under-turn on every command."""
        from ml_stack.vision import column_to_deg
        mid = column_to_deg(240, 320, 62.2)
        assert mid > (62.2 / 2 * 0.5) + 1.0, f"{mid} looks linear"

    def test_a_degenerate_width_does_not_divide_by_zero(self):
        from ml_stack.vision import column_to_deg
        assert column_to_deg(0, 1, 62.2) == 0.0
        assert column_to_deg(0, 0, 62.2) == 0.0

    def test_hfov_is_required_rather_than_guessed(self):
        """A defaulted field of view is a silent calibration everybody inherits."""
        import inspect

        from ml_stack.vision import column_to_deg, nearest_obstacle
        assert inspect.signature(column_to_deg).parameters["hfov_deg"].default \
            is inspect.Parameter.empty
        assert inspect.signature(nearest_obstacle).parameters["hfov_deg"].default \
            is inspect.Parameter.empty

    def test_the_scale_follows_the_field_of_view(self):
        from ml_stack.vision import column_to_deg
        narrow = column_to_deg(319, 320, 40.0)
        wide = column_to_deg(319, 320, 90.0)
        assert wide > narrow, "a wider lens must map the same column further off-axis"

    def test_an_empty_room_is_not_a_person(self):
        """THE regression. A back wall spans every column and is not floor, so
        a naive not-floor mask reports it as a full-width object dead ahead with
        maximum confidence -- identical to a room with someone standing in it."""
        from ml_stack.vision import nearest_obstacle
        assert nearest_obstacle(self._room(), hfov_deg=62.2).confidence == 0.0

    def test_an_empty_room_and_an_occupied_one_do_not_look_alike(self):
        from ml_stack.vision import nearest_obstacle
        empty = nearest_obstacle(self._room(), hfov_deg=62.2)
        occupied = nearest_obstacle(self._room(person=(250, 285)), hfov_deg=62.2)
        assert occupied.confidence > 0.2 > empty.confidence

    def test_it_points_at_where_the_person_actually_is(self):
        from ml_stack.vision import nearest_obstacle
        assert nearest_obstacle(self._room(person=(250, 285)), hfov_deg=62.2).deg > 10
        assert nearest_obstacle(self._room(person=(45, 80)), hfov_deg=62.2).deg < -10
        assert abs(nearest_obstacle(self._room(person=(150, 175)),
                                    hfov_deg=62.2).deg) < 5

    def test_a_textured_wall_is_wide_but_flat_and_so_is_rejected(self):
        """Width alone is not evidence. The wall has structure at every column
        and still ends at the same distance everywhere."""
        import numpy as np

        from ml_stack.vision import nearest_obstacle
        rng = np.random.default_rng(1)
        f = self._room()
        f[: int(240 * 0.45)] = rng.integers(120, 190, (int(240 * 0.45), 320, 3),
                                            dtype=np.uint8)
        assert nearest_obstacle(f, hfov_deg=62.2).confidence < 0.2

    def test_a_uniform_void_yields_nothing(self):
        import numpy as np

        from ml_stack.vision import nearest_obstacle
        b = nearest_obstacle(np.full((240, 320, 3), 128, np.uint8), hfov_deg=62.2)
        assert b.confidence == 0.0

    def test_speckle_noise_does_not_become_a_bearing(self):
        import numpy as np

        from ml_stack.vision import nearest_obstacle
        rng = np.random.default_rng(0)
        f = self._room()
        f[rng.integers(120, 230, 60), rng.integers(0, 320, 60)] = 255
        assert nearest_obstacle(f, hfov_deg=62.2).confidence < 0.2

    def test_a_nearer_object_beats_a_further_one(self):
        """Two objects: the one whose floor ends lower in the frame is closer,
        and closer is what 'come here' should walk toward."""
        from ml_stack.vision import nearest_obstacle
        import numpy as np
        f = self._room()
        f[int(240 * 0.30):int(240 * 0.60), 40:80] = 40      # far: ends higher
        f[int(240 * 0.30):int(240 * 0.95), 250:290] = 40    # near: ends lower
        assert nearest_obstacle(f, hfov_deg=62.2).deg > 10

    def test_the_floor_boundary_is_flat_for_an_empty_room(self):
        import numpy as np

        from ml_stack.vision import floor_boundary
        prof = floor_boundary(self._room())
        assert float(np.std(prof)) < 2.0, "an empty room should have a flat horizon"

    def test_the_floor_boundary_dips_nearer_where_someone_stands(self):
        import numpy as np

        from ml_stack.vision import floor_boundary
        prof = floor_boundary(self._room(person=(250, 285)))
        assert prof[265] > np.median(prof) + 10

    def test_to_gray_channel_order_is_not_silently_reversed(self):
        import numpy as np

        from ml_stack.vision import to_gray
        blue_first = np.array([[[255, 0, 0]]], np.uint8)
        assert to_gray(blue_first, bgr=True)[0, 0] == pytest.approx(0.114 * 255, abs=0.5)
        assert to_gray(blue_first, bgr=False)[0, 0] == pytest.approx(0.299 * 255, abs=0.5)

    def test_to_gray_passes_through_an_already_gray_frame(self):
        import numpy as np

        from ml_stack.vision import to_gray
        assert to_gray(np.full((8, 8), 100, np.uint8))[0, 0] == pytest.approx(100.0)
