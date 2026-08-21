"""Check that a vision model can actually see before believing what it says.

A model served without its multimodal projector, or one whose runtime advertises vision it
does not have, does not error when handed an image. It describes the picture — fluently,
confidently, and entirely from the prompt. There is nothing in the response to distinguish
that from a real description, so the only defence is to ask it about an image whose
contents are already known.

Two details make the difference between a gate that works and one that feels like it does:

**The palette is not primary colours.** Ask a blind model to name red, green, blue and
yellow bands and it will often get all four right, because those are the colours a model
guesses when guessing. The palette here is teal, orange, purple, brown, pink and olive —
plausible enough to be nameable, unusual enough that guessing the exact set in the exact
order does not happen.

**The probe image is built by hand.** ``mainspring.media.probe_png`` uses only zlib and
struct, so the gate runs on a machine with no imaging library installed. A gate that cannot
run **fails open**, which is the one outcome it exists to prevent.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from mainspring.media import probe_png, to_data_url

PALETTE: dict[str, tuple[tuple[int, int, int], tuple[str, ...]]] = {
    "teal": ((0, 128, 128), ("teal", "turquoise", "cyan", "aqua")),
    "orange": ((255, 140, 0), ("orange", "amber", "tangerine")),
    "purple": ((128, 0, 128), ("purple", "violet", "magenta", "mauve")),
    "brown": ((139, 69, 19), ("brown", "tan", "chocolate", "sienna", "rust")),
    "pink": ((255, 105, 180), ("pink", "rose", "fuchsia")),
    "olive": ((107, 142, 35), ("olive", "khaki", "moss")),
}

PROMPT = (
    "This image shows vertical colour bands, left to right. "
    "Name each band's colour in order, as a comma-separated list. "
    "Answer with colour names only."
)


class VisionUnverified(RuntimeError):
    """The model could not be shown to see."""


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    expected: tuple[str, ...]
    answered: tuple[str, ...]
    raw: str = ""
    detail: str = ""

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class VisionGate:
    """One gate per model. Caches its verdict, including a failure."""

    bands: int = 3
    seed: int | None = None
    _verdicts: dict[str, GateResult] = field(default_factory=dict, repr=False)

    def build_probe(self, *, size: int = 256) -> tuple[bytes, tuple[str, ...]]:
        """A probe image and the colours in it, in order."""
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        names = rng.sample(sorted(PALETTE), k=min(self.bands, len(PALETTE)))
        colours = [PALETTE[n][0] for n in names]
        return probe_png(colours, size=size), tuple(names)

    def read_answer(self, text: str) -> tuple[str, ...]:
        """Pull colour names out of a reply, in order of first appearance.

        Consecutive repeats collapse: a model that says "orange, orange, teal" about two
        bands has named two, and holding that against it tests its prose rather than its
        eyes.
        """
        lowered = text.lower()
        hits: list[tuple[int, str]] = []
        for canonical, (_rgb, synonyms) in PALETTE.items():
            # Word boundaries, not substrings. "tangerine" contains "tan", so a substring
            # search scores a model that correctly said orange as having said brown --
            # failing a gate it should pass.
            positions = [
                match.start()
                for word in synonyms
                if (match := re.search(rf"\b{re.escape(word)}\b", lowered))
            ]
            if positions:
                hits.append((min(positions), canonical))

        ordered = [name for _pos, name in sorted(hits)]
        collapsed: list[str] = []
        for name in ordered:
            if not collapsed or collapsed[-1] != name:
                collapsed.append(name)
        return tuple(collapsed)

    def check(self, describe, *, model: str = "default", refresh: bool = False) -> GateResult:
        """Run the gate. ``describe(image_bytes, prompt) -> str``.

        Failures are cached too. A model that cannot see will not start seeing on the next
        request, and re-probing per call spends a full inference to learn that again.
        """
        if not refresh and model in self._verdicts:
            return self._verdicts[model]

        image, expected = self.build_probe()
        try:
            raw = describe(image, PROMPT)
        except Exception as exc:
            result = GateResult(False, expected, (), detail=f"the probe request failed: {exc}")
            self._verdicts[model] = result
            return result

        answered = self.read_answer(raw or "")
        passed = answered == expected
        detail = (
            ""
            if passed
            else f"expected {list(expected)}, read {list(answered)} from {raw.strip()[:200]!r}"
        )

        result = GateResult(passed, expected, answered, raw=raw, detail=detail)
        self._verdicts[model] = result
        return result

    def require(self, describe, *, model: str = "default") -> None:
        """``check``, but raise when it fails."""
        result = self.check(describe, model=model)
        if not result:
            raise VisionUnverified(
                f"{model} did not identify a known test image, so its descriptions cannot "
                f"be trusted: {result.detail}"
            )

    def forget(self, model: str | None = None) -> None:
        """Drop a cached verdict, after swapping weights or a projector."""
        if model is None:
            self._verdicts.clear()
        else:
            self._verdicts.pop(model, None)


def describe_via_client(client, model: str | None = None):
    """Adapt a ``mainspring.client.Client`` into the ``describe`` callable the gate wants."""

    def describe(image: bytes, prompt: str) -> str:
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": to_data_url(image)}},
            ],
        }
        kwargs = {"model": model} if model else {}
        return client.chat([message], **kwargs).content or ""

    return describe
