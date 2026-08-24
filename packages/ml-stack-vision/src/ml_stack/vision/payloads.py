"""Getting an image into a request without blowing the context or the API's limits.

``ml_stack.media.image`` handles format sniffing and data URLs with no dependencies.
This module is the part that needs Pillow: resizing, re-encoding, and trimming an
attachment set down to what a given endpoint will actually accept.

Resizing is not optional in practice. A phone photo is 4000 px on its long edge and costs
thousands of tokens as tiles, for a description that a 1024 px version answers identically.
Sending the original is slower, more expensive, and frequently over the request limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.media import ImageError, kind, mime, to_data_url

DEFAULT_MAX_EDGE = 1024
DEFAULT_MAX_BYTES = 20 * 1024 * 1024


@dataclass
class NormalizationReport:
    """What had to be changed to make the images sendable. Worth logging."""

    resized: int = 0
    converted: int = 0
    dropped: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"{self.resized} resized, {self.converted} converted, {self.dropped} dropped"
            + (f"; {len(self.warnings)} warning(s)" if self.warnings else "")
        )


def load_bytes(source: Path | str | bytes) -> bytes:
    """Accept a path, a data URL, or raw bytes."""
    if isinstance(source, bytes):
        return source
    text = str(source)
    if text.startswith("data:"):
        from ml_stack.media import from_data_url

        return from_data_url(text)[0]
    return Path(text).read_bytes()


def _pillow():
    """Pillow, or an ``ImageError`` naming the extra that installs it.

    Pillow is an optional dependency because the vision *gate* needs nothing to run, but
    every path that changes pixels needs it. A bare ``ImportError`` escaping from inside
    ``normalize`` would abort the whole batch over one image; an ``ImageError`` is what
    the per-image handler there already knows how to report and carry on from.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageError(
            "Pillow is needed to convert or resize images -- "
            "install it with 'pip install ml-stack-vision[resize]'"
        ) from exc
    return Image


def _open_heif_too() -> None:
    """Teach Pillow to read HEIC, if the plugin for it is installed.

    Stock Pillow cannot decode HEIC at all, and iPhones produce it by default -- so the
    format the docstring below calls "the case that matters" is exactly the one that fails
    without ``pillow-heif``. Best effort: a box without the plugin gets a clear
    ``ImageError`` from the caller rather than an exception from here.
    """
    try:
        import pillow_heif
    except ImportError:
        return
    try:
        pillow_heif.register_heif_opener()
    except Exception:                                 # noqa: BLE001
        pass


def resize_to_fit(
    data: bytes,
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    quality: int = 85,
) -> tuple[bytes, bool]:
    """Shrink so the long edge is at most ``max_edge``. Returns ``(bytes, was_resized)``.

    An image already within the limit is returned untouched rather than re-encoded --
    re-encoding a JPEG is lossy every time, so doing it for no reason degrades the image
    the model sees.
    """
    import io

    Image = _pillow()
    try:
        image = Image.open(io.BytesIO(data))
    except OSError as exc:
        # UnidentifiedImageError subclasses OSError, so this is also the "Pillow does not
        # have a decoder for this" case -- which normalize must report, not crash on.
        raise ImageError(f"could not decode image for resizing: {exc}") from exc
    with image:
        width, height = image.size
        longest = max(width, height)
        if longest <= max_edge:
            return data, False

        scale = max_edge / longest
        resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))

        # Keep PNG as PNG: it is usually a screenshot or a diagram, where JPEG artefacts
        # land exactly on the text the model is being asked to read.
        source_kind = kind(data)
        buffer = io.BytesIO()
        if source_kind == "png":
            resized.save(buffer, format="PNG", optimize=True)
        else:
            resized.convert("RGB").save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue(), True


def to_supported_format(data: bytes) -> tuple[bytes, bool]:
    """Convert to PNG when the format is one that endpoints commonly reject.

    HEIC is the case that matters: iPhones produce it by default, and almost nothing
    accepts it.
    """
    detected = kind(data)
    if detected in ("jpeg", "png", "gif", "webp"):
        return data, False
    if detected is None:
        raise ImageError("unrecognised image format")

    import io

    Image = _pillow()
    _open_heif_too()
    try:
        image = Image.open(io.BytesIO(data))
    except OSError as exc:
        hint = ("; HEIC needs 'pip install ml-stack-vision[heic]'"
                if detected == "heic" else "")
        raise ImageError(f"cannot convert {detected}: {exc}{hint}") from exc
    with image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue(), True


def normalize(
    sources: list[Path | str | bytes],
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_images: int | None = None,
) -> tuple[list[str], NormalizationReport]:
    """Turn a list of image sources into data URLs an endpoint will accept.

    Trimming to ``max_images`` keeps the **first** ones. Later attachments in a message are
    usually context for earlier ones, so dropping from the end loses less than sampling.
    Whatever is dropped is reported rather than silently discarded.
    """
    report = NormalizationReport()
    urls: list[str] = []

    for index, source in enumerate(sources):
        if max_images is not None and len(urls) >= max_images:
            report.dropped += 1
            continue

        try:
            data = load_bytes(source)
        except (OSError, ImageError) as exc:
            report.dropped += 1
            report.warnings.append(f"image {index}: could not be read ({exc})")
            continue

        if len(data) > max_bytes:
            report.dropped += 1
            report.warnings.append(
                f"image {index}: {len(data) / 1e6:.1f} MB exceeds the "
                f"{max_bytes / 1e6:.0f} MB limit"
            )
            continue

        try:
            data, converted = to_supported_format(data)
            data, resized = resize_to_fit(data, max_edge=max_edge)
            urls.append(to_data_url(data, mime_type=mime(data)))
        except ImageError as exc:
            report.dropped += 1
            report.warnings.append(f"image {index}: {exc}")
            continue

        report.converted += int(converted)
        report.resized += int(resized)

    if max_images is not None and len(sources) > max_images:
        report.warnings.append(
            f"kept the first {max_images} of {len(sources)} images"
        )

    return urls, report


def build_message(
    text: str,
    images: list[Path | str | bytes],
    *,
    max_edge: int = DEFAULT_MAX_EDGE,
    max_images: int | None = None,
) -> tuple[dict[str, Any], NormalizationReport]:
    """An OpenAI-shaped multimodal user message, plus what had to be done to build it."""
    urls, report = normalize(images, max_edge=max_edge, max_images=max_images)
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in urls]
    return {"role": "user", "content": content}, report
