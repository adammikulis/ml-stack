"""Vision-language requests, and a gate that checks the model can actually see."""

from __future__ import annotations

from ml_stack.vision.gate import (
    PALETTE,
    PROMPT,
    GateResult,
    VisionGate,
    VisionUnverified,
    describe_via_client,
)
from ml_stack.vision.payloads import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_EDGE,
    NormalizationReport,
    build_message,
    load_bytes,
    normalize,
    resize_to_fit,
    to_supported_format,
)

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_EDGE",
    "PALETTE",
    "PROMPT",
    "GateResult",
    "NormalizationReport",
    "VisionGate",
    "VisionUnverified",
    "build_message",
    "describe_via_client",
    "load_bytes",
    "normalize",
    "resize_to_fit",
    "to_supported_format",
    "Bearing", "column_to_deg", "find_color_blob", "floor_boundary",
    "hfov_from_known_width", "nearest_obstacle", "to_gray",
]

from ml_stack.vision.geometry import (  # noqa: E402
    Bearing, column_to_deg, find_color_blob, floor_boundary,
    hfov_from_known_width, nearest_obstacle, to_gray,
)
