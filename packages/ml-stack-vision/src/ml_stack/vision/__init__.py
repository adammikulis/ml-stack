"""Vision-language requests, and a gate that checks the model can actually see.

Host tier. Pillow is needed for resizing; the gate itself needs nothing.

    from ml_stack.client import Client
    from ml_stack.vision import VisionGate, build_message, describe_via_client

    client = Client("http://127.0.0.1:8080")
    gate = VisionGate()
    gate.require(describe_via_client(client), model="my-vlm")   # fail now, not later

    message, report = build_message("What is in this photo?", ["photo.heic"])
    print(client.chat([message]).content, report)

The gate exists because a model served without its multimodal projector does not error
when handed an image -- it describes the picture confidently, entirely from the prompt.
"""

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
