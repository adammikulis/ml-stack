"""Installed decision and vision models exposed by the Gym catalogue."""

from pathlib import Path

from poolhouse import hub
from poolhouse.decide import registry
from poolhouse.decide.pins import STRANDS

SMOL = "ggml-org/SmolVLM-256M-Instruct-GGUF"


def vision_row(model):
    """Describe runtime support independently from downloaded model files."""
    supported = model.format == "gguf" and model.mmproj is not None
    installed = model.is_complete and model.path.exists()
    if model.mmproj is not None:
        installed = installed and model.mmproj.is_file()
    reason = "" if supported else "This format is not supported by the Gym llama.cpp vision runtime"
    if "fastvlm" in model.name.lower() and model.format != "gguf":
        reason = "FastVLM weights are downloaded; their native vision runtime is not implemented in Gym"
    return {"id": model.id, "label": model.name, "model": str(model.path),
            "format": model.format, "backend": "llama.cpp" if supported else None,
            "mmproj": str(model.mmproj) if model.mmproj else None,
            "available": bool(supported and installed),
            "status": "installed" if supported and installed else "missing" if supported else "unsupported",
            "reason": reason or ("Model or vision projector files are missing" if not installed else "")}


def model_choices():
    """List local choices without loading, fetching or verifying model weights."""
    decisions = [{"id": "strands", "label": "Strands Decider 2B (default)",
                  "checkpoint": None, "model": STRANDS, "device": "auto",
                  "status": "not_loaded", "reason": "Verified local files are checked when the model loads"}]
    decisions.extend({"id": row["name"], "label": row["name"], "checkpoint": row["path"],
                      "model": row["name"], "device": "auto",
                      "status": "installed" if Path(row["path"]).is_dir() else "missing"}
                     for row in registry.listing())
    vision = [vision_row(model) for model in hub.discover(kind="vision")]
    vision.sort(key=lambda row: ("smolvlm" not in row["label"].lower(),
                                "qwen3.5" not in row["label"].lower(), row["label"].lower()))
    default = next((row["id"] for row in vision if "smolvlm" in row["label"].lower()
                    and row["available"]), None)
    if default is None:
        vision.insert(0, {"id": "smolvlm-256m", "label": "SmolVLM 256M (speed default)",
                         "model": SMOL, "available": False, "status": "missing",
                         "reason": "Download SmolVLM 256M GGUF and its vision projector in Models"})
        default = "smolvlm-256m"
    return {"decision": decisions, "vision": vision, "vision_default": default}


def selected_vision(identifier):
    """Resolve only a supported installed vision model from the local catalogue."""
    for model in hub.discover(kind="vision"):
        if identifier in {model.id, str(model.path)}:
            row = vision_row(model)
            if not row["available"]:
                raise ValueError(row["reason"])
            return row
    raise ValueError("Vision model is not installed; select a downloaded model in Models")
