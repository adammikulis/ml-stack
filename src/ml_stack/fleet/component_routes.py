"""Optional component choices and background installation routes."""

from __future__ import annotations

from . import model_components
from .discovery import load_cluster_key
from .weights import ModelError


def selected(models, request: dict, key: bytes | None = None) -> list[dict]:
    choices = {"mtp": request.get("mtp", bool(request.get("draft"))), "vision": request.get("vision", False)}
    if any(not isinstance(value, bool) for value in choices.values()):
        raise ModelError("MTP and vision choices must be true or false")
    if not any(choices.values()):
        return []
    found = model_components.catalogue(models, request["name"], str(request.get("source") or ""), key)
    selected = []
    for row in found["components"]:
        if choices[row["kind"]] and row["status"] != "installed":
            if row["status"] != "available":
                raise ModelError(row["error"])
            selected.append(row)
    return selected


def route(request) -> bool:
    if request.path != "/ui/models/addons":
        return False
    ui = request.ui
    if ui.models is None:
        request.send(501, {"error": "No model store on this device"})
        return True
    try:
        if request.method == "GET":
            request.send(200, model_components.catalogue(ui.models, request.asked("name"), request.asked("source"),
                                                        load_cluster_key(ui.cluster_key_path)))
        elif request.method == "POST":
            body = request.body()
            name, kind = body.get("name", ""), body.get("kind", "")
            if kind not in model_components.KINDS:
                raise ModelError("Choose MTP or vision")
            if not ui.models.find(name):
                raise ModelError("Install the base model before adding a component")
            if ui.models.sources() not in {"internet", "lan", "both"}:
                raise ModelError("Choose download sources before installing components")
            offers = selected(ui.models, {"name": name, "mtp": kind == "mtp", "vision": kind == "vision"},
                              load_cluster_key(ui.cluster_key_path))
            if not offers:
                request.send(200, {"installed": True, "name": name, "kind": kind})
            elif ui.downloads is None:
                request.send(501, {"error": "Background downloads are unavailable"})
            else:
                row = ui.downloads.start(name, key=load_cluster_key(ui.cluster_key_path), components=offers)
                request.send(202, row.public())
        else:
            return False
    except (ModelError, ValueError, OSError) as exc:
        request.send(400, {"error": str(exc)})
    return True
