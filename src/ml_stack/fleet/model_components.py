"""Compatible model component offers and installed linkage."""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path, PurePosixPath

from ml_stack import hub, net
from ml_stack.http import ServerError
from ml_stack.safenames import safe_filename
from ml_stack.serve import mtp
from ml_stack.serve.preflight import read_gguf_header

from . import component_store
from .catalogue import SUGGESTED
from .weights import ModelError, is_a_piece

KINDS = {"mtp", "vision"}


def record(path: Path) -> dict:
    return component_store.read(path)


def remember(path: Path, source: str) -> None:
    component_store.source(path, source)


def linked(path: Path, kind: str) -> Path | None:
    row = record(path).get("components", {}).get(kind, {})
    if not isinstance(row, dict):
        return None
    name = row.get("name", "")
    if not name:
        return None
    try:
        if safe_filename(name) != name:
            return None
        target = path.parent / name
        if target.is_file() and target.stat().st_size == row.get("size_bytes"):
            return target
    except (OSError, ValueError):
        return None
    return None


def link(model: Path, component: Path, offer: dict) -> None:
    kind = offer["kind"]
    if kind not in KINDS or component.parent.resolve() != model.parent.resolve():
        raise ModelError("A model component must be stored beside its model")
    if kind == "mtp" and offer["packaging"] == "separate":
        if problem := mtp.mismatch(model, component):
            raise ModelError(problem)
    elif kind == "vision":
        header = read_gguf_header(component)
        if header.get("general.architecture") not in {"clip", "mtmd"}:
            raise ModelError("The vision component is not a multimodal projector")
    else:
        if problem := mtp.mismatch(model, component):
            raise ModelError(problem)
        if not mtp.embeds_head(component):
            raise ModelError("The replacement model carries no integrated prediction tensors")
    component_store.link(model, component, {**offer, "name": component.name,
                                            "size_bytes": component.stat().st_size})


def _reference(source: str) -> tuple[str, str]:
    if source.startswith("hf:"):
        parts = source[3:].split("/", 2)
        return ("/".join(parts[:2]), parts[2]) if len(parts) == 3 else ("", "")
    url = urllib.parse.urlsplit(source)
    parts = url.path.strip("/").split("/")
    if url.hostname == "huggingface.co" and len(parts) >= 5 and parts[2:4] == ["resolve", "main"]:
        return "/".join(parts[:2]), "/".join(parts[4:])
    return "", ""


def _family(name: str) -> str:
    stem = Path(name).stem.lower().removeprefix("mtp-").removeprefix("mmproj-")
    return re.split(r"-(?:ud|gsq|rco|iq\d|q\d|bf16|f16|f32)(?:[-_]|$)", stem, maxsplit=1)[0]


def discover(source: str, files: list[dict]) -> list[dict]:
    """Return unambiguous same-repository companion and integrated replacement offers."""
    repo, filename = _reference(source)
    if not repo or not filename:
        return []
    model_name = Path(filename).name
    family = _family(model_name)
    candidates = {"mtp": [], "vision": []}
    for row in files:
        if not isinstance(row, dict):
            raise ModelError("Invalid component repository entry")
        name = str(row.get("path", ""))
        if (str(PurePosixPath(name)) != name or name.startswith("/") or "\\" in name
                or any(part in {".", ".."} for part in PurePosixPath(name).parts)):
            raise ModelError("Unsafe component repository path")
        size = int(row.get("size") or (row.get("lfs") or {}).get("size") or 0)
        if not name.lower().endswith(".gguf") or is_a_piece(name) or size <= 0:
            continue
        basename = Path(name).name
        packaging, kind = "separate", ""
        if basename == model_name.removesuffix(".gguf") + "-mtp.gguf":
            kind, packaging = "mtp", "integrated"
        elif basename.lower().startswith("mtp-") and _family(basename) == family:
            kind = "mtp"
        elif basename.lower().startswith("mmproj-") and (_family(basename) == family or basename.lower()
                                                        in {"mmproj-bf16.gguf", "mmproj-f16.gguf", "mmproj-f32.gguf"}):
            kind = "vision"
        if kind:
            safe_filename(basename)
            candidates[kind].append({"kind": kind, "packaging": packaging, "ref": f"hf:{repo}/{name}",
                                     "name": basename, "size_bytes": size,
                                     "download_bytes": size, "storage_bytes": size})
    out = []
    for kind, rows in candidates.items():
        if not rows:
            continue
        preferred = [r for r in rows if r["packaging"] == "integrated"] if kind == "mtp" else []
        preferred = preferred or [r for r in rows if "bf16" in r["name"].lower() or "q8_0" in r["name"].lower()]
        selected = preferred or rows
        if len(selected) == 1:
            out.append({**selected[0], "status": "available", "error": ""})
        else:
            out.append({"kind": kind, "status": "unavailable", "error": "Multiple compatible components; choose an explicit manifest"})
    return out


def source_for(model: Path) -> str:
    source = record(model).get("source", "")
    if source:
        return source
    repo = hub.repo_of(model)
    if repo:
        return f"hf:{repo}/{model.name}"
    return next((pick.ref for pick in SUGGESTED if pick.file == model.name), "")


def effective(model: Path) -> Path:
    replacement = linked(model, "mtp")
    info = record(model).get("components", {}).get("mtp", {})
    return replacement if replacement and info.get("packaging") == "integrated" else model


def inventory(model: Path) -> dict:
    repo, filename = _reference(source_for(model))
    result = {"source": f"hf:{repo}/{filename}" if repo else "", "components": []}
    for kind in sorted(KINDS):
        if target := linked(model, kind):
            stored = record(model)["components"][kind]
            result["components"].append({**stored, "name": target.name, "size": target.stat().st_size})
    return result


def catalogue(models, name: str, source: str = "", key: bytes | None = None) -> dict:
    model = models.listed(name)       # a name from a request is compared, never opened
    source = source or (source_for(model.path) if model else "")
    offers, error = [], ""
    repo, _ = _reference(source)
    policy = models.sources()
    if key is not None and policy in {"lan", "both"}:
        from .remote import Peer, PeerError

        for peer in Peer.discover(key=key, timeout_s=1):
            try:
                for held in peer.models():
                    same = _reference(held.get("source", "")) == _reference(source) if source else held.get("name") == name
                    if not same:
                        continue
                    for offered in held.get("components", []):
                        if offered.get("kind") in KINDS:
                            offers.append({**offered, "status": "available", "error": "",
                                           "download_bytes": offered["size"], "storage_bytes": offered["size"]})
            except (PeerError, OSError, ValueError, KeyError):
                continue
    if repo and policy in {"internet", "both"}:
        try:
            files = net.default().json(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1",
                                       net.Ask(purpose="model components", tries=1))
            if not isinstance(files, list) or len(files) > 10_000:
                raise ModelError("Invalid component repository listing")
            internet = discover(source, files)
            offers.extend(row for row in internet if not any(local["kind"] == row["kind"] for local in offers))
        except (ServerError, ModelError, OSError, ValueError) as exc:
            error = str(exc)
    components = []
    for kind in sorted(KINDS):
        installed = linked(model.path, kind) if model else None
        integrated = kind == "mtp" and model and mtp.embeds_head(model.path)
        if integrated:
            components.append({"kind": kind, "packaging": "integrated", "status": "installed",
                               "name": model.name, "size_bytes": model.size, "download_bytes": 0,
                               "storage_bytes": 0, "error": ""})
        elif installed:
            stored = record(model.path)["components"][kind]
            components.append({**stored, "status": "installed", "download_bytes": 0, "storage_bytes": 0, "error": ""})
        else:
            components.append(next((row for row in offers if row["kind"] == kind),
                                   {"kind": kind, "status": "unavailable", "error": error or "No compatible component found in the allowed sources"}))
    return {"name": name, "source": source, "source_policy": policy, "components": components}
