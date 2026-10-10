"""Reading one folder of models into entries, once per on-disk layout.

``flat`` is a tree of files (llama.cpp's cache, LM Studio, GPT4All, a ``~/models``), ``hf``
is the Hugging Face cache with its ``models--owner--name/snapshots/<rev>/`` symlinks and
``ollama`` is manifests pointing at sha256 blobs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from poolhouse.hub.naming import aside
from poolhouse.hub.places import Place

PARTIAL = (".incomplete", ".part", ".partial", ".downloading", ".downloadinprogress",
           ".crdownload")
"""Suffixes of a download that is still arriving."""

SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".cache", "blobs", ".locks"})

REGISTRY = "registry.ollama.ai"


@dataclass(frozen=True, slots=True)
class Entry:
    """One model file or model folder, as found."""

    place: str
    path: Path
    name: str
    format: str
    size: int
    mtime: float
    verified: bool = True
    repo: str = ""
    revision: str = ""
    display: str = ""
    mmproj: Path | None = None
    complete: bool = True
    aux: bool = False
    sidecar_name: str = ""


def _stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def inside(path: Path, roots: Sequence[Path]) -> bool:
    """Whether ``path`` resolves to somewhere under one of ``roots``."""
    try:
        real = path.resolve()
    except OSError:
        return False
    return any(real == r or r in real.parents for r in roots)


def _allowed(path: Path, root: Path, allowed: Sequence[Path]) -> bool:
    """A symlink is followed only when its target is under its own root or another root."""
    return not path.is_symlink() or inside(path, [root.resolve(), *allowed])


def _weights_dir(names: set[str]) -> bool:
    return "config.json" in names and any(
        n.endswith(".safetensors") or n == "model.safetensors.index.json" for n in names)


def _config_format(directory: Path, repo: str) -> str:
    """``mlx`` for a model folder mlx-lm wrote, ``safetensors`` for any other."""
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        config = {}
    if repo.lower().startswith("mlx-community/") or "mlx" in repo.lower().split("/")[-1]:
        return "mlx"
    return "mlx" if isinstance(config, dict) and "quantization" in config else "safetensors"


def _folder_entry(place: Place, directory: Path, names: set[str], repo: str = "",
                  revision: str = "") -> Entry | None:
    sizes = []
    for n in names:
        if n.endswith(".safetensors"):
            st = _stat(directory / n)
            if st:
                sizes.append((st.st_size, st.st_mtime))
    if not sizes:
        return None
    return Entry(place.label, directory, directory.name, _config_format(directory, repo),
                 sum(s for s, _ in sizes), max(m for _, m in sizes), place.verified, repo,
                 revision,
                 complete=not any(n.lower().endswith(PARTIAL) for n in names))


def _etag_repo(path: Path) -> tuple[str, str]:
    """``(repo, file)`` from llama.cpp's ``<file>.json`` sidecar, which records the URL."""
    try:
        meta = json.loads(Path(f"{path}.json").read_text(encoding="utf-8"))
        url = str(meta.get("url", ""))
    except (OSError, ValueError, AttributeError):
        return "", ""
    parts = url.split("/")
    if "huggingface.co" in url and "resolve" in parts:
        at = parts.index("resolve")
        if at >= 2:
            return f"{parts[at - 2]}/{parts[at - 1]}", "/".join(parts[at + 2:])
    return "", ""


def _repo_hint(place: Place, rel: tuple[str, ...], path: Path) -> tuple[str, str]:
    if place.label == "llama.cpp":
        return _etag_repo(path)
    if place.label == "lmstudio" and len(rel) == 3:
        return f"{rel[0]}/{rel[1]}", rel[2]
    if place.label == "modelscope" and "models" in rel and len(rel) > rel.index("models") + 3:
        at = rel.index("models")
        return f"{rel[at + 1]}/{rel[at + 2]}", "/".join(rel[at + 3:])
    return "", ""


def _file_entry(place: Place, path: Path, rel: tuple[str, ...]) -> Entry | None:
    st = _stat(path)
    if st is None or not path.is_file():
        return None
    repo, sidecar = _repo_hint(place, rel, path)
    return Entry(place.label, path, path.name, "gguf", st.st_size, st.st_mtime,
                 place.verified, repo, sidecar_name=sidecar, aux=bool(aside(path.name)))


def scan_flat(place: Place, allowed: Sequence[Path] = ()) -> Iterator[Entry]:
    """Every GGUF file and every safetensors model folder under ``place``, to its depth."""
    root = place.path
    base = len(root.parts)
    for here, dirs, files in os.walk(root, followlinks=False):
        where = Path(here)
        depth = len(where.parts) - base
        keep = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("models--")]
        dirs[:] = sorted(keep) if depth < place.depth else []
        if place.depth >= 1 and _weights_dir(set(files)) and depth > 0:
            one = _folder_entry(place, where, set(files))
            if one:
                yield one
                dirs[:] = []
                continue
        for name in sorted(files):
            if not name.lower().endswith(".gguf"):
                continue
            path = where / name
            if _allowed(path, root, allowed):
                rel = path.relative_to(root).parts
                one = _file_entry(place, path, rel)
                if one:
                    held = any(n != name and n.startswith(name) and n.lower().endswith(PARTIAL)
                               for n in files)
                    yield replace(one, complete=not held)


def _hf_repo(directory: str) -> str:
    owner, _, name = directory[len("models--"):].partition("--")
    return f"{owner}/{name}" if name else ""


def scan_hf(place: Place, allowed: Sequence[Path] = ()) -> Iterator[Entry]:
    """The GGUF files and safetensors folders in every snapshot of a Hugging Face cache."""
    try:
        repos = sorted(p for p in place.path.iterdir()
                       if p.name.startswith("models--") and (p / "snapshots").is_dir())
    except OSError:
        return
    for repo_dir in repos:
        repo = _hf_repo(repo_dir.name)
        if not repo:
            continue
        for snap in sorted((repo_dir / "snapshots").iterdir()):
            if not snap.is_dir():
                continue
            names = {p.name for p in snap.iterdir()}
            if _weights_dir(names):
                one = _folder_entry(place, snap, names, repo, snap.name)
                if one:
                    yield one
            for here, _dirs, files in os.walk(snap, followlinks=False):
                for name in sorted(files):
                    path = Path(here) / name
                    if name.lower().endswith(".gguf") and _allowed(path, repo_dir, allowed):
                        one = _file_entry(place, path, ())
                        if one:
                            yield replace(one, repo=repo, revision=snap.name)


def _ollama_name(rel: tuple[str, ...]) -> str:
    host, *middle, tag = rel
    model = "/".join(middle[1:] if middle and middle[0] == "library" else middle)
    prefix = "" if host == REGISTRY else f"{host}/"
    return f"{prefix}{model}:{tag}"


def _layer(manifest: dict, kind: str) -> dict | None:
    for layer in manifest.get("layers", ()):
        if isinstance(layer, dict) and str(layer.get("mediaType", "")).endswith(kind):
            return layer
    return None


def _blob(root: Path, layer: dict) -> Path:
    return root / "blobs" / str(layer.get("digest", "")).replace(":", "-")


def scan_ollama(place: Place, allowed: Sequence[Path] = ()) -> Iterator[Entry]:
    """Each Ollama manifest that names a GGUF layer, as the blob that holds it."""
    manifests = place.path / "manifests"
    if not manifests.is_dir():
        return
    for here, _dirs, files in os.walk(manifests, followlinks=False):
        for name in sorted(files):
            file = Path(here) / name
            try:
                manifest = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            layer = _layer(manifest, "image.model") if isinstance(manifest, dict) else None
            if layer is None:
                continue
            blob = _blob(place.path, layer)
            st = _stat(blob)
            if st is None:
                continue
            projector = _layer(manifest, "image.projector")
            look = _blob(place.path, projector) if projector else None
            shown = _ollama_name(file.relative_to(manifests).parts)
            yield Entry(place.label, blob, shown, "gguf", st.st_size, file.stat().st_mtime,
                        place.verified, display=shown,
                        mmproj=look if look and look.exists() else None,
                        complete=st.st_size == int(layer.get("size", st.st_size)))


def detect(path: Path) -> str:
    """The layout of a folder somebody named: ``ollama``, ``hf`` or ``flat``."""
    if (path / "manifests").is_dir() and (path / "blobs").is_dir():
        return "ollama"
    try:
        if any(p.name.startswith("models--") for p in path.iterdir()):
            return "hf"
    except OSError:
        return "flat"
    return "flat"


def scan(place: Place, allowed: Sequence[Path] = ()) -> Iterator[Entry]:
    """The entries in one place, read the way its layout is read."""
    if not place.path.is_dir():
        return iter(())
    layout = detect(place.path) if place.layout == "auto" else place.layout
    reader = {"hf": scan_hf, "ollama": scan_ollama}.get(layout, scan_flat)
    return reader(place, allowed)
