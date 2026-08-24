"""Model files this machine holds, and getting one it does not."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Getting", "Model", "Models", "ModelError", "Downloads",
           "Suggestion", "default_roots", "family_of", "is_unfiltered",
           "families", "how_many", "popular", "searched_count",
           "searched_families", "suggestions"]

SUFFIXES = (".gguf", ".safetensors", ".bin", ".pt", ".onnx")
CHUNK = 1 << 20
MIN_SIZE = 1 << 20
# A download in progress writes continuously, so a part file untouched for this
# long belongs to one that stopped.
STALE_PART_S = 3600.0


class ModelError(RuntimeError):
    pass


def _read_stamp(stamp: Path) -> dict[str, Any]:
    """What a half-finished download recorded about where it came from."""
    try:
        raw = json.loads(stamp.read_text())
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_stamp(stamp: Path, url: str, headers: Any) -> None:
    validator = headers.get("ETag") or headers.get("Last-Modified") or ""
    try:
        stamp.write_text(json.dumps({"url": url, "validator": validator}))
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    path: Path
    size: int
    modified: float

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "modified": self.modified}


def default_roots(root: Path | str) -> list[Path]:
    """Where model files live. The llama.cpp cache is included because a model pulled
    by a server is one this machine already holds."""
    home = Path.home()
    return [
        Path(root).expanduser() / "models",
        home / ".cache" / "llama.cpp",
        home / ".cache" / "huggingface" / "hub",
        home / "models",
    ]


@dataclass
class Models:
    """The model files on this machine."""

    roots: list[Path]
    store: Path
    _digests: dict[tuple[str, int, int], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.store = Path(self.store).expanduser()
        self.roots = [Path(r).expanduser() for r in self.roots]

    def all(self) -> list[Model]:
        seen: dict[str, Model] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                    continue
                if DRAFT_MARK in path.suffixes:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size < MIN_SIZE:
                    continue
                if path.name not in seen:
                    seen[path.name] = Model(path.name, path, stat.st_size,
                                            stat.st_mtime)
        return sorted(seen.values(), key=lambda m: m.name.lower())

    def find(self, name: str) -> Model | None:
        needle = name.strip().lower()
        for model in self.all():
            if model.name.lower() == needle:
                return model
        for model in self.all():
            if needle in model.name.lower():
                return model
        return None

    def digest(self, model: Model) -> str:
        """sha256, cached against size and mtime so a large file is read once."""
        stat = model.path.stat()
        key = (str(model.path), stat.st_size, stat.st_mtime_ns)
        found = self._digests.get(key)
        if found:
            return found
        h = hashlib.sha256()
        with model.path.open("rb") as fh:
            while True:
                block = fh.read(CHUNK)
                if not block:
                    break
                h.update(block)
        self._digests[key] = h.hexdigest()
        return self._digests[key]

    def public(self, limit: int = 24) -> list[dict[str, Any]]:
        """What goes in the beacon. Names and sizes only -- a beacon is a UDP packet,
        and a full path is needless disclosure."""
        return [m.public() for m in self.all()[:limit]]

    # -- getting one ----------------------------------------------------
    def where(self, name: str, key: bytes, *, timeout_s: float = 2.0
              ) -> list[tuple[str, str, int]]:
        """Machines on this network holding a model. (peer, base_url, size)."""
        from .discovery import discover

        needle = name.strip().lower()
        out = []
        for beacon in discover(key, timeout_s=timeout_s):
            for row in (beacon.device.get("models") or []):
                if needle in str(row.get("name", "")).lower():
                    out.append((beacon.name, beacon.base_url, int(row.get("size") or 0)))
                    break
        return out

    def ensure(self, name: str, *, source: str = "", key: bytes | None = None,
               on_progress: "Callable[[int, int], None] | None" = None,
               on_note: "Callable[[str], None] | None" = None,
               autodownload: bool = True) -> Model:
        """Make sure this machine holds a model, preferring one on the network."""
        found = self.find(name)
        if found:
            return found
        if not autodownload:
            raise ModelError(
                f"{name} is not on this machine, and automatic downloading is off")

        if key is not None:
            for peer_name, base_url, _size in self.where(name, key):
                if on_note:
                    on_note(f"Copying {name} from {peer_name}")
                try:
                    return self._from_peer(name, base_url, key, on_progress)
                except (ModelError, OSError):
                    continue

        if not source:
            raise ModelError(
                f"no machine on this network has {name}, and no download was given")
        if on_note:
            on_note(f"Downloading {name}")
        return self._from_internet(name, source, on_progress)

    def _from_peer(self, name: str, base_url: str, key: bytes,
                   on_progress: Any) -> Model:
        from .discovery import derive_token
        from .remote import Peer

        peer = Peer(base_url, derive_token(key))
        holds = peer.models()
        match = next((m for m in holds
                      if name.strip().lower() in str(m.get("name", "")).lower()), None)
        if match is None:
            raise ModelError(f"{base_url} no longer has {name}")
        self.store.mkdir(parents=True, exist_ok=True)
        target = self.store / str(match["name"])
        peer.pull(str(match["name"]), target, on_progress=on_progress,
                  route="/models/")
        stat = target.stat()
        return Model(target.name, target, stat.st_size, stat.st_mtime)

    def _from_internet(self, name: str, source: str, on_progress: Any) -> Model:
        from .remote import range_total

        url = _resolve(source)
        self.store.mkdir(parents=True, exist_ok=True)
        # The name that was asked for wins: saving it under whatever the URL happened
        # to call it means find() will not match it afterwards.
        wanted = Path(name).name
        if Path(wanted).suffix.lower() not in SUFFIXES:
            wanted = Path(urllib.parse.urlparse(url).path).name or wanted
        target = self.store / wanted
        partial = target.with_suffix(target.suffix + ".part")
        stamp = Path(str(partial) + ".from")

        start = partial.stat().st_size if partial.exists() else 0
        origin = _read_stamp(stamp)
        if start and origin.get("url") not in (None, url):
            partial.unlink(missing_ok=True)
            stamp.unlink(missing_ok=True)
            start, origin = 0, {}

        req = urllib.request.Request(url, headers={"User-Agent": "ml-stack"})
        if start:
            req.add_header("Range", f"bytes={start}-")
            if origin.get("validator"):
                req.add_header("If-Range", str(origin["validator"]))
        try:
            response = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and start:
                size = range_total(exc.headers.get("Content-Range", ""))
                if size is not None and size == start:
                    os.replace(partial, target)
                    stamp.unlink(missing_ok=True)
                    stat = target.stat()
                    return Model(target.name, target, stat.st_size, stat.st_mtime)
                partial.unlink(missing_ok=True)
                stamp.unlink(missing_ok=True)
                raise ModelError(
                    f"{name}: the part here is {start} bytes but the file is "
                    f"{size}; discarded it, ask again") from None
            raise ModelError(f"could not download {name}: {exc.code}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise ModelError(f"could not download {name}: {exc}") from None

        # A server that does not honour Range answers 200 with the whole file.
        if start and response.status != 206:
            start = 0
        if start:
            total = range_total(response.headers.get("Content-Range", "")) or 0
        else:
            total = int(response.headers.get("Content-Length") or 0)

        _write_stamp(stamp, url, response.headers)
        done = start
        with response, partial.open("ab" if start else "wb") as fh:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if on_progress:
                    on_progress(done, total)
        if total and partial.stat().st_size != total:
            raise ModelError(
                f"{name}: got {partial.stat().st_size} of {total} bytes; "
                f"left {partial.name} to resume from")
        os.replace(partial, target)
        stamp.unlink(missing_ok=True)
        stat = target.stat()
        return Model(target.name, target, stat.st_size, stat.st_mtime)

    def ensure_draft(self, model: Model, source: str, *,
                     on_progress: "Callable[[int, int], None] | None" = None) -> Path:
        """Fetch the small model that guesses ahead for ``model``, beside it."""
        beside = model.path.with_suffix(DRAFT_MARK + model.path.suffix)
        if beside.is_file():
            return beside
        got = self._from_internet(beside.name, source, on_progress)
        if got.path != beside:
            os.replace(got.path, beside)
        return beside

    def remove(self, name: str) -> bool:
        found = self.find(name)
        if found is None or self.store not in found.path.parents:
            return False
        found.path.unlink(missing_ok=True)
        return True

    def unfinished(self, *, stale_s: float = STALE_PART_S) -> list[dict[str, Any]]:
        """Part files left by downloads that stopped, newest first."""
        import time

        if not self.store.exists():
            return []
        now = time.time()
        out = []
        for path in self.store.glob("*.part"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if now - stat.st_mtime < stale_s:
                continue
            out.append({"name": path.name, "size": stat.st_size,
                        "modified": stat.st_mtime})
        out.sort(key=lambda r: r["modified"], reverse=True)
        return out

    def discard(self, name: str = "") -> list[str]:
        """Delete a part file and what it recorded, or every stale one. Returns names."""
        wanted = [r["name"] for r in self.unfinished()] if not name else [Path(name).name]
        gone = []
        for part in wanted:
            path = self.store / part
            if path.suffix != ".part" or self.store not in path.parents:
                continue
            if not path.exists():
                continue
            path.unlink(missing_ok=True)
            Path(str(path) + ".from").unlink(missing_ok=True)
            gone.append(part)
        return gone

    def free_gb(self) -> float:
        try:
            self.store.mkdir(parents=True, exist_ok=True)
            return round(shutil.disk_usage(self.store).free / 2**30, 1)
        except OSError:
            return 0.0


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A model worth offering without anyone having to go looking for one."""

    name: str
    ref: str
    gb: float
    what: str
    family: str = ""
    params_b: float = 0.0
    active_b: float = 0.0
    takes: tuple[str, ...] = ("text",)
    gives: tuple[str, ...] = ("text",)
    unfiltered: bool = False
    draft_ref: str = ""
    draft_gb: float = 0.0

    @property
    def file(self) -> str:
        return self.ref.rsplit("/", 1)[-1]

    @property
    def moe(self) -> bool:
        return self.active_b > 0

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "ref": self.ref, "gb": self.gb,
                "what": self.what, "file": self.file,
                "family": self.family or family_of(self.name),
                "params_b": self.params_b, "active_b": self.active_b,
                "moe": self.moe,
                "takes": [MODALITY.get(m, m) for m in self.takes],
                "gives": [MODALITY.get(m, m) for m in self.gives],
                "unfiltered": self.unfiltered or is_unfiltered(self.name),
                "draft_ref": self.draft_ref, "draft_gb": self.draft_gb}


MODALITY = {"text": "💬", "image": "🖼", "audio": "🔊", "video": "🎬"}

# Names publishers give a model whose refusals have been trained or edited out.
UNFILTERED = ("uncensored", "abliterated", "obliterated", "unfiltered",
              "unleashed", "unchained", "heretic", "nsfw", "jailbreak",
              "norefusal", "no-refusal", "unaligned", "unsafe")


DRAFT_MARK = ".draft"


def draft_beside(model: Path) -> Path | None:
    """The small model kept next to ``model`` to guess ahead with, if one was got."""
    beside = model.with_suffix(DRAFT_MARK + model.suffix)
    return beside if beside.is_file() else None


def is_unfiltered(name: str) -> bool:
    """Whether a model is published as one that will not decline."""
    low = name.lower()
    return any(word in low for word in UNFILTERED)


# What to fall back on when Hugging Face cannot be reached. Anything shipped here is
# out of date the day it is written, so it is a backstop, not the list.
SUGGESTED: tuple[Suggestion, ...] = (
    Suggestion("Qwen3 4B", "hf:Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", 2.4,
               "Fast, and good on a laptop."),
    Suggestion("Qwen3 8B", "hf:Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", 4.7,
               "Better answers, still comfortable on 16 GB."),
    Suggestion("Qwen3 14B", "hf:unsloth/Qwen3-14B-GGUF/Qwen3-14B-Q4_K_M.gguf", 8.4,
               "For a machine with room to spare."),
    Suggestion("Gemma 3 4B",
               "hf:ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf", 2.3,
               "Google's small one."),
    Suggestion("Llama 3.2 3B",
               "hf:bartowski/Llama-3.2-3B-Instruct-GGUF/"
               "Llama-3.2-3B-Instruct-Q4_K_M.gguf", 1.9,
               "Small and widely used."),
    Suggestion("Tiny stories 15M",
               "hf:ggml-org/models/tinyllamas/stories15M-q4_0.gguf", 0.02,
               "Twenty megabytes, for seeing that all this works."),
)


def suggestions(free_gb: float = 0.0, ram_gb: float = 0.0) -> list[Suggestion]:
    """Models worth offering here, smallest first, leaving out what will not fit."""
    out = []
    for pick in SUGGESTED:
        if free_gb and pick.gb > free_gb:
            continue
        if ram_gb and pick.gb > ram_gb:
            continue
        out.append(pick)
    return sorted(out, key=lambda s: s.gb)


# Families worth recognising wherever they appear in a name, so Ternary-Bonsai-27B
# lands under Bonsai rather than starting a family of its own. Longest first: gpt-oss
# must win over gpt. A name matching nothing here still gets a family from its first
# word, so a release nobody has heard of yet still groups.
KNOWN = {
    "gpt-oss": "GPT-OSS", "command-r": "Command R", "deepseek": "DeepSeek",
    "nemotron": "Nemotron", "minicpm": "MiniCPM", "smollm": "SmolLM",
    "granite": "Granite", "mistral": "Mistral", "mixtral": "Mixtral",
    "starcoder": "StarCoder", "codestral": "Codestral", "exaone": "EXAONE",
    "internlm": "InternLM", "falcon": "Falcon", "bonsai": "Bonsai",
    "ornith": "Ornith", "gemma": "Gemma", "llama": "Llama", "qwen": "Qwen",
    "phi": "Phi", "olmo": "OLMo", "yi": "Yi", "glm": "GLM", "lfm": "LFM",
}
SPELLING = KNOWN

# Models published under a name of their own that are a fine-tune or an export of
# something else. The hub carries a base_model tag only sometimes -- Gemmable has
# none at all -- so a known lineage is written down rather than guessed at.
ALIAS = {
    "gemmable": "Gemma",
}
HUB = "https://huggingface.co/api/models"
POPULAR_TTL_S = 6 * 3600
SCAN = 40
WANT = 48
PER_PAGE = 12

# What a chat model is not. Asking for pipeline_tag=text-generation instead would drop
# anything the hub has not tagged, and the newest releases are often untagged.
NOT_CHAT = frozenset({
    "automatic-speech-recognition", "text-to-speech", "text-to-audio",
    "feature-extraction", "sentence-similarity", "fill-mask", "text-to-image",
    "image-to-image", "object-detection", "image-segmentation", "image-classification",
    "text-classification", "token-classification", "translation", "summarization",
    "audio-classification", "video-classification", "reinforcement-learning",
})
_popular: tuple[float, list[Suggestion]] = (0.0, [])
_drafts: dict[str, tuple[str, int] | None] = {}


def family_of(name: str) -> str:
    """The family a model name belongs to: what comes before the size or version.

    ``Qwen3-Coder-30B`` is Qwen, ``Ornith-1.5-9B`` is Ornith, ``gpt-oss-20b`` is
    GPT-OSS. Read from the name so a family nobody has heard of yet still groups.
    """
    bare = name.split("/")[-1]
    for tail in ("-GGUF", "-gguf", ".gguf"):
        bare = bare.removesuffix(tail)

    low = bare.lower()
    for needle in sorted(ALIAS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z])", low):
            return ALIAS[needle]
    for needle in sorted(KNOWN, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z])", low):
            return KNOWN[needle]

    head = re.match(r"[A-Za-z][A-Za-z\-_]*", bare)
    if not head:
        return "Other"
    word = head.group(0).rstrip("-_").replace("_", "-")
    while word and not word.split("-")[-1]:
        word = word.rsplit("-", 1)[0]
    low = word.lower()
    if low in SPELLING:
        return SPELLING[low]
    # A trailing word like "-Coder" or "-Instruct" is a variant, not a family.
    first = word.split("-")[0]
    return SPELLING.get(first.lower(), first[:1].upper() + first[1:])


def _hub(url: str, timeout: float = 25.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ml-stack"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _params_in(name: str) -> tuple[float, float]:
    """Total and active billions read off a name like ``Qwen3-Coder-30B-A3B``."""
    active = re.search(r"[-_]A(\d+(?:\.\d+)?)B\b", name, re.I)
    total = re.search(r"[-_](\d+(?:\.\d+)?)B\b", name, re.I)
    return (float(total.group(1)) if total else 0.0,
            float(active.group(1)) if active else 0.0)


def _modalities(facts: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """What a model reads and what it writes, from how the hub files it."""
    tags = {str(t).lower() for t in facts.get("tags") or ()}
    pipeline = str(facts.get("pipeline_tag") or "").lower()
    marks = tags | {pipeline}

    takes = ["text"]
    if any("image-text-to-text" in m or "vision" in m or "multimodal" in m
           for m in marks):
        takes.append("image")
    if any("audio" in m or "speech" in m for m in marks):
        takes.append("audio")
    if any("video" in m for m in marks):
        takes.append("video")

    gives = ["text"]
    if "text-to-image" in marks:
        gives = ["image"]
    elif "text-to-speech" in marks:
        gives = ["audio"]
    return tuple(takes), tuple(gives)


def _repo_facts(repo: str) -> dict[str, Any]:
    try:
        return _hub(f"{HUB}/{repo}")
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def _best_gguf(repo: str) -> tuple[str, int, bool] | None:
    """The Q4 build, its size, and whether the repository ships a vision projector."""
    try:
        tree = _hub(f"{HUB}/{repo}/tree/main?recursive=1")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    whole = []
    sees = False
    drafts: list[tuple[str, int]] = []
    for row in tree if isinstance(tree, list) else []:
        path = str(row.get("path", ""))
        if not path.lower().endswith(".gguf"):
            continue
        stem = path.rsplit("/", 1)[-1].lower()
        size = int(row.get("size") or (row.get("lfs") or {}).get("size") or 0)
        if "mmproj" in stem:
            sees = True
        if size and ("draft" in stem or "mtp" in stem):
            drafts.append((path, size))
        if _is_a_piece(path) or _is_beside(path):
            continue
        if size:
            whole.append((path, size))
    if not whole:
        return None
    _drafts[repo] = min(drafts, key=lambda x: x[1]) if drafts else None
    # A file at the top level is the model; one in a subdirectory is a variant.
    flat = [x for x in whole if "/" not in x[0]] or whole
    for want in QUANTS:
        for path, size in sorted(flat):
            if want in path.lower():
                return path, size, sees
    path, size = sorted(flat)[0]
    return path, size, sees


def _ranked() -> list[dict[str, Any]]:
    """The hub's two orderings, folded into one.

    Downloads alone is the last thirty days but favours whatever has been around all
    month; trending alone swings on a day's noise. A model near the top of either
    belongs on the first page, so each is scored by where it sits in both.
    """
    boards = []
    for order in ("downloads", "trendingScore"):
        try:
            got = _hub(f"{HUB}?filter=gguf&sort={order}&direction=-1&limit={SCAN}")
        except (urllib.error.URLError, OSError, ValueError):
            got = []
        boards.append([r for r in got if isinstance(r, dict)])
    if not any(boards):
        raise urllib.error.URLError("no board")

    score: dict[str, float] = {}
    rows: dict[str, dict[str, Any]] = {}
    for board in boards:
        for place, row in enumerate(board):
            repo = str(row.get("id") or "")
            if not repo:
                continue
            rows[repo] = row
            score[repo] = score.get(repo, 0.0) + 1.0 / (place + 1)
    return [rows[r] for r in sorted(score, key=lambda r: score[r], reverse=True)]


def popular(free_gb: float = 0.0, ram_gb: float = 0.0, *, limit: int = PER_PAGE,
            page: int = 0, rude: bool = False,
            query: str = "") -> list[Suggestion]:
    """What people are actually running, asked of Hugging Face rather than remembered.

    With a ``query`` the hub is searched instead, so typing narrows the same list.
    Falls back to ``suggestions`` when the hub cannot be reached.
    """
    global _popular
    if query.strip():
        return _searched(query.strip(), free_gb, ram_gb, limit=limit, page=page,
                         rude=rude)
    age, cached = _popular
    if time.time() - age > POPULAR_TTL_S or not cached:
        try:
            listed = _ranked()
        except (urllib.error.URLError, OSError, ValueError):
            return suggestions(free_gb, ram_gb)
        found = _resolve_rows(listed)
        if not found:
            return suggestions(free_gb, ram_gb)
        _popular = (time.time(), found)
        cached = found

    # Kept in the order the fold gave them: that order is the popularity. Filtered
    # before the page is cut, or a page would arrive half empty.
    out = _fitting(cached, free_gb, ram_gb, rude)
    start = max(0, page) * limit
    return out[start:start + limit]


def _fitting(rows: list[Suggestion], free_gb: float, ram_gb: float,
             rude: bool) -> list[Suggestion]:
    return [p for p in rows
            if (not free_gb or p.gb <= free_gb)
            and (not ram_gb or p.gb <= ram_gb)
            and (rude or not p.public()["unfiltered"])]


def _resolve_rows(listed: list[dict[str, Any]]) -> list[Suggestion]:
    """Turn hub rows into something the screen can show, stopping at ``WANT``."""
    found: list[Suggestion] = []
    for row in listed:
        repo = str(row.get("id") or "")
        if not repo or "/" not in repo:
            continue
        if str(row.get("pipeline_tag") or "") in NOT_CHAT:
            continue
        best = _best_gguf(repo)
        if best is None:
            continue
        path, size, sees = best
        draft = _drafts.get(repo)
        short = repo.split("/")[-1].removesuffix("-GGUF").removesuffix("-gguf")
        facts = _repo_facts(repo)
        total_b, active_b = _params_in(short)
        exact = int((facts.get("gguf") or {}).get("total") or 0)
        if exact:
            total_b = round(exact / 1e9, 1)
        takes, gives = _modalities(facts)
        if sees and "image" not in takes:
            takes = (*takes, "image")
        found.append(Suggestion(
            name=short, ref=f"hf:{repo}/{path}", gb=round(size / 2**30, 2),
            what=f"from {repo.split('/')[0]}", family=family_of(short),
            params_b=total_b, active_b=active_b, takes=takes, gives=gives,
            unfiltered=is_unfiltered(repo),
            draft_ref=f"hf:{repo}/{draft[0]}" if draft else "",
            draft_gb=round(draft[1] / 2**30, 2) if draft else 0.0))
        if len(found) >= WANT:
            break
    return found


_found: dict[str, list[Suggestion]] = {}


def _searched(query: str, free_gb: float, ram_gb: float, *, limit: int,
              page: int, rude: bool) -> list[Suggestion]:
    """The hub's answer to a search, resolved the same way the popular list is."""
    if query not in _found:
        try:
            listed = _hub(f"{HUB}?filter=gguf&search={urllib.parse.quote(query)}"
                          f"&sort=downloads&direction=-1&limit={SCAN}")
        except (urllib.error.URLError, OSError, ValueError):
            return []
        _found[query] = _resolve_rows(listed if isinstance(listed, list) else [])
    out = _fitting(_found[query], free_gb, ram_gb, rude)
    start = max(0, page) * limit
    return out[start:start + limit]


def searched_count(query: str, free_gb: float = 0.0, ram_gb: float = 0.0, *,
                   rude: bool = False) -> int:
    return len(_fitting(_found.get(query.strip(), []), free_gb, ram_gb, rude))


def searched_families(query: str, free_gb: float = 0.0, ram_gb: float = 0.0, *,
                      rude: bool = False) -> list[str]:
    return sorted({p.public()["family"]
                   for p in _fitting(_found.get(query.strip(), []), free_gb, ram_gb,
                                     rude)})


def how_many(free_gb: float = 0.0, ram_gb: float = 0.0, *, rude: bool = False) -> int:
    """How many models the last look found that fit here."""
    _, cached = _popular
    return len(_fitting(cached, free_gb, ram_gb, rude))


def families(free_gb: float = 0.0, ram_gb: float = 0.0, *,
             rude: bool = False) -> list[str]:
    """Every family across the whole list, not just the page being shown."""
    _, cached = _popular
    return sorted({p.public()["family"]
                   for p in _fitting(cached, free_gb, ram_gb, rude)})


@dataclass
class Getting:
    """One model being fetched, and how far it has got."""

    id: str
    name: str
    source: str = ""
    state: str = "getting"          # getting | done | failed
    note: str = ""
    done: int = 0
    total: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "source": self.source,
                "state": self.state, "note": self.note, "done": self.done,
                "total": self.total, "error": self.error,
                "started_at": self.started_at, "finished_at": self.finished_at}


class Downloads:
    """Model fetches running in the background, so a big one does not hold a request."""

    KEEP_S = 300.0

    def __init__(self, models: "Models", *, slots: int = 1) -> None:
        self.models = models
        self.getting: dict[str, Getting] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(slots)

    def start(self, name: str, *, source: str = "", key: bytes | None = None,
              autodownload: bool = True, draft: str = "") -> Getting:
        with self._lock:
            for row in self.getting.values():
                if row.state == "getting" and row.name == name:
                    return row
        row = Getting(id=f"{int(time.time())}-{secrets.token_hex(3)}",
                      name=name, source=source)
        with self._lock:
            self.getting[row.id] = row
        threading.Thread(target=self._run, args=(row, key, autodownload, draft),
                         daemon=True, name=f"get-{row.id}").start()
        return row

    def _run(self, row: Getting, key: bytes | None, autodownload: bool,
             draft: str = "") -> None:
        with self._sem:
            try:
                def progress(done: int, total: int) -> None:
                    row.done, row.total = done, total

                def note(text: str) -> None:
                    row.note = text

                got = self.models.ensure(row.name, source=row.source, key=key,
                                         on_progress=progress, on_note=note,
                                         autodownload=autodownload)
                if draft:
                    row.note = f"Getting the draft for {got.name}"
                    try:
                        self.models.ensure_draft(got, draft, on_progress=progress)
                    except (ModelError, OSError):
                        pass          # a model without its draft still runs
                row.state = "done"
                row.name = got.name
                row.done = row.total = got.size
            except Exception as exc:                  # noqa: BLE001
                row.state = "failed"
                row.error = str(exc)
            finally:
                row.finished_at = time.time()

    def active(self) -> list[Getting]:
        """Everything still running, plus anything that finished recently."""
        now = time.time()
        with self._lock:
            for key in [k for k, r in self.getting.items()
                        if r.finished_at and now - r.finished_at > self.KEEP_S]:
                del self.getting[key]
            rows = sorted(self.getting.values(), key=lambda r: r.started_at)
        return rows


QUANTS = ("q4_k_m", "q4_k_s", "q4_1", "q4_0", "q5_k_m", "q8_0")


def _resolve(source: str) -> str:
    """``hf:owner/repo/file.gguf``, ``hf:owner/repo``, or a plain URL.

    A reference naming only a repository is answered with its Q4 build, which is the
    one worth having on a machine that has to fit the model in memory.
    """
    source = source.strip()
    if source.startswith("hf:"):
        ref = source[3:].strip("/")
        parts = ref.split("/")
        if len(parts) == 2:
            parts = [parts[0], parts[1], _quant_in(parts[0], parts[1])]
        if len(parts) < 3:
            raise ModelError(f"{source!r} should look like hf:owner/repo/file.gguf")
        owner, repo, path = parts[0], parts[1], "/".join(parts[2:])
        return f"https://huggingface.co/{owner}/{repo}/resolve/main/{path}?download=true"
    if source.startswith(("http://", "https://")):
        return source
    raise ModelError(f"{source!r} is not a URL or an hf: reference")


def _read_repo_files(owner: str, repo: str) -> list[str]:
    """Every file Hugging Face lists in a repository."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{owner}/{repo}",
            headers={"User-Agent": "ml-stack"})
        with urllib.request.urlopen(req, timeout=30) as r:
            listed = json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ModelError(f"could not read hf:{owner}/{repo}: {exc}") from None
    return [str(f.get("rfilename", "")) for f in listed.get("siblings") or []]


def _quant_in(owner: str, repo: str) -> str:
    """The file to take from a repository nobody named a file in."""
    files = _read_repo_files(owner, repo)
    whole = [f for f in files
             if f.lower().endswith(".gguf") and not _is_a_piece(f)
             and not _is_beside(f)]
    if not whole:
        raise ModelError(f"hf:{owner}/{repo} holds no single-file gguf")
    for want in QUANTS:
        for name in sorted(whole):
            if want in name.lower():
                return name
    return sorted(whole)[0]


# Files that sit beside a model rather than being one: a vision projector, a
# multi-token-prediction head, a draft model, an importance matrix.
BESIDE = ("mmproj", "mtp", "draft", "imatrix", "vocab", "lora", "adapter")


def _is_a_piece(name: str) -> bool:
    """A shard such as ``model-00001-of-00003.gguf``, which is no use on its own."""
    return bool(re.search(r"-\d{5}-of-\d{5}", name))


def _is_beside(name: str) -> bool:
    """Whether a file is an accessory rather than the model itself."""
    stem = name.rsplit("/", 1)[-1].lower()
    return any(word in stem for word in BESIDE)
