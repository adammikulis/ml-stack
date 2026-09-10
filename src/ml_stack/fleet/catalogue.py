"""What the hub has to offer, and which of it fits this machine."""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass
from typing import Any

from ml_stack.http import ServerError, request_json

from .weights import QUANTS, is_a_piece, is_beside

__all__ = [
    "ALIAS",
    "HUB",
    "KNOWN",
    "MODALITY",
    "NOT_CHAT",
    "PER_PAGE",
    "SPELLING",
    "SUGGESTED",
    "UNFILTERED",
    "Suggestion",
    "families",
    "family_of",
    "how_many",
    "is_unfiltered",
    "popular",
    "searched_count",
    "searched_families",
    "suggestions",
]


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


def is_unfiltered(name: str) -> bool:
    """Whether a model is published as one that will not decline."""
    low = name.lower()
    return any(word in low for word in UNFILTERED)


# What to fall back on when Hugging Face cannot be reached. Anything shipped here is
# out of date the day it is written, so it is a backstop, not the list.
SUGGESTED: tuple[Suggestion, ...] = (
    Suggestion("Qwen3.5 4B", "hf:unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf", 2.6,
               "Fast, and good on a laptop."),
    Suggestion("Gemma 4 E4B",
               "hf:unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf", 4.7,
               "Google's small one."),
    Suggestion("Qwen3.5 9B", "hf:unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf", 5.3,
               "Better answers, still comfortable on 16 GB."),
    Suggestion("Gemma 4 12B",
               "hf:unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf", 6.7,
               "For a machine with room to spare."),
    Suggestion("Qwen3.8 27B",
               "hf:unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf", 15.4,
               "The best of these, for a machine with 24 GB or more."),
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
    return request_json(url, method="GET", timeout=timeout, tries=3)


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
    except (ServerError, urllib.error.URLError, OSError, ValueError):
        return {}


def _best_gguf(repo: str) -> tuple[str, int, bool] | None:
    """The Q4 build, its size, and whether the repository ships a vision projector."""
    try:
        tree = _hub(f"{HUB}/{repo}/tree/main?recursive=1")
    except (ServerError, urllib.error.URLError, OSError, ValueError):
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
        if is_a_piece(path) or is_beside(path):
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
        except (ServerError, urllib.error.URLError, OSError, ValueError):
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
        except (ServerError, urllib.error.URLError, OSError, ValueError):
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
        except (ServerError, urllib.error.URLError, OSError, ValueError):
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

