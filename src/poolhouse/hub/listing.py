"""What a repository on the Hub holds: searching it, listing its builds, downloading one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poolhouse import hub
from poolhouse.hub import remote
from poolhouse.hub.naming import _SHARD, _precision
from poolhouse.hub.transfer import pull

# Publishers whose quantisations tend to be there first and be right. Ordered: the first one
# that has a model wins. Override with --prefer; pass --prefer '' to rank by downloads alone.
PREFER = ("unsloth", "ggml-org", "google", "bartowski", "lmstudio-community")


@dataclass(frozen=True)
class Found:
    """One repository that looks like it holds the model asked for."""

    repo: str
    downloads: int = 0
    likes: int = 0

    @property
    def owner(self) -> str:
        return self.repo.split("/")[0]


def find(query: str, *, prefer: tuple[str, ...] = PREFER, gguf: bool = True,
         limit: int = 12) -> list[Found]:
    """Repositories matching ``query``, the trusted publishers first.

    The Hub's own relevance ordering puts whatever is popular first, which for a new model
    is usually somebody's uncensored remix rather than the release. Ranking by publisher
    fixes that without pretending to know which repo is "official".
    """
    seen: dict[str, Found] = {}
    for term in (query, f"{query} GGUF") if gguf else (query,):
        for model in remote.models(term, 100):
            name = str(model["id"])
            if gguf and "gguf" not in name.lower():
                continue
            seen[name] = Found(repo=name, downloads=int(model.get("downloads") or 0),
                               likes=int(model.get("likes") or 0))

    def rank(one: Found) -> tuple[int, int]:
        owner = one.owner.lower()
        where = prefer.index(owner) if owner in prefer else len(prefer)
        return (where, -one.downloads)

    return sorted(seen.values(), key=rank)[:limit]


def files(repo: str, *, ending: str = ".gguf") -> list[tuple[str, int]]:
    """What is in a repository, largest last, as ``(name, bytes)``.

    A GGUF repo holds one file per quantisation and sometimes a projector or a draft beside
    them; which one you want is a judgement about memory, and needs the sizes to make.
    """
    out = [(one.path, one.size) for one in remote.listing(repo)
           if one.path.lower().endswith(ending)]
    # Weights first, largest first, and the small things that travel with them last: a
    # listing sorted the other way buries the model itself under vision projectors and
    # draft heads, which is what happened the first time this was used in anger.
    return sorted(out, key=lambda kv: (hub.aside(kv[0]), -kv[1]))


def ref(repo: str, name: str = "") -> str:
    """The reference `poolhouse-serve up` takes, which downloads and caches on first use."""
    return f"hf:{repo}/{name}" if name else f"hf:{repo}"


def beside(repo: str, prefix: str, *, best: bool = False,
          prefer: str | tuple[str, ...] = (), avoid: str | tuple[str, ...] = ()) -> str:
    """A file whose name starts with ``prefix`` in one repository, root copy preferred.

    ``best`` takes the most precise when a repository offers several, which is what a
    vision projector wants. ``prefer`` is one or more substrings, most specific first,
    each tried in turn against the candidates found so far: the first that matches
    anything narrows to those matches, and nothing narrows when none does. ``avoid`` is
    substrings a candidate is set aside for while any candidate without them remains;
    nothing is set aside when every candidate matches.
    """
    words = (prefer,) if isinstance(prefer, str) else tuple(prefer)
    words = tuple(w for w in words if w)
    unwanted = (avoid,) if isinstance(avoid, str) else tuple(avoid)
    unwanted = tuple(w.lower() for w in unwanted if w)
    nested: list[str] = []
    root: list[str] = []
    try:
        listing = hub.files(repo)
    except Exception:  # noqa: BLE001 - a repository that is not there holds nothing
        return ""
    for name, _size in listing:
        plain = name.lower().rsplit("/", 1)[-1]
        if not plain.startswith(prefix):
            continue
        (nested if "/" in name else root).append(name)
    for found in (root, nested):
        if not found:
            continue
        kept = [f for f in found if not any(w in f.lower() for w in unwanted)]
        if kept:
            found = kept
        for word in words:
            matched = [f for f in found if word.lower() in f.lower()]
            if matched:
                found = matched
                break
        return ref(repo, min(found, key=_precision) if best else found[0])
    return ""


def mmproj_for(repo: str) -> str:
    """The vision projector shipped with a model, most precise first, or ''.

    Without one a multimodal model serves as a text model and says nothing about why an
    image was ignored -- which is the whole failure, since nothing errors. And a quantised
    projector is a false economy: it is a fraction of the weights and carries all of the
    seeing, so the precise one is taken whatever the model's own quantisation is.
    """
    return beside(repo, "mmproj-", best=True)


def builds(repo: str, *, ending: str = ".gguf") -> list[tuple[str, int, int]]:
    """What a repository offers, one row per build rather than per file.

    A large model is published in shards, one directory per quantisation, and a listing of
    forty files answers no question anybody has. What decides whether a model can be served
    is the *total* of a build -- Qwen3.8-Flash-Next is 329.7G at BF16 and 87.2G at
    UD-IQ4_XS, and reading that off a list of individual shards means adding up by hand.

    Returns ``(name, total bytes, shards)``, largest first, companions excluded.
    """
    grouped: dict[str, list[int]] = {}
    for name, size in hub.files(repo, ending=ending):
        if hub.aside(name):
            continue
        # the quantisation is the directory when there is one, else the shard-less filename
        stem = name.split("/")[0] if "/" in name else _SHARD.sub("", name.rsplit("/", 1)[-1])
        grouped.setdefault(stem, []).append(size)
    return sorted(((name, sum(sizes), len(sizes)) for name, sizes in grouped.items()),
                  key=lambda row: -row[1])


def build_files(repo: str, build: str, ending: str = ".gguf") -> list[tuple[str, int]]:
    """The filenames belonging to one build of a repository."""
    return [(name.rsplit("/", 1)[-1], size) for name, size in hub.files(repo, ending=ending)
            if not hub.aside(name) and (name.split("/")[0] == build
                                        or name.rsplit("/", 1)[-1] == build)]


def fetch(reference: str, *, peers: bool | None = None) -> Path:
    """Download an `hf:` reference into the Hugging Face hub cache, without serving it.

    A sharded model's *every* shard comes down, not only the one named: the file given is
    one member of a build, and a server started against a partial download fails at the far
    end of the load complaining about a missing shard.
    """
    from poolhouse.serve.backend import ServerSpec

    parts = ServerSpec.hf_parts(reference)
    if parts is None or not parts[1]:
        raise ValueError(f"{reference!r} should look like hf:owner/repo/file.gguf")
    repo, name = parts
    try:
        got = pull(f"hf:{repo}/{name}", peers=peers)
    except remote.NotFound as exc:
        raise ValueError(str(exc)) from exc
    except remote.RemoteError as exc:
        raise OSError(str(exc)) from exc
    return got.parent / name.rsplit("/", 1)[-1]
