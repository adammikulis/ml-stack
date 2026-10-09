"""Choosing a model for a new machine and deciding what happens to the model cache already on its disk."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poolhouse.files import promote

__all__ = ["ADOPTED", "DEFAULT_MODEL", "IN_PLACE", "LEFT_ALONE", "CachePlan", "choose_model",
           "models_in", "needs_bytes", "plan_cache"]

# -- what a new machine should start with ------------------------------------------------
DEFAULT_MODEL = "gemma-4-E2B-it-qat-UD-Q4_K_XL"
"""What a first install downloads unless asked otherwise: 2.6G, measured at 1.5 s a
question. The smallest thing that still answers, so the first thing a person does on a new
machine finishes while they are still watching. The bigger ones are a pick, not a default:
a machine with the room is offered E4B (4.4G, 3 s) and Flash-Next (104G, 27 s), and
``--models auto`` takes the best that fits, which is what a headless install does."""


def needs_bytes(fit: dict[str, Any], context: int = 0) -> int:
    """What one measured fit says a model costs to serve: weights, draft, cache, compute.

    The numbers come from `poolhouse.data.fit.json`, which is measured rather than guessed;
    ``context`` overrides the context that measurement used.
    """
    slots = int(context or fit.get("context") or 0)
    return (int(fit.get("weights") or 0) + int(fit.get("draft") or 0)
            + int(fit.get("per_token") or 0) * slots
            + int(fit.get("per_seq") or 0) + int(fit.get("compute") or 0))


def choose_model(room_bytes: int, *, want: str = "auto",
                 profiles: Sequence[Any] | None = None,
                 fits: Sequence[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """The best model this machine has room for, out of the ones that were measured.

    ``profiles`` is in the order the measurements ranked them, so this walks it and takes
    the first whose smallest measured fit leaves a fifth of the room spare -- serving a
    model with nothing left over is a machine that swaps the moment somebody asks it two
    questions. ``want`` narrows it: a word matched against the model's name (``flash-next``,
    ``small``), ``none`` picks nothing, ``auto`` takes the ranking as it stands.

    Returns the model, the draft head beside it and the build it needs, which is what
    ``poolhouse-models fetch`` and ``poolhouse-serve build`` take -- or None when nothing
    measured fits, which is an answer: the caller says so rather than fetching a model the
    machine cannot load.
    """
    asked = want.strip().lower()
    if asked in ("none", "off", ""):
        return None
    if asked == "default":
        asked = DEFAULT_MODEL.lower()
    if profiles is None:
        from poolhouse.serve.profile import profiles as read_profiles

        profiles = read_profiles()
    if fits is None:
        import json as _json

        from poolhouse.data import __file__ as data_init

        fits = _json.loads((Path(data_init).parent / "fit.json").read_text())
    by_model: dict[str, int] = {}
    for one in fits or ():
        name = str(one.get("model") or "")
        cost = needs_bytes(one)
        if name and (name not in by_model or cost < by_model[name]):
            by_model[name] = cost

    narrow = "" if asked == "auto" else asked
    for profile in profiles or ():
        model = str(getattr(profile, "model", "") or "")
        if not model or (narrow and narrow not in model.lower()):
            continue
        cost = by_model.get(model, 0)
        if cost and room_bytes and cost * 1.2 > room_bytes:
            continue
        return {"model": model,
                "draft": str(getattr(profile, "draft", "") or ""),
                "build": str(getattr(profile, "build", "") or ""),
                "bytes": cost}
    return None


# -- the models already on the disk ------------------------------------------------------
IN_PLACE, ADOPTED, LEFT_ALONE = "in place", "adopted", "left alone"


@dataclass(frozen=True, slots=True)
class CachePlan:
    """What a per-machine install did about the model cache that was already there.

    Three outcomes and no others. **in place**: the service runs as the person who
    installed it, so it opens their ``~/.cache/huggingface`` exactly where it is -- nothing
    is moved, linked or fetched. **adopted**: the service must run as another account, and
    they said yes, so the cache directory is *moved* to the shared location and a symlink
    left at the old path -- their own tools keep working and every file exists once.
    **left alone**: they said no, so their cache is untouched and the service starts with
    an empty one and will download what it needs.

    Never a copy. A cache is tens of gigabytes and a machine with two of them is a machine
    with a full disk and no idea why.
    """

    decision: str
    user_cache: Path
    service_cache: Path
    models: tuple[tuple[str, int], ...] = ()
    bytes: int = 0
    said: str = ""
    error: str = ""

    def public(self) -> dict[str, Any]:
        return {"decision": self.decision, "user_cache": str(self.user_cache),
                "service_cache": str(self.service_cache),
                "models": [{"name": n, "bytes": b} for n, b in self.models],
                "bytes": self.bytes, "said": self.said, "error": self.error}


def _size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def models_in(cache: Path | str) -> list[tuple[str, int]]:
    """Every model in a Hub cache, with its size on disk. Biggest first.

    The Hub's layout is one ``models--owner--repo`` directory per repository; anything else
    in there is not a model and is not counted.
    """
    hub = Path(cache).expanduser()
    hub = hub / "hub" if (hub / "hub").is_dir() else hub
    if not hub.is_dir():
        return []
    found = [(one.name.removeprefix("models--").replace("--", "/"), _size(one))
             for one in sorted(hub.iterdir())
             if one.is_dir() and one.name.startswith("models--")]
    return sorted(found, key=lambda pair: -pair[1])


def plan_cache(user_cache: Path | str, service_cache: Path | str, *,
               same_user: bool, adopt: bool = False) -> CachePlan:
    """Decide -- and carry out -- what happens to the models already on this machine.

    ``same_user`` is the good case and the default an installer should aim for: the service
    account *is* the person, so their cache is the service's cache and this does nothing at
    all. Otherwise ``adopt`` moves it (with a symlink left behind) and not adopting leaves
    it alone; neither ever happens silently -- the caller asks, having been given
    `models_in` to show what is at stake.

    Running it twice is a no-op: an already-adopted cache is a symlink at the old path
    pointing at the new one, and that is recognised rather than moved again.
    """
    mine = Path(user_cache).expanduser()
    theirs = Path(service_cache).expanduser()
    found = tuple(models_in(mine))
    total = sum(b for _, b in found)

    if same_user:
        return CachePlan(IN_PLACE, mine, mine, found, total,
                         said=f"the service runs as this user, so it reads {mine} where it "
                              f"is -- {len(found)} model(s), {_gb(total)}, nothing moved")
    if mine.is_symlink():
        target = mine.resolve()
        if target == theirs.resolve():
            return CachePlan(ADOPTED, mine, theirs, tuple(models_in(theirs)),
                             _size(theirs) if theirs.is_dir() else 0,
                             said=f"{mine} already points at {theirs}; nothing to do")
    if not adopt:
        listing = ", ".join(f"{n} ({_gb(b)})" for n, b in found[:5]) or "nothing"
        return CachePlan(LEFT_ALONE, mine, theirs, found, total,
                         said=f"{mine} is left alone ({listing}). The service runs as "
                              f"another account and starts with an empty cache, so it will "
                              f"download what it needs again -- {_gb(total)} of it. "
                              f"Re-run with --adopt-cache to move it instead.")
    if not mine.is_dir():
        return CachePlan(LEFT_ALONE, mine, theirs, said=f"there is no cache at {mine}")
    theirs.parent.mkdir(parents=True, exist_ok=True)
    if theirs.exists():
        return CachePlan(LEFT_ALONE, mine, theirs, found, total,
                         error=f"{theirs} already exists; move or remove it and re-run, "
                               f"rather than having two caches")
    try:
        promote(mine, theirs)
    except OSError as exc:
        # A cross-device rename would be a copy, and a copy is the one thing this must
        # never do: it doubles tens of gigabytes on a disk that has them once.
        return CachePlan(LEFT_ALONE, mine, theirs, found, total,
                         error=f"could not move {mine} to {theirs} without copying it "
                               f"({exc}); put the shared cache on the same filesystem, or "
                               f"run the service as this user instead")
    mine.symlink_to(theirs, target_is_directory=True)
    return CachePlan(ADOPTED, mine, theirs, found, total,
                     said=f"moved {len(found)} model(s), {_gb(total)}, to {theirs} and left "
                          f"a link at {mine} -- every file exists once, and both accounts "
                          f"read the same one")
