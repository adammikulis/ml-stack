"""The draft head a model serves with: what a repository ships, what is on this machine,
which build can load it, and the one line that says so."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ml_stack import home, hub
from ml_stack.hub.naming import DRAFT_KINDS
from ml_stack.units import human_bytes

DRAFT_DEPTH = 4
"""Tokens a chosen draft head guesses ahead of the model it drafts for."""

NO_HEAD = ("serving without a draft head: nothing guesses tokens ahead for the model to "
           "check in one pass, which is slower")

_DRAFT_NOTES: dict[str, str] = {}


def draft_for(repo: str, *, borrows: bool = False) -> str:
    """The draft head shipped beside the weights, as a reference, or ''.

    A repository that ships one carries a file named for the method it implements. It may
    sit at the root, under `MTP/`, or in a sibling `-MTP-GGUF` repository; the root copy is
    preferred, and among several at one level a `*shared*` head is.

    ``borrows`` says whether the binary about to serve this head can load one that needs a
    fork. A head whose own README says it needs one (``draft_note``) is offered only then.
    """
    found, where = _head_in(repo, prefer=("shared-q8_0", "shared"))
    if not found:
        return ""
    if not borrows and hub.draft_note(where):
        return ""
    return found


def _head_in(repo: str, *, prefer: tuple[str, ...] = (),
             avoid: tuple[str, ...] = ()) -> tuple[str, str]:
    """The draft head shipped with ``repo``, and which repository it was found in.

    ``(reference, repository)``, or ``("", "")``. The repository matters because its README
    is where a publisher says whether the head needs a fork (`draft_note`), and a head found
    in a sibling repository is warned about by the sibling's README, not the model's.

    Only a file actually named for a method in `DRAFT_KINDS` is taken, so a `-MTP-GGUF`
    repository that is the whole model rebuilt with the prediction layers in it -- 36G of
    weights, to be served with `--spec-type` -- correctly yields nothing.
    """
    stem = repo[: -len("-GGUF")] if repo.upper().endswith("-GGUF") else repo
    for where in (repo, f"{stem}-MTP-GGUF", f"{stem}-MTP"):
        for prefix in DRAFT_KINDS:
            found = hub.beside(where, prefix, prefer=prefer, avoid=avoid)
            if found:
                return found, where
    return "", ""


def draft_note(repo: str) -> str:
    """The sentence a draft head's own README says about needing a fork, or ''.

    `ml-stack-models files` prints this under the draft line it already reports, so a
    person is told a head needs a fork before spending a load on it. Read once per
    repository and cached for the life of the process.
    """
    if repo in _DRAFT_NOTES:
        return _DRAFT_NOTES[repo]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        # the note is a courtesy; a head is still chosen without the hub package, and the
        # choice must not raise where it used to be swallowed into "no head"
        return ""

    text = ""
    for name in ("MTP/README.md", "README.md"):
        try:
            text = Path(hf_hub_download(repo, name)).read_text(errors="replace")
            break
        except Exception:  # noqa: BLE001 - no README there, or the repo is not there at all
            continue

    note = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        low = sentence.lower()
        if "mainline" in low or "fork" in low or "requires" in low:
            note = " ".join(sentence.split())
            break
    _DRAFT_NOTES[repo] = note
    return note


def drafts_for(model: str | Path, *, files: Sequence[Path] | None = None) -> tuple[Path, ...]:
    """Every draft head on this machine that could draft for ``model``.

    Two sources: a head shipped beside the weights -- the same directory, an ``MTP/``
    folder under it, or the snapshot above it, which is where a Hub download of one
    repository puts one -- and a head anywhere else under the model roots whose name says
    it is for the same base model.
    """
    where = Path(str(model)).expanduser()
    if "/" not in str(model) and not where.is_file():
        where = hub.located(str(model)) or where
    wanted = hub.base_words(where.name)
    roots = {root.resolve() for root in hub.default_roots(home.home())}
    near = {where.parent}
    if where.parent.resolve() not in roots:
        near.add(where.parent.parent)
    close: list[Path] = []
    far: list[Path] = []
    for path in (list(files) if files is not None else hub.weight_paths()):
        if not hub.is_head(path.name):
            continue
        if path.parent in near or path.parent.parent in near:
            close.append(path)
            continue
        words = hub.base_words(path.name)
        if wanted and words and words == wanted[: len(words)]:
            far.append(path)
    # a Hub cache reaches the same blob from more than one root, and only one is a choice
    seen: dict[Path, Path] = {}
    for path in close + far:
        try:
            seen.setdefault(path.resolve(), path)
        except OSError:
            continue
    return tuple(seen.values())


def forks(binary: str | Path | None = None,
          builds: Sequence[tuple[str, Path]] | None = None) -> tuple[bool, list[str]]:
    """Whether the binary that would serve loads a head that borrows its target's
    embeddings, and the named builds here that do, the ones a record measured first.

    ``binary`` None is the one `find_binary` would pick.
    """
    from ml_stack.serve import binary as builds_module

    chosen = binary if binary is not None else builds_module.find_binary()
    if builds_module.borrows(chosen):
        return True, []
    from ml_stack.serve import profile as records

    measured = {str(one.build) for one in records.profiles() if one.build}
    named = list(builds) if builds is not None else builds_module.named_builds()
    found = [name for name, one in named if builds_module.borrows(one)]
    return False, sorted(found, key=lambda name: (name not in measured, name))


def drafting(draft: str = "", spec_type: str = "", depth: int | None = None,
             build: str = "", size: int = 0) -> str:
    """One line naming the draft head a model serves with, or that it serves without one."""
    if not draft:
        return NO_HEAD
    memory = f" ({human_bytes(size)} of memory)" if size else ""
    kind = f", {spec_type}" if spec_type else ""
    ahead = f", {depth} tokens ahead" if depth else ""
    on = f", on build {build}" if build else ""
    return f"drafting ahead with {Path(draft).name}{memory}{kind}{ahead}{on}"


@dataclass(frozen=True)
class Head:
    """A draft head on this machine: what to serve, how big, and which build loads it."""

    path: str
    bytes: int
    spec_type: str
    build: str = ""
    depth: int = DRAFT_DEPTH

    @property
    def name(self) -> str:
        return Path(self.path).name

    def said(self) -> str:
        """One line naming the head, its memory and the build it needs."""
        extra = f", needs --build {self.build}" if self.build else ""
        return f"{self.name} ({human_bytes(self.bytes)} of memory{extra})"

    def serving(self) -> str:
        """One line naming what the model is drafting ahead with."""
        return drafting(self.path, self.spec_type, self.depth, self.build, self.bytes)

    def over(self) -> dict[str, object]:
        """The `Config.over` fields that serve this head."""
        return {"draft": self.path, "spec_type": self.spec_type, "build": self.build,
                "draft_n_max": self.depth}


def head_choice(model: str | Path, asked: str = "auto", *,
                files: Sequence[Path] | None = None,
                heads: Sequence[Head] | None = None) -> Head | None:
    """The draft head to serve with ``model``, or None for none.

    'auto' takes the smallest head on this machine; 'none' takes no head; anything else
    names one of the heads found.
    """
    want = str(asked).strip().lower()
    if want in ("none", "no", "off", ""):
        return None
    found = list(heads) if heads is not None else hub.heads_for(model, files=files)
    if want == "auto":
        return found[0] if found else None
    wanted = Path(str(asked)).name.lower()
    for one in found:
        if one.name.lower() == wanted:
            return one
    named = ", ".join(one.name for one in found) or "none"
    raise ValueError(f"no draft head called {asked!r} for that model; here there is {named}")


def heads_for(model: str | Path, *, files: Sequence[Path] | None = None,
              builds: Sequence[tuple[str, Path]] | None = None,
              binary: str | Path | None = None) -> list[Head]:
    """The draft heads this machine could serve with ``model``, smallest first.

    The size is what the head adds to memory. A head that borrows its target's embeddings
    is named for the fork build that loads it, and is left out when this machine has no
    such build.
    """
    here, named = hub.forks(binary, builds)
    fork = named[0] if named else ""
    out: list[Head] = []
    for path in hub.drafts_for(model, files=files):
        needs = hub.borrowed_head(path.name)
        if needs and not here and not fork:
            continue
        try:
            size = path.resolve().stat().st_size
        except OSError:
            continue
        out.append(Head(path=str(path), bytes=size, spec_type=hub.spec_for(path.name),
                        build=fork if needs and not here else ""))
    return sorted(out, key=lambda h: (h.bytes, h.name))


@dataclass(frozen=True)
class Chosen:
    """What `choose_head` decided, and why in one sentence.

    ``path`` is what to serve as the draft -- an `hf:` reference, a local path, or '' when
    nothing should be. ``spec_type`` is the `--spec-type` it needs ('' when there is no
    head). ``borrows`` is what was decided about the binary. ``note`` is the repository's
    own sentence about needing a fork, when it has one, for printing under ``why``.
    """

    path: str
    spec_type: str
    why: str
    borrows: bool
    note: str = ""


def choose_head(model: str | Path, *, binary: str | Path | None, prefer: tuple[str, ...] = (),
                borrows: bool | None = None) -> Chosen:
    """The draft head to serve with ``model`` through ``binary``, with the reason.

    ``binary`` decides ``borrows`` (`serve.binary.borrows`): a named build or a fork's
    BUILD.json loads a head that borrows its target's embeddings, and `current`, brew and
    anything on PATH does not. ``None`` is the binary `find_binary` would pick; a caller
    that knows better -- a listing that serves nothing -- passes ``borrows`` outright.

    ``model`` is an `hf:` reference, a local path (`repo_of`) or a bare filename
    (`located`). The repository's listing is asked first, then the disk beside the
    weights. ``prefer`` narrows several heads by substring (`beside`); unset, a fork build
    takes a `shared-Q8_0` head and a mainline build avoids ``shared`` altogether.
    """
    if borrows is None:
        borrows = hub.forks(binary)[0]

    words = tuple(prefer) or (("shared-q8_0", "shared") if borrows else ("q8_0",))
    avoid = () if borrows else ("shared",)

    repo = hub.repo_of(model)
    found, where = _head_in(repo, prefer=words, avoid=avoid) if repo else ("", "")
    why = "shipped beside the weights"
    if found and where != repo:
        why = f"shipped in the sibling repository {where}"
    if not found:
        text = str(model)
        local = Path(text).expanduser()
        if "/" not in text and not local.is_file():
            local = hub.located(text) or local
        if local.is_file():
            from ml_stack.serve.ops import alongside

            for prefix in DRAFT_KINDS:
                found = alongside(str(local), "auto", prefix)
                if found:
                    break
            where = repo
            why = "found beside the weights on disk"
    if not found:
        return Chosen("", "", "no head shipped beside the weights", borrows)

    note = hub.draft_note(where) if where else ""
    if note and not borrows:
        return Chosen("", "", "withheld: the repository's README says it needs a fork and "
                      "this build is mainline", borrows, note)
    return Chosen(found, hub.spec_for(found), why, borrows, note)
