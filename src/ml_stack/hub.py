"""Finding a model that did not exist when you last learned anything.

A model released last week is on the Hub and in nobody's training data, so the way to find
it is to look rather than to remember. This asks the Hub what exists, prefers the publishers
you trust, and prints the reference `ml-stack-serve` already understands — `hf:owner/repo/file` —
so nothing between here and a running server needs writing again.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Found", "PREFER", "advice", "aside", "beside", "builds", "card",
           "draft_for", "draft_note", "fetch", "files", "find", "located", "main",
           "mmproj_for", "DRAFT_KINDS", "held", "in_gguf", "ref", "room", "spec_for"]

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
    from huggingface_hub import HfApi

    seen: dict[str, Found] = {}
    for term in (query, f"{query} GGUF") if gguf else (query,):
        for model in HfApi().list_models(search=term, limit=100):
            name = str(model.id)
            if gguf and "gguf" not in name.lower():
                continue
            seen[name] = Found(repo=name, downloads=int(model.downloads or 0),
                               likes=int(getattr(model, "likes", 0) or 0))

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
    from huggingface_hub import HfApi

    out = []
    for info in HfApi().model_info(repo, files_metadata=True).siblings or ():
        name = str(info.rfilename)
        if name.lower().endswith(ending):
            out.append((name, int(getattr(info, "size", 0) or 0)))
    # Weights first, largest first, and the small things that travel with them last: a
    # listing sorted the other way buries the model itself under vision projectors and
    # draft heads, which is what happened the first time this was used in anger.
    return sorted(out, key=lambda kv: (aside(kv[0]), -kv[1]))


def aside(name: str) -> int:
    """0 for the weights themselves, 1 for what merely travels alongside them.

    A subdirectory does not make something a companion. A large model is published one
    directory per quantisation -- `UD-Q4_K_XL/thing-00001-of-00004.gguf` -- and calling
    those "alongside" buries the weights under the projector and prints the model itself as
    an afterthought. What a file *is* is in its name, not its folder.
    """
    plain = name.lower().rsplit("/", 1)[-1]
    return 1 if plain.startswith(("mmproj", "mtp-", "imatrix")) else 0


# How a draft head names itself, and the `--spec-type` each one needs. A head is named by
# the *method* it implements, not by the fact that it is a draft: `mtp-` for
# multi-token prediction, `eagle3-` for EAGLE3. A rule that knew only one of them reported
# "no draft" for gpt-oss-20b, which ships two EAGLE3 heads and no mtp- file at all.
DRAFT_KINDS = {"mtp-": "draft-mtp", "eagle3-": "draft-eagle3"}


def draft_for(repo: str, *, borrows: bool = False) -> str:
    """The draft head shipped beside the weights, as a reference, or ''.

    A repository that ships one carries an `mtp-` file — a multi-token-prediction head
    trained with the model, a few tens of megabytes for a small model and around a gigabyte
    for a large one. It is the draft to serve with it.

    "Beside" is not always the same directory. Gemma's QAT repositories put the head at the
    root *and* under `MTP/`; Qwen3.8-27B puts it only under `MTP/`, and a rule that ignored
    subdirectories found nothing for it. Prefer the root copy when there is one, since a
    reference without a directory is the plainer thing to serve, but take a nested one
    rather than reporting no draft at all. Among several at the same level, one named
    `*shared*` is preferred -- unsloth's own recommendation for Qwen3.8-Flash-Next's head,
    which borrows the target's embeddings and output layer rather than carrying its own.

    ``borrows`` says whether the binary about to serve this head can actually load one that
    needs a fork. Measured for real: every `mtp-` head under
    `unsloth/Qwen3.8-Flash-Next-GGUF/MTP/` fails on mainline llama.cpp master with
    `check_tensor_dims: tensor 'output_hc_norm.weight' not found` -- mainline has no MTP
    graph for `qwen4exp` at all -- and the repository's own `MTP/README.md` says so: "these
    do not work on mainline ggml-org/llama.cpp yet". So a head whose repository says as much
    (``draft_note``) is offered only when ``borrows`` is true, which is only when serving
    through a named fork build (`ml-stack-serve up --build NAME`, a fork the README names).
    Offering it to mainline is offering something that fails at the far end of a
    multi-gigabyte load; ``borrows=False`` is the safe default. A repository whose README
    says nothing about a fork is unaffected either way.
    """
    where = repo
    found = ""
    for prefix in DRAFT_KINDS:
        found = beside(where, prefix, prefer=("shared-q8_0", "shared"))
        if found:
            break
    if not found:
        # Some publishers ship the head as its own repository beside the weights. Asking
        # costs one request and is the difference between speculating and not.
        #
        # Beware what a `-MTP-GGUF` repository actually is: for Qwen3.6-35B-A3B it is not a
        # draft head at all but the *whole model* rebuilt with the multi-token-prediction
        # layers in it, 36G of weights, to be served with `--spec-type draft-mtp` rather
        # than as a second model. Only a file actually named `mtp-` is taken here, so that
        # repository correctly yields nothing rather than a 36G "draft".
        stem = repo[: -len("-GGUF")] if repo.upper().endswith("-GGUF") else repo
        for sibling in (f"{stem}-MTP-GGUF", f"{stem}-MTP"):
            for prefix in DRAFT_KINDS:
                found = beside(sibling, prefix, prefer=("shared-q8_0", "shared"))
                if found:
                    where = sibling
                    break
            if found:
                break
    if not found:
        return ""
    if not borrows and draft_note(where):
        return ""
    return found


def spec_for(draft: str) -> str:
    """The `--spec-type` a draft head needs, read from what it is called, or ''.

    A head implements one method and only that method: an EAGLE3 head served as
    `draft-simple` is not slower, it is wrong about what it is being asked to do.
    """
    plain = str(draft).lower().rsplit("/", 1)[-1]
    for prefix, kind in DRAFT_KINDS.items():
        if plain.startswith(prefix):
            return kind
    return ""


# How a file says which precision it is, best first. A repository that ships more than one
# companion ships one per precision, and which to take depends on what the companion is.
_QUANTS = ("f32", "bf16", "f16", "q8_0", "q6_k", "q5_k_m", "q5_k", "q4_k_xl", "q4_k_m",
           "q4_k", "q4_0", "iq4_nl", "q3_k", "q2_k")


def _precision(name: str) -> int:
    """Where a file sits in ``_QUANTS``; lower is more precise. Unmarked sorts last."""
    plain = name.casefold()
    for n, quant in enumerate(_QUANTS):
        if quant in plain:
            return n
    return len(_QUANTS)


def beside(repo: str, prefix: str, *, best: bool = False,
          prefer: str | tuple[str, ...] = ()) -> str:
    """A file whose name starts with ``prefix`` in one repository, root copy preferred.

    The things that travel with a model are named by convention and filed wherever the
    publisher felt like: `mtp-` heads at the root, or under `MTP/`; `mmproj-` projectors
    usually at the root, and sometimes one per precision. One rule reads all of them.

    ``best`` takes the most precise when a repository offers several. That is what a vision
    projector wants: it is a fraction of the model's size -- DeepSeek-OCR-2's is 886M against
    5.5G of weights -- and quantising it costs sight out of all proportion to what it saves,
    so a Q4 model is still served with an F32 or BF16 projector. A draft head is the other
    way about and *precision is not what picks one*: unsloth's own recommendation for
    Qwen3.8-Flash-Next is `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` over its BF16 sibling
    ("the LM head is cheaper at 8 bits"), which `best`'s most-precise-wins rule would get
    backwards. ``prefer`` is for exactly this: one or more substrings, most specific first,
    each tried in turn against the candidates found so far -- the first that matches
    anything narrows to those matches; nothing narrows when none does. `draft_for` asks for
    ``("shared-q8_0", "shared")``.
    """
    words = (prefer,) if isinstance(prefer, str) else tuple(prefer)
    words = tuple(w for w in words if w)
    nested: list[str] = []
    root: list[str] = []
    try:
        held = files(repo)
    except Exception:  # noqa: BLE001 - a repository that is not there holds nothing
        return ""
    for name, _size in held:
        plain = name.lower().rsplit("/", 1)[-1]
        if not plain.startswith(prefix):
            continue
        (nested if "/" in name else root).append(name)
    for found in (root, nested):
        if not found:
            continue
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


# What a card calls a sampler setting, and what llama.cpp calls it. A card writes prose, so
# `temperature=1.0`, `temperature: 1.0`, `"temperature": 1.0` and `--temp 1.0` all appear.
_KNOBS = {"temperature": "temperature", "temp": "temperature", "top_p": "top_p",
          "top-p": "top_p", "top_k": "top_k", "top-k": "top_k", "min_p": "min_p",
          "min-p": "min_p", "repeat_penalty": "repeat_penalty",
          "repetition_penalty": "repeat_penalty"}
_NAMES = "|".join(sorted(_KNOBS, key=len, reverse=True))
# `temperature=1.0`, `temperature: 1.0`, `"temperature": 1.0`, and `--temp 1.0`. The last
# form has no separator, so a card that only shows a llama.cpp command line still reads.
_SETTING = re.compile(
    r"(?:--)?[`\"\'*]*\b(" + _NAMES + r")\b[`\"\'*]*\s*(?:[=:]\s*|(?<=--\w)\s+)"
    r"[`\"\'*]*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_FLAG = re.compile(r"--(" + _NAMES + r")\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Where a card puts what it recommends. Searched first, so a document that opens by warning
# against a setting is not read as recommending it -- "first mention wins" is only safe
# inside the section that is doing the recommending.
_ADVISING = re.compile(
    r"^[ \t]*(#{1,6})[ \t]*[^\n]*\b"
    r"(?:sampling|best practice|recommend|inference|usage|parameters)\b",
    re.IGNORECASE | re.MULTILINE)


def in_gguf(path: str | Path) -> dict[str, float]:
    """The sampler settings written into a GGUF's own metadata, or {}.

    Better than the card by every measure: it is in the file being served rather than in
    prose beside it, it cannot drift from the weights, and it needs no parsing. Qwen3.8
    carries `general.sampling.temp`, `.top_k` and `.top_p`; many models carry none, which is
    why the card is still read when this is empty.
    """
    import struct

    want = {"temp": "temperature", "temperature": "temperature", "top_k": "top_k",
            "top_p": "top_p", "min_p": "min_p"}
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
           11: "q", 12: "d"}
    out: dict[str, float] = {}
    try:
        with Path(path).expanduser().open("rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))                     # tensor count
            keys = struct.unpack("<Q", f.read(8))[0]

            def text() -> str:
                return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")

            def value(kind: int) -> object:
                if kind == 8:
                    return text()
                if kind == 9:
                    each = struct.unpack("<I", f.read(4))[0]
                    return [value(each) for _ in range(struct.unpack("<Q", f.read(8))[0])]
                return struct.unpack("<" + fmt[kind],
                                     f.read(struct.calcsize(fmt[kind])))[0]

            for _ in range(keys):
                name = text()
                held = value(struct.unpack("<I", f.read(4))[0])
                if name.startswith("general.sampling."):
                    tail = name.rsplit(".", 1)[-1]
                    if tail in want and isinstance(held, (int, float)):
                        out[want[tail]] = float(held)
    except Exception:  # noqa: BLE001 - a file that will not parse simply says nothing
        return {}
    return out


def card(repo: str) -> str:
    """The repository's README, which is where a publisher writes down what it wants."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, "README.md")).read_text(errors="replace")


def advice(text: str) -> dict[str, float]:
    """The sampler settings a card names, as llama.cpp spells them.

    Cards are prose, not configuration, so this reads what is written rather than pretending
    to a schema: gemma-4's says "Use the following standardized sampling configuration across
    all use cases: temperature=1.0, top_p=0.95, top_k=64", and gpt-oss's says nothing at all.
    An empty answer means the card was silent, which is a real answer — it is the difference
    between a publisher choosing a default and nobody having chosen one.

    A card's recommending section is read first when it has one -- a heading naming sampling,
    best practices, recommendations, inference or usage. Only within it does "first mention
    wins", which is what makes that rule safe: a document that opens by warning against a
    setting would otherwise be read as recommending it.

    This is regex over prose and it is only ever a *hypothesis*. Nothing applies it: it is
    what a benchmark starts from, which is why being roughly right is enough and being
    silently wrong is not dangerous.
    """
    body = text or ""
    for where in (_advising(body), body):
        found: dict[str, float] = {}
        for pattern in (_SETTING, _FLAG):
            for match in pattern.finditer(where):
                name = _KNOBS[match.group(1).lower()]
                found.setdefault(name, float(match.group(2)))
        if found:
            return found
    return {}


def _advising(text: str) -> str:
    """The part of a card that is making a recommendation, or '' when none is marked."""
    first = _ADVISING.search(text)
    if first is None:
        return ""
    after = text[first.start():]
    # Search past this section's own heading line, not past one character of it: a heading
    # may be indented, and skipping a single space leaves the hashes to match themselves.
    line_end = after.find("\n")
    if line_end == -1:
        return after
    # to the next heading of the same depth or shallower, so a section keeps its subsections
    depth = len(first.group(1))
    nxt = re.search(rf"^[ \t]*#{{1,{depth}}}[ \t]", after[line_end:], re.MULTILINE)
    return after[:line_end + nxt.start()] if nxt else after


def ref(repo: str, name: str = "") -> str:
    """The reference `ml-stack-serve up` takes, which downloads and caches on first use."""
    return f"hf:{repo}/{name}" if name else f"hf:{repo}"


_SHARD = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)


def builds(repo: str, *, ending: str = ".gguf") -> list[tuple[str, int, int]]:
    """What a repository offers, one row per build rather than per file.

    A large model is published in shards, one directory per quantisation, and a listing of
    forty files answers no question anybody has. What decides whether a model can be served
    is the *total* of a build -- Qwen3.8-Flash-Next is 329.7G at BF16 and 87.2G at
    UD-IQ4_XS, and reading that off a list of individual shards means adding up by hand.

    Returns ``(name, total bytes, shards)``, largest first, companions excluded.
    """
    grouped: dict[str, list[int]] = {}
    for name, size in files(repo, ending=ending):
        if aside(name):
            continue
        # the quantisation is the directory when there is one, else the shard-less filename
        stem = name.split("/")[0] if "/" in name else _SHARD.sub("", name.rsplit("/", 1)[-1])
        grouped.setdefault(stem, []).append(size)
    return sorted(((name, sum(sizes), len(sizes)) for name, sizes in grouped.items()),
                  key=lambda row: -row[1])


def held() -> dict[str, int]:
    """Every model file already on this machine, by filename, with its real size.

    `ml_stack.fleet.models` has known where they are all along; this asks it, because the
    alternative is what happened once: reaching for the Hub to fetch 87G that was already
    on the disk. A listing that does not say what you have invites downloading it twice.

    Sizes are resolved through symlinks on purpose. A Hub cache is symlinks into
    `blobs/`, so `ls -l` reports 79 bytes for a 46G shard and reading that as "not
    downloaded" is the same mistake wearing a different hat.
    """
    from pathlib import Path

    try:
        from ml_stack.fleet.models import Models, default_roots

        found = Models(roots=default_roots(Path.home() / ".ml-stack"),
                       store=Path.home() / ".ml-stack").all()
    except Exception:  # noqa: BLE001 - a machine with no models has no models
        return {}
    out: dict[str, int] = {}
    for model in found:
        where = Path(getattr(model, "path", "") or "")
        try:
            out[where.name] = where.resolve().stat().st_size
        except OSError:
            continue
    return out


def held_files(repo: str, build: str, ending: str = ".gguf") -> list[tuple[str, int]]:
    """The filenames belonging to one build of a repository."""
    return [(name.rsplit("/", 1)[-1], size) for name, size in files(repo, ending=ending)
            if not aside(name) and (name.split("/")[0] == build
                                    or name.rsplit("/", 1)[-1] == build)]


HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def located(name: str, *, cache: Path | None = None) -> Path | None:
    """A model already in the Hub cache, found by its exact filename, or ``None``.

    A name copied out of `ml-stack-models files` -- no directory, no `hf:` -- names a file
    already downloaded, but `ml-stack-serve up` used to read anything without a `/` or an
    `hf:` prefix as a relative path and report "shards missing" for a model that was on the
    machine the whole time. `graph.bench.find_model` already solved this for the bench by
    asking `fleet.models` where the file is; this is the same idea narrowed to the Hub cache
    itself and to an *exact* filename match, which is what a preflight is actually handed.

    Resolved through the symlink: a Hub cache entry is a name in `snapshots/<rev>/` pointing
    into `blobs/`, and reading the symlink's own size reports a few bytes rather than the
    weights it stands for.

    A sharded build's first shard is found when ``name`` is the shard-less stem -- what the
    quantisation itself is called, e.g. `thing-UD-Q4_K_XL.gguf`, rather than any one file
    inside it.
    """
    root = Path(cache) if cache is not None else HUB_CACHE
    if not root.is_dir():
        return None
    wanted = Path(name).name

    def _search(pattern: str) -> Path | None:
        for snapshot in sorted(root.glob("*/snapshots/*")):
            if not snapshot.is_dir():
                continue
            for candidate in sorted(snapshot.rglob(pattern)):
                if candidate.is_file():
                    return candidate.resolve()
        return None

    found = _search(wanted)
    if found is not None:
        return found
    base = wanted[: -len(".gguf")] if wanted.lower().endswith(".gguf") else wanted
    return _search(f"{base}-00001-of-*.gguf")


_DRAFT_NOTES: dict[str, str] = {}


def draft_note(repo: str) -> str:
    """The sentence a draft head's own README says about needing a fork, or ''.

    Measured for real: unsloth's `mtp-` heads for Qwen3.8-Flash-Next fail to load on
    mainline llama.cpp master with `check_tensor_dims: tensor 'output_hc_norm.weight' not
    found`, and `MTP/README.md` says why in one sentence -- "These do not work on mainline
    ggml-org/llama.cpp yet". `ml-stack-models files` prints this under the draft line it
    already reports, so a person is told a head needs a fork *before* spending a load on it,
    not after it fails at the far end of one.

    Read once per repository and cached for the life of the process -- `files` calls this on
    every listing, and a repository's README does not change between two calls of it.
    """
    if repo in _DRAFT_NOTES:
        return _DRAFT_NOTES[repo]
    from huggingface_hub import hf_hub_download

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


def fetch(reference: str) -> Path:
    """Download an `hf:` reference into the Hub cache, without serving it.

    The same cache llama-server's own `-hf` download fills, and `held()` reads back -- so a
    prefetch here and a lease afterward see the same file, and a benchmark that preflights
    a model before timing it never pays for the download inside the timed window.

    A sharded model's *every* shard comes down, not only the one named: the file given is
    one member of a build, and a server started against a partial download fails at the far
    end of the load complaining about a missing shard, which is exactly the fault a
    preflight exists to catch first.
    """
    from huggingface_hub import hf_hub_download

    from ml_stack.serve.backend import ServerSpec

    parts = ServerSpec.hf_parts(reference)
    if parts is None or not parts[1]:
        raise ValueError(f"{reference!r} should look like hf:owner/repo/file.gguf")
    repo, name = parts

    stem = name.split("/")[0] if "/" in name else _SHARD.sub("", name.rsplit("/", 1)[-1])
    members = [n for n, _size in files(repo)
              if (n.split("/")[0] if "/" in n
                  else _SHARD.sub("", n.rsplit("/", 1)[-1])) == stem] or [name]

    wanted: Path | None = None
    last: Path | None = None
    for member in members:
        last = Path(hf_hub_download(repo, member))
        if member == name:
            wanted = last
    return wanted or last


def room() -> int:
    """How much memory a model could actually use here, in bytes, or 0 when unknown.

    On a machine with unified memory this is not the whole of RAM: Metal will not wire more
    than `iogpu.wired_limit_mb`, and a model plus its KV cache has to fit under that. On a
    machine with a separate card it is that card's memory. Either way it is the number that
    decides whether a build can be served, and it is not the number `free` reports.
    """
    import subprocess

    try:
        got = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                             capture_output=True, text=True, timeout=5)
        if got.returncode == 0 and got.stdout.strip().isdigit():
            return int(got.stdout.strip()) * 1024 * 1024
        got = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        if got.returncode == 0 and got.stdout.strip().isdigit():
            return int(int(got.stdout.strip()) * 0.75)   # Metal's own default share
    except Exception:  # noqa: BLE001 - not every machine answers, and that is not a failure
        pass
    return 0


def _human(size: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if size < 1024 or unit == "G":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}G"


def main(argv: list[str] | None = None) -> int:
    """``ml-stack-models`` -- find a model on the Hub and print how to serve it."""
    ap = argparse.ArgumentParser(
        prog="ml-stack-models",
        description="Find a model that is newer than anything you remember, and serve it.")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{find,files,card,fetch}")

    look = sub.add_parser("find", help="repositories matching some words")
    look.add_argument("words", nargs="+", help="e.g. gemma-4 E4B")
    look.add_argument("--prefer", default=",".join(PREFER),
                      help="publishers to rank first, comma separated (default: %(default)s)")
    look.add_argument("--all", action="store_true", help="not only GGUF repositories")
    look.add_argument("--limit", type=int, default=12)

    what = sub.add_parser("files", help="what is in one repository, and how to serve each")
    what.add_argument("repo", help="owner/name")
    what.add_argument("--ending", default=".gguf")
    what.add_argument("--every", action="store_true",
                      help="one line per file rather than per build; a sharded model is "
                           "forty lines this way and its totals are what you wanted")

    said = sub.add_parser("card", help="what a model's own card asks for -- sampler settings "
                                       "first, because those are what get guessed at")
    said.add_argument("repo", help="owner/name")
    said.add_argument("--full", action="store_true", help="print the whole card as well")

    got = sub.add_parser("fetch", help="download hf: references into the cache, without "
                                       "serving them -- every shard of a sharded model")
    got.add_argument("refs", nargs="+", metavar="REF",
                     help="hf:owner/repo/file.gguf, one or more")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "find":
            prefer = tuple(p.strip().lower() for p in args.prefer.split(",") if p.strip())
            found = find(" ".join(args.words), prefer=prefer, gguf=not args.all,
                         limit=args.limit)
            if not found:
                print("nothing matched", file=sys.stderr)
                return 1
            for one in found:
                print(f"{one.downloads:>10}  {one.repo}")
            print(f"\nml-stack-models files {found[0].repo}")
            return 0

        if args.cmd == "card":
            text = card(args.repo)
            asked = advice(text)
            if asked:
                print(f"{args.repo} asks for:")
                for name, value in asked.items():
                    print(f"  {name:16} {value:g}")
                flags = " ".join(f"--{n.replace('_', '-')} {v:g}" for n, v in asked.items()
                                 if n in ("temperature", "top_p", "top_k", "min_p"))
                print(f"\nml-stack-bench run {flags}")
            else:
                print(f"{args.repo}'s card names no sampler settings. That is an answer: "
                      f"nobody has chosen one, so the caller's default stands.")
            if args.full:
                print("\n" + text)
            return 0

        if args.cmd == "fetch":
            for one in args.refs:
                path = fetch(one)
                size = path.stat().st_size if path.exists() else 0
                print(f"{_human(size):>8}  {path}")
            return 0

        listing = files(args.repo, ending=args.ending)
        if not listing:
            print(f"no {args.ending} in {args.repo}", file=sys.stderr)
            return 1
        if not args.every:
            fits = room()
            mine = held()
            grouped = builds(args.repo, ending=args.ending)
            if fits:
                print(f"this machine can serve about {_human(fits)}\n")
            for name, size, shards in grouped:
                on_disk = sum(1 for f, _s in held_files(args.repo, name, args.ending)
                              if f in mine)
                mark = "" if not fits else ("  fits" if size < fits * 0.95 else "  TOO BIG")
                if on_disk:
                    mark = ("  ON THIS MACHINE" if on_disk >= shards
                            else f"  {on_disk}/{shards} downloaded") + mark
                many = f"  {shards} shards" if shards > 1 else ""
                print(f"{_human(size):>8}  {name}{many}{mark}")
            for name, size in listing:
                if aside(name):
                    print(f"{_human(size):>8}  {ref(args.repo, name)}  (alongside)")
            print(f"\nml-stack-models files {args.repo} --every  for individual files")
            # This listing never serves anything, so it is told about a head even when it
            # needs a fork -- borrows=True -- and the note underneath it says so, which is
            # the whole point: know before spending a load, not after one fails at the end.
            drafted = draft_for(args.repo, borrows=True)
            if drafted:
                print(f"draft head shipped with it: {drafted}")
                warned = draft_note(args.repo)
                if warned:
                    print(f"  {warned}")
            return 0
        for name, size in listing:
            note = "  (alongside)" if aside(name) else ""
            print(f"{_human(size):>8}  {ref(args.repo, name)}{note}")
        drafted = draft_for(args.repo, borrows=True)
        if drafted:
            print(f"\ndraft head shipped with it: {drafted}")
            warned = draft_note(args.repo)
            if warned:
                print(f"  {warned}")
        return 0
    except Exception as exc:  # noqa: BLE001 - the Hub is somebody else's machine
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
