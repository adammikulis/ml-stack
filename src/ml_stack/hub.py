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

__all__ = ["Found", "PREFER", "advice", "aside", "card", "draft_for", "files",
           "find", "main", "ref"]

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
    """0 for the weights themselves, 1 for what merely travels alongside them."""
    plain = name.lower().rsplit("/", 1)[-1]
    return 1 if plain.startswith(("mmproj", "mtp-")) or "/" in name else 0


def draft_for(repo: str) -> str:
    """The draft head shipped beside the weights, as a reference, or ''.

    Gemma's QAT repositories carry an `mtp-` file of a few tens of megabytes — a
    multi-token-prediction head trained with the model. It is the draft to serve with it.
    """
    for name, _size in files(repo):
        if name.lower().rsplit("/", 1)[-1].startswith("mtp-") and "/" not in name:
            return ref(repo, name)
    return ""


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
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{find,files,card}")

    look = sub.add_parser("find", help="repositories matching some words")
    look.add_argument("words", nargs="+", help="e.g. gemma-4 E4B")
    look.add_argument("--prefer", default=",".join(PREFER),
                      help="publishers to rank first, comma separated (default: %(default)s)")
    look.add_argument("--all", action="store_true", help="not only GGUF repositories")
    look.add_argument("--limit", type=int, default=12)

    what = sub.add_parser("files", help="what is in one repository, and how to serve each")
    what.add_argument("repo", help="owner/name")
    what.add_argument("--ending", default=".gguf")

    said = sub.add_parser("card", help="what a model's own card asks for -- sampler settings "
                                       "first, because those are what get guessed at")
    said.add_argument("repo", help="owner/name")
    said.add_argument("--full", action="store_true", help="print the whole card as well")

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

        held = files(args.repo, ending=args.ending)
        if not held:
            print(f"no {args.ending} in {args.repo}", file=sys.stderr)
            return 1
        for name, size in held:
            note = "  (alongside)" if aside(name) else ""
            print(f"{_human(size):>8}  {ref(args.repo, name)}{note}")
        drafted = draft_for(args.repo)
        if drafted:
            print(f"\ndraft head shipped with it: {drafted}")
        return 0
    except Exception as exc:  # noqa: BLE001 - the Hub is somebody else's machine
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
