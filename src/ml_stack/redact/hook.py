"""The pre-commit hook: refuse a commit that carries a person's name, handle or contact details.

Two detectors, an exact list read from a local database and a Presidio recogniser with a
shape rule beside it, run over every staged file. Configuration is by environment variable:

    NAMES_GRAPH     a graph JSON; its person and org entries and its message senders are refused
    NAMES_SCRAPE    a JSONL file whose "sender" fields are refused
    NAMES_FIXTURES  the allow-list of invented names (default: tests/known-fixtures.txt,
                    relative to the repo root; an absolute path is used as given)
    SKIP_NAME_CHECK set to anything to skip the check
"""

from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from typing import Any, TextIO

__all__ = ["main", "recogniser"]

DEFAULT_FIXTURES = "tests/known-fixtures.txt"
SKIP_SUFFIX = (".png", ".jpg", ".gguf", ".lock", ".min.js", ".map")
FLOOR = 0.6
SHOWN = 25

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
# RFC 2606 / 6761 documentation and test domains
RESERVED_DOMAIN = re.compile(r"(?:^|\.)(?:example|test|invalid|localhost)$|^example\.(?:com|net|org)$", re.I)
UUIDISH = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
PHONE = re.compile(r"\+?\d[\d ().-]{8,}\d")
DATEISH = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
TIMESTAMP = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")
FRACTION = re.compile(r"\.\d{4,}")
# shapes that are code rather than a person
IDENTIFIER = re.compile(r"[_\[\](){}\"\n@/=]|^[a-z]+$")
# a Name-shaped run of one to four words inside a quoted string; a surname may be one letter
NAMEISH = re.compile(r"[\"\'`]([A-Z][a-z]{1,20}(?:\s+[A-Z][\w\'’.-]{0,20}){1,3})[\"\'`]")
# markers that make a name-shaped pair a place
PLACE_FIRST = {"north", "south", "east", "west", "new", "old", "upper", "lower", "san", "santa",
               "los", "las", "saint", "st.", "fort", "port", "mount", "lake", "cape", "united",
               "the", "greater", "central"}
PLACE_LAST = {"county", "river", "city", "island", "islands", "bay", "area", "kingdom", "states",
              "emirates", "beach", "lake", "valley", "park", "street", "road", "avenue",
              "mountain", "mountains", "hills", "falls", "springs", "harbor", "harbour",
              "coast", "province", "republic", "region", "district", "township", "heights"}
# last words that make a name-shaped pair a job title
ROLE_LAST = {"engineer", "manager", "designer", "analyst", "specialist", "writer", "marketer",
             "executive", "representative", "partner", "director", "lead", "officer",
             "scientist", "researcher", "developer", "administrator", "coordinator", "recruiter",
             "accountant", "counsel", "architect", "consultant", "technician", "assistant",
             "associate", "intern", "head", "chief", "professor", "fellow", "student", "lecturer",
             "maintainer", "contributor", "volunteer", "moderator", "founder", "president",
             "responder", "worker", "agent", "buyer", "clerk", "auditor", "nurse", "guide",
             "treasurer", "secretary", "controller", "strategist", "planner", "operator",
             "generalist", "advocate", "evangelist", "trainer", "educator", "librarian",
             "support", "success", "operations", "sales", "marketing", "finance", "legal",
             "security", "reliability", "quality", "product", "platform", "data", "people"}
# sender handles that are also ordinary words
COMMON = {"contact", "team", "admin", "support", "info", "hello", "help", "sales"}
# a file whose first lines say this turns the shape rule off for itself
SHAPES_OFF = "no-real-names: shapes off"


def is_role(body: str) -> bool:
    return body.casefold().split()[-1].strip(".,") in ROLE_LAST


def is_place(body: str) -> bool:
    words = body.casefold().split()
    return words[0] in PLACE_FIRST or words[-1].strip(".,") in PLACE_LAST


def permitted(root: str, fixtures: str, allow: str) -> set[str]:
    """Invented names and anything deliberately allowed, casefolded."""
    out: set[str] = set()
    if not os.path.isabs(fixtures):
        fixtures = f"{root}/{fixtures}"
    for path in (fixtures, f"{root}/{DEFAULT_FIXTURES}", allow):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    body = line.split("#")[0].strip()
                    if body:
                        out.add(body.casefold())
        except OSError:
            pass
    return out


def from_database(graph: str, scrape: str) -> set[str]:
    """Every person, named org and message sender the graph and the scrape hold."""
    names: set[str] = set()
    people: set[str] = set()
    try:
        with open(graph, encoding="utf-8") as fh:
            data = json.load(fh)
        nodes = data.get("nodes", [])
        people = {n["label"] for n in nodes if n.get("kind") == "person"}
        names |= people
        # an org counts only when it looks named: longer than four characters, capitalised
        names |= {n["label"] for n in nodes if n.get("kind") == "org"
                  and len(n["label"]) > 4 and n["label"][:1].isupper()}
        names |= {m["sender"] for m in (data.get("messages") or {}).values() if m.get("sender")}
    except (OSError, ValueError):
        pass
    try:
        with open(scrape, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("sender"):
                    names.add(row["sender"])
    except OSError:
        pass
    # the length floor is for handles and org names; a person keeps protection at any length
    return {n for n in names if n.casefold() not in COMMON and (n in people or len(n) > 3)}


def contacts(line: str, allowed: set[str]) -> Iterator[tuple[str, str]]:
    """Each email address and phone-shaped run in the line, with what it is."""
    for found in EMAIL.finditer(line):
        domain = found.group(0).rsplit("@", 1)[-1]
        if RESERVED_DOMAIN.search(domain):
            continue
        if "noreply" not in found.group(0) and found.group(0).casefold() not in allowed:
            yield "an email address", found.group(0)
    uuids = [m.group(0) for m in UUIDISH.finditer(line)]
    for found in PHONE.finditer(line):
        text = found.group(0)
        if not (text.startswith("+") or "(" in text or "-" in text):
            continue
        if any(text in u for u in uuids):
            continue
        if text.count(".") > 1 or DATEISH.match(text) or FRACTION.search(text):
            continue
        if TIMESTAMP.search(text):
            continue
        if 7 <= sum(c.isdigit() for c in text) <= 15:
            yield "something shaped like a phone number", text


@functools.lru_cache(maxsize=None)
def recogniser() -> Any:
    """A Presidio analyzer over spaCy's small English model, built once per process; None if
    presidio is not installed."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError:
        return None
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]})
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


def _git(root: str | None, *args: str) -> str:
    done = subprocess.run(("git", *args), cwd=root or None, capture_output=True, text=True)
    return done.stdout


def _staged(root: str | None) -> list[str]:
    listed = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR").split("\n")
    return [f for f in listed if f and not f.endswith(SKIP_SUFFIX)]


def _findings(path: str, blob: str, known: set[str], allowed: set[str],
              engine: Any) -> Iterator[tuple[str, int, str]]:
    lines = blob.split("\n")
    for i, line in enumerate(lines, 1):
        for name in known:
            if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", line, re.I):
                yield path, i, f"{name!r} is someone in the data"
        for what, text in contacts(line, allowed):
            yield path, i, f"{what}: {text}"
    shapes_off = SHAPES_OFF in blob[:2000] or path.endswith((".json", ".csv", ".geojson"))
    for i, line in enumerate([] if shapes_off else lines, 1):
        for found in NAMEISH.finditer(line):
            body = found.group(1).strip()
            if body.casefold() not in allowed and not is_place(body) and not is_role(body):
                yield path, i, f"{body!r} is shaped like a name"
    if engine is None:
        return
    for hit in engine.analyze(text=blob, language="en"):
        if hit.entity_type != "PERSON" or hit.score < FLOOR:
            continue
        body = blob[hit.start:hit.end].strip().strip("’‘'\"`,.:; ")
        # one CamelCase token is a class, not a person
        if not body or " " not in body or body.casefold() in allowed or IDENTIFIER.search(body):
            continue
        yield path, blob.count("\n", 0, hit.start) + 1, f"{body!r} reads as a person"


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None,
         root: str | os.PathLike[str] | None = None, stdout: TextIO | None = None) -> int:
    """Check the staged files of the repository at ``root`` (default: the one around the
    working directory). Reads its settings from ``env`` (default: the process environment),
    writes its report to ``stdout`` (default: ``sys.stderr``). Returns 1 when a commit is
    refused, else 0."""
    del argv
    env = os.environ if env is None else env
    out = sys.stderr if stdout is None else stdout
    if env.get("SKIP_NAME_CHECK"):
        return 0
    where = os.fspath(root) if root is not None else _git(None, "rev-parse", "--show-toplevel").strip()
    staged = _staged(where)
    if not staged:
        return 0

    fixtures = env.get("NAMES_FIXTURES", DEFAULT_FIXTURES)
    home = env.get("HOME") or os.path.expanduser("~")
    allowed = permitted(where, fixtures, f"{home}/.config/pii-allow.txt")
    known = {n for n in from_database(env.get("NAMES_GRAPH", ""), env.get("NAMES_SCRAPE", ""))
             if n.casefold() not in allowed}
    engine = recogniser()

    bad: list[tuple[str, int, str]] = []
    for path in staged:
        blob = _git(where, "show", f":{path}")
        if not blob or "\0" in blob[:2048]:
            continue
        bad.extend(_findings(path, blob, known, allowed, engine))

    if engine is None:
        print("pre-commit: presidio is not installed, so only known names are checked", file=out)
    if not bad:
        return 0
    unique = list(dict.fromkeys(bad))
    print("pre-commit: refusing to commit a person's details", file=out)
    for path, line, why in unique[:SHOWN]:
        print(f"           {path}:{line}  {why}", file=out)
    if len(unique) > SHOWN:
        print(f"           ...and {len(unique) - SHOWN} more", file=out)
    print(f"           invent the data. If the name is made up, add it to {fixtures}.", file=out)
    return 1


if __name__ == "__main__":
    sys.exit(main())
