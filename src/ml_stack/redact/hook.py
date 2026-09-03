"""The pre-commit hook: refuse a commit that carries a person's name, handle or contact details.

Two detectors, an exact list read from a local database and a Presidio recogniser with a
shape rule beside it, run over every staged file. The shape rule and everything that stands
it down -- a place, a job title, a reserved domain, the inside of a uuid, and a match that is
code rather than a person (an expression, a table cell of digits, a keyword and an identifier,
a product) -- are data in ``contracts/name-shapes.json``, one section per rule, so an
exception is a data change and not a code change; every finding, and every pair stood down,
names its section. The exact list is never stood down by any of it, and neither is a name
inside a quoted string, so a person in a JavaScript string is still refused. Configuration
is by environment variable:

    NAMES_GRAPH     a graph JSON; its person and org entries and its message senders are refused
    NAMES_SCRAPE    a JSONL file whose "sender" fields are refused
    NAMES_FIXTURES  the allow-list of invented names (default: tests/known-fixtures.txt,
                    relative to the repo root; an absolute path is used as given)
    NAMES_SHAPES    a shape-rules JSON to read instead of the shipped contract
    NAMES_WHY       set to anything (or pass ``--why``) to print, for every name-shaped pair
                    and contact-shaped run that was cleared, which rule cleared it --
                    and, after ``allow``, the nearest rule to each phrase allowed
    SKIP_NAME_CHECK set to anything to skip the check
"""

from __future__ import annotations

import difflib
import functools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from ml_stack.contracts import ContractError, load

__all__ = ["Shapes", "main", "recogniser", "shapes"]

CONTRACT = "name-shapes.json"
# every section the code reads; a rules file missing one is refused, not guessed at
SECTIONS = ("place_first", "place_last", "role_last", "shapes_off", "reserved_domains",
            "not_a_contact", "context", "patterns", "skip_suffixes")
CONTEXT_KEYS = ("operators", "brackets", "adjacent", "keywords", "products")
DEFAULT_FIXTURES = "tests/known-fixtures.txt"
FLOOR = 0.6
SHOWN = 25

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\d[\d ().-]{8,}\d")
DATEISH = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
TIMESTAMP = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")
FRACTION = re.compile(r"\.\d{4,}")
# shapes that are code rather than a person
IDENTIFIER = re.compile(r"[_\[\](){}\"\n@/=]|^[a-z]+$")
# `label.lower`, and not an initial between a forename and a surname, and not `St. Cloud`:
# the left side is two characters or more and a letter follows the dot with no space, so a
# middle initial is not read as attribute access
ATTRIBUTE = re.compile(r"[A-Za-z_]\w+\.[A-Za-z_]")
TABLE_ROW = re.compile(r"^\s*\|.*\|")
LETTER = re.compile(r"[A-Za-z]")
IDENT = re.compile(r"^[A-Za-z_][\w.]*$")
# sender handles that are also ordinary words
COMMON = {"contact", "team", "admin", "support", "info", "hello", "help", "sales"}


def _closest(word: str, pool: Any) -> tuple[str, float]:
    """The entry of ``pool`` nearest to ``word``, and how near, for ``near_misses``."""
    near, ratio = "", 0.0
    for entry in pool:
        score = difflib.SequenceMatcher(None, word, entry).ratio()
        if score > ratio:
            near, ratio = entry, score
    return near, ratio


@dataclass(frozen=True)
class Shapes:
    """The shape rules of one ``name-shapes.json``, compiled. Each ``stood_down`` /
    ``in_context`` / ``reserved`` / ``inside_uuid`` answer is the rule that fired, as
    ``section: word``, or None when nothing did -- that string is what ``--why`` prints."""

    place_first: frozenset[str]
    place_last: frozenset[str]
    role_last: frozenset[str]
    shapes_off_marker: str
    shapes_off_suffixes: tuple[str, ...]
    reserved_domains: tuple[str, ...]
    not_a_contact: tuple[str, ...]
    operators: tuple[str, ...]
    brackets: tuple[str, ...]
    adjacent: frozenset[str]
    keywords: frozenset[str]
    products: tuple[str, ...]
    uuid: re.Pattern[str]
    nameish: re.Pattern[str]
    skip_suffixes: tuple[str, ...]

    @classmethod
    def from_data(cls, data: Mapping[str, Any], where: str) -> Shapes:
        missing = [s for s in SECTIONS if s not in data]
        if missing:
            raise ContractError(f"{where} lacks the section(s) {', '.join(missing)}")
        patterns = data["patterns"]
        for key in ("uuid", "nameish"):
            if key not in patterns:
                raise ContractError(f"{where} lacks patterns.{key}")
        context = data["context"]
        for key in CONTEXT_KEYS:
            if key not in context:
                raise ContractError(f"{where} lacks context.{key}")
        off = data["shapes_off"]
        return cls(
            place_first=frozenset(w.casefold() for w in data["place_first"]),
            place_last=frozenset(w.casefold() for w in data["place_last"]),
            role_last=frozenset(w.casefold() for w in data["role_last"]),
            shapes_off_marker=off["marker"],
            shapes_off_suffixes=tuple(off["suffixes"]),
            reserved_domains=tuple(d.casefold() for d in data["reserved_domains"]),
            not_a_contact=tuple(data["not_a_contact"]),
            # longest first, so '||' is reported rather than the '|' inside it
            operators=tuple(sorted(context["operators"], key=len, reverse=True)),
            brackets=tuple(context["brackets"]),
            adjacent=frozenset(context["adjacent"]),
            keywords=frozenset(w.casefold() for w in context["keywords"]),
            products=tuple(context["products"]),
            uuid=re.compile(patterns["uuid"]),
            nameish=re.compile(patterns["nameish"]),
            skip_suffixes=tuple(data["skip_suffixes"]),
        )

    def stood_down(self, body: str) -> str | None:
        """The rule that makes a name-shaped pair a place or a job title, or None."""
        words = body.casefold().split()
        last = words[-1].strip(".,")
        if words[0] in self.place_first:
            return f"place_first: {words[0]}"
        if last in self.place_last:
            return f"place_last: {last}"
        if last in self.role_last:
            return f"role_last: {last}"
        return None

    def in_context(self, body: str, line: str = "", span: tuple[int, int] | None = None,
                   ) -> str | None:
        """The context rule that makes this match code rather than a person, or None. ``line``
        is the whole line it was found on and ``span`` where in that line it sits -- pass
        ``span`` only for a match that is not itself a quoted string, because a quoted string
        is a literal and the code around it says nothing about what is inside it. Most
        specific rule first, so ``assert label.lower`` is reported as the keyword and not as
        the dot."""
        body = body.strip()
        if not body:
            return None
        return (self._table_cell(body, line) or self._keyword(body) or self._product(body)
                or self._expression(body, line, span))

    def _table_cell(self, body: str, line: str) -> str | None:
        """`| ~3 |`: a markdown row, and a cell that is a number or a symbol. A cell with a
        letter in it is not stood down, so a name in a table is still refused."""
        if not TABLE_ROW.match(line) or LETTER.search(body) or not body.strip("| \t"):
            return None
        return f"context_table_cell: {body}"

    def _keyword(self, body: str) -> str | None:
        """`assert label.lower`: a keyword or builtin and one identifier after it. Two words
        after it, or one that is capitalised and has no dot, is a name again."""
        words = body.split()
        if len(words) != 2 or words[0].casefold() not in self.keywords:
            return None
        rest = words[1]
        if IDENT.match(rest) and (rest[:1].islower() or "." in rest):
            return f"context_keyword: {words[0].casefold()}"
        return None

    def _product(self, body: str) -> str | None:
        """`Windows Defender Firewall`, `Practical AI`: two words or more, one of them a
        product, an operating system or a vendor."""
        words = [w.strip(".,:;'\"()[]") for w in body.split()]
        if len(words) < 2:
            return None
        known = {p.casefold(): p for p in self.products}
        for word in words:
            if word.casefold() in known:
                return f"context_product: {known[word.casefold()]}"
        return None

    def _expression(self, body: str, line: str, span: tuple[int, int] | None) -> str | None:
        """`x1 - x0`, `null ||`, `label.lower`, `(a)`: a bracket, an operator with a space
        beside it, an attribute dot, or -- for an unquoted match -- operator characters
        immediately either side of it. An operator wedged between letters (`Anne-Marie`) is
        part of the word, not an operator."""
        for bracket in self.brackets:
            if bracket in body:
                return f"context_expression: {bracket}"
        for token in self.operators:
            escaped = re.escape(token)
            if re.search(rf"(?:^|\s){escaped}|{escaped}(?:\s|$)", body):
                return f"context_expression: {token}"
        if ATTRIBUTE.search(body):
            return "context_expression: attribute"
        if span and line:
            start, end = span
            before = line[start - 1] if start > 0 else ""
            after = line[end] if end < len(line) else ""
            if before in self.adjacent and after in self.adjacent:
                return f"context_expression: adjacent {before}{after}"
        return None

    def near_misses(self, body: str, limit: int = 3) -> list[str]:
        """The rules that came nearest to standing ``body`` down, closest first, each as
        ``section: what is missing from it``. ``allow --why`` prints these, so that a phrase
        a rule should cover becomes a line in that rule rather than a fixture of its own."""
        words = [w.strip(".,:;'\"()[]") for w in body.split()] or [body]
        scored: list[tuple[float, str]] = []
        for section, pool, word in (
                ("place_first", self.place_first, words[0].casefold()),
                ("place_last", self.place_last, words[-1].casefold()),
                ("role_last", self.role_last, words[-1].casefold()),
                ("context.keywords", self.keywords, words[0].casefold())):
            near, ratio = _closest(word, pool)
            scored.append((ratio, f"{section}: nothing there matches {word!r}"
                                  + (f" (nearest {near!r})" if near else "")))
        best = max(((*_closest(w.casefold(), {p.casefold() for p in self.products}), w)
                    for w in words), key=lambda t: t[1])
        scored.append((best[1], f"context.products: no word of it is a product token; "
                                f"{best[2]!r} is nearest {best[0]!r}"))
        if not LETTER.search(body):
            scored.append((0.75, "context.table_cell: it has no letters, so it stands down "
                                 "on a markdown table row -- but this was not one"))
        if any(t in body for t in self.operators) or ATTRIBUTE.search(body):
            scored.append((0.75, "context.operators: it holds an operator, but wedged "
                                 "between letters rather than used as one"))
        scored.sort(key=lambda t: -t[0])
        return [why for _, why in scored[:limit]]

    def reserved(self, domain: str) -> str | None:
        """The reserved-domain entry the address's domain falls under, or None. An entry
        without a dot is a top-level label; one with a dot is a whole domain."""
        domain = domain.casefold()
        tld = domain.rsplit(".", 1)[-1]
        for entry in self.reserved_domains:
            if (tld if "." not in entry else domain) == entry:
                return f"reserved_domains: {entry}"
        return None

    def not_contact(self, address: str) -> str | None:
        for word in self.not_a_contact:
            if word in address:
                return f"not_a_contact: {word}"
        return None

    def off_for(self, path: str, blob: str) -> str | None:
        """Why the shape rule is off for this file, or None when it is on."""
        if self.shapes_off_marker in blob[:2000]:
            return f"shapes_off: marker {self.shapes_off_marker!r}"
        for suffix in self.shapes_off_suffixes:
            if path.endswith(suffix):
                return f"shapes_off: suffix {suffix}"
        return None


@functools.lru_cache(maxsize=None)
def shapes(path: str | None = None) -> Shapes:
    """The shape rules, read once per process: from ``path`` when given, else from the
    shipped contract ``name-shapes.json``."""
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ContractError(f"cannot read the shape rules at {path}: {exc}") from exc
        return Shapes.from_data(data, path)
    return Shapes.from_data(load(CONTRACT), CONTRACT)


def _rules(env: Mapping[str, str] | None = None) -> Shapes:
    env = os.environ if env is None else env
    return shapes(env.get("NAMES_SHAPES") or None)


def is_role(body: str, rules: Shapes | None = None) -> bool:
    fired = (rules or _rules()).stood_down(body)
    return bool(fired and fired.startswith("role_last"))


def is_place(body: str, rules: Shapes | None = None) -> bool:
    fired = (rules or _rules()).stood_down(body)
    return bool(fired and fired.startswith("place_"))


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


def contacts(line: str, allowed: set[str], rules: Shapes | None = None,
             why: list[str] | None = None) -> Iterator[tuple[str, str]]:
    """Each email address and phone-shaped run in the line, with what it is. Anything a rule
    cleared is appended to ``why`` as ``'text' cleared by section: word`` when a list is
    given."""
    rules = rules or _rules()
    cleared = why.append if why is not None else (lambda _: None)
    for found in EMAIL.finditer(line):
        address = found.group(0)
        fired = rules.reserved(address.rsplit("@", 1)[-1]) or rules.not_contact(address)
        if fired:
            cleared(f"{address!r} cleared by {fired}")
            continue
        if address.casefold() in allowed:
            cleared(f"{address!r} cleared by fixtures")
            continue
        yield "an email address (no reserved_domains or not_a_contact match)", address
    uuids = [m.group(0) for m in rules.uuid.finditer(line)]
    for found in PHONE.finditer(line):
        text = found.group(0)
        if not (text.startswith("+") or "(" in text or "-" in text):
            continue
        if any(text in u for u in uuids):
            cleared(f"{text!r} cleared by patterns: uuid")
            continue
        if text.count(".") > 1 or DATEISH.match(text) or FRACTION.search(text):
            continue
        if TIMESTAMP.search(text):
            continue
        if 7 <= sum(c.isdigit() for c in text) <= 15:
            yield "something shaped like a phone number (not inside a patterns: uuid)", text


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


def _staged(root: str | None, rules: Shapes) -> list[str]:
    listed = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR").split("\n")
    return [f for f in listed if f and not f.endswith(rules.skip_suffixes)]


def _findings(path: str, blob: str, known: set[str], allowed: set[str], engine: Any,
              rules: Shapes, why: list[tuple[str, int, str]] | None = None,
              ) -> Iterator[tuple[str, int, str]]:
    """Every refusal in one file as ``(path, line, why)``. What a rule cleared goes to
    ``why`` in the same shape, when a list is given."""
    lines = blob.split("\n")
    for i, line in enumerate(lines, 1):
        for name in known:
            if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", line, re.I):
                yield path, i, f"{name!r} is someone in the data"
        stood: list[str] = []
        for what, text in contacts(line, allowed, rules, stood):
            yield path, i, f"{what}: {text}"
        if why is not None:
            why.extend((path, i, s) for s in stood)
    off = rules.off_for(path, blob)
    if off is not None and why is not None:
        why.append((path, 0, f"shape rule off: {off}"))
    for i, line in enumerate([] if off else lines, 1):
        for found in rules.nameish.finditer(line):
            # the punctuation a sentence hangs on a name is not part of it: 'Tinsley
            # Kilnworks.' was refused with 'Tinsley Kilnworks' on the list (2026-09-02)
            body = found.group(1).strip().strip("’‘'\"`,.:; ")
            if body.casefold() in allowed:
                fired: str | None = "fixtures"
            else:
                # no span: the match is a quoted string, and what surrounds a literal says
                # nothing about what is inside it -- a name in a JS string is still refused
                fired = rules.stood_down(body) or rules.in_context(body, line)
            if fired is None:
                yield path, i, f"{body!r} is shaped like a name (patterns: nameish; nothing stood it down)"
            elif why is not None:
                why.append((path, i, f"{body!r} cleared by {fired}"))
    if engine is None:
        return
    for hit in engine.analyze(text=blob, language="en"):
        if hit.entity_type != "PERSON" or hit.score < FLOOR:
            continue
        body = blob[hit.start:hit.end].strip().strip("’‘'\"`,.:; ")
        # one CamelCase token is a class, not a person
        if not body or " " not in body or body.casefold() in allowed or IDENTIFIER.search(body):
            continue
        at = blob.count("\n", 0, hit.start) + 1
        opened = blob.rfind("\n", 0, hit.start) + 1
        closed = blob.find("\n", hit.end)
        line = blob[opened:closed if closed != -1 else len(blob)]
        # the recogniser is left as strict as it was about people; what stands a hit down is
        # only ever a shape that is code -- a table cell of digits, an expression, a product
        fired = rules.in_context(body, line, (hit.start - opened, hit.end - opened))
        if fired is not None:
            if why is not None:
                why.append((path, at, f"{body!r} cleared by {fired}"))
            continue
        yield path, at, f"{body!r} reads as a person"


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None,
         root: str | os.PathLike[str] | None = None, stdout: TextIO | None = None) -> int:
    """Check the staged files of the repository at ``root`` (default: the one around the
    working directory). Reads its settings from ``env`` (default: the process environment),
    writes its report to ``stdout`` (default: ``sys.stderr``). ``--why`` (or ``NAMES_WHY``)
    also reports every pair a shape rule cleared, and which. Returns 1 when a commit is
    refused, else 0."""
    env = os.environ if env is None else env
    out = sys.stderr if stdout is None else stdout
    if argv and argv[0] == "allow":
        # `python -m ml_stack.redact.hook allow "Windows Defender Firewall"`: the phrase
        # is a product, a code fragment, an invented name -- never a person -- and goes on
        # the allow-list rather than being appended to the file by hand (three times on
        # 2026-09-02). A phrase already there is not added twice.
        where = os.fspath(root) if root is not None else _git(None, "rev-parse", "--show-toplevel").strip()
        fixtures = os.path.join(where, env.get("NAMES_FIXTURES", DEFAULT_FIXTURES))
        asked = "--why" in argv or bool(env.get("NAMES_WHY"))
        return allow(fixtures, [a for a in argv[1:] if a != "--why"], out,
                     rules=_rules(env) if asked else None)
    if env.get("SKIP_NAME_CHECK"):
        return 0
    where = os.fspath(root) if root is not None else _git(None, "rev-parse", "--show-toplevel").strip()
    rules = _rules(env)
    staged = _staged(where, rules)
    if not staged:
        return 0

    fixtures = env.get("NAMES_FIXTURES", DEFAULT_FIXTURES)
    home = env.get("HOME") or os.path.expanduser("~")
    allowed = permitted(where, fixtures, f"{home}/.config/pii-allow.txt")
    known = {n for n in from_database(env.get("NAMES_GRAPH", ""), env.get("NAMES_SCRAPE", ""))
             if n.casefold() not in allowed}
    engine = recogniser()
    explain = "--why" in (argv or ()) or bool(env.get("NAMES_WHY"))

    bad: list[tuple[str, int, str]] = []
    cleared: list[tuple[str, int, str]] | None = [] if explain else None
    for path in staged:
        blob = _git(where, "show", f":{path}")
        if not blob or "\0" in blob[:2048]:
            continue
        bad.extend(_findings(path, blob, known, allowed, engine, rules, cleared))

    if engine is None:
        print("pre-commit: presidio is not installed, so only known names are checked", file=out)
    if cleared:
        print(f"pre-commit: what a rule in {CONTRACT} stood down", file=out)
        for path, line, why in dict.fromkeys(cleared):
            at = f"{path}:{line}" if line else path
            print(f"           {at}  {why}", file=out)
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


def allow(fixtures: str, phrases: list[str], out: TextIO, rules: Shapes | None = None) -> int:
    """Add ``phrases`` to the allow-list at ``fixtures``, once each, under a dated heading.
    Refuses an empty list and says so. With ``rules`` (that is, ``--why`` or ``NAMES_WHY``)
    it first says, for each phrase, which rule already covers it or which rule came nearest
    -- because a fixture covers one phrase and a rule covers every phrase of that shape."""
    phrases = [p.strip() for p in phrases if p and p.strip()]
    if not phrases:
        print("allow what? e.g.: allow \"Windows Defender Firewall\" \"x1 - x0\"", file=out)
        return 2
    if rules is not None:
        for phrase in phrases:
            fired = rules.stood_down(phrase) or rules.in_context(phrase)
            if fired:
                print(f"why {phrase!r}: already stood down by {fired}; no fixture needed",
                      file=out)
                continue
            print(f"why {phrase!r}: no rule stood it down. The nearest:", file=out)
            for miss in rules.near_misses(phrase):
                print(f"           {miss}", file=out)
            print(f"           a line in {CONTRACT} covers every phrase of that shape; "
                  "a fixture covers only this one.", file=out)
    path = Path(fixtures)
    have = {ln.strip().casefold() for ln in path.read_text(encoding="utf-8").splitlines()} \
        if path.exists() else set()
    new = [p for p in phrases if p.casefold() not in have]
    if not new:
        print(f"already allowed in {fixtures}: {', '.join(phrases)}", file=out)
        return 0
    stamp = time.strftime("%Y-%m-%d")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    text += f"\n# allowed with `hook allow` on {stamp}: not people\n" + "".join(f"{p}\n" for p in new)
    path.write_text(text, encoding="utf-8")
    print(f"allowed in {fixtures}: {', '.join(new)}", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
