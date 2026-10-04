"""Labels for SQL text: which statements it holds and what each does to the data."""

from __future__ import annotations

import re

from ml_stack.guard.harm import Finding

__all__ = ["MAX_SQL", "looks_like_sql", "scan"]

MAX_SQL = 8000
WORD = re.compile(r"[A-Za-z_]+|\d+|[=<>!]+")
STRONG = re.compile(
    r"^\s*(?:select\b.{0,2000}?\bfrom\b|delete\s+from\b|drop\s+(?:table|database|schema|index|view|user|role)\b"
    r"|truncate\b|update\b.{0,500}?\bset\b|insert\s+(?:or\s+\w+\s+)?into\b|alter\s+table\b"
    r"|create\s+(?:or\s+replace\s+)?(?:table|database|index|view|user)\b|grant\b|revoke\b"
    r"|with\b.{0,500}?\bas\s*\()", re.I | re.S)
D, R, S, U = "destructive", "reversible", "safe", "unsure"
TAUTOLOGY = re.compile(r"(?:^| OR )(?:1 = 1|TRUE|S = S|1|0 = 0|2 = 2)$")


def looks_like_sql(text: str) -> bool:
    """Whether ``text`` opens like a SQL statement rather than prose."""
    return STRONG.search(text[:MAX_SQL]) is not None


def _blank(text: str) -> tuple[str, str]:
    """``text`` with comments removed and every literal replaced by ``S``; the second value is
    why that could not be done ('' when it could)."""
    out: list[str] = []
    at, n = 0, len(text)
    while at < n:
        c = text[at]
        two = text[at:at + 2]
        if two == "--" or c == "#":
            end = text.find("\n", at)
            at = n if end < 0 else end
            out.append(" ")
        elif two == "/*":
            end = text.find("*/", at + 2)
            if end < 0:
                return "", "an unterminated comment"
            at = end + 2
            out.append(" ")
        elif c in "'\"`":
            end = at + 1
            while True:
                end = text.find(c, end)
                if end < 0:
                    return "", "an unterminated quoted string"
                if text[end + 1:end + 2] == c:
                    end += 2
                    continue
                break
            out.append(" S " if c == "'" else " " + text[at + 1:end].replace(" ", "_") + " ")
            at = end + 1
        else:
            out.append(c)
            at += 1
    return "".join(out), ""


def _statement(words: list[str]) -> Finding:
    first = words[0] if words else ""
    text = " ".join(words)
    if first in ("SELECT", "SHOW", "DESCRIBE", "DESC", "VALUES", "TABLE", "EXPLAIN", "PRAGMA",
                 "WITH", "ANALYZE", "VACUUM", "BEGIN", "COMMIT", "ROLLBACK", "USE", "START",
                 "END", "SAVEPOINT", "RELEASE", "CHECKPOINT"):
        if first == "PRAGMA" and "=" in words:
            return Finding(R, "changes a database setting")
        if "OUTFILE" in words or "DUMPFILE" in words:
            return Finding(D, "writes query output to a file on the database server")
        for verb in ("DROP", "TRUNCATE", "DELETE", "UPDATE", "INSERT", "GRANT", "REVOKE", "ALTER"):
            if verb in words and first in ("WITH", "EXPLAIN"):
                return _statement(words[words.index(verb):])
        if "INTO" in words and first == "SELECT":
            return Finding(R, "creates a table from a query")
        return Finding(S, f"{first.lower()} only reads")
    if first in ("INSERT", "REPLACE", "MERGE", "UPSERT", "SET", "LOCK", "ATTACH", "DETACH", "COMMENT",
                 "RENAME", "CREATE", "COPY", "LOAD", "IMPORT", "REINDEX", "CLUSTER", "REFRESH"):
        if "PROGRAM" in words:
            return Finding(D, "runs a program from the database server")
        if first == "CREATE" and "REPLACE" in words:
            return Finding(D, "replaces an existing database object")
        return Finding(R, f"{first.lower()} adds or changes data")
    if first == "UPDATE":
        after = text.rsplit(" WHERE ", 1)[1] if " WHERE " in text else ""
        if not after or TAUTOLOGY.search(after):
            return Finding(D, "updates every row (UPDATE without a real WHERE)")
        return Finding(R, "updates the rows a WHERE picks")
    if first == "DELETE":
        after = text.rsplit(" WHERE ", 1)[1] if " WHERE " in text else ""
        every = not after or TAUTOLOGY.search(after)
        return Finding(D, "deletes every row (DELETE without a real WHERE)" if every
                       else "deletes rows")
    if first in ("DROP", "TRUNCATE", "GRANT", "REVOKE", "DEALLOCATE", "PURGE", "FLASHBACK"):
        return Finding(D, f"{first.lower()} removes data or permissions")
    if first == "ALTER":
        if "DROP" in words:
            return Finding(D, "drops a column, constraint or object (ALTER ... DROP)")
        return Finding(R, "changes the schema")
    return Finding(U, f"a SQL statement the classifier does not know ({first.lower() or 'empty'})")


def scan(sql: str) -> list[Finding]:
    """One finding per statement in ``sql``."""
    if len(sql) > MAX_SQL:
        return [Finding(U, f"SQL of {len(sql)} characters is too long to read")]
    cleaned, why = _blank(sql)
    if why:
        return [Finding(U, f"SQL with {why}")]
    found = []
    for part in cleaned.split(";"):
        words = [w.upper() for w in WORD.findall(part)]
        if words:
            found.append(_statement(words))
    return found or [Finding(S, "empty SQL")]
