"""Bounded native-quoted exact path languages."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

GROUP = 32
MAX_PATHS = 65536
MAX_PATH_BYTES = 4096
MAX_INPUT = 32 * 1024 * 1024
MAX_OUTPUT = 32 * 1024 * 1024
MAX_NODES = 65536
MAX_DEPTH = 64
MAX_EXPRESSION_PATHS = 1024
MAX_EXPRESSION_NODES = 4096
MAX_EXPRESSION_BYTES = 65536
CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class LanguageError(ValueError):
    """An exact language exceeds its representation bounds."""


@dataclass(frozen=True)
class Node:
    edge: str
    terminal: bool
    children: tuple[Node, ...]


def paths_checked(paths: Sequence[str]) -> list[str]:
    """Return sorted unique absolute path text within aggregate bounds."""
    if len(paths) > MAX_PATHS:
        raise LanguageError("exact paths: path count exceeds bound")
    size = 0
    for path in paths:
        if (not isinstance(path, str) or not path.startswith("/") or CONTROL.search(path)
                or "//" in path or (path != "/" and path.endswith("/"))
                or any(part in (".", "..") for part in path.split("/"))):
            raise LanguageError("exact paths: malformed path text")
        try:
            count = len(path.encode("utf-8"))
        except UnicodeError as error:
            raise LanguageError("exact paths: invalid UTF-8") from error
        size += count
        if count > MAX_PATH_BYTES or size > MAX_INPUT:
            raise LanguageError("exact paths: input bytes exceed bound")
    return sorted(set(paths))


def compact(raw: dict, edge: str = "", depth: int = 0) -> Node:
    """Compact unary nonterminal edges without crossing terminal nodes."""
    if depth > MAX_DEPTH:
        raise LanguageError("exact paths: branch depth exceeds bound")
    while "" not in raw and len(raw) == 1:
        character, raw = next(iter(raw.items()))
        edge += character
    children = tuple(compact(raw[key], key, depth + 1) for key in sorted(raw) if key)
    return Node(edge, "" in raw, children)


def terminals(root: Node) -> set[str]:
    """Enumerate complete terminal paths from the compact tree."""
    values = set()
    pending = [(root, "", 0)]
    visited = 0
    while pending:
        node, prefix, depth = pending.pop()
        visited += 1
        if depth > MAX_DEPTH or visited > MAX_NODES:
            raise LanguageError("exact paths: terminal traversal exceeds bound")
        path = prefix + node.edge
        if node.terminal:
            values.add(path)
        pending.extend((child, path, depth + 1) for child in node.children)
    return values


def tree(paths: Sequence[str]) -> Node:
    """Build one bounded group and verify its complete terminal set."""
    if not 1 <= len(paths) <= GROUP:
        raise LanguageError("exact paths: group count exceeds bound")
    raw = {}
    count = 1
    for path in paths:
        current = raw
        for character in path:
            if character not in current:
                count += 1
                if count > MAX_NODES:
                    raise LanguageError("exact paths: trie nodes exceed bound")
                current[character] = {}
            current = current[character]
        current[""] = True
    result = compact(raw)
    if terminals(result) != set(paths):
        raise LanguageError("exact paths: terminal language differs from input")
    return result


def render(node: Node, quote: Callable[[str], str]) -> str:
    """Render native-quoted edges and explicit terminal alternatives."""
    edge = "(regex-quote " + quote(node.edge) + ")"
    if not node.children:
        if not node.terminal:
            raise LanguageError("exact paths: nonterminal leaf")
        return edge
    branches = (["(regex-quote " + quote("") + ")"] if node.terminal else [])
    branches.extend(render(child, quote) for child in node.children)
    joined = ' "|" '.join(branches)
    return '(string-append ' + edge + ' "(" ' + joined + ' ")")'


def freeze_radix(raw: dict, depth: int = 0) -> Node:
    """Freeze sorted compact edges within the existing depth limit."""
    if depth > MAX_DEPTH:
        raise LanguageError("exact paths: branch depth exceeds bound")
    return Node(raw["edge"], raw["terminal"], tuple(
        freeze_radix(raw["children"][key], depth + 1) for key in sorted(raw["children"])))


def radix(paths: Sequence[str]) -> Node:
    """Build one charged compressed tree and verify every complete terminal."""
    root = {"edge": "", "terminal": False, "children": {}}
    nodes, size = 1, 0
    for path in paths:
        current, remaining, depth = root, path, 0
        while remaining:
            if depth > MAX_DEPTH:
                raise LanguageError("exact paths: branch depth exceeds bound")
            children = current["children"]
            child = children.get(remaining[0])
            if child is None:
                nodes += 1
                size += len(remaining.encode("utf-8"))
                if nodes > MAX_NODES or size > MAX_INPUT:
                    raise LanguageError("exact paths: radix storage exceeds bound")
                children[remaining[0]] = {"edge": remaining, "terminal": True, "children": {}}
                break
            edge = child["edge"]
            common = 0
            while common < min(len(edge), len(remaining)) and edge[common] == remaining[common]:
                common += 1
            if common < len(edge):
                nodes += 1
                if nodes > MAX_NODES:
                    raise LanguageError("exact paths: radix nodes exceed bound")
                suffix = {"edge": edge[common:], "terminal": child["terminal"], "children": child["children"]}
                child.update(edge=edge[:common], terminal=False, children={suffix["edge"][0]: suffix})
            remaining = remaining[common:]
            if not remaining:
                child["terminal"] = True
                break
            current, depth = child, depth + 1
    if len(root["children"]) == 1:
        root = next(iter(root["children"].values()))
    result = freeze_radix(root)
    if terminals(result) != set(paths):
        raise LanguageError("exact paths: terminal language differs from input")
    return result


def subtree_counts(root: Node) -> dict[int, tuple[int, int]]:
    """Count each compact subtree with bounded traversal."""
    counts = {}
    pending = [(root, False, 0)]
    while pending:
        node, complete, depth = pending.pop()
        if depth > MAX_DEPTH:
            raise LanguageError("exact paths: partition depth exceeds bound")
        if complete:
            counts[id(node)] = (int(node.terminal) + sum(counts[id(child)][0] for child in node.children),
                                1 + sum(counts[id(child)][1] for child in node.children))
            if counts[id(node)][1] > MAX_NODES:
                raise LanguageError("exact paths: partition nodes exceed bound")
        else:
            pending.append((node, True, depth))
            pending.extend((child, False, depth + 1) for child in node.children)
    return counts


def partition(root: Node, quote: Callable[[str], str]) -> list[tuple[Node, str]]:
    """Split only at compact subtree boundaries within expression limits."""
    counts = subtree_counts(root)
    pending = [(root, "")]
    groups = []
    output_bytes = 0
    while pending:
        node, prefix = pending.pop()
        candidate = Node(prefix + node.edge, node.terminal, node.children)
        path_count, node_count = counts[id(node)]
        if path_count <= MAX_EXPRESSION_PATHS and node_count <= MAX_EXPRESSION_NODES:
            value = '(regex (string-append "^" ' + render(candidate, quote) + ' "$"))'
            if len(value.encode("utf-8")) <= MAX_EXPRESSION_BYTES:
                output_bytes += len(value.encode("utf-8"))
                if output_bytes > MAX_OUTPUT:
                    raise LanguageError("exact paths: output bytes exceed bound")
                groups.append((candidate, value))
                continue
        if not node.children:
            raise LanguageError("exact paths: singleton expression exceeds bound")
        path = prefix + node.edge
        if node.terminal:
            terminal = Node(path, True, ())
            value = '(regex (string-append "^" ' + render(terminal, quote) + ' "$"))'
            if len(value.encode("utf-8")) > MAX_EXPRESSION_BYTES:
                raise LanguageError("exact paths: terminal expression exceeds bound")
            output_bytes += len(value.encode("utf-8"))
            if output_bytes > MAX_OUTPUT:
                raise LanguageError("exact paths: output bytes exceed bound")
            groups.append((terminal, value))
        pending.extend((child, path) for child in reversed(node.children))
    return groups


def exact_language(paths: Sequence[str], quote: Callable[[str], str]) -> tuple[list[str], dict]:
    """Return verified native expressions and complete input/terminal digests."""
    unique = paths_checked(paths)
    branch = radix(unique) if unique else None
    recovered = set()
    groups = partition(branch, quote) if branch is not None else []
    result = []
    for node, value in groups:
        complete = terminals(node)
        if recovered & complete:
            raise LanguageError("exact paths: partition terminal groups overlap")
        recovered.update(complete)
        result.append(value)
    size = sum(len(value.encode("utf-8")) for value in result)
    if size > MAX_OUTPUT:
        raise LanguageError("exact paths: output bytes exceed bound")
    if recovered != set(unique):
        raise LanguageError("exact paths: aggregate terminal language differs from input")
    original = json.dumps(unique, ensure_ascii=False).encode("utf-8")
    complete = json.dumps(sorted(recovered), ensure_ascii=False).encode("utf-8")
    return result, {"representation": "subtree-radix", "groups": len(result),
                    "max_expression_bytes": max((len(value.encode("utf-8")) for value in result), default=0),
                    "paths": len(unique), "input_sha256": hashlib.sha256(original).hexdigest(),
                    "terminals_sha256": hashlib.sha256(complete).hexdigest(), "output_bytes": size,
                    "expressions_sha256": hashlib.sha256("\n".join(result).encode("utf-8")).hexdigest()}


def expressions(paths: Sequence[str], quote: Callable[[str], str]) -> list[str]:
    """Return anchored native expressions for exactly the supplied complete paths."""
    return exact_language(paths, quote)[0]
