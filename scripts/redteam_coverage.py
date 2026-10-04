#!/usr/bin/env python3
"""Every place untrusted input reaches a model, a tool, a process, a path or the network, and the
red-team test that covers it.

    scripts/redteam_coverage.py            print the counts and the uncovered surfaces
    scripts/redteam_coverage.py --write    write docs/redteam/coverage.json
    scripts/redteam_coverage.py --check    fail on a surface with no row in the map, a row that
                                           matches nothing, a test that is gone, a stale
                                           coverage.json or an uncovered count above the ratchet
    scripts/redteam_coverage.py --update   lower the ratchet to what the map holds

Surfaces are found in the tree by AST: route paths, HTTP handlers, MCP and chat tools (read
from the registries), console scripts, subprocess spawns, listeners, outbound requests, human
grants, desktop and key-store calls, model-context dicts and input parsers. The map
`docs/redteam/coverage-map.toml` says which test covers each one.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from gates._util import calls, dotted, parse, python_files, rel  # noqa: E402

SRC = "src/ml_stack"
OUT = ROOT / "docs/redteam/coverage.json"
MAP = ROOT / "docs/redteam/coverage-map.toml"
STATUSES = ("covered", "partial", "uncovered", "n/a")
SKIP = ("src/ml_stack/testing/", "src/ml_stack/redteam/")

ATTACKER = {
    "route": "reach a protected path, crash the handler, read or write a file outside the root",
    "handler": "malformed, oversized or unauthenticated requests",
    "mcp-tool": "a prompt-injected model calls the tool with hostile arguments",
    "chat-tool": "a prompt-injected model calls the tool with hostile arguments",
    "slash-command": "text read from outside is taken for a person's command",
    "cli": "hostile argument text or path reaches the command",
    "spawn": "attacker text becomes a command line, environment or working directory",
    "listener": "an unauthenticated or cross-origin client reaches the port",
    "egress": "a redirect or address sends the request somewhere the person did not choose",
    "human": "an agent or a forged grant takes an action only a person may take",
    "desktop": "attacker text reaches a dialog, notification or shell command",
    "keystore": "a failure path leaks, downgrades or overwrites a secret",
    "context": "text from outside is read by a model as an instruction",
    "parser": "a crafted file or page crashes the reader, exhausts it or smuggles text",
}
TRUST = {
    "route": "network peer or local browser",
    "handler": "network peer or local browser",
    "mcp-tool": "tool arguments written by a model that may have read hostile text",
    "chat-tool": "tool arguments written by a model that may have read hostile text",
    "slash-command": "a line typed at the prompt, or text echoed from outside",
    "cli": "the person at a terminal; arguments may carry text from outside",
    "spawn": "arguments built from model output, peer messages or file names",
    "listener": "anything that can connect to the address",
    "egress": "addresses and redirects named by pages, repositories and peers",
    "human": "an agent process or a peer",
    "desktop": "names of peers, sessions, files",
    "keystore": "the process environment and the files on disk",
    "context": "tool results, pages, files, peers, stored facts",
    "parser": "files, pages and archives from outside",
}
SPAWN = {
    "subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call",
    "subprocess.call", "subprocess.getoutput", "os.system", "os.popen", "os.execv", "os.execve",
    "os.execvp", "os.execl", "os.spawnv", "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
}
LISTEN = {
    "http.server.HTTPServer", "http.server.ThreadingHTTPServer", "socketserver.TCPServer",
    "socketserver.ThreadingTCPServer", "asyncio.start_server", "socket.create_server",
}
LISTEN_ATTRS = {"serve_forever", "listen"}
EGRESS = {
    "urllib.request.urlopen", "http.client.HTTPConnection", "http.client.HTTPSConnection",
    "socket.create_connection",
}
EGRESS_PREFIX = ("ml_stack.http.", "ml_stack.httpguard.")
HUMAN = {"require_person", "mint", "mint_pressed", "mint_clicked", "protect", "agent_may"}
DESKTOP_WORDS = ("osascript", "notify-send", "powershell", "toast")
KEYSTORE_WORDS = ("add-generic-password", "find-generic-password", "keyring", "secret-tool")
READ_DIRS = ("ingest", "hub", "scrape", "sources", "datasheet", "gguf", "media", "net", "vision",
             "speech", "memory", "workspace", "sentinel", "decide", "taint", "guard", "sandbox",
             "fleet/onboard", "agent", "graph")
READ_FILES = ("web.py", "markup.py", "chat.py", "do.py", "files.py", "messages.py", "roles.py",
              "rules.py", "extraction.py", "records.py", "jsonl/__init__.py",
              "gym/world_files.py", "gym/car_definition.py", "gym/traffic_world.py",
              "fleet/request_fields.py", "fleet/gym_recording_routes.py", "fleet/workspace_routes.py")
FORCED = ("hub/cards.py", "hub/discover.py", "hub/listing.py", "workspace/notes.py",
          "workspace/service.py", "memory/recall.py", "memory/tools.py", "sandbox/policies.py",
          "sentinel/review.py", "decide/questions.py", "decide/pointer_prompt.py",
          "decide/logprob.py")
READ_CALLS = {"json.loads", "tomllib.loads", "tomllib.load", "struct.unpack",
              "struct.unpack_from", "xml.etree.ElementTree.fromstring",
              "xml.etree.ElementTree.parse", "zipfile.ZipFile", "tarfile.open", "pickle.loads"}
READ_ATTRS = {"read_text", "read_bytes", "readlines"}
ROUTE_NAME = re.compile(r"path|route|tail|rest|action", re.I)
ROUTE_DIRS = ("fleet/", "graph/", "sentinel/", "ui/")
ROLE_VALUES = {"tool", "user", "system"}
TRUST_BY_FILE = {
    "fleet/api.py": "peer holding the cluster key; /health open to anyone",
    "fleet/routes.py": "browser session on this machine behind the UI header",
    "graph/": "browser on this machine",
    "fleet/onboard/pairing.py": "anyone on the LAN, unauthenticated until the owner accepts",
    "fleet/onboard/bootstrap.py": "whoever holds the unguessable address",
}


class Surface(dict):
    """One discovered surface: a plain dict that sorts by id."""


def add(found: dict[str, Surface], kind: str, name: str, source: str, trust: str = "") -> None:
    sid = f"{kind}:{name}"
    found.setdefault(sid, Surface(
        id=sid, kind=kind, source=source, trust=trust or TRUST[kind], attacker=ATTACKER[kind]))


def spans(tree: ast.Module) -> list[tuple[int, int, str]]:
    """The (first line, last line, qualified name) of every function, outermost first."""
    out: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((child.lineno, child.end_lineno or child.lineno,
                            f"{prefix}{child.name}"))
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def symbol(table: list[tuple[int, int, str]], line: int) -> str:
    """The innermost function holding the line, or `<module>`."""
    best = "<module>"
    for first, last, name in table:
        if first <= line <= last:
            best = name
    return best.split(".<locals>")[0]


def short(where: str) -> str:
    return where.removeprefix(SRC + "/")


def tree_files() -> list[tuple[str, ast.Module]]:
    out = []
    for path in python_files(ROOT, (SRC,)):
        where = rel(path, ROOT)
        tree = parse(path)
        if tree is not None and not where.startswith(SKIP):
            out.append((where, tree))
    return out


def route_names(node: ast.Compare | ast.Call, where: str) -> list[str]:
    """The path constants a comparison or `startswith` tests."""
    names: list[str] = []
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "startswith" and node.args \
                and ROUTE_NAME.search(ast.unparse(func.value)):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.append(arg.value + "*")
        return [n for n in names if n.startswith("/") and len(n) > 2]
    if not any(isinstance(op, (ast.Eq, ast.In)) for op in node.ops):
        return []
    sides = [node.left, *node.comparators]
    if not any(ROUTE_NAME.search(ast.unparse(s)) for s in sides
               if not isinstance(s, (ast.Constant, ast.Tuple, ast.Set, ast.List))):
        return []
    onboard = "fleet/onboard/" in where
    for side in sides:
        for item in side.elts if isinstance(side, (ast.Tuple, ast.Set, ast.List)) else [side]:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                text = item.value
                if (text.startswith("/") and len(text) > 1) or (onboard and text.isalpha()):
                    names.append(text)
    return names


def find_routes(found: dict[str, Surface], where: str, tree: ast.Module) -> None:
    sub = short(where)
    if not sub.startswith(ROUTE_DIRS):
        return
    table = spans(tree)
    trust = next((t for k, t in TRUST_BY_FILE.items() if sub.startswith(k)), "")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Compare, ast.Call)):
            for name in route_names(node, where):
                add(found, "route", f"{sub}:{name}", f"{sub}:{symbol(table, node.lineno)}", trust)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("handle_"):
            add(found, "route", f"{sub}:{node.name}", f"{sub}:{node.name}", trust)


def find_handlers(found: dict[str, Surface], where: str, tree: ast.Module) -> None:
    sub = short(where)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            verbs = [n.name for n in node.body
                     if isinstance(n, ast.FunctionDef) and n.name.startswith("do_")]
            if verbs:
                add(found, "handler", f"{sub}:{node.name}", f"{sub}:{node.name}")
        if isinstance(node, ast.FunctionDef) and node.name in ("dispatch", "_route") and any(
                "Call" in ast.unparse(a.annotation) for a in node.args.args if a.annotation):
            add(found, "handler", f"{sub}:{node.name}", f"{sub}:{node.name}")


def find_calls(found: dict[str, Surface], where: str, tree: ast.Module) -> None:
    sub = short(where)
    table = spans(tree)
    for node, name in calls(tree):
        sym = symbol(table, node.lineno)
        last = name.rsplit(".", 1)[-1]
        if name in SPAWN:
            add(found, "spawn", f"{sub}:{sym}", f"{sub}:{sym}")
        elif name in LISTEN or (last in LISTEN_ATTRS and not name.startswith("ml_stack")) or (last == "bind" and node.args):
            add(found, "listener", f"{sub}:{sym}", f"{sub}:{sym}")
        elif name in EGRESS or name.startswith(EGRESS_PREFIX):
            add(found, "egress", f"{sub}:{sym}", f"{sub}:{sym}")
        elif last in HUMAN and (name == last or "human" in name or name.startswith("ml_stack")):
            add(found, "human", f"{sub}:{sym}", f"{sub}:{sym}")


def find_words(found: dict[str, Surface], where: str, tree: ast.Module) -> None:
    sub = short(where)
    table = spans(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 80:
            for words, kind in ((DESKTOP_WORDS, "desktop"), (KEYSTORE_WORDS, "keystore")):
                if any(w == node.value or node.value.startswith(w + " ") for w in words):
                    sym = symbol(table, node.lineno)
                    add(found, kind, f"{sub}:{sym}", f"{sub}:{sym}")
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) \
                and dotted(node.exc.func).endswith("HumanRequired"):
            sym = symbol(table, node.lineno)
            add(found, "human", f"{sub}:{sym}", f"{sub}:{sym}")
        if isinstance(node, ast.Compare) and any(
                isinstance(n, ast.Name) and n.id == "HUMAN" for n in [node.left, *node.comparators]):
            sym = symbol(table, node.lineno)
            add(found, "human", f"{sub}:{sym}", f"{sub}:{sym}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
            if "keyring" in names:
                add(found, "keystore", f"{sub}:<module>", f"{sub}:<module>")


def find_context(found: dict[str, Surface], where: str, tree: ast.Module) -> None:
    sub = short(where)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "role" \
                        and isinstance(value, ast.Constant) and value.value in ROLE_VALUES:
                    add(found, "context", sub, sub)


def find_parsers(found: dict[str, Surface], where: str, tree: ast.Module) -> None:
    sub = short(where)
    if sub in FORCED:
        add(found, "parser", sub, sub)
        return
    if not (sub.startswith(tuple(d + "/" for d in READ_DIRS)) or sub in READ_FILES):
        return
    if sub.endswith(("__main__.py", "cli.py")) or (sub.rsplit("/", 1)[-1] == "__init__.py" \
            and sub not in READ_FILES):
        return
    for node, name in calls(tree):
        func = node.func
        if name in READ_CALLS or (isinstance(func, ast.Attribute) and func.attr in READ_ATTRS):
            add(found, "parser", sub, sub)
            return


def find_slash(found: dict[str, Surface], where: str, tree: ast.Module) -> None:
    sub = short(where)
    if sub not in ("chat.py", "do.py", "chatpolicy.py", "rules.py"):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and re.fullmatch(r"/[a-z]+", node.value) and node.value != "/":
            add(found, "slash-command", f"{sub}:{node.value}", f"{sub}:{node.value}")


def find_scripts(found: dict[str, Surface]) -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for script, target in sorted(data["project"]["scripts"].items()):
        add(found, "cli", script, target)


def find_tools(found: dict[str, Surface]) -> None:
    import io

    from ml_stack import chat, do, mcp
    for tool in mcp.TOOLS:
        add(found, "mcp-tool", tool.name, f"mcp.py:{tool.name}")
    person = do.Person(io.StringIO(""), io.StringIO(""))
    session = chat.Chat(None, person, role="approve-first", extension=chat.extensions(person))
    for spec, _ in session.offered:
        add(found, "chat-tool", spec["function"]["name"], f"chat.py:{spec['function']['name']}")


def discover() -> dict[str, Surface]:
    """Every surface in the tree, keyed by id."""
    found: dict[str, Surface] = {}
    for where, tree in tree_files():
        for finder in (find_routes, find_handlers, find_calls, find_words, find_context,
                       find_parsers, find_slash):
            finder(found, where, tree)
    find_scripts(found)
    find_tools(found)
    for kind, name, source in (
        ("desktop", "app/src-tauri/capabilities/main.json", "app/src-tauri/capabilities/main.json"),
        ("spawn", "scripts/test-on-linux", "scripts/test-on-linux"),
        ("spawn", "gym/transport.py:Process.start", "src/ml_stack/gym/transport.py"),
        ("spawn", "sandbox/bubblewrap.py:_probe", "src/ml_stack/sandbox/bubblewrap.py"),
        ("route", "fleet/gym_recording_routes.py:/ui/gym/recordings*", "src/ml_stack/fleet/gym_recording_routes.py"),
    ):
        if (ROOT / source).is_file():
            add(found, kind, name, source)
    return dict(sorted(found.items()))


def load_map() -> dict:
    return tomllib.loads(MAP.read_text(encoding="utf-8")) if MAP.exists() else {}


def resolve(found: dict[str, Surface], table: dict) -> tuple[dict[str, dict], list[str]]:
    """Each surface's first matching group, and the problems in the groups themselves."""
    groups = table.get("group", [])
    problems: list[str] = []
    owner: dict[str, dict] = {}
    hits = [0] * len(groups)
    for sid in found:
        for index, group in enumerate(groups):
            if any(fnmatch.fnmatchcase(sid, pattern) for pattern in group["match"]):
                owner.setdefault(sid, group)
                hits[index] += 1
                break
    for index, group in enumerate(groups):
        problems += group_problems(group, hits[index])
    return owner, problems


def group_problems(group: dict, hit: int) -> list[str]:
    label = f"group {group['match']}"
    out = []
    if group.get("status") not in STATUSES:
        out.append(f"{label}: status must be one of {STATUSES}")
    if hit == 0:
        out.append(f"{label}: matches no surface; delete it or fix the pattern")
    if "count" in group and group["count"] != hit:
        out.append(f"{label}: expected {group['count']} surfaces, found {hit}. A surface was added "
                   "or removed here: check that a test covers the new one, then set count.")
    status = group.get("status")
    if status in ("covered", "partial") and not group.get("tests"):
        out.append(f"{label}: {status} needs at least one test")
    if status in ("n/a", "uncovered") and not group.get("note"):
        out.append(f"{label}: {status} needs a note saying why")
    for ref in group.get("tests", []):
        out += ref_problems(label, ref)
    return out


def ref_problems(label: str, ref: str) -> list[str]:
    path, _, name = ref.partition("::")
    file = ROOT / path
    if not file.is_file():
        return [f"{label}: test file {path} does not exist"]
    if name and not re.search(rf"def {re.escape(name)}\b", file.read_text(encoding="utf-8")):
        return [f"{label}: {path} has no test named {name}"]
    return []


def rows(found: dict[str, Surface], owner: dict[str, dict]) -> list[dict]:
    out = []
    for sid, surface in found.items():
        group = owner.get(sid)
        row = dict(surface)
        row["status"] = group["status"] if group else "unmapped"
        row["tests"] = list(group.get("tests", [])) if group else []
        row["note"] = group.get("note", "") if group else ""
        out.append(row)
    return out


def counts(table: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in table:
        bucket = out.setdefault(row["kind"], dict.fromkeys((*STATUSES, "unmapped"), 0))
        bucket[row["status"]] += 1
    return out


def totals(table: list[dict]) -> dict[str, int]:
    out = dict.fromkeys((*STATUSES, "unmapped"), 0)
    for row in table:
        out[row["status"]] += 1
    return out


def render(table: list[dict]) -> str:
    body = {"surfaces": len(table), "by_status": totals(table), "by_kind": counts(table),
            "rows": table}
    return json.dumps(body, indent=1, sort_keys=False) + "\n"


def stanza(sid: str, surface: Surface) -> str:
    return (f'[[group]]\nmatch = ["{sid}"]\nstatus = "covered"   # or partial, uncovered, n/a\n'
            f'tests = ["tests/test_redteam_....py::test_name"]\nnote = "{surface["attacker"]}"\n')


def ratchet() -> dict[str, int]:
    return load_map().get("ratchet", {})


def check(found: dict[str, Surface], table: list[dict], owner: dict[str, dict],
          problems: list[str]) -> list[str]:
    out = list(problems)
    for sid, surface in found.items():
        if sid not in owner:
            out.append(f"no row in docs/redteam/coverage-map.toml for the surface {sid} "
                       f"({surface['source']}). Write a red-team test that attacks it and add:\n\n"
                       + stanza(sid, surface))
    held = totals(table)
    limit = ratchet()
    for key in ("uncovered", "partial"):
        if held[key] > limit.get(key, 0):
            out.append(f"{key} surfaces rose to {held[key]} from {limit.get(key, 0)}: cover them")
        elif held[key] < limit.get(key, 0):
            out.append(f"{key} fell to {held[key]} from {limit.get(key, 0)}: run "
                       "scripts/redteam_coverage.py --update and commit docs/redteam/coverage-map.toml")
    if not OUT.exists() or OUT.read_text(encoding="utf-8") != render(table):
        out.append("docs/redteam/coverage.json is stale: run scripts/redteam_coverage.py --write")
    return out


def update_ratchet(table: list[dict]) -> None:
    """Rewrite the [ratchet] numbers in the map, never upward."""
    held = totals(table)
    text = MAP.read_text(encoding="utf-8")
    for key in ("uncovered", "partial"):
        if os.environ.get("CLAUDECODE") and held[key] > ratchet().get(key, 0):
            raise SystemExit(f"{key} would rise; only the owner raises a number")
        text = re.sub(rf"(?m)^{key} = \d+$", f"{key} = {held[key]}", text, count=1)
    MAP.write_text(text, encoding="utf-8")


def summary(table: list[dict]) -> str:
    held = totals(table)
    lines = [f"{len(table)} surfaces: " + ", ".join(f"{k} {v}" for k, v in held.items())]
    for kind, bucket in counts(table).items():
        lines.append(f"  {kind:14} " + " ".join(f"{k}={v}" for k, v in bucket.items() if v))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--list", metavar="STATUS", help="print the surfaces with this status")
    args = parser.parse_args(argv)
    found = discover()
    owner, problems = resolve(found, load_map())
    table = rows(found, owner)
    if args.write:
        OUT.write_text(render(table), encoding="utf-8")
    if args.update:
        update_ratchet(table)
    if args.list:
        print("\n".join(f"{r['id']}  {r['source']}  {r['note']}" for r in table
                        if r["status"] == args.list))
    print(summary(table))
    if args.check:
        failures = check(found, table, owner, problems)
        for failure in failures:
            print("FAIL " + failure, file=sys.stderr)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
