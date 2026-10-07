"""Read-only inspection of registered application and dataset graphs."""

from contextlib import nullcontext
from itertools import islice
from pathlib import Path

from ml_stack import lock
from ml_stack.graph.columns import column
from ml_stack.graph.store import GraphStore

from .files import safe_relpath
from .jobs import DaemonError
from .room_routes import _origin_ok

PAGE_SIZE = 60
FIELD_CHARS = 32_768
NODE_FIELDS = (
    "n.id AS id, n.kind AS kind, substring(n.label, 1, 2000) AS label, "
    "n.mentions AS mentions, substring(n.attrs, 1, 32769) AS attrs, "
    "substring(n.data, 1, 32769) AS data"
)


def preview(raw, label):
    if raw and len(raw) > FIELD_CHARS:
        return {"preview": raw[:FIELD_CHARS], "truncated": True}
    return column(raw, label)


def decoded(row):
    return {**{key: value for key, value in row.items() if key not in {"data", "attrs"}},
            "attrs": preview(row["attrs"], "node attributes"),
            "details": preview(row["data"], "node details")}


class KnowledgeRoutes:
    def route(self) -> bool:
        if not self.path.startswith("/ui/knowledge/"):
            return super().route()
        if not _origin_ok(self.header("Origin"), self.host_header):
            self.send(403, {"error": "Graph inspection requires the same origin."})
            return True
        if self.method != "GET":
            self.send(405, {"error": "The graph inspector is read-only."})
            return True
        try:
            root = Path(self.ui.runner.files_root).resolve() if self.ui.runner else None
            sources = self._sources()
            if self.path == "/ui/knowledge/stores":
                return self._stores(root, sources)
            if self.path not in ("/ui/knowledge/nodes", "/ui/knowledge/node"):
                return super().route()
            wanted = self.asked("store")
            if wanted.startswith(("app:", "project:")):
                if wanted not in sources:
                    self.send(403, {"error": "This application graph is unavailable to this session."})
                    return True
                path, lock_path, _label = sources[wanted]
            else:
                if root is None:
                    self.send(501, {"error": "Start the local daemon to inspect dataset graphs."})
                    return True
                path, lock_path = safe_relpath(root, wanted), None
            if path.suffix not in (".db", ".lbug", ".kuzu") or not path.is_file() or path.is_symlink():
                raise ValueError("Choose an existing graph store from the list.")
            guard = (lock.only_one(lock_path, timeout=2, announce=lambda _text: None)
                     if lock_path else nullcontext())
            with guard, GraphStore(path, read_only=True, buffer_pool_size=32 << 20) as graph:
                if self.path.endswith("/node"):
                    return self._node(graph)
                return self._nodes(graph)
        except lock.Busy:
            self.send(409, {"error": "This graph is being updated. Try again shortly."})
            return True
        except (DaemonError, ValueError, OSError, RuntimeError) as exc:
            self.send(400, {"error": str(exc)})
            return True

    def _sources(self):
        sources = {}
        conversations = getattr(self.ui, "conversations", None)
        if conversations is not None:
            base = Path(conversations.root)
            if (base / "conversations.db").is_file():
                sources["app:conversations"] = (base / "conversations.db", base / "conversations.lock", "Conversations and models")
        projects = getattr(self.ui, "projects", None)
        if projects is not None and self._local_person():
            for project in islice(projects.boards(), 200):
                if project["authority_machine"] != projects.machine:
                    continue
                base = projects.workspace_base(project["id"])
                if (base / "coordination.db").is_file():
                    key = "project:" + project["id"] + ":coordination"
                    sources[key] = (base / "coordination.db", base / "coordination.lock", project["name"] + " · tasks and coordination")
        return sources

    def _local_person(self):
        return (self.client_ip in {"127.0.0.1", "::1"} and self.ui.authed(self.cookie)
                and self.ui.host_ok(self.host_header))

    def _stores(self, root, sources):
        relative, entries, truncated = self.asked("path"), [], False
        if root is not None:
            folder = safe_relpath(root, relative) if relative else root
            if not folder.is_dir():
                raise ValueError("Choose a folder inside the files root.")
            for index, child in enumerate(folder.iterdir()):
                if index == 1000 or len(entries) == 200:
                    truncated = True
                    break
                if child.is_symlink() or not child.resolve().is_relative_to(root):
                    continue
                if child.is_dir() or child.suffix in (".db", ".lbug", ".kuzu"):
                    entries.append({"path": child.relative_to(root).as_posix(),
                                    "name": child.name, "directory": child.is_dir()})
        self.send(200, {"path": relative, "entries": sorted(entries, key=lambda item: item["name"].casefold()),
                        "truncated": truncated, "files_available": root is not None,
                        "sources": [{"path": key, "name": value[2]} for key, value in sources.items()]})
        return True

    def _nodes(self, graph):
        offset = int(self.asked("offset", "0"))
        if offset < 0 or offset > 1_000_000:
            raise ValueError("Node offset must be between 0 and 1000000.")
        text, kind = self.asked("q")[:200], self.asked("kind")[:200]
        params = {"q": text.lower(), "kind": kind}
        where = ("WHERE ($q = '' OR lower(n.label) CONTAINS $q OR lower(n.id) CONTAINS $q) "
                 "AND ($kind = '' OR n.kind = $kind) ")
        rows = graph.query("MATCH (n:Node) " + where + "RETURN " + NODE_FIELDS
                           + f" ORDER BY label, id SKIP {offset} LIMIT {PAGE_SIZE + 1}", params)
        kinds = graph.query("MATCH (n:Node) RETURN n.kind AS kind, count(n) AS count ORDER BY kind LIMIT 200")
        self.send(200, {"nodes": [decoded(row) for row in rows[:PAGE_SIZE]],
                        "more": len(rows) > PAGE_SIZE, "offset": offset,
                        "kinds": kinds, "counts": graph.counts()})
        return True

    def _node(self, graph):
        node_id = self.asked("id")
        rows = graph.query("MATCH (n:Node {id:$id}) RETURN " + NODE_FIELDS, {"id": node_id})
        if not rows:
            self.send(404, {"error": "This node is no longer in the graph."})
            return True
        relations = graph.query(
            "MATCH (a:Node)-[e:Edge]->(b:Node) WHERE a.id = $id OR b.id = $id "
            "RETURN a.id AS source, substring(a.label, 1, 200) AS source_label, b.id AS target, "
            "substring(b.label, 1, 200) AS target_label, e.rel AS rel, e.weight AS weight "
            "ORDER BY weight DESC, source, target LIMIT 201", {"id": node_id})
        self.send(200, {"node": decoded(rows[0]), "relations": relations[:200],
                        "truncated": len(relations) > 200})
        return True
