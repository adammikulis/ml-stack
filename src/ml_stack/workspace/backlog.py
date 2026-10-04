"""Authorized repository issues cached and leased in the workspace graph."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from ml_stack.graph.store import GraphStore
from ml_stack.serve.process import pid_exists
from ml_stack.workspace import localagent as la, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.plain import line

POLL_S = 60
MAX_FAILURES = 3
REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def configure(ws, token, name, repo, project):
    """Authorize one coding worker to select local work from a repository's open issues."""
    parent = ws.auth(token)
    agent = la.load(ws, name)
    if agent is None or agent.profile != "coding":
        raise ValueError("repository backlog requires an existing coding worker")
    child = ws.auth(tokens.load(ws.base, agent.identity or name))
    if parent.role != HUMAN and (not child.parent or child.parent != parent.id):
        raise Denied("only the person or this worker's registered parent may enable its backlog")
    if not REPO.fullmatch(repo) or ".." in repo:
        raise ValueError("repository must be owner/name")
    folder = la.check_project(project, ws.base)
    if not folder:
        raise ValueError("repository backlog requires an isolated project folder")
    if not (Path(folder) / ".git").is_file():
        raise ValueError("repository backlog requires an isolated git worktree")
    scope = {"repo": repo, "project": folder, "authority": parent.id, "enabled": True}
    la.save(ws, replace(agent, extra={**agent.extra, "backlog": scope}))
    ws.audit("local-agent.backlog", parent.id, agent=name, repo=repo, project=folder)
    return scope


def fetch(repo):
    """Read at most one hundred open issues through the installed GitHub CLI."""
    binary = shutil.which("gh")
    if not binary:
        raise RuntimeError("install and authenticate gh to read repository issues")
    result = subprocess.run([binary, "issue", "list", "--repo", repo, "--state", "open",
        "--limit", "100", "--json", "number,title,body,url,updatedAt,assignees,labels"],
        capture_output=True, text=True, timeout=30, check=True)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise ValueError("GitHub returned an invalid issue list")
    return [r for r in rows if isinstance(r, dict) and isinstance(r.get("number"), int)
            and not isinstance(r["number"], bool) and r["number"] > 0
            and r.get("url") == f"https://github.com/{repo}/issues/{r['number']}"]


def _store(ws):
    return GraphStore(ws.base / "issue-backlog.db")


def _scope(ws, agent):
    fresh = la.load(ws, agent.name)
    return (fresh.extra.get("backlog") or {}) if fresh else {}


def _record(graph, key):
    return next((row["attrs"] for row in graph.nodes("issue") if row["id"] == key), {})


def _save(graph, key, repo, record):
    graph.upsert_node({"id": f"repo:{repo}", "kind": "repository", "label": repo})
    graph.upsert_node({"id": key, "kind": "issue", "label": key, "attrs": record})
    graph.upsert_edge({"source": f"repo:{repo}", "target": key, "rel": "issue"})
    if record.get("owner"):
        owner = f"agent:{record['owner']}"
        graph.upsert_node({"id": owner, "kind": "agent", "label": record["owner"]})
        graph.upsert_edge({"source": owner, "target": key, "rel": "work"})


def _cache(ws, scope, fetcher, now):
    repo = scope["repo"]
    with held(ws.base / "issue-backlog.lock"), _store(ws) as graph:
        before = graph.get_doc(f"poll:{repo}", {})
    if now < before.get("next_poll", 0):
        return before.get("error", "")
    try:
        issues, error = fetcher(repo), ""
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        issues, error = [], line(str(exc), 240)
    with held(ws.base / "issue-backlog.lock"), _store(ws) as graph:
        graph.put_doc(f"poll:{repo}", {"next_poll": now + POLL_S, "error": error})
        if not error:
            graph.put_doc(f"issues:{repo}", {"rows": issues})
    return error


def pick(ws, agent, *, fetcher=fetch, clock=time.time):
    """Lease the next eligible issue, or return its idle/blocked explanation."""
    scope = _scope(ws, agent)
    if not scope.get("enabled"):
        return None, ""
    now, repo = clock(), scope["repo"]
    error = _cache(ws, scope, fetcher, now)
    if error:
        return None, f"Repository intake blocked: {error}"
    with held(ws.base / "issue-backlog.lock"), _store(ws) as graph:
        rows = graph.get_doc(f"issues:{repo}", {}).get("rows", [])
        for issue in sorted(rows, key=lambda r: r["number"]):
            if issue.get("assignees") or any(label.get("name", "").lower() in
                    ("blocked", "wontfix", "duplicate") for label in issue.get("labels", [])):
                continue
            key = f"issue:{repo}:{issue['number']}"
            old = _record(graph, key)
            if old.get("state") == "working" and pid_exists(old.get("pid", 0)):
                continue
            if old.get("revision") == issue.get("updatedAt") and (
                    old.get("state") == "proposed" or old.get("failures", 0) >= MAX_FAILURES
                    or old.get("next_try", 0) > now):
                continue
            record = {**old, "revision": issue.get("updatedAt"), "state": "working",
                      "owner": agent.identity or agent.name, "pid": os.getpid(), "started": now}
            if old.get("revision") != issue.get("updatedAt"):
                record["failures"] = 0
            _save(graph, key, repo, record)
            return {**issue, "key": key, "repo": repo, "scope": scope}, ""
    return None, "Repository backlog has no unclaimed eligible work; retrying after the next poll"


def finish(ws, issue, agent, result, *, clock=time.time):
    """Record a proposed result or bounded retry without awarding completion credit."""
    kind, text = result
    with held(ws.base / "issue-backlog.lock"), _store(ws) as graph:
        record = _record(graph, issue["key"])
        if record.get("owner") != (agent.identity or agent.name):
            raise Denied("this issue lease belongs to another worker")
        failed = kind != "answer"
        failures = record.get("failures", 0) + int(failed)
        _save(graph, issue["key"], issue["repo"], {**record, "state": "blocked" if failures >= MAX_FAILURES
            else "retry" if failed else "proposed", "failures": failures, "finished": clock(),
            "next_try": clock() + min(900, 30 * 2 ** failures), "summary": line(text, 400)})

