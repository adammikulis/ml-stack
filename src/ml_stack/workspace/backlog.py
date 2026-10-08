"""Authorized repository issues cached and leased in the workspace graph."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from uuid import uuid4

from ml_stack import worktreerules
from ml_stack.graph.store import GraphStore
from ml_stack.serve.process import pid_exists
from ml_stack.workspace import localagent as la, project as projects, tokens
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import HUMAN, Denied
from ml_stack.workspace.plain import line

POLL_S = 60
REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
ISSUE_REF = re.compile(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#([1-9][0-9]*)$")


def repository(project):
    """Return the GitHub owner/repository named by the checkout's origin, if any."""
    try:
        result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=project,
                                capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    value = result.stdout.strip()
    match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", value)
    return match.group(1) if match and REPO.fullmatch(match.group(1)) \
        and '..' not in match.group(1) else ""


def subscribe_issue(ws, token, ref, enabled=True):
    """Follow or stop following one ``OWNER/REPO#NUMBER`` issue in the caller's inbox."""
    who = ws.auth(token)
    ws._may(who, 'send')
    match = ISSUE_REF.fullmatch(ref)
    if not match or not REPO.fullmatch(match.group(1)) or '..' in match.group(1):
        raise ValueError('issue reference must be OWNER/REPO#NUMBER')
    repo, number = match.group(1), int(match.group(2))
    key = f'issue-watch:{repo}:{number}:{who.id}'
    with held(ws.base / 'issue-backlog.lock'), _store(ws) as graph:
        graph.upsert_node({'id': key, 'kind': 'issue-watch', 'label': f'{repo}#{number}',
                           'attrs': {'repo': repo, 'number': number, 'subscriber': who.id,
                                     'enabled': bool(enabled)}})
    ws.audit('issue.subscribe' if enabled else 'issue.unsubscribe', who.id,
             repo=repo, issue=number)
    return {'repo': repo, 'issue': number, 'subscriber': who.id, 'subscribed': bool(enabled)}


def issue_watchers(ws, repo, number):
    """Return identities following one source issue."""
    with _store(ws) as graph:
        return sorted(row['attrs']['subscriber'] for row in graph.nodes('issue-watch')
                      if row['attrs'].get('repo') == repo and row['attrs'].get('number') == number
                      and row['attrs'].get('enabled'))


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
        raise ValueError("repository backlog requires a source repository")
    if not worktreerules.checkouts(folder):
        raise ValueError("repository backlog requires a registered git checkout")
    if parent.role != HUMAN and ws.registry.info(parent.id).get('project') != projects.describe(folder):
        raise Denied('repository backlog requires the parent registered project')
    if folder != agent.project:
        if parent.role != HUMAN:
            raise Denied("only the person may redirect a worker to a different project")
        ws.registry.set_project(parent, agent.identity or name, projects.describe(folder))
        agent = replace(agent, project=folder)
        la.save(ws, agent)
    scope = {"repo": repo, "project": folder, "authority": parent.id, "enabled": True}
    la.save(ws, replace(agent, extra={**agent.extra, "backlog": scope}))
    ws.audit("local-agent.backlog", parent.id, agent=name, repo=repo, project=folder)
    return scope



def supersede(ws, token, name, number, reason):
    """Record an authorized repository decision that excludes an obsolete issue."""
    who, agent = ws.auth(token), la.load(ws, name)
    if agent is None:
        raise ValueError('the coding worker does not exist')
    child = ws.auth(tokens.load(ws.base, agent.identity or name))
    if who.role != HUMAN and child.parent != who.id:
        raise Denied('only the person or registered worker parent may supersede an issue')
    scope = _scope(ws, agent)
    if not scope.get('enabled'):
        raise ValueError('configure the worker repository backlog first')
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise ValueError('issue number must be a positive integer')
    reason = line(reason, 400).strip()
    if not reason:
        raise ValueError('an explicit superseding decision reason is required')
    repo, key = scope['repo'], f"issue:{scope['repo']}:{number}"
    with held(ws.base / 'issue-backlog.lock'), _store(ws) as graph:
        record = {**_record(graph, key), 'state': 'superseded', 'decision_by': who.id,
                  'decision_at': ws.clock(), 'summary': reason}
        _save(graph, key, repo, record)
        decision = f'issue-decision:{repo}:{number}'
        graph.upsert_node({'id': decision, 'kind': 'issue-decision', 'label': key,
                           'attrs': {'issue': key, 'actor': who.id, 'reason': reason, 'state': 'superseded'}})
        graph.upsert_edge({'source': decision, 'target': key, 'rel': 'supersedes-issue'})
    ws.audit('local-agent.issue-superseded', who.id, repo=repo, issue=number, reason=reason)
    return record



def resume(ws, token, name, number, reason):
    """Authorize one blocked legacy intake retry while preserving its graph history."""
    who, agent = ws.auth(token), la.load(ws, name)
    ws._may(who, 'send')
    if agent is None:
        raise ValueError('the coding worker does not exist')
    child = ws.auth(tokens.load(ws.base, agent.identity or name))
    if who.role != HUMAN and child.parent != who.id:
        raise Denied('only the person or registered worker parent may resume an issue')
    scope = _scope(ws, agent)
    if not scope.get('enabled') or not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise ValueError('resume requires a configured repository and positive issue number')
    reason = line(reason, 400).strip()
    if not reason:
        raise ValueError('an explicit changed-condition reason is required')
    key = f"issue:{scope['repo']}:{number}"
    with held(ws.base / 'issue-backlog.lock'), _store(ws) as graph:
        prior = _record(graph, key)
        if prior.get('state') != 'blocked':
            raise ValueError('only an explicitly blocked issue projection can resume')
        if prior.get('owner') != (agent.identity or name):
            raise Denied('the blocked projection belongs to another worker')
        if any(node['attrs'].get('issue', {}).get('key') == key for node in graph.nodes('issue-dispatch')):
            raise Denied('use canonical TaskBoard resume with its existing retry budget')
        before, decision = f'issue-attempt:{uuid4().hex}', f'issue-recovery:{uuid4().hex}'
        graph.upsert_node({'id': before, 'kind': 'issue-attempt', 'label': key, 'attrs': prior})
        graph.upsert_node({'id': decision, 'kind': 'issue-recovery', 'label': key,
                           'attrs': {'issue': key, 'actor': who.id, 'reason': reason,
                                     'at': ws.clock(), 'retry_budget': 1}})
        graph.upsert_edge({'source': decision, 'target': before, 'rel': 'recovers-blocked-attempt'})
        graph.upsert_edge({'source': before, 'target': key, 'rel': 'attempted-issue'})
        graph.upsert_edge({'source': decision, 'target': key, 'rel': 'resumes-issue'})
        record = {**prior, 'state': 'ready', 'retry_budget': 1, 'resume_by': who.id,
                  'resume_reason': reason, 'prior_attempt': before, 'resume_decision': decision}
        _save(graph, key, scope['repo'], record)
    ws.audit('local-agent.issue-resumed', who.id, issue=number, reason=reason)
    return record


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


def pick(ws, agent, *, fetcher=fetch, clock=time.time, exclude=frozenset()):
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
            if key in exclude:
                continue
            old = _record(graph, key)
            if old.get('state') in ('blocked', 'superseded'):
                continue
            if old.get("state") == "working" and pid_exists(old.get("pid", 0)):
                continue
            if old.get("revision") == issue.get("updatedAt") and (
                    old.get("state") == "proposed"
                    or old.get("next_try", 0) > now):
                continue
            if old.get("resume_decision") and old.get("state") != "proposed" and old.get("retry_budget", 0) < 1:
                continue
            record = {**old, "revision": issue.get("updatedAt"), "state": "working",
                      "owner": agent.identity or agent.name, "pid": os.getpid(), "started": now}
            if old.get("resume_decision") and old.get("state") == "ready":
                record["retry_budget"] = old["retry_budget"] - 1
            if old.get("revision") != issue.get("updatedAt") and old.get("state") != "ready":
                record["failures"] = 0
            _save(graph, key, repo, record)
            return {**issue, "key": key, "repo": repo, "scope": scope}, ""
    return None, "Repository backlog has no unclaimed eligible work; retrying after the next poll"


def finish(ws, issue, agent, result, *, clock=time.time):
    """Project a proposed or blocked outcome without completion credit."""
    kind, text = result
    with held(ws.base / "issue-backlog.lock"), _store(ws) as graph:
        record = _record(graph, issue["key"])
        if record.get("owner") != (agent.identity or agent.name):
            raise Denied("this issue lease belongs to another worker")
        failed = kind != "answer"
        failures = record.get("failures", 0) + int(failed)
        _save(graph, issue["key"], issue["repo"], {**record, "state": "blocked" if failed else "proposed",
            "failures": failures, "finished": clock(), "summary": line(text, 400)})
