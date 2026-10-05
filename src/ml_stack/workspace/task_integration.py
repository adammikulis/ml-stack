"""Independently reviewed task branches land through the maintained development gates."""

import hashlib
import subprocess
from pathlib import Path
from uuid import uuid4

from ml_stack.graph.store import GraphStore
from ml_stack.workspace import integration_git as repo
from ml_stack.workspace.chain import held
from ml_stack.workspace.claims import Conflict
from ml_stack.workspace.identity import HUMAN, Denied, Identity
from ml_stack.workspace.task_graph import link, record, save
from ml_stack.workspace.task_schema import fingerprint, review_decision
from ml_stack.workspace.taskboard import TaskBoard


def reviewed(ws, token: str, task_id: str) -> tuple[dict, dict]:
    who = ws.auth(token)
    ws._may(who, 'send')
    board = TaskBoard(ws)
    task = board.get(token, task_id)
    proposal, review = task.get('proposal'), task.get('review')
    if task['workspace'] != board.workspace_id or task['state'] != 'completed' \
            or not proposal or not review or not review['accepted'] or review['outcome'] != 'accepted':
        raise Denied('integration requires an independently accepted canonical proposal')
    parent = ws.registry.info(task['worker'])['parent']
    if who.role != HUMAN and who.id not in (task['worker'], task['created_by'], review['verifier'], parent):
        raise Denied('only the task worker, registered parent, creator or independent reviewer may request integration')
    reviewer = ws.registry.info(review['verifier'])
    if not ws.registry.role_of(review['verifier']):
        raise Denied('the independent reviewer authority expired or was revoked')
    reviewer_identity = Identity(review['verifier'], reviewer['role'], reviewer['parent'],
                                  tuple(reviewer['can']))
    ws._may(reviewer_identity, 'send')
    board._reviewer(reviewer_identity, task)
    for value, key in ((proposal, 'proposal_hash'), (review, 'review_hash')):
        if fingerprint({name: item for name, item in value.items() if name != key}) != value[key]:
            raise Denied('the independently reviewed proposal or outcome was changed')
    review_decision({key: review[key] for key in
                     ('accepted', 'outcome', 'reason', 'checks', 'quality', 'review', 'verified_usage')
                     if key in review}, task['acceptance'], proposal['artifacts'])
    if review['proposal_id'] != proposal['id'] or review['proposal_hash'] != proposal['proposal_hash'] \
            or review['worker'] != proposal['worker'] or task['worker'] != proposal['worker'] \
            or review['task'] != task_id \
            or proposal['task'] != task_id or not all(check['passed'] for check in review['checks']):
        raise Denied('the independent acceptance does not match this exact task proposal')
    with GraphStore(ws.base / 'coordination.db') as graph:
        worktree = record(graph, 'task-worktree:' + task_id.removeprefix('task:'), 'task-worktree')
    if worktree['task'] != task_id or worktree['worker'] != task['worker']:
        raise Denied('the prepared worktree belongs to another task or worker')
    lease = task.get('lease')
    if not lease or lease['active'] or lease['worker'] != task['worker'] \
            or proposal['lease_id'] != lease['id']:
        raise Denied('the reviewed task must have released its exact native execution lease')
    return task, worktree


def _source(task: dict, worktree: dict) -> tuple[Path, str]:
    source = Path(worktree['project']).resolve()
    commit = task['proposal']['provenance'].get('commit', '')
    if not repo.COMMIT.fullmatch(commit):
        raise Denied('the proposal needs the full committed source SHA, not an uncommitted patch')
    repo.clean(source)
    if repo.git(source, 'rev-parse', 'HEAD') != commit \
            or repo.git(source, 'branch', '--show-current') != worktree['branch']:
        raise Denied('the source branch changed after review; submit it for a new independent review')
    if not repo.ancestor(source, worktree['baseline_commit'], commit):
        raise Denied('the reviewed source diverged from its prepared task baseline')
    repo.reviewed_files(source, commit, task['proposal']['artifacts'])
    repo.native_patch(source, commit, worktree['baseline_commit'], task['proposal']['artifacts'])
    return source, commit


def _event(ws, integration: dict, state: str, **details) -> dict:
    event = {'id': 'integration-event:' + uuid4().hex, 'integration': integration['id'],
             'task': integration['task'], 'state': state, 'at': ws.clock(), **details}
    with held(ws.base / 'coordination.lock'), GraphStore(ws.base / 'coordination.db') as graph, graph.transaction():
        previous = next((row['attrs'] for row in graph.nodes('integration') if row['id'] == integration['id']), {})
        if previous and any(previous[key] != integration[key]
                            for key in ('task', 'owner', 'source_commit', 'review_id', 'review_hash')):
            raise Denied('another owner or immutable review already owns this integration record')
        result = {**previous, **integration, 'state': state, **details}
        save(graph, 'integration', result)
        save(graph, 'integration-event', event)
        link(graph, integration['task'], integration['id'], 'integration')
        link(graph, integration['id'], event['id'], 'integration-event')
        link(graph, integration['id'], integration['review_id'], 'authorized-review')
        _proof(graph, integration['id'], details)
    return result


def _proof(graph, integration_id: str, details: dict) -> None:
    if details.get('commit'):
        commit = {'id': 'git-commit:' + details['commit'], 'sha': details['commit']}
        save(graph, 'git-commit', commit)
        link(graph, integration_id, commit['id'], 'gated-commit')
    for index, check in enumerate(details.get('checks', [])):
        proof = {**check, 'id': integration_id + ':gate:' + str(index)}
        save(graph, 'integration-gate', proof)
        link(graph, integration_id, proof['id'], 'required-gate')


class DevelopmentIntegration:
    """Own a candidate checkout, gate its combined tree and publish only development."""

    def __init__(self, ws, token: str, task_id: str):
        self.ws, self.token = ws, token
        self.task, self.worktree = reviewed(ws, token, task_id)
        self.source, self.commit = _source(self.task, self.worktree)
        self.primary, self.development, self.before = repo.repository(self.source)
        digest = hashlib.sha256(f'{self.task["workspace"]}:{task_id}:{self.task["review"]["id"]}'.encode()).hexdigest()
        self.ident = 'integration:' + digest
        self.branch = ws.auth(token).id + '/integrate-' + digest[:16]
        self.candidate = self.primary.parent / (self.primary.name + '-integrate-' + digest[:16])
        self.record = {'id': self.ident, 'task': task_id, 'owner': ws.auth(token).id,
                       'source_commit': self.commit, 'review_id': self.task['review']['id'],
                       'review_hash': self.task['review']['review_hash'],
                       'proposal_hash': self.task['proposal']['proposal_hash'],
                       'development': self.development, 'before': self.before,
                       'candidate': str(self.candidate), 'branch': self.branch,
                       'policy': 'development-integration-v1'}

    def prepare(self) -> None:
        self.ws.claim(self.token, 'branch', self.development)
        self.ws.claim(self.token, 'area', str(self.primary / '.ml-stack-integration'))
        self.ws.claim(self.token, 'branch', self.branch)
        self.ws.claim(self.token, 'worktree', str(self.candidate))
        source_claim = self.ws.who_owns('worktree', str(self.source))
        if source_claim and source_claim['owner'] == self.worktree['worker']:
            who = self.ws.auth(self.token)
            if who.role != HUMAN and self.ws.registry.info(self.worktree['worker'])['parent'] != who.id:
                raise Denied('the current child worktree claim requires its registered parent to return it')
            source_claim = self.ws.claims.return_worktree(who, self.worktree)
            _event(self.ws, self.record, 'scope_returned', scope_owner=source_claim['owner'],
                   worker=self.worktree['worker'])
            self.ws.audit('task.claim_returned', who.id, task=self.task['id'], worker=self.worktree['worker'])
        if not source_claim or source_claim['owner'] != self.worktree['owner']:
            raise Denied('the prepared task source no longer has its recorded worktree owner')
        if (source_claim['owner'] == self.record['owner'] or self.ws.auth(self.token).role == HUMAN) \
                and not Path(repo.git(self.source, 'rev-parse', '--git-path', 'locked')).exists():
            repo.git(self.primary, 'worktree', 'lock', '--reason', 'active reviewed task integration', str(self.source))
        if self.candidate.exists():
            raise Denied('the owned candidate already exists; inspect its recorded outcome before retrying')
        repo.git(self.primary, 'worktree', 'add', '-b', self.branch, str(self.candidate), self.before)
        repo.git(self.primary, 'worktree', 'lock', '--reason', 'active reviewed task integration', str(self.candidate))
        repo.git(self.candidate, 'fetch', 'origin', self.development)
        remote = repo.git(self.candidate, 'rev-parse', 'FETCH_HEAD')
        if not repo.ancestor(self.candidate, remote, self.before):
            raise Denied('development moved remotely; update the integration baseline before retrying')
        repo.git(self.candidate, 'merge', '--no-edit', self.commit)
        tip = repo.git(self.candidate, 'rev-parse', 'HEAD')
        repo.reviewed_files(self.candidate, tip, self.task['proposal']['artifacts'])

    def publish(self, checks: list[dict]) -> dict:
        self.ws.claim(self.token, 'branch', self.development)
        primary, branch, before = repo.repository(self.source)
        if (primary, branch, before) != (self.primary, self.development, self.before):
            raise Denied('development changed during gating; preserve candidate and recheck the new baseline')
        current, worktree = reviewed(self.ws, self.token, self.task['id'])
        if current['review']['review_hash'] != self.record['review_hash'] \
                or current['proposal']['proposal_hash'] != self.record['proposal_hash'] \
                or worktree != self.worktree:
            raise Denied('the review or prepared task binding changed during gating')
        _source(self.task, self.worktree)
        tip = repo.git(self.candidate, 'rev-parse', 'HEAD')
        repo.git(self.primary, 'merge', '--ff-only', tip)
        _event(self.ws, self.record, 'integrated', commit=tip, checks=checks)
        if repo.git(self.primary, 'rev-parse', 'HEAD') != tip:
            raise Denied('development moved after its fast-forward; publication must be rechecked')
        self.ws.claim(self.token, 'branch', self.development)
        repo.git(self.primary, 'push', 'origin', f'{tip}:refs/heads/{self.development}')
        remote = repo.git(self.primary, 'ls-remote', 'origin', f'refs/heads/{self.development}').split()[0]
        if remote != tip:
            raise Denied('remote development did not confirm the exact gated commit')
        return _event(self.ws, self.record, 'published', commit=tip, checks=checks)

    def run(self) -> dict:
        with held(self.primary / '.git' / 'ml-stack-integration.lock'):
            try:
                self.prepare()
                _event(self.ws, self.record, 'candidate')
                checks = repo.gates(self.candidate, self.before)
                return self.publish(checks)
            except (Denied, OSError, RuntimeError, subprocess.SubprocessError) as error:
                details = {'reason': str(error)}
                if isinstance(error, repo.GateFailed):
                    details['checks'] = error.checks
                if isinstance(error, Conflict):
                    details['blocking_owner'] = error.owner['owner']
                    details['blocking_claim'] = error.owner
                return _event(self.ws, self.record, 'blocked', **details)
            finally:
                area = str(self.primary / '.ml-stack-integration')
                claim = self.ws.who_owns('area', area)
                if claim and claim['owner'] == self.record['owner']:
                    self.ws.release(self.token, 'area', area)


def integrate(ws, token: str, task_id: str) -> dict:
    """Integrate a committed independently reviewed task without granting new authority."""
    operation = DevelopmentIntegration(ws, token, task_id)
    with GraphStore(ws.base / 'coordination.db') as graph:
        previous = next((row['attrs'] for row in graph.nodes('integration') if row['id'] == operation.ident), None)
    if previous and previous['state'] == 'published':
        return previous
    return operation.run()
