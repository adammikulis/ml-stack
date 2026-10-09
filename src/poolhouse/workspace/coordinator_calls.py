"""Bounded authenticated command dispatch and persistent reconnect outcomes."""

import hashlib
import json
import re

from poolhouse.graph.store import GraphStore
from poolhouse.workspace.chain import held
from poolhouse.workspace.identity import HUMAN, Denied
from poolhouse.workspace.taskboard import TaskBoard

READS = frozenset({'thread', 'outbox', 'status', 'tasks', 'task', 'audit-verify', 'audit-head'})
WRITES = frozenset({'ack', 'task-create', 'task-claim', 'task-credit', 'task-review', 'task-heartbeat', 'task-checkpoint', 'task-submit'})
REQUEST_ID = re.compile(r'[a-f0-9]{32}')
MAX_OUTCOME = 256 * 1024
MAX_CACHED = 10000
OUTCOME_TTL_S = 86400.0


def prepare(ws, token, argv, parser, handlers):
    who = ws.auth(token)
    if who.role == HUMAN:
        raise Denied('the coordinator RPC accepts agent credentials, never person credentials')
    if not argv or argv[0] not in READS | WRITES:
        raise Denied('this command is not a remote coordination operation')
    ws._may(who, 'read')
    ws._may(who, 'send' if argv[0] in WRITES else 'read')
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        raise ValueError('invalid coordination command arguments') from error
    if args.agent or args.token_file:
        raise Denied('the remote actor is authenticated by its token')
    args.json = True
    if any(getattr(args, field, '') == '-' for field in ('body', 'text', 'payload')):
        raise ValueError('remote command text is supplied in the bounded request')
    if argv[0] == 'send':
        ws._message_rights(who, {'to': args.to, 'subject': args.subject, 'body': args.body,
                                'reply_to': args.reply_to, 'thread': 0}, False)
    if argv[0] == 'announce':
        ws._screen(who, 'send', ws.limits.body_bytes, args.text)
    if argv[0].startswith('task') and argv[0] not in ('tasks', 'task-create'):
        TaskBoard(ws).get(token, args.id)
        if argv[0] == 'task-review':
            TaskBoard(ws).assert_reviewer(token, args.id)
    return who, args, handlers[argv[0]]


def execute(ws, token, document, parser, handlers):
    if set(document) != {'workspace', 'request_id', 'argv'}:
        raise ValueError('a coordination call contains workspace, request_id and argv')
    argv, request_id = document['argv'], document['request_id']
    if type(argv) is not list or len(argv) > 80 or any(type(arg) is not str for arg in argv):
        raise ValueError('argv is a bounded string list')
    if type(request_id) is not str or not REQUEST_ID.fullmatch(request_id):
        raise ValueError('a stable request ID is required')
    who, args, handler = prepare(ws, token, argv, parser, handlers)
    if argv[0] in ('claim', 'release', 'who') and args.kind not in ('branch', 'area'):
        raise Denied('device-local resource claims require a verified paired-device namespace')
    if argv[0] in READS:
        return handler(args, ws, token)
    digest = hashlib.sha256(json.dumps(argv, ensure_ascii=True).encode()).hexdigest()
    ident = f'coordinator-call:{who.id}:{request_id}'
    with held(ws.base / 'coordinator-rpc.lock'):
        who, args, handler = prepare(ws, token, argv, parser, handlers)
        with GraphStore(ws.base / 'coordination.db') as graph:
            expire(graph, ws.clock())
            old = next((node['attrs'] for node in graph.nodes('coordinator-call') if node['id'] == ident), None)
            if old:
                if old['digest'] != digest:
                    raise Denied('a request ID cannot name different operations')
                if old['state'] != 'finished':
                    raise Denied('the original outcome is uncertain; inspect shared state before a new request')
                return old['result']
            graph.upsert_node({'id': ident, 'kind': 'coordinator-call', 'label': argv[0],
                               'attrs': {'actor': who.id, 'workspace': document['workspace'],
                                         'digest': digest, 'state': 'started', 'at': ws.clock()}})
            graph.upsert_edge({'source': document['workspace'], 'target': ident, 'rel': 'coordination-request'})
        result = handler(args, ws, token)
        if len(json.dumps(result, ensure_ascii=True).encode()) > MAX_OUTCOME:
            raise Denied('the coordination outcome exceeds the response limit')
        with GraphStore(ws.base / 'coordination.db') as graph:
            graph.upsert_node({'id': ident, 'kind': 'coordinator-call', 'label': argv[0],
                               'attrs': {'actor': who.id, 'workspace': document['workspace'],
                                         'digest': digest, 'state': 'finished', 'at': ws.clock(), 'result': result}})
        return result


def expire(graph, now):
    finished = sorted((node for node in graph.nodes('coordinator-call')
                       if node['attrs']['state'] == 'finished'), key=lambda node: node['attrs']['at'])
    for index, node in enumerate(finished):
        attrs = node['attrs']
        if attrs['at'] < now - OUTCOME_TTL_S or index < len(finished) - MAX_CACHED + 1:
            graph.upsert_node({**node, 'attrs': {key: value for key, value in attrs.items() if key != 'result'} |
                               {'state': 'expired'}})
