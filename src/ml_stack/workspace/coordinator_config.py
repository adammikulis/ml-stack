"""Explicit Fleet coordinator routing retained by each device."""

from urllib.parse import urlsplit

from ml_stack.files import read_json, write_json
from ml_stack.workspace.identity import Denied


def load(base):
    document = read_json(base / 'coordinator.json', {})
    if not document:
        return {}
    if document.get('version') != 1 or document.get('mode') not in ('host', 'remote'):
        raise Denied('coordinator configuration is invalid')
    if not str(document.get('workspace', '')).startswith('workspace:'):
        raise Denied('coordinator workspace identity is missing')
    if document['mode'] == 'remote':
        validate_endpoint(document.get('endpoint', ''))
    return document


def validate_endpoint(endpoint):
    parts = urlsplit(endpoint)
    if parts.username or parts.password or parts.path not in ('', '/') or parts.query or parts.fragment:
        raise Denied('the coordinator endpoint is an enrolled Fleet origin')
    if not parts.hostname or parts.scheme not in ('http', 'https'):
        raise Denied('the coordinator endpoint is HTTP or HTTPS')
    if parts.scheme != 'https' and parts.hostname not in ('127.0.0.1', '::1', 'localhost'):
        raise Denied('a remote coordinator requires encrypted Fleet transport')


def save(base, document):
    write_json(base / 'coordinator.json', {'version': 1, **document})
    return load(base)
