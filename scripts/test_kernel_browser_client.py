"""Fixture-side listener leases over the supervisor's private Unix channel."""
from __future__ import annotations

import array
import ipaddress
import json
import os
import socket

from test_kernel_endpoint import connection


def received_descriptors(ancillary) -> list[int]:
    received = []
    malformed = False
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors = array.array('i')
            length = len(value) - len(value) % descriptors.itemsize
            descriptors.frombytes(value[:length])
            received.extend(descriptors)
            malformed |= length != len(value)
        else:
            malformed = True
    if malformed:
        for descriptor in received:
            os.close(descriptor)
        raise RuntimeError('browser confinement: invalid descriptor response')
    return received


class Listener:
    def __init__(self, resource: str = 'browser-endpoints'):
        if resource not in {'browser-endpoints', 'lan-negative'}:
            raise ValueError('browser confinement: unknown listener resource')
        self.resource = resource
        self.socket = None
        self.lease = None
        self.parent = os.environ['DEV_TEST_REMOTE_LEASE']
        with self.channel() as stream:
            self.send(stream, 'allocate')
            raw, ancillary, flags, _ = stream.recvmsg(1024, socket.CMSG_SPACE(array.array('i').itemsize))
            received = []
            try:
                received = received_descriptors(ancillary)
                if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or len(received) != 1:
                    raise RuntimeError('browser confinement: listener transfer failed')
                while raw and not raw.endswith(b'\n') and len(raw) <= 1024:
                    block = stream.recv(1025 - len(raw))
                    if not block:
                        break
                    raw += block
                if not raw.endswith(b'\n') or len(raw) > 1024:
                    raise RuntimeError('browser confinement: incomplete bounded listener response')
                response = json.loads(raw)
                if (set(response) != {'lease', 'host', 'port', 'resource'} or response['resource'] != self.resource
                        or not isinstance(response['host'], str)
                        or not isinstance(response['lease'], str) or len(response['lease']) != 48
                        or any(c not in '0123456789abcdef' for c in response['lease'])
                        or type(response['port']) is not int):
                    raise RuntimeError('browser confinement: invalid listener receipt')
                address = ipaddress.IPv4Address(response['host'])
                if (str(address) != response['host'] or address.is_unspecified or address.is_multicast
                        or (self.resource == 'browser-endpoints' and response['host'] != '127.0.0.1')
                        or (self.resource == 'lan-negative' and address.is_loopback)):
                    raise RuntimeError('browser confinement: invalid listener address')
                self.socket = socket.socket(fileno=received[0])
                received.clear()
                self.socket.set_inheritable(False)
                if (self.socket.family != socket.AF_INET or self.socket.type != socket.SOCK_STREAM
                        or self.socket.getsockname() != (response['host'], response['port'])
                        or self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1):
                    raise RuntimeError('browser confinement: transferred descriptor is not the reserved listener')
                self.lease = response['lease']
            except BaseException:
                if self.socket is not None:
                    self.socket.close()
                raise
            finally:
                for descriptor in received:
                    os.close(descriptor)

    def channel(self):
        return connection('unix:' + os.environ['DEV_TEST_BROWSER_ENDPOINT'], os.environ['DEV_TEST_BROWSER_IDENTITY'])

    def send(self, stream, operation):
        value = {'operation': operation, 'token': os.environ['DEV_TEST_BROWSER_TOKEN'], 'parent': self.parent}
        if operation == 'release':
            value['lease'] = self.lease
        else:
            value['resource'] = self.resource
        stream.sendall(json.dumps(value).encode() + b'\n')

    def close(self):
        if self.socket is None:
            return
        self.socket.close()
        self.socket = None
        with self.channel() as stream:
            self.send(stream, 'release')
            with stream.makefile('rb') as reader:
                if reader.readline(1025) != b'{"released":true}\n':
                    raise RuntimeError('browser confinement: listener release failed')
