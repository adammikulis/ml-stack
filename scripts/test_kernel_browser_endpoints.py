"""Supervisor-held loopback listeners transferred to admitted test leases."""
from __future__ import annotations

import array
import ipaddress
import json
import os
import secrets
import socket
import threading
from pathlib import Path

import psutil
from test_kernel_endpoint import private_directory, socket_identity, verify_socket


def identifier(value) -> bool:
    return isinstance(value, str) and len(value) == 48 and all(c in '0123456789abcdef' for c in value)


def local_interface(address: str) -> str:
    try:
        parsed = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError:
        raise ValueError('browser confinement: invalid LAN address') from None
    if str(parsed) != address or parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
        raise ValueError('browser confinement: LAN listener requires a host-local unicast address')
    statistics = psutil.net_if_stats()
    for name, entries in psutil.net_if_addrs().items():
        if name in statistics and statistics[name].isup and any(
                entry.family == socket.AF_INET and entry.address == address for entry in entries):
            return name
    raise PermissionError('browser confinement: LAN address is not on an active owned interface')


class EndpointBank:
    def __init__(self, count: int, active, *, lan_address: str | None = None):
        if type(count) is not int or not 1 <= count <= (64 if lan_address is not None else 128):
            raise ValueError('browser confinement: invalid listener budget')
        self.lan_interface = local_interface(lan_address) if lan_address is not None else None
        self.lan_address = lan_address
        self.active = active
        self.token = secrets.token_hex(24)
        self.guard = threading.Lock()
        self.listeners = []
        self.kinds = []
        self.leases = {}
        self.stopped = threading.Event()
        self.thread = None
        self.directory = private_directory()
        self.endpoint = str(self.directory / 'browser.sock')
        self.channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            targets = [('browser-endpoints', '127.0.0.1')]
            if lan_address is not None:
                targets.append(('lan-negative', lan_address))
            for kind, address in targets:
                for _ in range(count):
                    self.reserve(kind, address)
            self.addresses = tuple(listener.getsockname() for listener in self.listeners)
            self.ports = tuple(address[1] for kind, address in zip(self.kinds, self.addresses, strict=True)
                               if kind == 'browser-endpoints')
            self.lan_endpoints = tuple(address for kind, address in zip(self.kinds, self.addresses, strict=True)
                                       if kind == 'lan-negative')
            self.identities = tuple((os.fstat(listener.fileno()).st_dev, os.fstat(listener.fileno()).st_ino) for listener in self.listeners)
            self.channel.bind(self.endpoint)
            Path(self.endpoint).chmod(0o600)
            self.identity = socket_identity(self.endpoint)
            self.channel.listen(count)
            self.channel.settimeout(.2)
            self.thread = threading.Thread(target=self.serve)
            self.thread.start()
        except BaseException:
            self.close()
            raise

    def reserve(self, kind: str, address: str) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind((str(ipaddress.ip_address(address)), 0))
            listener.listen(16)
        except BaseException:
            listener.close()
            raise
        self.listeners.append(listener)
        self.kinds.append(kind)

    def recheck(self) -> None:
        verify_socket(self.endpoint, self.identity)
        if self.lan_address is not None and local_interface(self.lan_address) != self.lan_interface:
            raise RuntimeError('browser confinement: LAN interface identity changed')
        for listener, address, expected in zip(self.listeners, self.addresses, self.identities, strict=True):
            if listener.getsockname() != address or (os.fstat(listener.fileno()).st_dev, os.fstat(listener.fileno()).st_ino) != expected:
                raise RuntimeError('browser confinement: listener identity changed')

    def proof(self) -> dict:
        self.recheck()
        return {'ports': self.ports, 'listeners': self.identities, 'endpoint': self.identity,
                'capacity': len(self.listeners), 'transport': 'SCM_RIGHTS',
                'endpoints': tuple(zip(self.kinds, self.addresses, self.identities, strict=True)),
                'lan_interface': self.lan_interface}

    def operate(self, request: dict, connection: socket.socket) -> None:
        if (not isinstance(request, dict) or request.get('token') != self.token
                or not identifier(request.get('parent'))):
            raise PermissionError('browser confinement: invalid endpoint admission')
        operation = request.get('operation')
        fields = {'operation', 'token', 'parent'} | ({'lease'} if operation == 'release' else {'resource'})
        if set(request) != fields or operation not in {'allocate', 'release'}:
            raise ValueError('browser confinement: invalid endpoint operation')
        with self.active(request['parent']) as authorized, self.guard:
            self.recheck()
            if operation == 'release':
                lease = self.leases.get(request['lease'])
                if lease is None or lease[1] != request['parent'] or lease[2]:
                    raise PermissionError('browser confinement: foreign or released listener')
                self.authorized(authorized, self.kinds[lease[0]])
                self.leases[request['lease']] = (*lease[:2], True)
                connection.sendall(b'{"released":true}\n')
                return
            if request['resource'] not in {'browser-endpoints', 'lan-negative'}:
                raise ValueError('browser confinement: unknown listener resource')
            self.authorized(authorized, request['resource'])
            used = {value[0] for value in self.leases.values()}
            index = next((index for index, kind in enumerate(self.kinds)
                          if kind == request['resource'] and index not in used), None)
            if index is None:
                raise RuntimeError('browser confinement: reserved endpoint bank exhausted')
            lease = secrets.token_hex(24)
            self.leases[lease] = (index, request['parent'], False)
            value = json.dumps({'lease': lease, 'host': self.addresses[index][0], 'port': self.addresses[index][1], 'resource': self.kinds[index]}).encode() + b'\n'
            descriptor = array.array('i', [self.listeners[index].fileno()])
            sent = connection.sendmsg([value], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptor)])
            if sent < len(value):
                connection.sendall(value[sent:])

    @staticmethod
    def authorized(granted, resource: str) -> None:
        if not isinstance(granted, frozenset) or resource not in granted:
            raise PermissionError('browser confinement: active test has no typed resource grant')

    def serve(self) -> None:
        accepted = 0
        while not self.stopped.is_set() and accepted < 4096:
            try:
                connection, _ = self.channel.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with connection:
                accepted += 1
                connection.settimeout(2)
                try:
                    with connection.makefile('rb') as stream:
                        raw = stream.readline(1025)
                    if not raw or len(raw) > 1024 or not raw.endswith(b'\n'):
                        raise ValueError('browser confinement: invalid bounded endpoint frame')
                    self.operate(json.loads(raw), connection)
                except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                    continue

    def close(self) -> None:
        self.stopped.set()
        self.channel.close()
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                raise RuntimeError('browser confinement: endpoint service did not stop')
        for listener in self.listeners:
            listener.close()
        self.listeners.clear()
