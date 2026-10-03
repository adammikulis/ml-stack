"""Native SUMO procedural networks and imported single-intersection worlds."""

import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from defusedxml.ElementTree import fromstring

from ml_stack.gym.world_files import digest, directory, imported, record
from ml_stack.lock import only_one


def bounded(value, low, high, name):
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def build(spec=None, seed=0):
    spec = dict(spec or {})
    mode = spec.pop('mode', 'procedural')
    seed = int(spec.pop('seed', seed))
    if not 0 <= seed < 2**31:
        raise ValueError('World seed must be between zero and 2147483647')
    if mode == 'manual':
        if not spec.get('net_file') or not spec.get('route_file'):
            raise ValueError('Manual SUMO worlds require net_file and route_file')
        source_paths = {name: spec[name] for name in ('net_file', 'route_file')}
        net, routes = imported(spec.pop('net_file')), imported(spec.pop('route_file'))
        xml(net, 'net')
        xml(routes, 'routes')
        definition = {'mode': mode, 'seed': seed, 'sources': {'net': digest(net), 'routes': digest(routes)}, 'source_paths': source_paths, 'sumo_version': version('eclipse-sumo')}
        path = directory('sumo', definition)
        shutil.copyfile(net, path / 'network.net.xml')
        shutil.copyfile(routes, path / 'routes.rou.xml')
    elif mode == 'procedural':
        arm = bounded(spec.pop('arm_length', np.random.default_rng(seed).integers(90, 241)), 60, 500, 'arm_length')
        lanes = bounded(spec.pop('lanes', 1), 1, 2, 'lanes')
        if not lanes.is_integer():
            raise ValueError('lanes must be an integer')
        lanes = int(lanes)
        period = bounded(spec.pop('vehicle_period', 3), .5, 60, 'vehicle_period')
        demand = int(bounded(spec.pop('demand_seconds', 3600), 60, 86400, 'demand_seconds'))
        definition = {'mode': mode, 'seed': seed, 'arm_length': arm, 'lanes': lanes,
                      'vehicle_period': period, 'demand_seconds': demand, 'sumo_version': version('eclipse-sumo')}
        path = directory('sumo', definition)
        with only_one(path / 'generation.lock', timeout=180):
            generate(path, definition)
    else:
        raise ValueError('World mode must be procedural or manual')
    if spec:
        raise ValueError(f'Unknown traffic world settings: {sorted(spec)}')
    net, routes = path / 'network.net.xml', path / 'routes.rou.xml'
    if len({node.attrib['id'] for node in xml(net, 'net').findall('tlLogic')}) != 1:
        raise ValueError('Traffic worlds require exactly one signal-controlled intersection')
    provenance = record(path, 'SUMO netgenerate/randomTrips' if mode == 'procedural' else 'SUMO imported XML',
                        definition, {'network': net, 'routes': routes})
    return {'net_file': str(net), 'route_file': str(routes), 'sumo_seed': seed}, provenance

def xml(path, expected):
    data = Path(path).read_bytes()
    if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('World XML cannot contain document types or entities')
    node = fromstring(data, forbid_dtd=True)
    if node.tag != expected or any(item.tag == 'include' for item in node.iter()):
        raise ValueError(f'Expected standalone SUMO {expected} XML')
    return node


def canonical_xml(path, expected):
    ElementTree.ElementTree(xml(path, expected)).write(path, encoding='utf-8', xml_declaration=True)


def generate(path, definition):
    if not (path / 'routes.rou.xml').is_file():
        home = Path(os.environ['SUMO_HOME'])
        command = [str(home / 'bin' / 'netgenerate'), '--grid', '--grid.number', '1',
                   '--grid.attach-length', str(definition['arm_length']), '--tls.set', 'A0', '--default.lanenumber', str(definition['lanes']),
                   '--seed', str(definition['seed']), '-o', str(path / 'network.net.xml')]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        subprocess.run([sys.executable, str(home / 'tools' / 'randomTrips.py'),
                        '-n', str(path / 'network.net.xml'), '-r', str(path / 'routes.rou.xml'),
                        '-o', str(path / 'trips.xml'), '--seed', str(definition['seed']), '--end', str(definition['demand_seconds']),
                        '--period', str(definition['vehicle_period']), '--fringe-factor', '10', '--validate'],
                       check=True, capture_output=True, text=True, timeout=120)
        canonical_xml(path / 'network.net.xml', 'net')
        canonical_xml(path / 'routes.rou.xml', 'routes')
