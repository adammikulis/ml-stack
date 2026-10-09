"""Immutable allocation-backed attribution for worker proposals."""

from copy import deepcopy


def snapshot(actor, resource):
    """Return the authenticated actor and allocation's observed execution metadata."""
    observed = resource.get('broker_runtime') or {}
    environment = observed.get('environment') or {}
    broker = {key: observed.get(key) for key in
              ('protocol', 'source_commit', 'source_dirty', 'implementation_sha256')}
    broker['environment'] = {key: environment.get(key) for key in ('package_version', 'python')}
    broker['source'] = 'broker-observed' if observed.get('state') == 'observed' else 'unknown'
    return {'actor': actor, 'actor_source': 'authenticated-task-worker',
            'alias': resource.get('actor_alias'), 'alias_source': resource.get('actor_alias_source', 'unknown'),
            'model': resource['model'], 'model_source': 'verified-serving-resource',
            'artifact': {'id': None, 'source': 'unknown'},
            'allocation_id': resource['allocation_id'], 'device_id': resource.get('device_id'),
            'runtime': resource.get('harness') or resource.get('profile') or None,
            'runtime_source': 'registered-worker-configuration', 'backend_artifact': {'id': None, 'source': 'unknown'},
            'broker_runtime': broker, 'execution_config': deepcopy(resource.get('execution_config') or {}),
            'execution_config_source': 'registered-worker-configuration',
            'scheduler_interpreter': resource.get('interpreter'), 'scheduler_python': resource.get('python'),
            'scheduler_source': 'allocation-scheduler-observed',
            'worker_interpreter': {'value': None, 'source': 'unknown'}}
