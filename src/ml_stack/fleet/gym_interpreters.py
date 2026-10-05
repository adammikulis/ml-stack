"""Reusable native simulator interpreters from saved choices and managed scopes."""

import json
import os
from pathlib import Path

from ml_stack.fleet.environment import CATALOG
from ml_stack.gym import manager

LIBRARY_ENVIRONMENTS = {'gym-drone': ['drone']}


def configure_interpreters(ui):
    """Refresh installed simulator scopes without changing setup preferences."""
    settings = getattr(ui, 'settings', None)
    environment = getattr(ui, 'environment', None)
    if not os.environ.get('ML_STACK_GYM_PYTHON'):
        saved = getattr(settings, 'gym_python', '')
        if saved:
            if not Path(saved).is_file():
                raise ValueError('Saved Gym interpreter is missing; select an existing interpreter with --gym-python')
            manager.configure(saved)
        elif environment is not None and environment.exists:
            manager.configure(environment.python)
    choices = {}
    if environment is not None and hasattr(environment, 'for_library'):
        for library in CATALOG:
            if library.name in LIBRARY_ENVIRONMENTS:
                target = environment.for_library(library)
                if target.exists:
                    choices.update(dict.fromkeys(LIBRARY_ENVIRONMENTS[library.name], str(target.python)))
    saved = getattr(settings, 'gym_pythons', {})
    if not isinstance(saved, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in saved.items()):
        raise ValueError('Saved per-example Gym interpreters must map environment IDs to Python paths')
    choices.update(saved)
    explicit = json.loads(os.environ.get('ML_STACK_GYM_PYTHONS', '{}'))
    if not isinstance(explicit, dict):
        raise ValueError('Per-example Gym interpreters must be an object')
    choices.update(explicit)
    manager.configure_map(choices)
