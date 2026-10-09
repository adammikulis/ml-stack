"""Metal working-set allowances are separate from physical memory readings."""

import sys
from types import ModuleType, SimpleNamespace

from poolhouse.train.accelerator import mlx_report


def test_metal_limit_is_not_reported_as_free_physical_memory(monkeypatch):
    core = ModuleType('mlx.core')
    core.default_device = lambda: 'gpu'
    core.metal = SimpleNamespace(device_info=lambda: {
        'max_recommended_working_set_size': 112 * 2**30})
    mlx = ModuleType('mlx')
    mlx.core = core
    monkeypatch.setitem(sys.modules, 'mlx', mlx)
    monkeypatch.setitem(sys.modules, 'mlx.core', core)
    got = mlx_report()
    assert got['gpu_working_set_limit_gb'] == 112
    assert got['unified_memory'] is True
    assert 'vram_free_gb' not in got
    assert 'vram_total_gb' not in got
