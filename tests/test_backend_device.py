"""What `detect_device` says about a machine."""

from __future__ import annotations

import hashlib

from ml_stack.backend import device
from ml_stack.backend.device import DeviceProfile, Vendor


def test_apple_fingerprint_is_the_chip_and_its_cores_whatever_the_memory():
    small = DeviceProfile(Vendor.APPLE, "chip-a", 36.0, True, gpu_cores=40, ram_gb=36.0)
    large = DeviceProfile(Vendor.APPLE, "chip-a", 128.0, True, gpu_cores=40, ram_gb=128.0)
    fewer = DeviceProfile(Vendor.APPLE, "chip-a", 128.0, True, gpu_cores=32)
    assert small.fingerprint == large.fingerprint != fewer.fingerprint
    assert small.fingerprint == hashlib.sha256(b"chip-a|40").hexdigest()[:16]
    assert small.label == "chip-a (40-core GPU)"


def test_card_fingerprint_is_the_card_and_the_cpu_beside_it():
    one = DeviceProfile(Vendor.NVIDIA, "card-b", 24.0, False, cpu_model="cpu-c")
    other = DeviceProfile(Vendor.NVIDIA, "card-b", 24.0, False, cpu_model="cpu-d")
    assert one.fingerprint != other.fingerprint
    assert one.label == "card-b"


def test_nvidia_reads_name_memory_capability_and_driver(monkeypatch):
    answers = {"name,memory.total,compute_cap,driver_version": "card-b, 24564, 8.9, 550.54",
               "memory.free": "20480"}
    monkeypatch.setattr(device, "_nvidia_query", answers.get)
    found = device._detect_nvidia()
    assert found is not None
    assert (found.name, found.compute_capability, found.driver_version) == (
        "card-b", "sm_89", "550.54")
    assert round(found.total_memory_gb, 2) == 23.99
    assert found.free_gb() == 20.0


def test_this_machine_reports_memory_it_has():
    found = device.detect_device()
    assert found.total_memory_gb > 0
    assert 0 < found.free_gb() <= found.total_memory_gb + 1
