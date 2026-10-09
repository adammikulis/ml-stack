"""The directory server logs go to stays bounded however many servers have been started."""

import os
import time

import pytest

from ml_stack import home
from ml_stack.serve import logs
from ml_stack.serve.backend import server_log


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    for name in (logs.FILES_ENV, logs.MEGABYTES_ENV, logs.DAYS_ENV):
        monkeypatch.delenv(name, raising=False)


def _logs(count, *, size=10, age_s=None):
    """``count`` logs on as many ports, the first the oldest."""
    age_s = age_s or (lambda n: count - n)
    out = []
    directory = home.state("logs")
    directory.mkdir(parents=True, exist_ok=True)
    for n in range(count):
        one = directory / f"llama-server-{9000 + n}-20260101-0000{n:02d}-1.log"
        one.write_bytes(b"x" * size)
        then = time.time() - age_s(n)
        os.utime(one, (then, then))
        out.append(one)
    return out


def test_each_start_on_a_new_port_does_not_grow_the_directory_without_end(monkeypatch):
    monkeypatch.setenv(logs.FILES_ENV, "20")
    made = _logs(50)
    server_log("llama-server", 9999)
    left = sorted(home.state("logs").glob("*.log"))
    assert len(left) <= 20
    assert made[-1] in left and made[0] not in left, "the newest stay, the oldest go"


def test_the_total_size_is_bounded(monkeypatch):
    monkeypatch.setenv(logs.MEGABYTES_ENV, "1")
    made = _logs(30, size=100_000)
    server_log("llama-server", 9999)
    assert sum(one.stat().st_size for one in home.state("logs").glob("*.log")) <= 1 << 20
    assert made[-1].exists() and not made[0].exists()


def test_a_log_past_the_age_limit_goes(monkeypatch):
    monkeypatch.setenv(logs.DAYS_ENV, "1")
    made = _logs(4, age_s=lambda n: 5 * 86400 if n < 2 else 60)
    server_log("llama-server", 9999)
    assert [one.exists() for one in made] == [False, False, True, True]


def test_a_log_a_recorded_server_is_still_writing_to_is_kept_whatever_its_age(monkeypatch):
    from ml_stack.files import write_json
    from ml_stack.serve.leases import lease_file

    monkeypatch.setenv(logs.FILES_ENV, "3")
    made = _logs(10)
    write_json(lease_file(), {"9000": {"port": 9000, "pid": os.getpid(), "log": str(made[0])}})
    server_log("llama-server", 9999)
    assert made[0].exists()
    assert len(list(home.state("logs").glob("*.log"))) <= 4


def test_zero_means_no_limit(monkeypatch):
    monkeypatch.setenv(logs.FILES_ENV, "0")
    monkeypatch.setenv(logs.MEGABYTES_ENV, "0")
    monkeypatch.setenv(logs.DAYS_ENV, "0")
    made = _logs(80, age_s=lambda n: 400 * 86400)
    assert logs.prune(home.state("logs")) == []
    assert all(one.exists() for one in made)


def test_a_limit_that_is_not_a_number_is_the_default(monkeypatch):
    monkeypatch.setenv(logs.FILES_ENV, "many")
    monkeypatch.setenv(logs.MEGABYTES_ENV, "-5")
    assert logs.limits()[0] == logs.FILES and logs.limits()[1] == 0


def test_the_directory_is_private():
    server_log("llama-server", 9999)
    assert home.state("logs").stat().st_mode & 0o077 == 0
