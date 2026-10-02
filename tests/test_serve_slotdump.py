"""Slot dumps and the guard file beside each: saved and restored against a llama-server on a
real socket, and refused when the server in front of them is not the one that wrote them."""

from __future__ import annotations

import json

import pytest

from ml_stack.serve import cli
from ml_stack.serve.slotdump import (
    SlotGuardRefused,
    current_guard,
    dump_name,
    guard_path,
    restore_all,
    restore_slot,
    save_all,
    save_slot,
)
from ml_stack.testing import FakeLlamaServer, Served

MODEL = "tinyfixture-4B-Q4_K_M.gguf"


@pytest.fixture
def server():
    started: list[FakeLlamaServer] = []

    def start(**served: object) -> FakeLlamaServer:
        fake = FakeLlamaServer(Served(model=MODEL, **served))
        started.append(fake)
        return fake

    yield start
    for fake in started:
        fake.close()


def test_the_guard_is_what_the_server_reports(server) -> None:
    fake = server(context=8192, slots=2, build_info="b1-x")
    assert current_guard(fake.base_url) == {
        "model": MODEL, "model_bytes": None, "ctx_size": 4096, "parallel": 2,
        "llama_server_version": "b1-x"}


def test_save_asks_the_server_and_writes_the_guard(server, tmp_path) -> None:
    fake = server(slots=2)
    dump = save_slot(fake.base_url, 1, "a.bin", directory=tmp_path)
    assert (dump.slot, dump.filename) == (1, "a.bin")
    assert fake.saved == [(1, "a.bin")]
    written = json.loads(guard_path(tmp_path, "a.bin").read_text())
    assert written["slot_id"] == 1 and written["parallel"] == 2 and written["model"] == MODEL


def test_a_failed_save_leaves_no_guard(server, tmp_path) -> None:
    fake = server()
    fake.refuse["/slots/0"] = 500
    with pytest.raises(Exception):  # noqa: B017 - ServerError
        save_slot(fake.base_url, 0, "a.bin", directory=tmp_path)
    assert not guard_path(tmp_path, "a.bin").exists()


def test_restore_after_save_on_the_same_layout(server, tmp_path) -> None:
    fake = server()
    save_slot(fake.base_url, 0, "a.bin", directory=tmp_path)
    restore_slot(fake.base_url, 0, "a.bin", directory=tmp_path)
    assert fake.restored == [(0, "a.bin")]


@pytest.mark.parametrize("changed", [
    {"context": 16384}, {"slots": 2}, {"build_info": "b2-other"}, {"model": "other.gguf"}])
def test_restore_refuses_a_different_layout(server, tmp_path, changed) -> None:
    first = server()
    save_slot(first.base_url, 0, "a.bin", directory=tmp_path)
    kwargs = {"model": MODEL, **changed}
    second = FakeLlamaServer(Served(**kwargs))
    try:
        with pytest.raises(SlotGuardRefused, match="different server layout"):
            restore_slot(second.base_url, 0, "a.bin", directory=tmp_path)
        assert second.restored == []
    finally:
        second.close()


def test_restore_refuses_a_dump_with_no_guard(server, tmp_path) -> None:
    fake = server()
    with pytest.raises(SlotGuardRefused, match="no guard file"):
        restore_slot(fake.base_url, 0, "a.bin", directory=tmp_path)
    assert fake.restored == []


def test_all_slots_round_trip_and_missing_dumps_are_passed_over(server, tmp_path) -> None:
    fake = server(slots=3)
    slots = {"main": 0, "side": 2}
    saved = save_all(fake.base_url, slots, directory=tmp_path)
    assert [d.filename for d in saved] == [dump_name(MODEL, 0, "main"),
                                           dump_name(MODEL, 2, "side")]
    assert dump_name(MODEL, 0, "main") == "tinyfixture-4B-Q4_K_M.slot0.main.bin"
    guard_path(tmp_path, saved[1].filename).unlink()
    (tmp_path / saved[0].filename).write_bytes(b"cache")
    restored = restore_all(fake.base_url, {**slots, "never": 1}, directory=tmp_path)
    assert [d.slot for d in restored] == [0]
    assert fake.restored == [(0, saved[0].filename)]


def test_the_cli_saves_and_refuses(server, tmp_path, capsys) -> None:
    fake = server(slots=2)
    base = ["slots", "--port", str(fake.port), "--dir", str(tmp_path)]
    assert cli.main([*base[:1], "save", *base[1:], "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [row["slot"] for row in out] == [0, 1]
    assert cli.main(["slots", "restore", *base[1:]]) == 0
    assert fake.restored == [(0, out[0]["filename"]), (1, out[1]["filename"])]
    guard_path(tmp_path, out[0]["filename"]).unlink()
    (tmp_path / out[0]["filename"]).write_bytes(b"cache")
    assert cli.main(["slots", "restore", *base[1:]]) == 2
