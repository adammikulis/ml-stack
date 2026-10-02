"""The join-request state machine and the limits around it, on a real file and a movable clock."""

import multiprocessing

import pytest
from onboard_support import Clock, Recorder, info, requests

from ml_stack.fleet.onboard.requests import Limits, Refused, Requests, State, clean

FP1, FP2, FP3 = "1" * 64, "2" * 64, "3" * 64


def make(tmp_path, **limits):
    rec, clock = Recorder(), Clock()
    return requests(tmp_path, rec, clock, Limits(**limits) if limits else None), rec, clock


def test_a_request_waits_then_is_accepted_with_a_code_then_pairs(tmp_path):
    rq, rec, _ = make(tmp_path)
    r = rq.submit(info(FP1), "10.0.0.5")
    assert r.state is State.PENDING and r.code == ""
    got = rq.accept(r.id)
    assert got.state is State.ACCEPTED and len(got.code) == 6 and got.code.isdigit()
    assert "code" not in got.public()
    device = rq.paired(r.id, shared_cluster_key=True)
    assert device.fingerprint == FP1 and rq.get(r.id).state is State.PAIRED
    assert rq.get(r.id).code == ""
    assert rec.kinds() == ["onboard.request.received", "onboard.request.accepted",
                           "onboard.pair.succeeded"]


def test_unanswered_requests_expire_and_the_code_expires_sooner(tmp_path):
    rq, rec, clock = make(tmp_path)
    r = rq.submit(info(FP1), "10.0.0.5")
    clock.advance(301)
    assert rq.get(r.id).state is State.EXPIRED
    with pytest.raises(Refused):
        rq.accept(r.id)
    r2 = rq.submit(info(FP2), "10.0.0.6")
    rq.accept(r2.id)
    clock.advance(121)
    assert rq.get(r2.id).state is State.EXPIRED and rq.get(r2.id).code == ""
    with pytest.raises(Refused):
        rq.take_attempt(r2.id)
    assert rec.kinds().count("onboard.request.expired") == 2


def test_accepting_twice_does_not_mint_a_second_code(tmp_path):
    rq, _, _ = make(tmp_path)
    r = rq.submit(info(FP1), "10.0.0.5")
    code = rq.accept(r.id).code
    with pytest.raises(Refused) as why:
        rq.accept(r.id)
    assert why.value.status == 409
    assert rq.get(r.id).code == code


def test_one_active_request_per_device_and_per_address(tmp_path):
    rq, rec, _ = make(tmp_path)
    rq.submit(info(FP1), "10.0.0.5")
    with pytest.raises(Refused) as same_device:
        rq.submit(info(FP1, nonce="cd" * 16), "10.0.0.9")
    with pytest.raises(Refused) as same_address:
        rq.submit(info(FP2), "10.0.0.5")
    assert same_device.value.status == same_address.value.status == 409
    assert len(rq.pending()) == 1
    assert rec.kinds().count("onboard.request.refused") == 2


def test_an_address_that_keeps_asking_is_cut_off_inside_the_window(tmp_path):
    rq, _, clock = make(tmp_path, per_address=3)
    for i in range(3):
        r = rq.submit(info(f"{i}" * 64), "10.0.0.5")
        rq.decline(r.id)
        clock.advance(61)           # past the decline wait, still inside the window
    with pytest.raises(Refused) as why:
        rq.submit(info("9" * 64), "10.0.0.5")
    assert why.value.status == 429
    clock.advance(700)
    rq.submit(info("9" * 64), "10.0.0.5")


def test_the_number_of_waiting_requests_is_capped(tmp_path):
    rq, _, _ = make(tmp_path, max_pending=3)
    for i in range(3):
        rq.submit(info(f"{i}" * 64), f"10.0.0.{i}")
    with pytest.raises(Refused) as why:
        rq.submit(info("9" * 64), "10.0.0.99")
    assert why.value.status == 503


def test_a_declined_device_waits_and_three_declines_block_it_for_an_hour(tmp_path):
    rq, _, clock = make(tmp_path)
    r = rq.submit(info(FP1), "10.0.0.5")
    rq.decline(r.id)
    with pytest.raises(Refused) as soon:
        rq.submit(info(FP1, nonce="cd" * 16), "10.0.0.5")
    assert soon.value.status == 429
    clock.advance(61)
    for _ in range(2):
        r = rq.submit(info(FP1, nonce="cd" * 16), "10.0.0.5")
        rq.decline(r.id)
        clock.advance(61)
    with pytest.raises(Refused) as blocked:         # third decline: now an hour, not a minute
        rq.submit(info(FP1, nonce="ef" * 16), "10.0.0.5")
    assert blocked.value.status == 429
    clock.advance(3601)
    rq.submit(info(FP1, nonce="ef" * 16), "10.0.0.5")


def test_three_wrong_codes_close_the_request_and_the_right_code_no_longer_helps(tmp_path):
    rq, rec, _ = make(tmp_path)
    r = rq.submit(info(FP1), "10.0.0.5")
    code = rq.accept(r.id).code
    left = []
    for _ in range(3):
        rq.take_attempt(r.id)
        left.append(rq.wrong(r.id))
    assert left == [2, 1, 0]
    assert rq.get(r.id).state is State.FAILED
    with pytest.raises(Refused):
        rq.take_attempt(r.id)
    assert rq.get(r.id).code == "" and code not in str(rq.get(r.id).public())
    assert rec.of("onboard.pair.locked")[0].severity == "critical"
    with pytest.raises(Refused) as again:           # and the device waits before asking again
        rq.submit(info(FP1, nonce="cd" * 16), "10.0.0.5")
    assert again.value.status == 429


def test_a_revoked_device_cannot_ask_again(tmp_path):
    rq, rec, _ = make(tmp_path)
    r = rq.submit(info(FP1), "10.0.0.5")
    rq.accept(r.id)
    rq.paired(r.id, shared_cluster_key=True)
    gone = rq.devices.revoke("kitchen-pi")
    assert gone.status == "revoked" and gone.shared_cluster_key
    with pytest.raises(Refused) as why:
        rq.submit(info(FP1, nonce="cd" * 16), "10.0.0.5")
    assert why.value.status == 403
    assert "onboard.revoked" in rec.kinds()
    assert rq.devices.allow_again(FP1)
    rq.submit(info(FP1, nonce="cd" * 16), "10.0.0.5")


def test_revoke_by_fingerprint_prefix_and_ambiguity(tmp_path):
    rq, _, _ = make(tmp_path)
    for fp, addr in ((FP1, "10.0.0.5"), (FP2, "10.0.0.6")):
        r = rq.submit(info(fp), addr)
        rq.accept(r.id)
        rq.paired(r.id, shared_cluster_key=False)
    with pytest.raises(ValueError):
        rq.devices.revoke("kitchen-pi")             # two devices share that name
    with pytest.raises(KeyError):
        rq.devices.revoke("ffff")
    assert rq.devices.revoke("1111").fingerprint == FP1
    assert [d.status for d in rq.devices.all()] == ["revoked", "active"]


@pytest.mark.parametrize("fp,nonce", [("short", "ab" * 16), ("G" * 64, "ab" * 16),
                                      ("1" * 64, ""), ("1" * 64, "zz" * 16)])
def test_malformed_identity_is_refused(tmp_path, fp, nonce):
    rq, _, _ = make(tmp_path)
    with pytest.raises(Refused) as why:
        rq.submit({"name": "x", "fingerprint": fp, "nonce": nonce}, "10.0.0.5")
    assert why.value.status == 400


def test_text_from_a_stranger_becomes_one_short_printable_line(tmp_path):
    rq, _, _ = make(tmp_path)
    evil = "pi\n\x1b[31mACCEPT\x1b[0m‮gnp.exe" + "A" * 300
    r = rq.submit(info(FP1, name=evil, hostname=evil, model=evil), "10.0.0.5")
    for text in (r.name, r.hostname, r.model):
        assert "\n" not in text and "\x1b" not in text and "‮" not in text
        assert len(text) <= 64
    assert clean(None) == "" and clean("  a \t b  ") == "a b"


def test_state_survives_a_second_process_on_the_same_file(tmp_path):
    rq = Requests(tmp_path / "requests.json", bus=Recorder().bus)   # real clock: two processes
    r = rq.submit(info(FP1), "10.0.0.5")
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    p = ctx.Process(target=_accept_elsewhere, args=(str(tmp_path), r.id, out))
    p.start()
    code = out.get(timeout=60)
    p.join(30)
    seen = Requests(tmp_path / "requests.json").get(r.id)
    assert seen.state is State.ACCEPTED and seen.code == code


def _accept_elsewhere(root, request_id, out):
    from pathlib import Path

    from ml_stack.fleet.onboard.requests import Requests
    out.put(Requests(Path(root) / "requests.json").accept(request_id).code)


def test_the_file_is_private_and_carries_a_schema_version(tmp_path):
    import json
    rq, _, _ = make(tmp_path)
    rq.submit(info(FP1), "10.0.0.5")
    path = tmp_path / "requests.json"
    assert json.loads(path.read_text())["schema_version"] == 1
    assert path.stat().st_mode & 0o077 == 0
