"""When a machine will take work.

The overnight window is the case worth testing hardest: "22:00-06:00" is one span
through the night, and treating it as start-before-end makes it an empty window that
silently protects nothing.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest
from ml_stack.fleet.availability import Availability, parse_window

MON, SAT = "2026-08-24", "2026-08-22"


def at(day: str, hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{day} {hhmm}")


class TestWindows:
    def test_a_working_day_is_blocked_and_the_evening_is_not(self):
        a = Availability.from_specs(busy=["mon-fri 09:00-17:00"])
        assert not a.open_at(at(MON, "10:00"))
        assert a.open_at(at(MON, "08:30"))
        assert a.open_at(at(MON, "17:00")), "the end of a window is exclusive"

    def test_the_weekend_is_not_a_working_day(self):
        a = Availability.from_specs(busy=["mon-fri 09:00-17:00"])
        assert a.open_at(at(SAT, "10:00"))

    @pytest.mark.parametrize("day, hhmm, busy", [
        (MON, "23:30", True), ("2026-08-25", "03:00", True),
        ("2026-08-25", "07:00", False), (MON, "21:00", False),
    ])
    def test_a_window_that_wraps_past_midnight_is_one_span_not_none(self, day, hhmm, busy):
        a = Availability.from_specs(busy=["22:00-06:00"])
        assert a.open_at(at(day, hhmm)) is not busy

    def test_an_allowed_window_carves_a_hole_in_a_busy_one(self):
        a = Availability.from_specs(busy=["mon-fri 09:00-17:00"],
                                    free=["mon-fri 12:00-13:00"])
        assert not a.open_at(at(MON, "11:00"))
        assert a.open_at(at(MON, "12:30")), "lunch should be usable"

    def test_it_says_when_work_resumes(self):
        a = Availability.from_specs(busy=["mon-fri 09:00-17:00"])
        assert a.opens_at(at(MON, "10:00")) == at(MON, "17:00")

    def test_a_machine_with_no_windows_works_at_any_hour(self):
        assert Availability().open_at(at(MON, "03:00"))

    @pytest.mark.parametrize("bad", ["nonsense", "25:00-26:00", "funday 09:00-17:00"])
    def test_an_unparseable_window_is_refused_rather_than_ignored(self, bad):
        with pytest.raises(ValueError):
            parse_window(bad)

    def test_a_window_survives_being_saved_and_read_back(self, tmp_path):
        """The prose form is not the storage form. Conflating them meant a saved
        schedule did not parse on the next start, so a machine silently lost the
        working hours it was supposed to be protecting."""
        a = Availability.from_specs(busy=["mon-fri 09:00-17:00", "22:00-06:00"])
        a.save(tmp_path / "avail.json")
        back = Availability.load(tmp_path / "avail.json")

        assert [w.spec() for w in back.windows] == [w.spec() for w in a.windows]
        assert not back.open_at(at(MON, "10:00"))

    def test_an_unreadable_schedule_leaves_the_box_available(self, tmp_path):
        """Available is the safe failure: a run at the wrong time gets noticed far
        sooner than a machine that quietly never accepts anything."""
        bad = tmp_path / "avail.json"
        bad.write_text("{ not json")
        assert Availability.load(bad).open_at(at(MON, "10:00"))


class TestPause:
    def test_pausing_stops_work_immediately(self):
        a = Availability()
        a.pause(reason="gaming")
        allowed, why = a.may_start()
        assert not allowed and "gaming" in why

    def test_an_indefinite_pause_does_not_promise_a_resume_time(self):
        """Showing "resumes at 17:00" for a machine paused indefinitely is a promise
        the box has no way to keep."""
        a = Availability.from_specs(busy=["mon-fri 09:00-17:00"])
        a.pause()
        assert a.public(at(MON, "10:00"))["next_open"] is None

    def test_a_timed_pause_lifts_itself(self):
        a = Availability()
        a.pause(minutes=1)
        assert a.paused
        a.paused_until = 0.0
        assert not a.paused
        assert a.may_start()[0]

    def test_resuming_takes_work_again(self):
        a = Availability()
        a.pause(reason="gaming")
        a.resume()
        assert a.may_start()[0]

    def test_a_pause_survives_a_restart(self, tmp_path):
        """A pause that quietly lifted on reboot would hand the GPU back at the worst
        possible moment."""
        a = Availability()
        a.pause(reason="gaming")
        a.save(tmp_path / "avail.json")

        back = Availability.load(tmp_path / "avail.json")
        assert back.paused
        assert back.paused_until == math.inf
        assert "gaming" in back.may_start()[1]

    def test_a_pause_beats_an_otherwise_free_schedule(self):
        a = Availability.from_specs(busy=["mon-fri 09:00-17:00"])
        a.pause(reason="rendering")
        assert not a.may_start(when=at(SAT, "10:00"))[0]


class TestReservations:
    def test_a_peer_can_hold_a_box(self):
        a = Availability()
        a.reserve("gpubox", 60)
        assert not a.may_start("someone else")[0]
        assert a.may_start("gpubox")[0]

    def test_two_peers_cannot_hold_the_same_box(self):
        a = Availability()
        a.reserve("gpubox", 60)
        with pytest.raises(PermissionError, match="gpubox"):
            a.reserve("other", 60)

    def test_the_holder_can_extend_its_own_hold(self):
        a = Availability()
        a.reserve("gpubox", 10)
        assert a.reserve("gpubox", 60).until > 0

    def test_a_hold_expires(self):
        a = Availability()
        a.reserve("gpubox", 60)
        a.reservation = a.reservation.__class__("gpubox", 0.0)
        assert a.may_start("anyone")[0]

    def test_a_hold_cannot_take_a_machine_out_of_the_fleet_forever(self):
        """A reservation with no ceiling is a way to lose a machine by accident."""
        import time
        a = Availability(max_hold_s=60)
        held = a.reserve("gpubox", 10 ** 9)
        assert held.until <= time.time() + a.max_hold_s + 1

    def test_only_the_holder_can_release(self):
        a = Availability()
        a.reserve("gpubox", 60)
        assert not a.release("someone else")
        assert a.release("gpubox")
