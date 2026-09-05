"""Output that reads as `print` did, and goes elsewhere when a caller asks."""

from __future__ import annotations

import pytest

from ml_stack import log


class TestTheConsole:
    def test_say_writes_a_line_to_stdout(self, capsys):
        log.say("two", "words")
        assert capsys.readouterr() == ("two words\n", "")

    def test_warn_writes_a_line_to_stderr(self, capsys):
        log.warn("that did not work")
        assert capsys.readouterr() == ("", "that did not work\n")

    def test_it_takes_the_separator_and_ending_print_takes(self, capsys):
        log.say("a", "b", sep="", end="")
        assert capsys.readouterr().out == "ab"

    def test_an_empty_call_is_a_blank_line(self, capsys):
        log.say()
        assert capsys.readouterr().out == "\n"

    def test_it_writes_what_print_would_have(self, capsys):
        rows = [("one", 2), (None, 3.5)]
        for row in rows:
            print(*row)
        expected = capsys.readouterr().out
        for row in rows:
            log.say(*row)
        assert capsys.readouterr().out == expected


class TestStopping:
    def test_die_warns_and_exits_with_one(self, capsys):
        with pytest.raises(SystemExit) as stopped:
            log.die("no model there")
        assert stopped.value.code == 1
        assert capsys.readouterr() == ("", "no model there\n")

    def test_die_takes_its_own_code(self):
        with pytest.raises(SystemExit) as stopped:
            log.die("nothing to do", code=2)
        assert stopped.value.code == 2

    def test_die_with_nothing_to_say_says_nothing(self, capsys):
        with pytest.raises(SystemExit):
            log.die(code=3)
        assert capsys.readouterr() == ("", "")


class TestRedirection:
    def test_a_callback_takes_both_streams_and_the_console_gets_neither(self, capsys):
        seen: list[tuple[str, str]] = []
        with log.to(lambda stream, text: seen.append((stream, text))):
            log.say("output")
            log.warn("trouble")
        assert seen == [("out", "output\n"), ("err", "trouble\n")]
        assert capsys.readouterr() == ("", "")

    def test_the_console_comes_back_after_the_block(self, capsys):
        with log.to(lambda stream, text: None):
            log.say("swallowed")
        log.say("heard")
        assert capsys.readouterr().out == "heard\n"

    def test_a_file_takes_everything_a_command_would_have_printed(self, tmp_path, capsys):
        where = tmp_path / "logs" / "daemon.log"
        with log.to_file(where):
            log.say("started")
            log.warn("port busy")
        assert where.read_text(encoding="utf-8") == "started\nport busy\n"
        assert capsys.readouterr() == ("", "")

    def test_a_second_run_appends_unless_asked_otherwise(self, tmp_path):
        where = tmp_path / "daemon.log"
        with log.to_file(where):
            log.say("first")
        with log.to_file(where):
            log.say("second")
        assert where.read_text(encoding="utf-8") == "first\nsecond\n"
        with log.to_file(where, append=False):
            log.say("only")
        assert where.read_text(encoding="utf-8") == "only\n"

    def test_die_reaches_the_listener_too(self, tmp_path):
        where = tmp_path / "daemon.log"
        with pytest.raises(SystemExit), log.to_file(where):
            log.die("stopping")
        assert where.read_text(encoding="utf-8") == "stopping\n"
