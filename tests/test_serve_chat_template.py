"""A model's chat template, with the guard that refuses Claude Code lifted."""

from __future__ import annotations

import pytest
from ml_stack.serve.chat_template import forgiving, needs_forgiving

GUARD = ("{%- for message in messages %}\n"
         "    {%- if message.role == \"system\" %}\n"
         "        {%- if not loop.first %}\n"
         "            {{- raise_exception('System message must be at the beginning.') }}\n"
         "        {%- endif %}\n"
         "    {%- endif %}\n"
         "{%- endfor %}")


def test_a_template_that_refuses_a_late_system_message_is_recognised():
    assert needs_forgiving(GUARD)


def test_a_template_without_the_guard_is_left_alone():
    assert not needs_forgiving("{{ messages }}")
    assert forgiving("{{ messages }}") == "{{ messages }}"


def test_forgiving_it_removes_the_refusal():
    assert not needs_forgiving(forgiving(GUARD))


def test_the_late_system_message_is_rendered_rather_than_dropped():
    jinja = pytest.importorskip("jinja2")
    env = jinja.Environment(loader=jinja.BaseLoader())
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    body = ("{%- for message in messages %}{%- set content = message.content %}"
            "{%- if message.role == \"system\" %}{%- if not loop.first %}"
            "{{- raise_exception('System message must be at the beginning.') }}"
            "{%- endif %}{%- endif %}{%- endfor %}")
    said = [{"role": "system", "content": "first"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "a reminder"}]
    with pytest.raises(RuntimeError, match="System message must be at the beginning"):
        env.from_string(body).render(messages=said)
    assert "a reminder" in env.from_string(forgiving(body)).render(messages=said)


def test_a_model_that_names_no_template_needs_none(tmp_path):
    from ml_stack.serve.chat_template import template_of, written_beside

    empty = tmp_path / "nothing.gguf"
    empty.write_bytes(b"not a gguf")
    assert template_of(empty) == ""
    assert written_beside(empty) is None
