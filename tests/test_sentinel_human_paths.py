"""Protected state paths remain denied across native and serialized tool arguments."""

import pytest

from poolhouse.sentinel import human

pytestmark = pytest.mark.redteam


@pytest.mark.parametrize("spelling", ["native", "posix", "windows", "nested"])
def test_protected_unicode_token_directory_denied_for_every_argument_spelling(tmp_path, monkeypatch, spelling):
    directory = tmp_path / "private-\u03bb" / "tokens"
    monkeypatch.setattr(human, "_PROTECTED", set())
    human.protect(directory)
    path = {"native": str(directory), "posix": directory.as_posix(),
            "windows": directory.as_posix().replace("/", "\\")}
    arguments = {"path": path.get(spelling, str(directory)) + "/lead"}
    if spelling == "nested":
        arguments = {"value": {"paths": [str(directory)]}}
    assert human.agent_may("workspace_tasks", arguments)
    assert human.agent_may("Read", {"path": str(directory.parent / "public")}) == ""


@pytest.mark.parametrize("separator", ["/", "\\"])
def test_sentinel_state_and_forbidden_source_remain_denied(tmp_path, monkeypatch, separator):
    directory = tmp_path / "sentinel"
    monkeypatch.setattr(human.home, "state", lambda name: directory)
    assert human.agent_may("Read", {"path": directory.as_posix().replace("/", separator) + "/held"})
    assert human.agent_may("Bash", {"command": "cat poolhouse" + separator + "sentinel" + separator + "cli.py"})
