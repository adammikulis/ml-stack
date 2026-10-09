"""The table of model folders, resolved for each operating system and environment."""

from __future__ import annotations

import json
import os
from pathlib import Path

from poolhouse.hub import places

HOME = Path("/h")
STATE = Path("/h/.poolhouse")


def where(system: str, **env: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for one in places.places(env=env, where=system, home=HOME, state=STATE,
                             volumes=Path("/nowhere")):
        out.setdefault(one.label, []).append(str(one.path))
    return out


def test_macos_folders():
    got = where("Darwin")
    assert got["llama.cpp"] == ["/h/Library/Caches/llama.cpp", "/h/.cache/llama.cpp"]
    assert got["gpt4all"] == ["/h/Library/Application Support/nomic.ai/GPT4All"]
    assert got["huggingface"] == ["/h/.cache/huggingface/hub"]
    assert got["lmstudio"] == ["/h/.lmstudio/models", "/h/.cache/lm-studio/models"]
    assert got["ollama"] == ["/h/.ollama/models"]
    assert got["poolhouse"] == ["/h/.poolhouse/models"]
    assert "/h/Library/Application Support/Jan/data/llamacpp/models" in got["jan"]


def test_linux_folders():
    got = where("Linux")
    assert got["llama.cpp"] == ["/h/.cache/llama.cpp", "/h/.cache/llama.cpp"][:1]
    assert got["gpt4all"] == ["/h/.local/share/nomic.ai/GPT4All"]
    assert got["ollama"] == ["/h/.ollama/models", "/usr/share/ollama/.ollama/models"]
    assert "/opt/models" in got["manual"]


def test_windows_folders():
    got = where("Windows", LOCALAPPDATA="C:/Users/x/AppData/Local", APPDATA="C:/R")
    assert got["gpt4all"] == ["C:/Users/x/AppData/Local/nomic.ai/GPT4All"]
    assert got["llama.cpp"][0] == "C:/Users/x/AppData/Local/llama.cpp"
    assert "C:/R/Jan/data/models" in got["jan"]


def test_xdg_cache_home_moves_the_linux_llama_and_hugging_face_caches():
    got = where("Linux", XDG_CACHE_HOME="/x")
    assert got["llama.cpp"][0] == "/x/llama.cpp"
    assert got["huggingface"] == ["/x/huggingface/hub"]


def test_the_hugging_face_variables_win_in_order():
    assert where("Linux", HF_HOME="/a", XDG_CACHE_HOME="/x")["huggingface"] == ["/a/hub"]
    both = where("Linux", HF_HOME="/a", HF_HUB_CACHE="/b")
    assert both["huggingface"] == ["/b"]


def test_cache_variables_replace_the_default_folders():
    got = where("Darwin", LLAMA_CACHE="/l", OLLAMA_MODELS="/o", MODELSCOPE_CACHE="/m")
    assert got["llama.cpp"] == ["/l"] and got["ollama"] == ["/o"]
    assert got["modelscope"] == ["/m/hub"]


def test_extra_folders_come_from_the_path_list():
    sep = os.pathsep
    got = where("Linux", POOLHOUSE_MODEL_PATHS=f"/one{sep}{sep}/two")
    assert got["extra"] == ["/one", "/two"]


def test_volumes_are_searched_only_when_asked(tmp_path):
    (tmp_path / "Disk").mkdir()
    (tmp_path / "file").write_text("")
    off = places.places(env={}, where="Darwin", home=HOME, state=STATE, volumes=tmp_path)
    on = places.places(env={places.VOLUMES_ENV: "1"}, where="Darwin", home=HOME,
                       state=STATE, volumes=tmp_path)
    assert not [p for p in off if p.label == "volume"]
    assert [str(p.path) for p in on if p.label == "volume"] == [
        str(tmp_path / "Disk" / "models"), str(tmp_path / "Disk" / "Models")]
    linux = places.places(env={places.VOLUMES_ENV: "1"}, where="Linux", home=HOME,
                          state=STATE, volumes=tmp_path)
    assert not [p for p in linux if p.label == "volume"]


def test_lm_studio_names_its_own_download_folder(tmp_path):
    (tmp_path / ".lmstudio").mkdir()
    (tmp_path / ".lmstudio" / "settings.json").write_text(
        json.dumps({"downloadsFolder": "/big/disk/lms"}))
    got = places.places(env={}, where="Linux", home=tmp_path, state=STATE)
    lm = [str(p.path) for p in got if p.label == "lmstudio"]
    assert lm[0] == "/big/disk/lms" and str(tmp_path / ".lmstudio/models") in lm


def test_a_folder_named_twice_is_listed_once():
    found = places.places(env={"HF_HUB_CACHE": "/h/.cache/huggingface/hub",
                               places.EXTRA_ENV: "/h/.cache/huggingface/hub"},
                          where="Linux", home=HOME, state=STATE)
    assert [str(p.path) for p in found].count("/h/.cache/huggingface/hub") == 1


def test_unverified_layouts_are_marked():
    marked = {p.label for p in places.places(env={}, where="Linux", home=HOME, state=STATE)
              if not p.verified}
    assert marked == {"jan", "kagglehub", "lmstudio", "gpt4all", "modelscope"}
