"""The tools `poolhouse-chat` offers: the registry, the lookups and the worked examples.

The model is a `ScriptedModel`, the commands are a fake registry that records what it was
called with, the person is a string on stdin. Nothing here serves a model, touches a port or
reads ``~/.poolhouse``.
"""

from __future__ import annotations

import io

import pytest

from poolhouse import do, mcp
from poolhouse.testing import ScriptedModel
from poolhouse.testing.fakes import reply_from


def call(name, **args):
    return (name, args)


class Scripted(ScriptedModel):
    """`ScriptedModel` whose script may also hold words (an answer with no call) and
    callables asked with ``(messages, tools)``; a call is issued only when its tool is on
    offer."""

    def chat(self, messages, *, tools=None, **extra):
        self.seen.append(list(messages))
        if not self.script:
            return reply_from(self.answer, messages, tools)
        entry = self.script[0]
        offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
        if isinstance(entry, tuple) and entry[0] not in offered:
            return reply_from(self.answer, messages, tools)
        return reply_from(self.script.pop(0), messages, tools)


def recording(name: str, seen: list, *, answer=None, **params):
    """A registry entry that records its arguments and answers ``answer``."""
    def fn(**args):
        seen.append((name, dict(args)))
        return answer if answer is not None else {"log": f"/tmp/{name}.log", "pid": 7}

    fn.__annotations__ = {**{k: v for k, v in params.items()}, "return": dict}
    fn.__signature__ = _signature(params)
    return mcp.Tool(name, f"The {name} command.", fn)


def _signature(params):
    import inspect

    return inspect.Signature([inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY,
                                                annotation=v) for k, v in params.items()])


def registry(seen: list) -> list[mcp.Tool]:
    """A fake ``mcp.TOOLS``: every bench tool the loop offers, none of which measures."""
    return [recording("bench_run", seen, argv=list[str]),
            recording("bench_status", seen, answer={"text": "nothing is measuring"}),
            recording("bench_compare", seen, args=list[str]),
            recording("bench_animate", seen, args=list[str]),
            recording("bench_standard", seen, args=list[str]),
            recording("bench_speed", seen, args=list[str]),
            recording("serve_status", seen, answer=[])]


def tools_over(seen: list, *, files=(), fetch=None):
    return do.command_tools(registry(seen), files=list(files), fetch=fetch)


# -- the tools -------------------------------------------------------------------------

def test_command_tools_carry_every_mcp_tool_and_the_bench_subcommands_that_follow_the_cli():
    offered = {s["function"]["name"] for s, _ in do.command_tools()}
    assert {t.name for t in mcp.TOOLS} <= offered
    assert {"bench_compare", "bench_animate", "bench_standard", "bench_speed"} <= offered
    assert {"jobs_status", "jobs_wait", "models_on_disk", "ollama_models"} <= offered
    assert {"ask_user", "plan", "done"}.isdisjoint(offered), "the loop's own are added by it"


def test_every_offered_tool_carries_a_worked_example():
    for schema, _ in [*do.command_tools(), *do.own_tools(stdin=io.StringIO(),
                                                          stdout=io.StringIO())]:
        text = schema["function"]["description"]
        assert "Example" in text and "->" in text, schema["function"]["name"]


def test_every_bench_example_parses_as_the_bench_command_line(capsys):
    import ast
    import re

    from poolhouse.bench.run import _parser

    found = []
    for name, pairs in do.EXAMPLES.items():
        for _asked, said in pairs:
            for sub, args in re.findall(r"bench_(\w+)\(args=(\[[^\]]*\])\)", said):
                found.append((name, sub, ast.literal_eval(args)))
    assert found, "the examples name bench subcommands with their arguments"
    for name, sub, args in found:
        try:
            _parser().parse_args([sub, *args])
        except SystemExit:
            pytest.fail(f"{name}: `poolhouse-bench {sub} {' '.join(args)}` -- "
                        + capsys.readouterr().err.strip().splitlines()[-1])


def test_a_bench_subcommand_not_in_mcp_is_registered_once_and_calls_the_cli(monkeypatch):
    ran = []
    monkeypatch.setattr(do, "bench_cli", lambda sub, args, detach: ran.append((sub, args, detach))
                        or {"sub": sub})
    tools = dict((s["function"]["name"], fn) for s, fn in do.command_tools())
    assert tools["bench_animate"](args=["--out", "x.mp4"]) == {"sub": "animate"}
    assert tools["bench_standard"](args=["quince-2b.gguf"]) == {"sub": "standard"}
    assert ran == [("animate", ["--out", "x.mp4"], False), ("standard", ["quince-2b.gguf"], True)]
    names = [s["function"]["name"] for s, _ in do.command_tools()]
    assert len(names) == len(set(names))


def test_models_on_disk_lists_the_weights_with_the_head_and_projector_beside_them(tmp_path):
    models_dir = tmp_path / "models"
    (models_dir / "UD-Q4_K_XL").mkdir(parents=True)
    weights = [models_dir / "UD-Q4_K_XL" / f"Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf"
               for i in (1, 2, 3, 4)]
    head = models_dir / "UD-Q4_K_XL" / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    proj = models_dir / "UD-Q4_K_XL" / "mmproj-F16.gguf"
    other = models_dir / "quince-2b.gguf"
    for p in (*weights, head, proj, other):
        p.write_bytes(b"gguf")
    found = do.models_on_disk("qwen3.8-flash-next", files=[*weights, head, proj, other])
    assert [f["model"] for f in found] == ["Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"]
    assert found[0]["draft"] == "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
    assert found[0]["mmproj"] == "mmproj-F16.gguf"
    assert found[0]["shards"] == 4 and found[0]["path"] == str(weights[0])
    assert found[0]["backend"] == "llama.cpp"
    assert [f["model"] for f in do.models_on_disk("quince", files=[*weights, head, proj, other])] \
        == ["quince-2b.gguf"]
    assert do.models_on_disk("larch", files=[other]) == []


def test_a_head_in_a_sibling_folder_of_the_repository_counts_as_beside(tmp_path):
    """A Hub snapshot keeps the head under MTP/ beside the quant folders. Driven
    2026-09-05: poolhouse-do said no head was on disk and planned to fetch one."""
    snap = tmp_path / "snapshots" / "abc"
    (snap / "UD-Q4_K_XL").mkdir(parents=True)
    (snap / "MTP").mkdir()
    weights = snap / "UD-Q4_K_XL" / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
    head = snap / "MTP" / "mtp-Qwen3.8-Flash-Next-BF16.gguf"
    for p in (weights, head):
        p.write_bytes(b"gguf")
    [found] = do.models_on_disk("flash-next", files=[weights, head])
    assert found["draft"] == "mtp-Qwen3.8-Flash-Next-BF16.gguf"


def test_ollama_models_reads_the_tags_and_the_show_of_each_match():
    asked = []

    def fetch(path, payload=None):
        asked.append((path, payload))
        if path == "/api/tags":
            return {"models": [
                {"name": "qwen3.8-flash-next:125b-mlx", "size": 70_000_000_000,
                 "details": {"format": "safetensors", "family": "qwen3next",
                             "parameter_size": "125B", "quantization_level": "nvfp4"}},
                {"name": "quince:2b", "size": 1_500_000_000,
                 "details": {"format": "gguf", "family": "quince", "parameter_size": "2B",
                             "quantization_level": "Q4_K_M"}}]}
        if path == "/api/show":
            return {"details": {"format": "safetensors", "quantization_level": "nvfp4",
                                "family": "qwen3next", "parameter_size": "125B"},
                    "model_info": {"general.architecture": "qwen3next"}}
        raise AssertionError(path)

    found = do.ollama_models("qwen3.8 flash next", fetch=fetch)
    assert found == [{"name": "qwen3.8-flash-next:125b-mlx", "backend": "ollama",
                      "bytes": 70_000_000_000, "format": "safetensors",
                      "quantization": "nvfp4", "family": "qwen3next", "parameters": "125B",
                      "architecture": "qwen3next"}]
    assert asked == [("/api/tags", None), ("/api/show", {"model": "qwen3.8-flash-next:125b-mlx"})]


def test_ollama_that_is_not_running_is_an_empty_list_that_says_so():
    def fetch(path, payload=None):
        raise OSError("connection refused")

    found = do.ollama_models("quince", fetch=fetch)
    assert len(found) == 1 and "not answering" in found[0]["error"]
