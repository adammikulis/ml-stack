"""Pinning the tool schemas a dataset was made for, and scoring a served model on the
held-out rows -- the model being a llama-server on a real socket that follows a script."""

from __future__ import annotations

import json

import pytest

from poolhouse.testing.tool_server import ToolCallingServer, Turn
from poolhouse.train.tools import (
    CHAT,
    SchemaDrift,
    evaluate,
    fingerprint,
    main,
    schema_hash,
    score,
    signatures,
    write_dataset,
)
from poolhouse.train.tools.drift import check

FIND = {"type": "function", "function": {
    "name": "find_recipe", "description": "Search the cookbook.",
    "parameters": {"type": "object", "required": ["words"],
                   "properties": {"words": {"type": "array", "items": {"type": "string"}}}}}}
SHELF = {"type": "function", "function": {
    "name": "list_shelf", "description": "Read out one kind of thing.",
    "parameters": {"type": "object", "required": ["kind"],
                   "properties": {"kind": {"type": "string"}}}}}


def row(tool: str, arguments: dict | None = None, *, side: str = "holdout") -> dict:
    assistant = ({"role": "assistant", "content": "Hello."} if tool == CHAT else
                 {"role": "assistant", "content": None, "tool_calls": [
                     {"id": "call_0", "type": "function",
                      "function": {"name": tool, "arguments": arguments or {}}}]})
    return {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": f"about {tool}"}, assistant],
            "tools": [FIND, SHELF], "tool": tool, "from": f"q-{tool}", "split": side}


def call(name: str, arguments: str) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def test_the_hash_ignores_key_order_and_sees_any_change() -> None:
    reordered = {"function": dict(reversed(list(FIND["function"].items()))), "type": "function"}
    assert schema_hash([FIND, SHELF]) == schema_hash([reordered, SHELF])
    assert schema_hash([FIND]) != schema_hash([FIND, SHELF])
    other = json.loads(json.dumps(FIND))
    other["function"]["description"] = "Search the pantry."
    assert schema_hash([other]) != schema_hash([FIND])
    assert signatures([FIND]) == {"find_recipe": {"required": ["words"],
                                                   "properties": ["words"]}}


def test_drift_names_the_tools_that_moved() -> None:
    pinned = fingerprint([FIND, SHELF])
    check(pinned, [SHELF, FIND])
    check({}, [FIND])
    wider = json.loads(json.dumps(FIND))
    wider["function"]["parameters"]["required"] = []
    with pytest.raises(SchemaDrift, match=r"removed: list_shelf; changed: find_recipe"):
        check(pinned, [wider])
    with pytest.raises(SchemaDrift, match=r"added: extra"):
        check(pinned, [FIND, SHELF, {**SHELF, "function": {**SHELF["function"],
                                                           "name": "extra"}}])
    reworded = json.loads(json.dumps(FIND))
    reworded["function"]["description"] = "Other."
    with pytest.raises(SchemaDrift, match="descriptions or nested schemas"):
        check(pinned, [reworded, SHELF])


def test_a_dataset_manifest_pins_its_schemas(tmp_path) -> None:
    write_dataset(tmp_path, [row("find_recipe", {"words": ["a"]}), row("list_shelf")],
                  base="b")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_hash"] == schema_hash([FIND, SHELF])
    assert set(manifest["signatures"]) == {"find_recipe", "list_shelf"}


def test_synth_refuses_data_made_for_other_tools(tmp_path, capsys) -> None:
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps([FIND, SHELF]))
    data = tmp_path / "out" / "data"
    write_dataset(data, [row("find_recipe", {"words": ["a"]}, side="train")], base="b",
                  tools=str(tools))
    args = ["--tools", str(tools), "--out", str(tmp_path / "out"), "--only", "synth"]
    assert main(args) == 0
    tools.write_text(json.dumps([FIND]))
    assert main(args) == 2
    assert "differ from the ones this was made for" in capsys.readouterr().err


def test_scoring_each_measure() -> None:
    rows = [row("find_recipe", {"words": ["a"]}), row("find_recipe", {"words": ["a"]}),
            row("list_shelf", {"kind": "tin"}), row("list_shelf", {"kind": "tin"}),
            row(CHAT)]
    calls = [call("find_recipe", '{"words": ["a"]}'),
             call("find_recipe", '{"words": '),
             call("find_recipe", '{"kind": "tin"}'),
             None,
             None]
    got = score(rows, calls)
    assert got["n"] == 5
    assert got["tool_name_accuracy"] == 0.6      # rows 0, 1 and the chat row
    assert got["valid_json_rate"] == 0.6         # rows 0, 2 and the chat row
    assert got["end_to_end"] == 0.4              # row 0 and the chat row
    assert got["arguments_match"] == 0.4
    assert got["per_tool"]["find_recipe"]["end_to_end"] == 0.5
    assert got["per_tool"]["list_shelf"]["n"] == 2
    assert got["per_tool"]["chat"]["end_to_end"] == 1.0
    wrong = score([row("find_recipe", {"words": ["a"]})], [call("find_recipe", "{}")])
    assert (wrong["required_arg_fill"], wrong["end_to_end"]) == (0.0, 0.0)
    with pytest.raises(ValueError, match="5 rows but 1"):
        score(rows, calls[:1])


def test_evaluate_asks_a_served_model_with_each_rows_tools() -> None:
    from poolhouse.client import Client

    rows = [row("find_recipe", {"words": ["a"]}), row(CHAT)]
    server = ToolCallingServer([Turn(calls=(("find_recipe", '{"words": ["a"]}'),)),
                                Turn(text=("Hello.",))])
    try:
        got = evaluate(Client(server.base_url), rows)
        sent = server.bodies
    finally:
        server.close()
    assert got["end_to_end"] == 1.0
    assert [m["role"] for m in sent[0]["messages"]] == ["system", "user"]
    assert [t["function"]["name"] for t in sent[0]["tools"]] == ["find_recipe", "list_shelf"]


def test_the_eval_command_scores_and_checks_drift(tmp_path, capsys) -> None:
    write_dataset(tmp_path, [row("find_recipe", {"words": ["a"]})], base="b")
    live = tmp_path / "live.json"
    live.write_text(json.dumps([FIND, SHELF]))
    server = ToolCallingServer([Turn(calls=(("find_recipe", '{"words": ["a"]}'),))])
    try:
        code = main(["eval", "--data", str(tmp_path), "--url", server.base_url,
                     "--tools", str(live), "--json"])
        assert code == 0
        assert json.loads(capsys.readouterr().out)["end_to_end"] == 1.0
        live.write_text(json.dumps([SHELF]))
        assert main(["eval", "--data", str(tmp_path), "--url", server.base_url,
                     "--tools", str(live)]) == 2
        assert len(server.bodies) == 1
    finally:
        server.close()
