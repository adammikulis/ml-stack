"""What a run record identifies, and what it says when a run could not say it.

Every fixture here is invented. Nothing reads a real store, a real graph, or a real server.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ml_stack.bench.record import Measured, Spread, prompt_digest

SYSTEM = "Answer from the graph. Show only the entries the question is about."

TOOLS = [{"type": "function",
          "function": {"name": "look_up",
                       "description": "Search the graph for entries whose name matches.",
                       "parameters": {"type": "object",
                                      "properties": {"text": {"type": "string"}}}}}]


def _run(**over):
    """A kept run, as the store holds one."""
    server = {"model": "sundial-8b.bin", "binary": "/opt/builds/thornfell/llama-server",
              "context": 32768, "slots": 4, "cache_type": "q8_0", "host": "kiln",
              "commit": "abc1234", "sampling": {"temperature": 0.0},
              "graph": "0f0f0f0f", "finder": "words"}
    server.update(over.pop("server", {}))
    one = {"at": "2026-09-05T09:00:00", "label": "sundial-plain", "server": server,
           "asking": {"tight": True, "terse": False, "sampling": {"temperature": 0.0}},
           "prompts": prompt_digest(SYSTEM, TOOLS),
           "rows": [{"question": "who fires kilns?", "expected": ["person:iris"],
                     "shown": ["person:iris"], "unread_named": 1, "seconds": 2.0},
                    {"question": "who glazes?", "expected": ["person:otto"],
                     "shown": [], "unread_named": 0, "seconds": 3.0}]}
    one.update(over)
    return one


class TestWhatWasShown:
    """The digest of the system prompt and the tool schemas."""

    def test_one_character_of_the_system_prompt_is_a_different_run(self):
        """The point of the record: the answer cache hashes the prompt into its key and
        the bench did not, so two runs across a prompt edit were compared as one."""
        before = Measured.from_dict(_run(prompts=prompt_digest(SYSTEM, TOOLS)))
        after = Measured.from_dict(_run(prompts=prompt_digest(SYSTEM + ".", TOOLS)))

        assert before.prompts != after.prompts
        assert before.identity != after.identity
        assert before.fingerprint != after.fingerprint
        # and nothing else about them differs
        assert before.serving == after.serving
        assert before.ways == after.ways

    def test_an_edited_tool_description_is_a_different_run(self):
        """A tool's words change what the model does with it, so an edit to them misses."""
        edited = [{"type": "function",
                   "function": {**TOOLS[0]["function"],
                                "description": "Search the graph. Call this first."}}]
        assert prompt_digest(SYSTEM, TOOLS) != prompt_digest(SYSTEM, edited)

    def test_a_renamed_tool_is_a_different_run(self):
        renamed = [{"type": "function",
                    "function": {**TOOLS[0]["function"], "name": "search"}}]
        assert prompt_digest(SYSTEM, TOOLS) != prompt_digest(SYSTEM, renamed)

    def test_the_same_prompt_and_tools_digest_the_same_whatever_the_order(self):
        second = {"type": "function", "function": {"name": "show", "description": "light",
                                                   "parameters": {}}}
        assert prompt_digest(SYSTEM, [TOOLS[0], second]) \
            == prompt_digest(SYSTEM, [second, TOOLS[0]])

    def test_whitespace_in_the_system_prompt_is_not_a_different_run(self):
        assert prompt_digest(SYSTEM, TOOLS) == prompt_digest(SYSTEM.replace(" ", "  "), TOOLS)


class TestOneAccessorEach:

    def test_the_build_is_the_name_a_serve_command_takes(self, monkeypatch):
        import pathlib

        from ml_stack.serve import build as build_module

        monkeypatch.setattr(build_module, "named_dir", lambda: pathlib.Path("/opt/builds"))
        assert Measured.from_dict(_run()).build == "thornfell"

    def test_a_binary_that_is_no_named_build_names_none(self, monkeypatch):
        import pathlib

        from ml_stack.serve import build as build_module

        monkeypatch.setattr(build_module, "named_dir", lambda: pathlib.Path("/opt/builds"))
        one = _run(server={"binary": "/usr/local/bin/llama-server"})
        assert Measured.from_dict(one).build == ""

    def test_another_program_is_named_as_what_served_it(self):
        one = _run(server={"served_by": {"program": "ollama", "version": "0.33.3",
                                         "runtime": "mlx", "quant": "nvfp4"}})
        assert Measured.from_dict(one).build == "ollama 0.33.3 · mlx · nvfp4"

    def test_the_head_is_said_with_how_far_it_guessed(self):
        assert Measured.from_dict(_run()).head_said == "-"
        drafted = _run(server={"draft_model": "mtp-sundial.bin", "spec_draft_max": 4})
        assert Measured.from_dict(drafted).head_said == "mtp-sundial.bin@n4"
        assert Measured.from_dict(drafted).head == "mtp-sundial.bin"

    def test_what_was_made_up_is_per_question(self):
        """One total over two questions and over thirty-four is not one number; the table
        lists runs of both lengths side by side."""
        one = Measured.from_dict(_run())
        assert one.made_per_question == pytest.approx(0.5)
        assert one.made == "0.5"

    def test_a_run_from_before_the_count_says_so_rather_than_zero(self):
        one = _run(rows=[{"question": "q", "expected": ["a"], "shown": ["a"]}])
        assert Measured.from_dict(one).made_per_question is None
        assert Measured.from_dict(one).made == "-"


class TestARunKeptBeforeTheRecord:
    """Nothing here invents a value the run did not keep."""

    def test_a_run_with_no_asking_reads_back_and_says_it_has_none(self):
        one = _run()
        one.pop("asking")
        one.pop("prompts")
        got = Measured.from_dict(one)

        assert got.knows_asking is False
        assert got.asked is None
        assert got.knows_prompts is False
        assert got.prompts == ""
        assert got.label == "sundial-plain"
        assert got.model == "sundial-8b.bin"
        assert got.questions == 2

    def test_a_run_with_nothing_at_all_reads_back(self):
        got = Measured.from_dict({})
        assert (got.label, got.model, got.questions, got.made) == ("", "", 0, "-")

    def test_the_asking_comes_back_as_the_record_it_was_written_from(self):
        from ml_stack.graph.asking import Asking

        one = _run(asking={"tight": True, "terse": False, "batch": True, "rounds": 8,
                           "shortlist": 12, "sampling": {"temperature": 0.0}})
        got = Measured.from_dict(one).asked
        assert got == Asking(tight=True, terse=False, batch=True, rounds=8)


class TestTheStoreBoundary:

    def test_a_run_reads_back_as_what_went_in(self):
        one = _run()
        assert Measured.from_dict(one).to_dict() == one

    def test_an_old_run_reads_back_as_what_went_in(self):
        one = _run()
        one.pop("asking")
        one.pop("prompts")
        one["traced"] = 0
        assert Measured.from_dict(one).to_dict() == one

    def test_the_key_is_carried_but_is_not_part_of_the_record(self):
        got = Measured.from_dict({**_run(), "key": "bench:sundial-plain:20260905T090000"})
        assert got.key == "bench:sundial-plain:20260905T090000"
        assert "key" not in got.to_dict()


class TestSeveralSeeds:

    def test_a_metric_over_seeds_carries_its_mean_and_its_spread(self):
        got = Spread.over([2.0, 4.0, 6.0])
        assert got.mean == pytest.approx(4.0)
        assert got.std == pytest.approx(2.0)
        assert got.n == 3

    def test_one_seed_has_no_spread(self):
        got = Spread.over([7.5])
        assert (got.mean, got.std, got.n) == (7.5, 0.0, 1)

    def test_no_seed_at_all_is_not_a_zero_mean(self):
        assert Spread.over([]).n == 0

    def test_seeds_and_their_spread_survive_the_store(self):
        one = _run(seeds=[0, 1, 2],
                   metrics={"loss": {"mean": 4.0, "std": 2.0, "n": 3,
                                     "values": [2.0, 4.0, 6.0]}},
                   failures=["seed 3: RuntimeError: no card"], seconds=12.5)
        got = Measured.from_dict(one)

        assert got.seeds == (0, 1, 2)
        assert got.metrics["loss"].mean == pytest.approx(4.0)
        assert got.failures == ("seed 3: RuntimeError: no card",)
        assert got.to_dict() == one


class TestWhatTheRunRanOn:

    def test_a_dirty_tree_is_read_off_the_commit(self):
        assert Measured.from_dict(_run()).dirty is False
        one = Measured.from_dict(_run(server={"commit": "abc1234 (dirty)"}))
        assert one.dirty is True
        assert one.sha == "abc1234"

    def test_the_host_is_the_machine_that_measured_it(self):
        assert Measured.from_dict(_run()).host == "kiln"


class TestWhatAMeasuredRunKeeps:
    """Through the client that counts, into the store, and back."""

    class _Answering:
        def __init__(self) -> None:
            self.seen = []

        def chat(self, messages, tools=None, **_):
            self.seen.append((messages, tools))
            return SimpleNamespace(
                content="Iris fires the kilns.", tool_calls=(), thinking="",
                finish_reason="stop",
                raw={"usage": {"prompt_tokens": 10, "completion_tokens": 4}})

    def test_the_counting_client_digests_what_the_model_was_shown(self):
        from ml_stack.bench.measure import Counting

        counting = Counting(self._Answering())
        counting.chat([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": "who fires kilns?"}], tools=TOOLS)

        assert counting.prompts == prompt_digest(SYSTEM, TOOLS)
        # taken once, on the first call: the prompt does not change inside a question
        counting.chat([{"role": "system", "content": "something else"}], tools=[])
        assert counting.prompts == prompt_digest(SYSTEM, TOOLS)

    def test_a_saved_run_carries_the_digest_and_its_rows_do_not(self, tmp_path):
        from ml_stack.bench import Row, runs, save

        row = Row(label="sundial-plain", question="who fires kilns?",
                  expected=["person:iris"], shown=["person:iris"])
        row.prompts = prompt_digest(SYSTEM, TOOLS)
        store = tmp_path / "runs.ladybug"
        key = save(store, [row], held={"model": "sundial-8b.bin"})

        kept = next(r for r in runs(store) if r["key"] == key)
        assert kept["prompts"] == prompt_digest(SYSTEM, TOOLS)
        assert "prompts" not in kept["rows"][0], "a digest of the run is not a field of a row"
        assert Measured.from_dict(kept).knows_prompts is True

    def test_two_runs_across_a_prompt_edit_are_two_runs_in_the_store(self, tmp_path):
        from ml_stack.bench import Row, runs, save

        store = tmp_path / "runs.ladybug"
        for system in (SYSTEM, SYSTEM + "."):
            row = Row(label="sundial-plain", question="q", expected=["a"], shown=["a"])
            row.prompts = prompt_digest(system, TOOLS)
            save(store, [row], held={"model": "sundial-8b.bin"})

        kept = [Measured.from_dict(r) for r in runs(store)]
        assert len({one.fingerprint for one in kept}) == 2, \
            "an edited prompt is a different measurement, whatever the label says"
