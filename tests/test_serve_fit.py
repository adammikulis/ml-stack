"""How many people fit on one machine, from what llama.cpp said it allocated.

Everything here is a fake load log and a fake GGUF header written into ``tmp_path``. No
model is served, no GPU is touched, nothing under ``~/.ml-stack`` or ``~/.cache`` is read
or written -- the file the records live in is pointed at ``tmp_path`` by
``$MLSTACK_FIT_FILE`` and by replacing ``package_file``, in an autouse fixture, so a test
that forgets cannot reach the real one.

The log fixtures reproduce the *shape* of llama.cpp's own lines (llama-kv-cache.cpp's
constructor summary, llama-memory-recurrent.cpp's, the compute-buffer line
``sched_reserve`` prints per backend) over invented models. The numbers are round so the
arithmetic can be checked by hand rather than against the code that produced it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from ml_stack.serve import fit as fit_mod
from ml_stack.serve.fit import Fit, Measured, parse_load_log, parse_room, records, render

MIB = 1024 * 1024
GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _fit_files_in_tmp(tmp_path, monkeypatch):
    """Both halves of the source of truth, pointed at an empty directory.

    ``package_file`` is a function for exactly this reason; the local half moves with
    ``$MLSTACK_FIT_FILE``. Without both, a test would read the measurements this repository
    ships and a `--measure` test would write into it.
    """
    shipped = tmp_path / "ssot" / "fit.json"
    shipped.parent.mkdir(parents=True, exist_ok=True)
    shipped.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(fit_mod, "package_file", lambda: shipped)
    monkeypatch.setenv("MLSTACK_FIT_FILE", str(tmp_path / "local" / "fit.json"))
    monkeypatch.setattr(fit_mod, "writable_file", lambda: shipped)
    return shipped


# -- the logs ---------------------------------------------------------------------------
#
# `-lv 4` prefixes every line with a timestamp and a level, which is why nothing in the
# parser anchors at the start of a line.

DENSE_LOG = """\
0.00.100.000 I llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors \
from /models/thornfield-8B-Q4_K_M.gguf (version GGUF V3 (latest))
0.00.101.000 I cmn  common_param: build 1 (a1b2c3d) with Apple clang for arm64-apple-darwin
0.00.200.000 I llama_kv_cache:      Metal KV buffer size =  1024.00 MiB
0.00.200.100 I llama_kv_cache: size =  1024.00 MiB ( 32768 cells,  32 layers,  2/1 seqs), \
K (f16):   512.00 MiB, V (f16):   512.00 MiB
0.00.300.000 I sched_reserve:      Metal compute buffer size =   304.00 MiB
0.00.300.001 I sched_reserve:        CPU compute buffer size =    72.00 MiB
0.00.300.002 I sched_reserve: graph nodes  = 1030
"""

# A hybrid: one layer in four holds a token cache, the other 36 keep a state per sequence.
HYBRID_LOG = """\
0.00.100.000 I llama_model_loader: loaded meta data with 44 key-value pairs and 700 tensors \
from /models/marrowgate-A3B-UD-Q4_K_XL.gguf (version GGUF V3 (latest))
0.00.200.000 I llama_kv_cache:      Metal KV buffer size =   384.00 MiB
0.00.200.100 I llama_kv_cache: size =   384.00 MiB ( 32768 cells,  12 layers,  2/1 seqs), \
K (f16):   192.00 MiB, V (f16):   192.00 MiB
0.00.210.000 I llama_memory_recurrent:      Metal RS buffer size =    24.00 MiB
0.00.210.100 I llama_memory_recurrent: size =    24.00 MiB (     2 cells,  36 layers,  \
2 seqs  2 rs_seq), R (f32):    12.00 MiB, S (f32):    10.00 MiB, P (f32):     2.00 MiB
0.00.300.000 I sched_reserve:      Metal compute buffer size =   512.00 MiB
"""

# Sliding-window layers: llama_kv_cache_iswa builds two caches and so prints the summary
# twice -- the base one at the context, the SWA one at a few hundred cells per sequence.
ISWA_LOG = """\
0.00.100.000 I llama_model_loader: loaded meta data with 38 key-value pairs and 444 tensors \
from /models/quillhaven-E2B-it-qat-UD-Q4_K_XL.gguf (version GGUF V3 (latest))
0.00.190.000 I llama_kv_cache_iswa: creating non-SWA KV cache, size = 32768 cells
0.00.200.100 I llama_kv_cache: size =   256.00 MiB ( 32768 cells,   8 layers,  2/1 seqs), \
K (f16):   128.00 MiB, V (f16):   128.00 MiB
0.00.205.000 I llama_kv_cache_iswa: creating     SWA KV cache, size =  1536 cells
0.00.206.100 I llama_kv_cache: size =    48.00 MiB (  1536 cells,  24 layers,  2/1 seqs), \
K (f16):    24.00 MiB, V (f16):    24.00 MiB
0.00.300.000 I sched_reserve:      Metal compute buffer size =   128.00 MiB
"""

# The target, then a draft head. The head's lines are the same lines again, and reading
# them as the target's would report a 3 MiB cache for an 8B model.
DRAFTED_LOG = DENSE_LOG + """\
0.00.400.000 I llama_model_loader: loaded meta data with 24 key-value pairs and 30 tensors \
from /models/mtp-thornfield-8B.gguf (version GGUF V3 (latest))
0.00.450.100 I llama_kv_cache: size =     3.00 MiB ( 32768 cells,   1 layers,  2/1 seqs), \
K (f16):     1.50 MiB, V (f16):     1.50 MiB
0.00.460.000 I sched_reserve:      Metal compute buffer size =    16.00 MiB
"""


class TestParsingALoadLog:
    def test_a_dense_model_is_its_base_cache_divided_by_its_cells(self):
        """1024 MiB over 32768 cells is 32 KiB a token, and nothing fixed per sequence.

        Mutation: divide by the layers, or by the sequences.
        """
        got = parse_load_log(DENSE_LOG)
        assert got.measured
        assert got.per_token == 1024 * MIB // 32768 == 32768
        assert got.per_seq == 0
        assert got.compute == (304 + 72) * MIB
        assert got.cache_type == "f16"
        assert got.kv_layers == 32
        assert got.recurrent_layers == 0
        assert got.cells == 32768
        assert got.model_file == "thornfield-8B-Q4_K_M.gguf"
        assert got.build == "a1b2c3d"

    def test_a_recurrent_state_is_charged_per_sequence_and_not_per_token(self):
        """The recurrent cache does not grow with the context -- 24 MiB for two sequences
        is 12 MiB each however long they get, which is the whole reason the architecture
        is worth serving at a long context.

        Mutation: add the recurrent bytes into per_token, or forget to divide by the seqs.
        """
        got = parse_load_log(HYBRID_LOG)
        assert got.per_token == 384 * MIB // 32768 == 12288
        assert got.per_seq == 24 * MIB // 2 == 12 * MIB
        assert got.recurrent_layers == 36
        assert got.kv_layers == 12
        assert got.compute == 512 * MIB

    def test_a_sliding_window_cache_is_the_second_line_and_costs_per_sequence(self):
        """Two `llama_kv_cache: size` lines: the base at the context, the SWA one at 1536
        cells. Only the first is per-token; the second is a fixed cost per sequence.

        Mutation: sum both into per_token, which is what counting every layer as full
        attention does and what this whole module exists because of.
        """
        got = parse_load_log(ISWA_LOG)
        assert got.per_token == 256 * MIB // 32768 == 8192
        assert got.per_seq == 48 * MIB // 2 == 24 * MIB
        assert got.swa_cells == 1536
        assert got.kv_layers == 8 + 24
        assert got.cells == 32768

    def test_a_draft_head_loaded_after_the_target_is_not_read_as_the_target(self):
        """Both models print the same lines; the target loads first. Mutation: parse the
        whole log at once, and an 8B model reports a 3 MiB cache."""
        assert parse_load_log(DRAFTED_LOG).per_token == parse_load_log(DENSE_LOG).per_token
        assert parse_load_log(DRAFTED_LOG).compute == (304 + 72) * MIB

    def test_a_log_at_the_servers_own_verbosity_says_nothing_rather_than_raising(self):
        """Every line this reads is an LLAMA_LOG_INFO, which maps to LOG_LEVEL_TRACE, so
        verbosity 3 -- the server's default -- prints none of them. That is an empty
        measurement, not a crash and not a zero pretending to be a number."""
        quiet = ("0.12.434.595 I cmn  common_param: verbosity = 3\n"
                 "0.12.434.597 W srv  llama_server: CORS is set to allow all origins\n")
        got = parse_load_log(quiet)
        assert got == Measured()
        assert not got.measured

    def test_the_kv_buffer_line_is_not_mistaken_for_the_size_line(self):
        """`llama_kv_cache: Metal KV buffer size = 1024.00 MiB` is the allocation, not the
        summary, and it carries no cells to divide by. Mutation: match on 'buffer size'."""
        only_buffers = ("llama_kv_cache:      Metal KV buffer size =  1024.00 MiB\n"
                        "llama_memory_recurrent:  Metal RS buffer size =    24.00 MiB\n")
        assert parse_load_log(only_buffers) == Measured()

    def test_a_reserve_that_ran_twice_is_counted_once_per_backend(self):
        """`sched_reserve` prints one line per backend and can run again after the memory
        changes. Mutation: sum every line, and the compute buffers double."""
        twice = DENSE_LOG + (
            "0.00.500.000 I sched_reserve:      Metal compute buffer size =   304.00 MiB\n"
            "0.00.500.001 I sched_reserve:        CPU compute buffer size =    72.00 MiB\n")
        assert parse_load_log(twice).compute == (304 + 72) * MIB

    def test_a_quantised_cache_is_named_and_a_mixed_one_says_both(self):
        mixed = DENSE_LOG.replace("K (f16)", "K (q8_0)").replace("V (f16)", "V (q4_0)")
        assert parse_load_log(mixed).cache_type == "q8_0/q4_0"


class TestWhoFits:
    """The arithmetic, over numbers chosen so it can be done in one's head."""

    def one(self, **over) -> Fit:
        base = dict(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, draft=0,
                    room=24 * GIB, per_token=32768, per_seq=0, compute=GIB)
        return Fit(**{**base, **over})

    def test_free_is_the_room_less_the_weights_the_draft_and_the_compute(self):
        assert self.one().free() == 18 * GIB
        assert self.one(draft=2 * GIB).free() == 16 * GIB

    def test_a_model_that_does_not_fit_at_all_has_no_room_for_anyone(self):
        """Never a negative free(), and never a negative user count out of it."""
        cramped = self.one(room=2 * GIB)
        assert cramped.free() == 0
        assert cramped.users(4096) == 0
        assert cramped.longest(1) == 0

    def test_users_is_the_free_room_divided_by_what_one_user_costs(self):
        """32 KiB a token at 32768 tokens is 1 GiB each, so 18 GiB holds eighteen of them.

        Mutation: divide by per_token alone, ignoring the fixed per-sequence cost.
        """
        fit = self.one()
        assert fit.cost(32768) == GIB
        assert fit.users(32768) == 18
        assert fit.users(65536) == 9
        assert fit.users(4096) == 144

    def test_a_fixed_per_sequence_cost_is_charged_once_per_user_not_once_per_token(self):
        """A user at 32768 tokens costs 1 GiB of cache plus 2 GiB of window and state, so
        six fit where eighteen did. Mutation: multiply per_seq by the context."""
        fit = self.one(per_seq=2 * GIB)
        assert fit.cost(32768) == 3 * GIB
        assert fit.users(32768) == 6

    def test_longest_is_the_same_question_solved_for_the_context(self):
        """One user gets all 18 GiB: 18 GiB / 32 KiB is 589,824 tokens. Two get half each.

        Mutation: forget to subtract the per-sequence cost before dividing.
        """
        fit = self.one()
        assert fit.longest(1) == 18 * GIB // 32768 == 589824
        assert fit.longest(2) == 294912
        assert fit.users(fit.longest(2)) == 2

        withstate = self.one(per_seq=2 * GIB)
        assert withstate.longest(1) == (18 * GIB - 2 * GIB) // 32768

    def test_a_model_whose_cache_costs_nothing_per_token_has_no_longest_context(self):
        """per_token 0 is an unmeasured model, not a machine of infinite context."""
        assert self.one(per_token=0).longest(1) == 0

    def test_at_room_asks_the_same_measurement_about_a_different_machine(self):
        """A 24 GB card asked about from a 96 GB laptop. Mutation: re-measure, or ignore it."""
        bigger = self.one(room=96 * GIB)
        assert bigger.at_room(24 * GIB).users(32768) == self.one().users(32768)
        assert bigger.at_room(24 * GIB).room == 24 * GIB


class TestTheSourceOfTruth:
    def write(self, path: Path, rows: list[dict]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_a_local_measurement_replaces_the_shipped_one_with_the_same_key(self, tmp_path,
                                                                           _fit_files_in_tmp):
        """A machine that measured a model again means the newer number, not both.

        Mutation: concatenate the two files, and a listing shows one model twice with two
        different answers and no way to choose.
        """
        shipped = _fit_files_in_tmp
        self.write(shipped, [
            Fit(model="thornfield-8B-Q4_K_M.gguf", per_token=32768, room=GIB).as_dict(),
            Fit(model="marrowgate-A3B-UD-Q4_K_XL.gguf", per_token=12288).as_dict()])
        self.write(fit_mod.local_file(), [
            Fit(model="thornfield-8B-Q4_K_M.gguf", per_token=99, room=GIB).as_dict()])

        got = {row.model: row for row in records()}
        assert set(got) == {"thornfield-8B-Q4_K_M.gguf", "marrowgate-A3B-UD-Q4_K_XL.gguf"}
        assert got["thornfield-8B-Q4_K_M.gguf"].per_token == 99

    def test_the_same_model_at_two_cache_types_is_two_records(self, _fit_files_in_tmp):
        """The key is the model, the cache type and the speculation kind: q8_0 halves the
        per-token cost, and that is a different fact about the same weights."""
        self.write(_fit_files_in_tmp, [
            Fit(model="thornfield-8B-Q4_K_M.gguf", cache_type="f16", per_token=32768).as_dict(),
            Fit(model="thornfield-8B-Q4_K_M.gguf", cache_type="q8_0", per_token=17408).as_dict()])
        assert len(records()) == 2

    def test_records_can_be_asked_about_another_machines_room(self, _fit_files_in_tmp):
        self.write(_fit_files_in_tmp, [
            Fit(model="thornfield-8B-Q4_K_M.gguf", room=96 * GIB, per_token=32768).as_dict()])
        assert records(room=24 * GIB)[0].room == 24 * GIB

    def test_a_file_that_is_missing_or_broken_contributes_nothing(self, tmp_path,
                                                                  _fit_files_in_tmp):
        """There is no such thing as a half-measured model. Mutation: raise, and every
        command that reads the records dies on one bad byte."""
        _fit_files_in_tmp.write_text("{not json", encoding="utf-8")
        assert records() == []
        _fit_files_in_tmp.write_text('[{"no": "model"}, 7, "x"]', encoding="utf-8")
        assert records() == []

    def test_a_record_written_by_a_newer_version_still_loads(self, _fit_files_in_tmp):
        """An unknown key is ignored rather than raising, so a file written by a version
        that knows more does not make this one unusable."""
        row = Fit(model="thornfield-8B-Q4_K_M.gguf", per_token=8).as_dict()
        self.write(_fit_files_in_tmp, [{**row, "something_new": [1, 2, 3]}])
        assert records()[0].per_token == 8

    def test_adding_replaces_the_record_it_supersedes_and_says_where_it_went(
            self, _fit_files_in_tmp):
        first = Fit(model="thornfield-8B-Q4_K_M.gguf", per_token=1, measured_at="2026-09-01")
        second = Fit(model="thornfield-8B-Q4_K_M.gguf", per_token=2, measured_at="2026-09-02")
        assert fit_mod.add(first) == _fit_files_in_tmp
        fit_mod.add(second)
        assert [(r.per_token, r.measured_at) for r in records()] == [(2, "2026-09-02")]

    def test_the_shipped_file_is_a_list_this_can_read(self):
        """The real one, not a fixture: `src/ml_stack/data/fit.json` has to parse or the
        command that reads it is broken for everybody."""
        real = Path(fit_mod.__file__).resolve().parent.parent / "data" / "fit.json"
        assert isinstance(json.loads(real.read_text(encoding="utf-8")), list)


class TestMeasuring:
    def test_measuring_asks_for_the_verbosity_that_prints_the_lines(self):
        """`-lv 4` is not decoration: at the server's default of 3 the library's own INFO
        lines are filtered out and there is nothing to read. Mutation: drop the flag, and
        every measurement comes back empty."""
        from ml_stack.serve.backend import ServerSpec

        seen: list[ServerSpec] = []

        def fake(spec, **_):
            seen.append(spec)
            return DENSE_LOG

        got = fit_mod.measure(ServerSpec(model="thornfield-8B-Q4_K_M.gguf", context=32768),
                              serve=fake)
        assert seen[0].extra_args[-2:] == ("-lv", "4")
        assert got.per_token == 32768

    def test_a_spec_that_already_asked_for_a_verbosity_is_left_alone(self):
        from ml_stack.serve.backend import ServerSpec

        seen: list[ServerSpec] = []

        def fake(spec, **_):
            seen.append(spec)
            return DENSE_LOG

        fit_mod.measure(ServerSpec(model="m.gguf", extra_args=("-lv", "5")), serve=fake)
        assert seen[0].extra_args == ("-lv", "5")

    def test_a_measurement_becomes_a_record_carrying_what_it_was_measured_with(self):
        got = parse_load_log(ISWA_LOG)
        record = Fit.of(got, model="quillhaven-E2B-it-qat-UD-Q4_K_XL.gguf",
                        weights=3 * GIB, room=24 * GIB, context=32768, parallel=2,
                        when="2026-09-02T00:00:00Z")
        assert record.per_token == 8192
        assert record.per_seq == 24 * MIB
        assert record.cache_type == "f16"
        assert record.swa_cells == 1536
        assert record.key == ("quillhaven-E2B-it-qat-UD-Q4_K_XL.gguf", "f16", "")


class TestSayingIt:
    def rows(self) -> list[Fit]:
        return [Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, room=24 * GIB,
                    compute=GIB, per_token=32768, per_seq=0, kv_layers=32,
                    cache_type="f16", measured_at="2026-09-02T00:00:00Z")]

    def test_the_listing_names_the_model_the_two_numbers_and_who_fits(self):
        text = render(self.rows(), [32768])
        assert "thornfield-8B (Q4_K_M)" in text
        assert "per token" in text and "per sequence" in text
        assert "32,768" in text and " 18 " in f" {text} "

    def test_markdown_is_a_table(self):
        text = render(self.rows(), [32768], None, True)
        assert text.startswith("### thornfield-8B (Q4_K_M)")
        assert "| per user context | users that fit | each costs |" in text

    def test_a_different_room_changes_the_answer_and_not_the_measurement(self):
        text = render(self.rows(), [32768], 12 * GIB)
        assert "of 12.0G room, 6.0G is left" in text
        assert "one user, longest context: 196,608 tokens" in text

    def test_nothing_measured_says_how_to_measure_something(self):
        assert "--measure" in render([])


class TestReadingARoom:
    @pytest.mark.parametrize(("text", "expected"), [
        ("24G", 24 * GIB), ("24g", 24 * GIB), ("24GiB", 24 * GIB), ("24GB", 24 * GIB),
        ("24576M", 24 * GIB), ("0.5G", GIB // 2), ("25769803776", 24 * GIB),
    ])
    def test_a_room_is_read_the_way_a_person_writes_one(self, text, expected):
        assert parse_room(text) == expected

    @pytest.mark.parametrize("text", ["", "lots", "24 gigs", "-4G", "G"])
    def test_anything_else_is_refused_rather_than_guessed_at(self, text):
        """A room misread by a factor of 1024 answers the question confidently and
        wrongly. Mutation: fall back to 0, and every model fits nobody."""
        with pytest.raises(ValueError):
            parse_room(text)


class TestTheCommand:
    def run(self, argv: list[str]) -> int:
        from ml_stack.serve.cli import main

        return main(["fit", *argv])

    def test_it_measures_writes_and_then_reports(self, tmp_path, monkeypatch, capsys,
                                                 _fit_files_in_tmp):
        """The whole path with one seam: the serving. Everything else -- resolving the
        model, sizing the weights off disk, parsing, recording, rendering -- runs for real.

        Mutation: record the estimate instead of the parsed log, and the per-token number
        stops matching what the log said.
        """
        model = tmp_path / "thornfield-8B-Q4_K_M.gguf"
        model.write_bytes(b"\0" * (4 * MIB))
        monkeypatch.setattr("ml_stack.hub.room", lambda: 24 * GIB)
        monkeypatch.setattr(fit_mod, "_load_log", lambda spec, **_: DENSE_LOG)

        assert self.run([str(model), "--measure", "--context", "32768"]) == 0
        out, err = capsys.readouterr()
        assert "thornfield-8B (Q4_K_M)" in out
        assert str(_fit_files_in_tmp) in err

        [row] = records()
        assert row.per_token == 32768
        assert row.weights == 4 * MIB
        assert row.room == 24 * GIB
        assert row.context == 32768

    def test_measuring_serves_two_slots_so_a_per_sequence_cost_can_be_seen(
            self, tmp_path, monkeypatch, _fit_files_in_tmp):
        """Divided by one sequence, a fixed cost and a constant are the same number.
        Mutation: serve at --parallel, and `--parallel 1` measures nothing about sequences.
        """
        model = tmp_path / "quillhaven-E2B-it-qat-UD-Q4_K_XL.gguf"
        model.write_bytes(b"\0" * MIB)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 24 * GIB)
        seen: list[int] = []

        def fake(spec, **_):
            seen.append(spec.parallel)
            return ISWA_LOG

        monkeypatch.setattr(fit_mod, "_load_log", fake)
        assert self.run([str(model), "--measure", "--parallel", "1"]) == 0
        assert seen == [2]
        assert records()[0].per_seq == 24 * MIB

    def test_a_room_given_on_the_command_line_overrides_this_machines(
            self, monkeypatch, capsys, _fit_files_in_tmp):
        """A 24 GB card asked about from somewhere that is not one. Mutation: ignore
        --room, and every number describes the wrong machine."""
        _fit_files_in_tmp.write_text(json.dumps([
            Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, compute=GIB,
                per_token=32768, room=96 * GIB).as_dict()]), encoding="utf-8")
        monkeypatch.setattr("ml_stack.hub.room", lambda: 96 * GIB)

        assert self.run(["--room", "24G", "--per-user", "32768"]) == 0
        assert "18" in capsys.readouterr().out

    def test_a_room_that_cannot_be_read_is_refused_before_anything_is_printed(
            self, monkeypatch, capsys):
        monkeypatch.setattr("ml_stack.hub.room", lambda: 96 * GIB)
        assert self.run(["--room", "lots"]) == 2
        assert "lots" in capsys.readouterr().err

    def test_asking_about_a_model_nobody_measured_says_so_and_exits_nonzero(
            self, monkeypatch, capsys):
        monkeypatch.setattr("ml_stack.hub.room", lambda: 96 * GIB)
        assert self.run(["nevermeasured-3B.gguf"]) == 1
        assert "--measure" in capsys.readouterr().err

    def test_write_puts_the_markdown_where_it_was_asked_to(self, tmp_path, monkeypatch,
                                                          capsys, _fit_files_in_tmp):
        _fit_files_in_tmp.write_text(json.dumps([
            Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, compute=GIB,
                per_token=32768, room=96 * GIB).as_dict()]), encoding="utf-8")
        monkeypatch.setattr("ml_stack.hub.room", lambda: 96 * GIB)
        where = tmp_path / "fit.md"

        assert self.run(["--room", "24G", "--write", str(where)]) == 0
        written = where.read_text(encoding="utf-8")
        assert written.startswith("# What fits")
        assert "## This machine" in written
        assert "## A machine with 24.0G" in written
        capsys.readouterr()

    def test_parallel_says_the_longest_context_that_many_users_could_share(
            self, monkeypatch, capsys, _fit_files_in_tmp):
        _fit_files_in_tmp.write_text(json.dumps([
            Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, compute=GIB,
                per_token=32768, room=24 * GIB).as_dict()]), encoding="utf-8")
        monkeypatch.setattr("ml_stack.hub.room", lambda: 24 * GIB)

        assert self.run(["--parallel", "4"]) == 0
        assert "4 users fit at 147,456 tokens each" in capsys.readouterr().out

    def test_measure_without_a_model_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr("ml_stack.hub.room", lambda: 96 * GIB)
        assert self.run(["--measure"]) == 2
        assert "needs a model" in capsys.readouterr().err


class TestDrawingIt:
    """Two panels over the records, written to a real file in ``tmp_path``.

    matplotlib is driven through the Agg backend `fit._pyplot` selects, so nothing here
    opens a window or needs a display. The assertions are about what was drawn -- the
    legend's entries, the axis labels, the file on disk -- rather than about pixels: a
    chart test that compares images fails on a font and passes on a wrong axis.
    """

    def rows(self) -> list[Fit]:
        """Two invented models with opposite shapes -- the crossing the picture is for.

        thornfield is small and its cache is fat; marrowgate is three times the weights and
        its cache costs a quarter as much per token, because most of its layers keep a state
        rather than a history.
        """
        return [
            Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, room=110 * GIB,
                compute=GIB, per_token=32768, per_seq=0, cache_type="f16",
                kv_layers=32, build="a1b2c3d", measured_at="2026-09-02T00:00:00Z"),
            Fit(model="marrowgate-A3B-UD-Q4_K_XL.gguf", weights=15 * GIB, room=110 * GIB,
                compute=GIB // 2, per_token=8192, per_seq=12 * MIB, cache_type="q8_0",
                spec="draft-simple", kv_layers=12, recurrent_layers=36, build="a1b2c3d",
                measured_at="2026-09-02T00:10:00Z"),
        ]

    @pytest.mark.parametrize("suffix", [".png", ".svg"])
    def test_the_file_is_written_in_the_format_its_name_asks_for(self, tmp_path, suffix):
        """Mutation: hard-code png, and `--plot fit.svg` writes a png with an svg name."""
        where = tmp_path / f"fit{suffix}"
        assert fit_mod.plot(self.rows(), where) == str(where)
        assert where.stat().st_size > 0
        start = where.read_bytes()[:2048]
        assert start.startswith(b"\x89PNG") if suffix == ".png" else b"<svg" in start

    def test_a_format_nobody_can_draw_is_refused_by_name(self, tmp_path):
        """Before the figure is built, not inside savefig, so the message names the flag's
        own value rather than a matplotlib backend."""
        with pytest.raises(ValueError, match="csv"):
            fit_mod.plot(self.rows(), tmp_path / "fit.csv")

    def test_nothing_measured_is_refused_rather_than_drawn_empty(self, tmp_path):
        with pytest.raises(ValueError, match="no model has been measured"):
            fit_mod.plot([], tmp_path / "fit.png")

    def test_the_legend_names_every_record_by_model_cache_and_draft(self, tmp_path):
        """Two records of the same weights differ by their cache type and whether a draft
        was served, so the label carries both. Mutation: label by model alone, and two
        measurements of one model become two indistinguishable lines.
        """
        import matplotlib.pyplot as plt

        fit_mod.plot(self.rows(), tmp_path / "fit.png")
        assert fit_mod.label_of(self.rows()[0]) == "thornfield-8B (Q4_K_M)"
        assert fit_mod.label_of(self.rows()[1]) == "marrowgate-A3B (Q4_K_XL) q8_0 +draft"
        plt.close("all")

    def test_every_record_appears_in_the_legend_of_both_panels(self, tmp_path,
                                                              monkeypatch):
        """Read off the figure itself rather than off the file: a record silently dropped
        from a panel is invisible in a png and obvious here."""
        import matplotlib.pyplot as plt

        drawn: list = []
        real = plt.subplots

        def spy(*a, **kw):
            made = real(*a, **kw)
            drawn.append(made)
            return made

        monkeypatch.setattr(plt, "subplots", spy)
        fit_mod.plot(self.rows(), tmp_path / "fit.png")
        _, (left, right) = drawn[0]
        for panel in (left, right):
            named = [t.get_text() for t in panel.get_legend().get_texts()]
            for record in self.rows():
                assert any(fit_mod.label_of(record) in entry for entry in named), \
                    f"{fit_mod.label_of(record)} is in no legend entry of a panel"
        plt.close("all")

    def test_a_model_that_does_not_fit_says_so_in_the_legend_and_draws_nothing(
            self, tmp_path, monkeypatch):
        """An empty space in a chart is read as 'not measured'. A named line with no points
        is read as what it is. Mutation: skip the record, and the reader cannot tell a model
        that was never measured from one this machine cannot hold.
        """
        import matplotlib.pyplot as plt

        drawn: list = []
        real = plt.subplots
        monkeypatch.setattr(plt, "subplots",
                            lambda *a, **kw: drawn.append(real(*a, **kw)) or drawn[-1])

        huge = Fit(model="cragmoor-400B-Q4_K_M.gguf", weights=200 * GIB, room=24 * GIB,
                   compute=GIB, per_token=16384, cache_type="f16")
        fit_mod.plot([*self.rows(), huge], tmp_path / "fit.png", rooms=[24 * GIB])
        _, (left, _right) = drawn[0]
        named = [t.get_text() for t in left.get_legend().get_texts()]
        assert any("cragmoor-400B (Q4_K_M)" in entry and "does not fit" in entry
                   for entry in named), named
        empty = [line for line in left.get_lines() if len(line.get_xdata()) == 0]
        assert empty, "the model that does not fit should draw a line with no points"
        plt.close("all")

    def test_a_second_room_is_drawn_dashed_beside_the_first(self, tmp_path, monkeypatch):
        """110G solid, 24G dashed: the same models, two machines, one picture. Mutation:
        draw only the first room, and --room's repeatability means nothing.
        """
        import matplotlib.pyplot as plt

        drawn: list = []
        real = plt.subplots
        monkeypatch.setattr(plt, "subplots",
                            lambda *a, **kw: drawn.append(real(*a, **kw)) or drawn[-1])

        fit_mod.plot(self.rows(), tmp_path / "fit.png", rooms=[110 * GIB, 24 * GIB])
        _, (left, right) = drawn[0]
        styles = {line.get_linestyle() for line in left.get_lines()
                  if len(line.get_xdata())}
        assert "-" in styles and "--" in styles
        # and one horizontal line per room on the second panel
        flat = [line for line in right.get_lines()
                if len(set(line.get_ydata())) == 1 and len(line.get_ydata()) > 1]
        assert len(flat) >= 2
        plt.close("all")

    def test_the_title_names_the_machine_the_room_and_the_build(self, tmp_path,
                                                                monkeypatch):
        """A chart with no machine on it is a chart nobody can check a year later."""
        import matplotlib.pyplot as plt

        drawn: list = []
        real = plt.subplots
        monkeypatch.setattr(plt, "subplots",
                            lambda *a, **kw: drawn.append(real(*a, **kw)) or drawn[-1])

        fit_mod.plot(self.rows(), tmp_path / "fit.png", machine="hollowmere")
        title = drawn[0][0].get_suptitle()
        assert "hollowmere" in title and "110.0G" in title and "a1b2c3d" in title
        plt.close("all")

    def test_the_second_panel_charges_at_the_context_it_was_given(self, tmp_path,
                                                                  monkeypatch):
        """`--at` is the whole reason the second panel is comparable: at 4k the small model
        holds more, at 128k the one with the cheap cache does. Mutation: hard-code 32768.
        """
        import matplotlib.pyplot as plt

        seen: list = []
        real = plt.subplots
        monkeypatch.setattr(plt, "subplots",
                            lambda *a, **kw: seen.append(real(*a, **kw)) or seen[-1])

        fit_mod.plot(self.rows()[:1], tmp_path / "fit.png", at=4096)
        _, (_left, right) = seen[0]
        assert "4,096 tokens" in right.get_xlabel()
        # the line starts at zero users, where the height is the model with an empty cache,
        # and climbs by one user's worth of cache a step
        line = right.get_lines()[0]
        assert line.get_xdata()[0] == 0
        assert line.get_ydata()[0] == pytest.approx(6 * GIB / GIB, rel=1e-6)
        assert line.get_ydata()[1] == pytest.approx(
            (6 * GIB + 4096 * 32768) / GIB, rel=1e-6)
        plt.close("all")


class TestTheLineAModelDraws:
    """The second panel is a straight line, and `Fit.line` is the two numbers it is drawn
    from -- so the chart is testable as arithmetic rather than as a picture."""

    def one(self, **over) -> Fit:
        base = dict(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, draft=GIB // 2,
                    room=24 * GIB, per_token=32768, per_seq=8 * MIB, compute=GIB)
        return Fit(**{**base, **over})

    def test_loaded_is_the_model_sitting_there_with_an_empty_cache(self):
        """Weights, a draft head and the compute buffers: everything that does not grow
        with the users. Mutation: leave the compute buffers out, and every intercept is
        low by the one part of the figure a GGUF header cannot tell you."""
        assert self.one().loaded() == 5 * GIB + GIB // 2 + GIB
        assert self.one().free() == 24 * GIB - self.one().loaded()

    def test_the_intercept_and_the_slope_are_what_the_chart_plots(self):
        """`loaded() + users * cost(at)`, and nothing else. Mutation: put the per-sequence
        cost in the intercept, and a model is charged for a user who never arrived."""
        intercept, each = self.one().line(32768)
        assert intercept == self.one().loaded()
        assert each == 32768 * 32768 + 8 * MIB == self.one().cost(32768)

    def test_a_longer_context_tilts_the_line_and_leaves_the_intercept_alone(self):
        """The whole shape of the answer: the context changes what a user costs, never what
        the model costs before anybody arrives."""
        short, steep = self.one().line(4096), self.one().line(131072)
        assert short[0] == steep[0]
        assert steep[1] > short[1]

    def test_the_line_meets_the_room_at_the_last_user_that_fits(self):
        """The two ends of the same fact: where the line crosses a room is `users()`.

        Mutation: any disagreement between the chart's arithmetic and the table's, which is
        exactly the bug a chart drawn from its own formula would hide.
        """
        fit = self.one()
        intercept, each = fit.line(32768)
        held = fit.users(32768)
        assert intercept + held * each <= fit.room
        assert intercept + (held + 1) * each > fit.room


class TestTheCardsBehindIt:
    def rows(self) -> list[Fit]:
        return [Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, room=110 * GIB,
                    compute=GIB, per_token=32768, cache_type="f16")]

    def panels(self, tmp_path, monkeypatch, **kw):
        import matplotlib.pyplot as plt

        drawn: list = []
        real = plt.subplots
        monkeypatch.setattr(plt, "subplots",
                            lambda *a, **k: drawn.append(real(*a, **k)) or drawn[-1])
        fit_mod.plot(self.rows(), tmp_path / "fit.png", **kw)
        return drawn[0]

    def test_the_familiar_card_sizes_are_drawn_behind_the_lines(self, tmp_path,
                                                                monkeypatch):
        """A chart that only knows about this machine answers "will it fit here"; the grey
        lines are what turn it into "and what would I need". Mutation: drop them, and a
        reader with a 24 GB card has to do the arithmetic themselves.
        """
        import matplotlib.pyplot as plt

        _figure, (_left, right) = self.panels(tmp_path, monkeypatch, rooms=[24 * GIB])
        flat = {round(line.get_ydata()[0]) for line in right.get_lines()
                if len(set(line.get_ydata())) == 1 and len(line.get_ydata()) > 1}
        assert {6, 8, 12, 16, 24}.issubset(flat)
        assert "128G" not in {a.get_text() for a in right.texts}, \
            "a card far above everything drawn only squashes the chart"
        plt.close("all")

    def test_the_top_of_the_chart_clears_the_room_being_asked_about(self, tmp_path,
                                                                    monkeypatch):
        """Mutation: fit the axis to the lines alone, and the room line -- the thing the
        reader is looking for -- falls off the top."""
        import matplotlib.pyplot as plt

        _figure, (_left, right) = self.panels(tmp_path, monkeypatch, rooms=[96 * GIB])
        assert right.get_ylim()[1] > 96
        assert right.get_xlim()[0] == 0
        plt.close("all")

    def test_the_legend_says_the_intercept_and_the_slope_in_words(self, tmp_path,
                                                                  monkeypatch):
        """"6.0G + 1.00G/user at 32k" -- the whole model in one line, which is what makes
        two models comparable at a glance. Mutation: label by model name alone."""
        import matplotlib.pyplot as plt

        _figure, (left, right) = self.panels(tmp_path, monkeypatch, at=32768)
        [entry] = [t.get_text() for t in right.get_legend().get_texts()]
        assert entry == "thornfield-8B (Q4_K_M): 6.0G + 1.00G/user at 32k"
        [first] = [t.get_text() for t in left.get_legend().get_texts()]
        assert first == "thornfield-8B (Q4_K_M) (6.0G loaded)"
        plt.close("all")


class TestTheCommandDraws:
    def run(self, argv: list[str]) -> int:
        from ml_stack.serve.cli import main

        return main(["fit", *argv])

    def some_records(self, path: Path) -> None:
        path.write_text(json.dumps([
            Fit(model="thornfield-8B-Q4_K_M.gguf", weights=5 * GIB, compute=GIB,
                per_token=32768, room=110 * GIB, build="a1b2c3d").as_dict(),
            Fit(model="marrowgate-A3B-UD-Q4_K_XL.gguf", weights=15 * GIB, compute=GIB // 2,
                per_token=8192, per_seq=12 * MIB, room=110 * GIB,
                build="a1b2c3d").as_dict()]), encoding="utf-8")

    def test_plot_writes_the_picture_and_says_where(self, tmp_path, monkeypatch, capsys,
                                                    _fit_files_in_tmp):
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)
        where = tmp_path / "fit.png"

        assert self.run(["--plot", str(where)]) == 0
        assert where.exists()
        assert str(where) in capsys.readouterr().err

    def test_open_shows_the_picture_with_the_desktops_opener(self, tmp_path, monkeypatch,
                                                            capsys, _fit_files_in_tmp):
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)
        opened = []
        monkeypatch.setattr("ml_stack.platform.open_path", lambda p: opened.append(str(p)) or "open")
        where = tmp_path / "fit.png"
        assert self.run(["--plot", str(where), "--open"]) == 0
        assert opened == [str(where)]
        assert "opened with open" in capsys.readouterr().err

    def test_two_rooms_on_the_command_line_reach_the_chart(self, tmp_path, monkeypatch,
                                                           capsys, _fit_files_in_tmp):
        """`--room` is repeatable and both reach `plot`; the listing still answers for the
        first. Mutation: keep --room a single value, and the second is silently dropped."""
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)
        seen: list = []
        monkeypatch.setattr(fit_mod, "plot",
                            lambda rows, where, **kw: seen.append(kw) or str(where))

        assert self.run(["--room", "110G", "--room", "24G", "--plot",
                         str(tmp_path / "fit.png")]) == 0
        assert seen[0]["rooms"] == [110 * GIB, 24 * GIB]
        out = capsys.readouterr().out
        assert "110.0G room" in out

    def test_a_bad_room_is_refused_before_anything_is_drawn(self, tmp_path, monkeypatch,
                                                            capsys, _fit_files_in_tmp):
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)
        where = tmp_path / "fit.png"
        assert self.run(["--room", "24G", "--room", "heaps", "--plot", str(where)]) == 2
        assert not where.exists()
        capsys.readouterr()

    def test_at_reaches_the_second_panel(self, tmp_path, monkeypatch, capsys,
                                         _fit_files_in_tmp):
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)
        seen: list = []
        monkeypatch.setattr(fit_mod, "plot",
                            lambda rows, where, **kw: seen.append(kw) or str(where))

        assert self.run(["--at", "8192", "--plot", str(tmp_path / "fit.svg")]) == 0
        assert seen[0]["at"] == 8192
        capsys.readouterr()

    def test_a_chart_drawn_beside_a_written_page_is_embedded_in_it(
            self, tmp_path, monkeypatch, capsys, _fit_files_in_tmp):
        """The pair travel together: the Markdown names the picture by its basename, so
        moving both to a docs directory keeps the link. Mutation: write an absolute path,
        and the image is broken everywhere but this machine.
        """
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)
        page, picture = tmp_path / "fit.md", tmp_path / "fit.png"

        assert self.run(["--plot", str(picture), "--write", str(page)]) == 0
        written = page.read_text(encoding="utf-8")
        assert "](fit.png)" in written
        assert str(tmp_path) not in written
        capsys.readouterr()

    def test_a_page_written_without_a_chart_embeds_nothing(self, tmp_path, monkeypatch,
                                                           capsys, _fit_files_in_tmp):
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)
        page = tmp_path / "fit.md"
        assert self.run(["--write", str(page)]) == 0
        assert "![" not in page.read_text(encoding="utf-8")
        capsys.readouterr()

    def test_without_matplotlib_the_refusal_says_how_to_get_it(self, tmp_path, monkeypatch,
                                                               capsys, _fit_files_in_tmp):
        """Optional the way every other heavy dependency here is optional. Mutation: let
        the ImportError out, and a person gets a traceback instead of an install line."""
        self.some_records(_fit_files_in_tmp)
        monkeypatch.setattr("ml_stack.hub.room", lambda: 110 * GIB)

        def missing():
            raise RuntimeError(
                "drawing the fit chart needs matplotlib: pip install 'ml-stack[plot]'")

        monkeypatch.setattr(fit_mod, "_pyplot", missing)
        assert self.run(["--plot", str(tmp_path / "fit.png")]) == 2
        assert "pip install 'ml-stack[plot]'" in capsys.readouterr().err


# -- the estimate, for when nothing has been measured -------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_serve_preflight import write_gguf  # noqa: E402


class TestTheEstimateWithoutAMeasurement:
    """`preflight._kv_estimate_bytes` over the three metadata shapes that used to be
    counted as full attention, each off by a different multiple."""

    def estimate(self, meta: dict, context: int) -> int:
        from ml_stack.serve.preflight import _kv_estimate_bytes

        return _kv_estimate_bytes(meta, context, "", "")

    def test_a_recurrent_interval_charges_only_the_layers_that_hold_a_cache(self):
        """`full_attention_interval = 4` means one layer in four -- (il+1) % 4 == 0, the
        way `models/qwen4exp.cpp` reads it. Twelve of forty-eight, not forty-eight.

        Mutation: count every layer, and the estimate is four times what loads.
        """
        meta = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 48,
                "qwen4exp.attention.head_count_kv": 2, "qwen4exp.attention.key_length": 128}
        full = self.estimate(meta, 1000)
        sparse = self.estimate({**meta, "qwen4exp.full_attention_interval": 4}, 1000)
        assert full == 48 * 2 * 128 * 1000 * 4
        assert sparse == 12 * 2 * 128 * 1000 * 4
        assert sparse * 4 == full

    def test_an_explicit_recurrent_layer_array_wins_over_the_interval(self):
        """llama.cpp reads `attention.recurrent_layers` first and only falls back to the
        interval. Mutation: read the interval first, and a model that named its layers
        exactly is estimated as though it had not."""
        meta = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 4,
                "qwen4exp.attention.head_count_kv": 2, "qwen4exp.attention.key_length": 8,
                "qwen4exp.full_attention_interval": 4,
                "qwen4exp.attention.recurrent_layers": [True, True, True, False]}
        assert self.estimate(meta, 100) == 1 * 2 * 8 * 100 * 4

    def test_a_sliding_window_pattern_charges_the_window_and_not_the_context(self):
        """gemma4: a bool per layer, a 512-token window, its own key_length_swa, and
        `shared_kv_layers` layers at the end that hold nothing at all.

        Mutation: give a sliding layer the full context, and a 128k estimate is an order
        of magnitude over what loads.
        """
        meta = {"general.architecture": "gemma4", "gemma4.block_count": 6,
                "gemma4.attention.head_count_kv": 2, "gemma4.attention.key_length": 256,
                "gemma4.attention.key_length_swa": 128,
                "gemma4.attention.sliding_window": 512,
                "gemma4.attention.sliding_window_pattern":
                    [True, True, False, True, True, False]}
        # four sliding layers at the 512-token window with 128-wide keys, two full ones at
        # the context with 256-wide keys
        assert self.estimate(meta, 8192) == (
            4 * 2 * 128 * 512 * 4 + 2 * 2 * 256 * 8192 * 4)

    def test_shared_kv_layers_hold_nothing_of_their_own(self):
        """gemma4's last `shared_kv_layers` read the cache the layers before them wrote --
        `hparams.n_layer_kv_from_start = n_layer - shared`. Mutation: charge them."""
        meta = {"general.architecture": "gemma4", "gemma4.block_count": 6,
                "gemma4.attention.head_count_kv": 2, "gemma4.attention.key_length": 8}
        assert self.estimate({**meta, "gemma4.attention.shared_kv_layers": 2}, 100) == \
            4 * 2 * 8 * 100 * 4

    def test_a_window_with_no_pattern_alternates_the_way_llama_cpp_does(self, tmp_path):
        """gpt-oss names a 128-token window and no pattern; llama.cpp defaults the period
        to 2, so the even layers slide. Written as a real GGUF and read back through the
        header reader, because that is the path a preflight takes.

        Mutation: treat a missing pattern as no sliding at all.
        """
        from ml_stack.serve.preflight import read_gguf_header

        path = write_gguf(tmp_path / "amberlin-20B-Q4_K_M.gguf", {
            "general.architecture": "gpt-oss", "gpt-oss.block_count": 4,
            "gpt-oss.attention.head_count_kv": 8, "gpt-oss.attention.key_length": 64,
            "gpt-oss.attention.sliding_window": 128})
        meta = read_gguf_header(path)
        # layers 0 and 2 slide (128 tokens), layers 1 and 3 see the whole 4096
        assert self.estimate(meta, 4096) == (
            2 * 8 * 64 * 128 * 4 + 2 * 8 * 64 * 4096 * 4)

    def test_a_plain_model_is_still_the_flat_multiplication_it_always_was(self):
        """No pattern, no interval, no shared layers: nothing changes. Mutation: any of the
        new branches firing on a model that named none of these keys."""
        meta = {"general.architecture": "llama", "llama.block_count": 32,
                "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128}
        assert self.estimate(meta, 4096) == 32 * 8 * 128 * 4096 * 4

    def test_a_value_length_that_differs_from_the_key_length_is_charged_separately(self):
        """A model whose V is narrower than its K. Mutation: use the key length for both."""
        meta = {"general.architecture": "x", "x.block_count": 2,
                "x.attention.head_count_kv": 1, "x.attention.key_length": 16,
                "x.attention.value_length": 8}
        assert self.estimate(meta, 100) == 2 * 1 * 100 * (16 * 2 + 8 * 2)
