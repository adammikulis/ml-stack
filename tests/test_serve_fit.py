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
        assert "thornfield-8B-Q4_K_M.gguf" in text
        assert "per token" in text and "per sequence" in text
        assert "32,768" in text and " 18 " in f" {text} "

    def test_markdown_is_a_table(self):
        text = render(self.rows(), [32768], None, True)
        assert text.startswith("### thornfield-8B-Q4_K_M.gguf")
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
        assert "thornfield-8B-Q4_K_M.gguf" in out
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
