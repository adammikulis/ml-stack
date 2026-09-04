"""``ml-stack-bench standard`` and ``animate`` hand their words to their own modules; the
``compare`` and ``speed`` lines take the shapes ``ml-stack-do`` writes.

Every fixture here is invented. Nothing reads a real store, a real graph, or a real server.
"""

from __future__ import annotations

import json

import pytest

from ml_stack.graph.bench import _parser


def test_standard_and_animate_hand_their_words_to_their_own_mains(monkeypatch):
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import animate, standard

    seen = {}
    monkeypatch.setattr(standard, "main", lambda argv: seen.setdefault("standard", list(argv)) and 0)
    monkeypatch.setattr(animate, "main", lambda argv: seen.setdefault("animate", list(argv)) and 0)
    assert bench._main(["standard", "--url", "http://127.0.0.1:1/v1/chat/completions",
                        "--model", "quince-2b", "--limit", "10", "--dry-run"]) == 0
    assert seen["standard"] == ["--url", "http://127.0.0.1:1/v1/chat/completions",
                                "--model", "quince-2b", "--limit", "10", "--dry-run"]
    assert bench._main(["animate", "compare.json", "--out", "compare.mp4", "--dry-run"]) == 0
    assert seen["animate"] == ["compare.json", "--out", "compare.mp4", "--dry-run"]


def test_standard_and_animate_are_not_measuring_commands_here():
    """`standard` takes the measuring lock itself and `animate` needs none; neither is sent
    through this parser's lock, its self-check or its estimate."""
    from ml_stack.graph.bench import MEASURING
    from ml_stack.graph.bench.run import HANDED_OVER

    assert set(HANDED_OVER) == {"standard", "animate"}
    assert not set(HANDED_OVER) & set(MEASURING)


def test_the_bench_parser_knows_the_flags_standard_and_animate_take():
    args = _parser().parse_args(["standard", "--url", "http://127.0.0.1:1/v1/chat/completions",
                                 "--model", "quince-2b", "--tasks", "gsm8k", "--limit", "5",
                                 "--think", "--dry-run"])
    assert args.cmd == "standard" and args.model == "quince-2b" and args.limit == 5
    args = _parser().parse_args(["animate", "c.json", "--out", "c.mp4", "--quality", "l",
                                 "--seconds", "20", "--only", "title", "--dry-run"])
    assert args.cmd == "animate" and args.out == "c.mp4" and args.quality == "l"
    with pytest.raises(SystemExit):
        _parser().parse_args(["animate", "c.json"])          # --out is required


def test_compare_takes_positional_labels_and_last(tmp_path, monkeypatch):
    from ml_stack.graph.bench import invented_digest, save
    from ml_stack.graph.bench.comparison import newest_labels
    import ml_stack.graph.bench as bench

    from conftest import scored_rows

    kept = tmp_path / "runs.ladybug"
    mine = invented_digest()
    for label in ("one-plain", "two-plain", "three-plain", "four-plain"):
        save(kept, scored_rows(label, questions=2, hits=1, seconds=4.0),
             held={"graph": mine, "context": 4096})
    labels = newest_labels(bench.runs(kept), 3)
    assert len(set(labels)) == 3
    out = tmp_path / "c.json"
    assert bench._main(["compare", "one-plain", "two-plain", "--kept", str(kept),
                        "--export", str(out)]) == 0
    assert [c["run"] for c in json.loads(out.read_text())["configs"]] == ["one-plain", "two-plain"]
    assert bench._main(["compare", "--last", "--kept", str(kept), "--export", str(out)]) == 0
    assert len(json.loads(out.read_text())["configs"]) == 3
    assert bench._main(["compare", "--last", "2", "--labels", "one-plain", "--kept", str(kept),
                        "--export", str(out)]) == 0
    got = [c["run"] for c in json.loads(out.read_text())["configs"]]
    assert got[0] == "one-plain" and 2 <= len(got) <= 3, "the named label first, then --last's"


def test_speed_takes_a_positional_model_and_users_beside_streams():
    args = _parser().parse_args(["speed", "quince-2b.gguf", "--users", "1,2,4"])
    assert args.model == "quince-2b.gguf" and args.streams == "1,2,4"
    args = _parser().parse_args(["speed", "--serve", "quince-2b.gguf", "--streams", "1"])
    assert args.model == "" and args.serve == ["quince-2b.gguf"]
