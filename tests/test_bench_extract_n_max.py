"""`ml-stack-bench extract --n-max N`: a draft length over the profile's, for the workload
that repeats what it just read."""

import contextlib

from tests.test_bench_extract import _world_dir


def _args(tmp_path, **over):
    world = tmp_path / "world"
    fields = {"serve": ["x.gguf"], "serve_port": 1, "context": 8192, "parallel": 2,
              "profile": True, "smoke": True,
              "world": str(world if world.exists() else _world_dir(tmp_path)),
              "per_message": 1, "seed": 1, "sample": 3, "kept": str(tmp_path / "k.ladybug"),
              "label": "t", "temperature": None, "top_p": None, "top_k": None, "min_p": None,
              "twice": False, "anyway": False, "base_url": "", "no_smoke": False, "yes": True,
              "ceiling": 0, "no_selfcheck": True, "detach": False, "no_queue": True,
              "no_prefetch": True, "n_max": None}
    fields.update(over)
    return type("A", (), fields)()


def _serving_seam(monkeypatch, seen, *, draft):
    from ml_stack.graph.bench import extract as ex
    from ml_stack.serve import Shape

    class Found:
        def shape(self, port, seats):
            return Shape(model="x.gguf", port=port, seats=seats, seat_context=4096,
                         cache_type="q8_0", draft=draft, draft_n_max=4 if draft else None)

        def said(self):
            return "measured"

    monkeypatch.setattr("ml_stack.serve.profile.profile_for", lambda m: Found())
    monkeypatch.setattr(ex, "find_model", lambda m: "x.gguf")

    def fake_serve(model, manager=None, **lease):
        seen["lease"] = lease
        raise SystemExit(0)

    monkeypatch.setattr("ml_stack.serve.serve", fake_serve)


def test_n_max_lengthens_the_profiles_draft(monkeypatch, tmp_path):
    from ml_stack.graph.bench import extract as ex

    seen: dict = {}
    _serving_seam(monkeypatch, seen, draft="mtp.gguf")
    with contextlib.suppress(SystemExit):
        ex.main(_args(tmp_path))
    assert seen["lease"]["spec_draft_max"] == 4, "the profile's own length when not told"
    seen.clear()
    with contextlib.suppress(SystemExit):
        ex.main(_args(tmp_path, n_max=8))
    assert seen["lease"]["spec_draft_max"] == 8 and seen["lease"]["draft"] == "mtp.gguf"


def test_n_max_without_a_head_is_refused_rather_than_ignored(monkeypatch, tmp_path, capsys):
    from ml_stack.graph.bench import extract as ex

    seen: dict = {}
    _serving_seam(monkeypatch, seen, draft="")
    assert ex.main(_args(tmp_path, n_max=8)) == 2
    assert "no draft head" in capsys.readouterr().err
    assert "lease" not in seen, "nothing was served"


def test_the_subcommand_parses_n_max():
    from ml_stack.graph.bench import _parser

    args = _parser().parse_args(["extract", "x", "--world", "w", "--n-max", "6"])
    assert args.n_max == 6
