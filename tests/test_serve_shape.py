"""One shape per port, written down once, and one held server to take a seat on."""

import pytest

import ml_stack.serve
from ml_stack.serve import shape as shape_mod
from ml_stack.serve.shape import Shape, draft_for, held, projector_for, release_all, seat


class Held:
    """What a fake `serve` yields: a context manager with a base url."""

    def __init__(self, url):
        self.base_url = url
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        return False


@pytest.fixture
def leases(monkeypatch):
    """`serve` replaced by a record of what it was asked for. No server is started."""
    asked: list[tuple[str, dict]] = []
    servers: list[Held] = []

    def fake_serve(model, **kwargs):
        asked.append((str(model), dict(kwargs)))
        servers.append(Held(f"http://127.0.0.1:{kwargs['port']}"))
        return servers[-1]

    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(shape_mod, "_STACKS", {})
    monkeypatch.setattr(shape_mod, "_URLS", {})
    yield asked, servers


def test_a_lease_says_only_what_was_asked_for():
    plain = Shape(model="weights.gguf", port=8080, seats=2, seat_context=32768)
    assert plain.lease() == {"port": 8080, "context": 65536, "parallel": 2}
    assert plain.context == 65536, "what the server is asked for is every seat added up"


def test_the_whole_shape_becomes_the_arguments_serve_takes():
    full = Shape(model="weights.gguf", port=8082, seats=4, seat_context=32768,
                 cache_type="q8_0", draft="hf:owner/repo/mtp-Q8_0.gguf", draft_n_max=4,
                 mmproj="/models/mmproj-F16.gguf", reasoning_budget=0)
    assert full.lease() == {
        "port": 8082, "context": 131072, "parallel": 4,
        "cache_type_k": "q8_0", "cache_type_v": "q8_0",
        "draft": "hf:owner/repo/mtp-Q8_0.gguf", "spec_type": "draft-mtp", "spec_draft_max": 4,
        "mmproj": "/models/mmproj-F16.gguf", "reasoning_budget": 0,
    }


def test_a_head_is_served_as_the_method_it_implements():
    """An EAGLE3 head served as draft-simple is not slower, it is wrong."""
    assert Shape(model="m", draft="eagle3-head.gguf").lease()["spec_type"] == "draft-eagle3"
    assert Shape(model="m", draft="mtp-head.gguf").lease()["spec_type"] == "draft-mtp"
    # thinking off is a decision and 0 is not "nothing was said"
    assert Shape(model="m", reasoning_budget=0).lease()["reasoning_budget"] == 0
    assert "reasoning_budget" not in Shape(model="m").lease()


def test_a_named_build_gets_its_own_manager_and_the_default_gets_none():
    assert Shape(model="m").manager() is None
    manager = Shape(model="m", build="unsloth").manager()
    assert manager.backend._build == "unsloth"


def test_two_models_on_two_ports_are_two_servers_and_neither_is_leased_twice(leases):
    """A single held server used to serve whichever job asked first, at four times the cost."""
    asked, _servers = leases
    large = Shape(model="/models/large.gguf", port=8080, seats=2, seat_context=32768)
    small = Shape(model="/models/small.gguf", port=8082, seats=4, seat_context=32768)

    reading = seat(large, index=0, n_predict=100)
    answering = seat(small, index=0, n_predict=100)

    assert [model for model, _ in asked] == ["/models/large.gguf", "/models/small.gguf"]
    assert [lease["port"] for _, lease in asked] == [8080, 8082]
    assert [lease["parallel"] for _, lease in asked] == [2, 4]
    assert reading.base_url != answering.base_url

    seat(small, index=1, n_predict=100)
    assert len(asked) == 2, "the server is held per port, not started per ask"
    assert held() == {8080: reading.base_url, 8082: answering.base_url}


def test_every_seat_is_its_own_slot_and_a_busy_port_cycles_through_them(leases):
    small = Shape(model="/models/small.gguf", port=8082, seats=4)
    assert [seat(small, index=i, n_predict=100).slot for i in range(6)] == [0, 1, 2, 3, 0, 1]
    # a shape with no seats named still asks for a slot that exists
    assert seat(Shape(model="/m.gguf", port=8083, seats=0), index=3, n_predict=100).slot == 0


def test_the_client_gets_the_ceiling_and_the_timeout_it_was_asked_for(leases):
    client = seat(Shape(model="/m.gguf", port=8080), index=0, n_predict=16384, timeout=42.0)
    assert (client.n_predict, client.timeout) == (16384, 42.0)


def test_letting_go_releases_every_held_server(leases):
    _asked, servers = leases
    seat(Shape(model="/a.gguf", port=8080), index=0, n_predict=100)
    seat(Shape(model="/b.gguf", port=8082), index=0, n_predict=100)
    release_all()
    assert [s.closed for s in servers] == [True, True]
    assert held() == {}


def test_a_draft_that_cannot_be_found_is_served_without_it_out_loud(monkeypatch):
    """Said in silence once, and a model ran undrafted for an hour with nothing to show."""
    import ml_stack.serve.cli as cli

    said: list[str] = []
    monkeypatch.setattr(cli, "drafted", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no repository listing")))
    assert draft_for("/models/weights.gguf", "auto", log=said.append) == ""
    assert said == ["no draft head: no repository listing"]


def test_a_draft_is_resolved_for_the_build_that_has_to_load_it(monkeypatch):
    import ml_stack.serve.cli as cli

    asked: list[tuple] = []

    def fake_drafted(model, want, *, binary=None, borrows=None):
        asked.append((model, want, binary))
        return "/models/mtp-Q8_0.gguf"

    monkeypatch.setattr(cli, "drafted", fake_drafted)
    from ml_stack.serve import backend

    monkeypatch.setattr(backend.LlamaServerBackend, "binary",
                        property(lambda self: f"/builds/{self._build}/llama-server"))
    assert draft_for("/models/weights.gguf", "auto",
                     build="unsloth") == "/models/mtp-Q8_0.gguf"
    assert asked == [("/models/weights.gguf", "auto", "/builds/unsloth/llama-server")]


def test_auto_takes_the_most_precise_projector_and_a_missing_one_is_no_projector(monkeypatch):
    import ml_stack.serve.cli as cli

    asked: list[tuple] = []

    def fake_alongside(model, wanted, prefix, *, best=False):
        asked.append((model, wanted, prefix, best))
        return "/models/mmproj-BF16.gguf"

    monkeypatch.setattr(cli, "alongside", fake_alongside)
    assert projector_for("/models/weights.gguf", "auto") == "/models/mmproj-BF16.gguf"
    assert asked == [("/models/weights.gguf", "auto", "mmproj-", True)]

    monkeypatch.setattr(cli, "alongside", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    said: list[str] = []
    assert projector_for("/models/weights.gguf", "auto", log=said.append) == ""
    assert said == ["no projector: gone"]
