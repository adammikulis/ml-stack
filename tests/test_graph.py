"""Graph container, message passing, DAG sweeps, topology construction."""

from __future__ import annotations

import numpy as np
import pytest
from ml_stack.backend import available, get_backend
from ml_stack.graph import (
    Graph,
    NotADAG,
    batch_graphs,
    build_topology,
    clear_cache,
    decompose_to_dags,
    degree,
    knn_edges,
    morton_codes,
    mst_edges,
    normalize_by_degree,
    propagate,
    require_topological_order,
    resolvent_sweep,
    topological_order,
)
from ml_stack.testing import assert_forward_parity, needs_both

BACKENDS = available()
each_backend = pytest.mark.parametrize("name", BACKENDS)

PATH = [(0, 1), (1, 2), (2, 3)]
DIAMOND = [(0, 1), (0, 2), (1, 3), (2, 3)]


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()


def _identity(backend, n: int):
    return backend.ops.array(np.eye(n, dtype=np.float32))


class TestGraph:
    def test_edge_arrays_must_agree_in_length(self):
        backend = get_backend()
        ops = backend.ops
        with pytest.raises(ValueError, match="edges"):
            Graph(
                num_nodes=3,
                src=ops.array([0, 1], dtype=ops.int32),
                dst=ops.array([1], dtype=ops.int32),
            )

    def test_weights_must_match_the_edge_count(self):
        backend = get_backend()
        ops = backend.ops
        with pytest.raises(ValueError, match="w has"):
            Graph(
                num_nodes=3,
                src=ops.array([0, 1], dtype=ops.int32),
                dst=ops.array([1, 2], dtype=ops.int32),
                w=ops.array([1.0], dtype=ops.float32),
            )

    def test_negative_node_count_is_rejected(self):
        backend = get_backend()
        ops = backend.ops
        with pytest.raises(ValueError, match="non-negative"):
            Graph(num_nodes=-1, src=ops.array([], dtype=ops.int32),
                  dst=ops.array([], dtype=ops.int32))

    def test_num_edges(self):
        assert Graph.from_edges(4, PATH).num_edges == 3

    def test_to_networkx_round_trips_the_structure(self):
        nx = pytest.importorskip("networkx")
        graph = Graph.from_edges(4, PATH).to_networkx()
        assert graph.number_of_nodes() == 4
        assert sorted(graph.edges()) == PATH
        # The point of the escape hatch: real algorithms, not reimplemented ones.
        assert nx.shortest_path_length(graph, 0, 3) == 3

    def test_isolated_nodes_survive_the_networkx_conversion(self):
        """A node with no edges is still a node. Building the graph from the edge list
        alone would silently drop it and renumber everything after it."""
        graph = Graph.from_edges(6, PATH).to_networkx()
        assert graph.number_of_nodes() == 6


class TestBatching:
    def test_edge_indices_are_offset_per_graph(self):
        batched = batch_graphs([Graph.from_edges(3, [(0, 1)]), Graph.from_edges(2, [(0, 1)])])
        assert batched.num_nodes == 5
        assert batched.num_graphs == 2
        assert np.asarray(batched.src).tolist() == [0, 3]
        assert np.asarray(batched.dst).tolist() == [1, 4]

    def test_graph_ids_label_every_node(self):
        batched = batch_graphs([Graph.from_edges(3, [(0, 1)]), Graph.from_edges(2, [(0, 1)])])
        assert np.asarray(batched.graph_ids).tolist() == [0, 0, 0, 1, 1]
        assert np.asarray(batched.node_offsets).tolist() == [0, 3, 5]

    def test_empty_batch_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            batch_graphs([])


class TestMessagePassing:
    @each_backend
    def test_propagate_moves_values_along_edges(self, name):
        backend = get_backend(name)
        graph = Graph.from_edges(4, PATH, backend=backend)
        out = np.asarray(propagate(backend, graph, _identity(backend, 4)))
        # Node 0 has no in-edges; each other node receives its predecessor's row.
        assert out[0].tolist() == [0, 0, 0, 0]
        assert out[1].tolist() == [1, 0, 0, 0]
        assert out[3].tolist() == [0, 0, 1, 0]

    @each_backend
    def test_a_node_with_two_parents_sums_both(self, name):
        backend = get_backend(name)
        graph = Graph.from_edges(4, DIAMOND, backend=backend)
        out = np.asarray(propagate(backend, graph, _identity(backend, 4)))
        assert out[3].tolist() == [0, 1, 1, 0]

    @each_backend
    def test_mean_reduction_does_not_produce_nan_for_isolated_nodes(self, name):
        """Dividing by a zero count would give NaN, which then propagates through the rest
        of the layer and destroys the whole batch."""
        backend = get_backend(name)
        graph = Graph.from_edges(5, PATH, backend=backend)
        out = np.asarray(propagate(backend, graph, _identity(backend, 5), reduce="mean"))
        assert np.isfinite(out).all()
        assert out[4].tolist() == [0, 0, 0, 0, 0]

    @each_backend
    def test_edge_weights_scale_the_message(self, name):
        backend = get_backend(name)
        graph = Graph.from_edges(2, [(0, 1)], backend=backend, weights=[2.5])
        out = np.asarray(propagate(backend, graph, _identity(backend, 2)))
        assert out[1].tolist() == [2.5, 0.0]

    @each_backend
    def test_weights_can_be_ignored(self, name):
        backend = get_backend(name)
        graph = Graph.from_edges(2, [(0, 1)], backend=backend, weights=[2.5])
        out = np.asarray(propagate(backend, graph, _identity(backend, 2), use_weights=False))
        assert out[1].tolist() == [1.0, 0.0]

    @each_backend
    def test_degree_counts_in_edges(self, name):
        backend = get_backend(name)
        graph = Graph.from_edges(4, DIAMOND, backend=backend)
        assert np.asarray(degree(backend, graph)).tolist() == [0.0, 1.0, 1.0, 2.0]

    @each_backend
    def test_degree_normalisation_leaves_isolated_nodes_finite(self, name):
        """rsqrt(0) is infinity, and an infinity in the features is not recoverable."""
        backend = get_backend(name)
        graph = Graph.from_edges(5, PATH, backend=backend)
        out = np.asarray(normalize_by_degree(backend, graph, _identity(backend, 5)))
        assert np.isfinite(out).all()

    def test_unknown_reduction_is_rejected(self):
        backend = get_backend()
        graph = Graph.from_edges(4, PATH, backend=backend)
        with pytest.raises(ValueError, match="reduce"):
            propagate(backend, graph, _identity(backend, 4), reduce="median")


class TestDAG:
    def test_topological_order_of_a_path(self):
        graph = Graph.from_edges(4, PATH)
        assert topological_order(4, graph.src, graph.dst) == [0, 1, 2, 3]

    def test_a_cycle_returns_none_rather_than_raising(self):
        """Callers routinely want to *ask* whether a graph is a DAG. That is a question,
        not an error."""
        graph = Graph.from_edges(3, [(0, 1), (1, 2), (2, 0)])
        assert topological_order(3, graph.src, graph.dst) is None

    def test_require_raises_on_a_cycle(self):
        graph = Graph.from_edges(3, [(0, 1), (1, 2), (2, 0)])
        with pytest.raises(NotADAG):
            require_topological_order(graph)

    def test_the_order_is_a_valid_one(self):
        graph = Graph.from_edges(4, DIAMOND)
        order = require_topological_order(graph)
        position = {node: i for i, node in enumerate(order)}
        for u, v in DIAMOND:
            assert position[u] < position[v], f"edge {u}->{v} violates the order"

    def test_the_cache_returns_a_copy_not_the_stored_list(self):
        """A caller mutating the returned list would corrupt every later lookup."""
        graph = Graph.from_edges(4, PATH)
        first = topological_order(4, graph.src, graph.dst)
        first.append(999)
        assert topological_order(4, graph.src, graph.dst) == [0, 1, 2, 3]

    @each_backend
    def test_resolvent_accumulates_along_the_path(self, name):
        """y[u] = x[u] + sum over parents. Down a path with identity input, row sums
        should be 1, 2, 3, 4."""
        backend = get_backend(name)
        graph = Graph.from_edges(4, PATH, backend=backend)
        out = np.asarray(resolvent_sweep(backend, graph, _identity(backend, 4)))
        assert out.sum(axis=1).tolist() == [1.0, 2.0, 3.0, 4.0]

    @each_backend
    def test_resolvent_does_not_mutate_its_input(self, name):
        """Writing into the input in place corrupts the autograd graph, and on an
        immutable array type it does not work at all."""
        backend = get_backend(name)
        graph = Graph.from_edges(4, PATH, backend=backend)
        x = _identity(backend, 4)
        before = np.asarray(x).copy()
        resolvent_sweep(backend, graph, x)
        assert np.array_equal(np.asarray(x), before)

    @each_backend
    def test_resolvent_matches_the_matrix_inverse(self, name):
        """The whole justification for the sweep: it computes (I - A)^-1 x in O(V+E)."""
        backend = get_backend(name)
        graph = Graph.from_edges(4, DIAMOND, backend=backend)

        adjacency = np.zeros((4, 4), dtype=np.float64)
        for u, v in DIAMOND:
            adjacency[v, u] = 1.0
        x = np.eye(4, dtype=np.float64)
        expected = np.linalg.inv(np.eye(4) - adjacency) @ x

        out = np.asarray(resolvent_sweep(backend, graph, _identity(backend, 4)))
        assert np.allclose(out, expected, atol=1e-5)

    def test_decompose_produces_dags_in_both_directions(self):
        graph = Graph.from_edges(4, PATH)
        dags = decompose_to_dags(4, graph.src, graph.dst)
        assert len(dags) == 2
        forward, backward = dags
        assert forward[0] == backward[1] and forward[1] == backward[0]

    def test_decompose_is_stable_across_calls(self):
        """Two backends must produce the same decomposition, or a parity test fails for a
        reason that has nothing to do with the arithmetic."""
        graph = Graph.from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
        first = decompose_to_dags(6, graph.src, graph.dst)
        assert decompose_to_dags(6, graph.src, graph.dst) == first

    def test_four_directions_adds_a_second_tree(self):
        graph = Graph.from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
        assert len(decompose_to_dags(5, graph.src, graph.dst, directions=4)) == 4

    def test_every_decomposed_direction_is_actually_a_dag(self):
        graph = Graph.from_edges(6, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5)])
        for src, dst in decompose_to_dags(6, graph.src, graph.dst):
            assert topological_order(6, src, dst) is not None, "decomposition left a cycle"


class TestTopology:
    def test_knn_connects_each_node_to_its_nearest(self):
        points = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        edges = knn_edges(points, k=1)
        assert len(edges) == 3
        pairs = set(zip(edges.src.tolist(), edges.dst.tolist()))
        assert (0, 1) in pairs and (1, 0) in pairs

    def test_knn_never_connects_a_node_to_itself(self):
        edges = knn_edges(np.random.default_rng(0).standard_normal((10, 3)), k=3)
        assert not (edges.src == edges.dst).any()

    def test_knn_is_deterministic_under_ties(self):
        """Equidistant neighbours must be taken in index order, or two backends build
        different graphs from the same points."""
        points = np.array([[0.0], [1.0], [-1.0]])  # nodes 1 and 2 are equidistant from 0
        first = knn_edges(points, k=1)
        assert np.array_equal(first.dst, knn_edges(points, k=1).dst)

    def test_mst_spans_every_node(self):
        points = np.random.default_rng(1).standard_normal((8, 2))
        edges = mst_edges(points)
        assert len(edges) == 7  # a spanning tree on N nodes has N-1 edges
        assert set(edges.src.tolist()) | set(edges.dst.tolist()) == set(range(8))

    def test_mst_connects_coincident_points_rather_than_dropping_them(self):
        """A zero distance is indistinguishable from 'no edge' in a sparse representation,
        and duplicate feature rows are common."""
        points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        edges = mst_edges(points)
        assert len(edges) == 2
        assert (edges.weight > 0).all(), "a coincident pair was given a zero-weight edge"

    def test_mst_weight_is_minimal_for_a_known_layout(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert mst_edges(points).weight.sum() == pytest.approx(3.0)

    def test_mst_of_a_single_point_is_empty(self):
        assert len(mst_edges(np.array([[0.0, 0.0]]))) == 0

    def test_morton_codes_are_exact_integers(self):
        """Bit arithmetic on quantised coordinates: identical on every platform, with no
        floating-point comparison anywhere in the result."""
        points = np.random.default_rng(2).standard_normal((16, 3))
        codes = morton_codes(points)
        assert codes.dtype == np.int64
        assert np.array_equal(codes, morton_codes(points))

    def test_morton_codes_give_nearby_points_nearby_values(self):
        points = np.array([[0.0, 0.0], [0.01, 0.01], [1.0, 1.0]])
        codes = morton_codes(points)
        assert abs(int(codes[0]) - int(codes[1])) < abs(int(codes[0]) - int(codes[2]))

    def test_build_topology_is_connected_thanks_to_the_mst(self):
        """kNN alone can leave a cluster with no edge out of it, and no amount of message
        passing will ever move information across that gap."""
        nx = pytest.importorskip("networkx")
        points = np.concatenate([
            np.random.default_rng(0).standard_normal((6, 2)),
            np.random.default_rng(1).standard_normal((6, 2)) + 100.0,  # a far-away cluster
        ])
        edges = build_topology(points, k=2, include_mst=True)
        graph = Graph.from_edges(
            12, list(zip(edges.src.tolist(), edges.dst.tolist()))
        ).to_networkx(directed=False)
        assert nx.number_connected_components(graph) == 1

    def test_without_the_mst_the_clusters_stay_separate(self):
        nx = pytest.importorskip("networkx")
        points = np.concatenate([
            np.random.default_rng(0).standard_normal((6, 2)),
            np.random.default_rng(1).standard_normal((6, 2)) + 100.0,
        ])
        edges = build_topology(points, k=2, include_mst=False)
        graph = Graph.from_edges(
            12, list(zip(edges.src.tolist(), edges.dst.tolist()))
        ).to_networkx(directed=False)
        assert nx.number_connected_components(graph) == 2

    def test_build_topology_deduplicates_and_drops_self_loops(self):
        points = np.random.default_rng(3).standard_normal((10, 2))
        edges = build_topology(points, k=3, include_mst=True, window=2)
        pairs = list(zip(edges.src.tolist(), edges.dst.tolist()))
        assert len(pairs) == len(set(pairs)), "duplicate edges survived"
        assert not any(u == v for u, v in pairs), "a self-loop survived"

    def test_build_topology_is_reproducible(self):
        points = np.random.default_rng(4).standard_normal((20, 3))
        first = build_topology(points, k=3)
        second = build_topology(points, k=3)
        assert np.array_equal(first.src, second.src)
        assert np.array_equal(first.dst, second.dst)

    def test_symmetrized_edges_go_both_ways(self):
        edges = knn_edges(np.random.default_rng(5).standard_normal((6, 2)), k=1).symmetrized()
        pairs = set(zip(edges.src.tolist(), edges.dst.tolist()))
        assert all((v, u) in pairs for u, v in pairs)


@needs_both
def test_message_passing_agrees_across_backends():
    """A graph model rebuilt each forward pass depends on this holding exactly."""
    graph_edges = [(0, 1), (1, 2), (0, 3), (3, 2), (2, 0)]
    x = np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)

    results = []
    for name in ("torch", "mlx"):
        backend = get_backend(name)
        graph = Graph.from_edges(4, graph_edges, backend=backend)
        out = propagate(backend, graph, backend.ops.array(x))
        results.append(np.asarray(out))

    assert_forward_parity(results[0], results[1], label="propagate")


@needs_both
def test_resolvent_sweep_agrees_across_backends():
    x = np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)

    results = []
    for name in ("torch", "mlx"):
        backend = get_backend(name)
        graph = Graph.from_edges(4, DIAMOND, backend=backend)
        results.append(np.asarray(resolvent_sweep(backend, graph, backend.ops.array(x))))

    assert_forward_parity(results[0], results[1], label="resolvent_sweep")
