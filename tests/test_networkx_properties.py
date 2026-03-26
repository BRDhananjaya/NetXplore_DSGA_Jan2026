"""Tests written for the E0 2510 NetworkX property-based testing project.

The idea here is simple: generate graphs using Hypothesis and check whether
important properties still hold for different algorithms.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
from hypothesis import HealthCheck, given, settings, strategies as st


def _build_connected_graph(node_count: int, extra_edges: list[tuple[int, int]]) -> nx.Graph:
    """Build a connected graph from a simple path plus extra edges."""
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from((node, node + 1) for node in range(node_count - 1))
    graph.add_edges_from(extra_edges)
    return graph


def _assign_weights(graph: nx.Graph, weights: list[int]) -> nx.Graph:
    """Add weight and capacity values to every edge in a copied graph."""
    weighted = graph.copy()
    for (u, v), weight in zip(weighted.edges(), weights, strict=True):
        weighted[u][v]["weight"] = weight
        weighted[u][v]["capacity"] = weight
    return weighted


def _path_weight(graph: nx.Graph, path: list[int]) -> int:
    """Return the total weight along a path."""
    return sum(graph[u][v]["weight"] for u, v in zip(path, path[1:]))


def _scaled_weighted_copy(graph: nx.Graph, factor: int) -> nx.Graph:
    """Return a copy of the graph with all weights multiplied by a factor."""
    scaled = graph.copy()
    for u, v in scaled.edges():
        scaled[u][v]["weight"] *= factor
    return scaled


def _tree_edge_key(u: int, v: int) -> frozenset[int]:
    """Store an undirected edge in a consistent way."""
    return frozenset((u, v))


def _minimum_spanning_tree(graph: nx.Graph) -> nx.Graph:
    """Return the minimum spanning tree using the weight field."""
    return nx.minimum_spanning_tree(graph, weight="weight")


@st.composite
def connected_graph(draw, min_nodes: int = 3, max_nodes: int = 15) -> nx.Graph:
    """Generate a connected graph."""
    node_count = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    non_tree_edges = [
        edge
        for edge in combinations(range(node_count), 2)
        if edge[1] != edge[0] + 1
    ]
    extra_edges = draw(st.lists(st.sampled_from(non_tree_edges), unique=True))
    return _build_connected_graph(node_count, extra_edges)


@st.composite
def weighted_graph(draw, min_nodes: int = 3, max_nodes: int = 15) -> nx.Graph:
    """Generate a connected graph with weights and capacities on each edge."""
    graph = draw(connected_graph(min_nodes=min_nodes, max_nodes=max_nodes))
    weights = draw(
        st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=graph.number_of_edges(),
            max_size=graph.number_of_edges(),
        )
    )
    return _assign_weights(graph, weights)


@st.composite
def small_weighted_graph(draw) -> nx.Graph:
    """Generate a small weighted graph for brute-force shortest-path checks."""
    node_count = draw(st.integers(min_value=3, max_value=5))
    graph = nx.complete_graph(node_count)
    weights = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=graph.number_of_edges(),
            max_size=graph.number_of_edges(),
        )
    )
    return _assign_weights(graph, weights)


common_settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(connected_graph())
@common_settings
def test_triangle_inequality(graph: nx.Graph) -> None:
    """Shortest-path distances should satisfy triangle inequality."""
    nodes = list(graph.nodes())
    u, v, w = nodes[:3]

    assert nx.shortest_path_length(graph, u, v) <= (
        nx.shortest_path_length(graph, u, w) + nx.shortest_path_length(graph, w, v)
    )


@given(connected_graph())
@common_settings
def test_isolated_node_invariance(graph: nx.Graph) -> None:
    """Adding an isolated node should not affect old shortest-path distances."""
    nodes = list(graph.nodes())
    start, end = nodes[0], nodes[-1]

    before = nx.shortest_path_length(graph, start, end)
    graph.add_node("isolated")
    after = nx.shortest_path_length(graph, start, end)

    assert before == after


@given(weighted_graph())
@common_settings
def test_shortest_path_weight_scaling(graph: nx.Graph) -> None:
    """Scaling all weights equally should keep the shortest path unchanged."""
    nodes = list(graph.nodes())
    start, end = nodes[0], nodes[-1]

    original_path = nx.shortest_path(graph, start, end, weight="weight")
    scaled_graph = _scaled_weighted_copy(graph, factor=2)
    scaled_path = nx.shortest_path(scaled_graph, start, end, weight="weight")

    assert original_path == scaled_path


@given(weighted_graph())
@common_settings
def test_shortest_path_symmetry(graph: nx.Graph) -> None:
    """Shortest-path distance should be symmetric in an undirected graph."""
    nodes = list(graph.nodes())
    start, end = nodes[0], nodes[-1]

    assert nx.shortest_path_length(graph, start, end, weight="weight") == nx.shortest_path_length(
        graph, end, start, weight="weight"
    )


@given(weighted_graph())
@common_settings
def test_shortest_path_subpaths_are_optimal(graph: nx.Graph) -> None:
    """Subpaths of a shortest path should also be shortest."""
    nodes = list(graph.nodes())
    start, end = nodes[0], nodes[-1]
    path = nx.shortest_path(graph, start, end, weight="weight")

    for left in range(len(path) - 1):
        for right in range(left + 1, len(path)):
            subpath = path[left : right + 1]
            assert nx.shortest_path_length(
                graph,
                subpath[0],
                subpath[-1],
                weight="weight",
            ) == _path_weight(graph, subpath)


@given(weighted_graph())
@common_settings
def test_shortest_path_consistent_length(graph: nx.Graph) -> None:
    """The reported path length should match the sum of weights on that path."""
    nodes = list(graph.nodes())
    start, end = nodes[0], nodes[-1]
    path = nx.shortest_path(graph, start, end, weight="weight")
    length = nx.shortest_path_length(graph, start, end, weight="weight")

    assert length == _path_weight(graph, path)


@given(small_weighted_graph())
@settings(max_examples=20, deadline=None)
def test_dijkstra_matches_bruteforce(graph: nx.Graph) -> None:
    """Dijkstra should match brute-force search on small graphs."""
    nodes = list(graph.nodes())
    start, end = nodes[0], nodes[-1]

    dijkstra_length = nx.dijkstra_path_length(graph, start, end, weight="weight")
    brute_force_length = min(
        _path_weight(graph, path)
        for path in nx.all_simple_paths(graph, start, end)
    )

    assert dijkstra_length == brute_force_length


@given(weighted_graph())
@common_settings
def test_mst_edge_count(graph: nx.Graph) -> None:
    """An MST should have exactly n - 1 edges."""
    tree = _minimum_spanning_tree(graph)
    assert tree.number_of_edges() == tree.number_of_nodes() - 1


@given(weighted_graph())
@common_settings
def test_mst_acyclic(graph: nx.Graph) -> None:
    """The MST should not contain any cycle."""
    assert nx.is_tree(_minimum_spanning_tree(graph))


@given(weighted_graph())
@common_settings
def test_mst_idempotence(graph: nx.Graph) -> None:
    """Running MST again on an MST should not change it."""
    tree = _minimum_spanning_tree(graph)
    assert set(tree.edges()) == set(_minimum_spanning_tree(tree).edges())


@given(weighted_graph())
@common_settings
def test_mst_edge_removal_disconnects(graph: nx.Graph) -> None:
    """Removing an edge from the MST should disconnect it."""
    tree = _minimum_spanning_tree(graph)
    removed_edge = next(iter(tree.edges()))
    tree.remove_edge(*removed_edge)
    assert not nx.is_connected(tree)


@given(weighted_graph())
@common_settings
def test_mst_spans_all_nodes(graph: nx.Graph) -> None:
    """The MST should still include all nodes from the original graph."""
    tree = _minimum_spanning_tree(graph)
    assert set(tree.nodes()) == set(graph.nodes())


@given(weighted_graph())
@common_settings
def test_mst_cycle_property(graph: nx.Graph) -> None:
    """Check the usual cycle property of the MST."""
    tree = _minimum_spanning_tree(graph)
    tree_edges = {_tree_edge_key(u, v) for u, v in tree.edges()}

    for u, v in graph.edges():
        if _tree_edge_key(u, v) in tree_edges:
            continue

        path = nx.shortest_path(tree, u, v)
        max_tree_weight = max(
            graph[a][b]["weight"] for a, b in zip(path, path[1:])
        )
        assert graph[u][v]["weight"] >= max_tree_weight


@given(weighted_graph())
@common_settings
def test_mst_cut_property(graph: nx.Graph) -> None:
    """Check the cut property for the MST."""
    tree = _minimum_spanning_tree(graph)
    nodes = list(graph.nodes())
    left_partition = set(nodes[: len(nodes) // 2])
    right_partition = set(nodes) - left_partition
    crossing_edges = [
        (u, v)
        for u, v in graph.edges()
        if (u in left_partition and v in right_partition)
        or (u in right_partition and v in left_partition)
    ]

    min_crossing_edge = min(
        crossing_edges,
        key=lambda edge: graph[edge[0]][edge[1]]["weight"],
    )
    assert _tree_edge_key(*min_crossing_edge) in {
        _tree_edge_key(u, v) for u, v in tree.edges()
    }


@given(connected_graph())
@common_settings
def test_components_single_component(graph: nx.Graph) -> None:
    """A generated connected graph should have only one component."""
    assert len(list(nx.connected_components(graph))) == 1


@given(connected_graph())
@common_settings
def test_all_nodes_reachable(graph: nx.Graph) -> None:
    """All nodes should be reachable in a connected graph."""
    reachable = nx.node_connected_component(graph, next(iter(graph.nodes())))
    assert set(reachable) == set(graph.nodes())


def test_single_node_component() -> None:
    """A graph with one node should still be one component."""
    graph = nx.Graph()
    graph.add_node(0)
    assert len(list(nx.connected_components(graph))) == 1


@given(weighted_graph())
@common_settings
def test_maxflow_equals_mincut(graph: nx.Graph) -> None:
    """Max flow should be equal to min cut."""
    nodes = list(graph.nodes())
    source, sink = nodes[0], nodes[-1]
    flow_value, _ = nx.maximum_flow(graph, source, sink, capacity="capacity")
    cut_value, _ = nx.minimum_cut(graph, source, sink, capacity="capacity")

    assert flow_value == cut_value


@given(weighted_graph())
@common_settings
def test_flow_respects_capacity(graph: nx.Graph) -> None:
    """Flow on any edge should not cross its capacity."""
    nodes = list(graph.nodes())
    source, sink = nodes[0], nodes[-1]
    _, flow_dict = nx.maximum_flow(graph, source, sink, capacity="capacity")

    for u, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if graph.has_edge(u, v):
                assert flow <= graph[u][v]["capacity"]


@given(weighted_graph())
@common_settings
def test_flow_conservation(graph: nx.Graph) -> None:
    """Intermediate nodes should satisfy flow conservation."""
    nodes = list(graph.nodes())
    source, sink = nodes[0], nodes[-1]
    _, flow_dict = nx.maximum_flow(graph, source, sink, capacity="capacity")

    for node in graph.nodes():
        if node in (source, sink):
            continue

        inflow = sum(neighbors.get(node, 0) for neighbors in flow_dict.values())
        outflow = sum(flow_dict[node].values())
        assert inflow == outflow
    