"""Team members: BRDhananjaya.

Algorithms tested:
- Dijkstra shortest path length
- Minimum spanning tree
- Clustering coefficient
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st


def edge_set(graph: nx.Graph) -> set[frozenset[int]]:
    """Return undirected edges in a hashable form for graph comparisons."""
    return {frozenset((u, v)) for u, v in graph.edges()}


def path_weight(graph: nx.Graph | nx.DiGraph, path: list[int]) -> int:
    """Compute the total weight of a node path."""
    return sum(graph[u][v]["weight"] for u, v in zip(path, path[1:]))


@st.composite
def connected_weighted_graphs(
    draw: st.DrawFn,
    *,
    min_nodes: int = 2,
    max_nodes: int = 7,
    distinct_weights: bool = False,
) -> nx.Graph:
    """Generate a connected undirected graph with positive integer edge weights."""
    node_count = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    tree_edges: list[tuple[int, int]] = []
    for node in range(1, node_count):
        parent = draw(st.integers(min_value=0, max_value=node - 1))
        tree_edges.append((parent, node))

    tree_edge_keys = {tuple(sorted(edge)) for edge in tree_edges}
    candidate_edges = [
        (u, v)
        for u in range(node_count)
        for v in range(u + 1, node_count)
        if (u, v) not in tree_edge_keys
    ]
    max_extra_edges = min(len(candidate_edges), node_count + 1)
    if candidate_edges:
        extra_edges = draw(
            st.lists(
                st.sampled_from(candidate_edges),
                unique=True,
                max_size=max_extra_edges,
            )
        )
    else:
        extra_edges: list[tuple[int, int]] = []

    all_edges = tree_edges + extra_edges
    if distinct_weights:
        for index, (u, v) in enumerate(sorted(all_edges), start=1):
            graph.add_edge(u, v, weight=index)
    else:
        weights = draw(
            st.lists(
                st.integers(min_value=1, max_value=25),
                min_size=len(all_edges),
                max_size=len(all_edges),
            )
        )
        for (u, v), weight in zip(all_edges, weights):
            graph.add_edge(u, v, weight=weight)

    return graph


@st.composite
def connected_weighted_graph_cases(draw: st.DrawFn) -> tuple[nx.Graph, int, int]:
    """Generate a connected weighted graph together with distinct terminals."""
    graph = draw(connected_weighted_graphs(min_nodes=2, max_nodes=6))
    source = draw(st.integers(min_value=0, max_value=graph.number_of_nodes() - 1))
    target_index = draw(st.integers(min_value=0, max_value=graph.number_of_nodes() - 2))
    target = target_index if target_index < source else target_index + 1
    return graph, source, target


@st.composite
def reachable_weighted_digraph_cases(draw: st.DrawFn) -> tuple[nx.DiGraph, int, int]:
    """Generate a directed weighted graph with a guaranteed path from source to target."""
    node_count = draw(st.integers(min_value=2, max_value=7))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))

    base_edges = [(node, node + 1) for node in range(node_count - 1)]
    candidate_edges = [
        (u, v)
        for u in range(node_count)
        for v in range(node_count)
        if u != v and (u, v) not in base_edges
    ]
    max_extra_edges = min(len(candidate_edges), node_count + 2)
    extra_edges = draw(
        st.lists(st.sampled_from(candidate_edges), unique=True, max_size=max_extra_edges)
    )
    all_edges = base_edges + extra_edges

    weights = draw(
        st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=len(all_edges),
            max_size=len(all_edges),
        )
    )
    for (u, v), weight in zip(all_edges, weights):
        graph.add_edge(u, v, weight=weight)

    source = draw(st.integers(min_value=0, max_value=node_count - 2))
    target = draw(st.integers(min_value=source + 1, max_value=node_count - 1))
    return graph, source, target


@st.composite
def random_trees(draw: st.DrawFn) -> nx.Graph:
    """Generate a random labeled tree with at least three vertices."""
    node_count = draw(st.integers(min_value=3, max_value=10))
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    for node in range(1, node_count):
        parent = draw(st.integers(min_value=0, max_value=node - 1))
        graph.add_edge(node, parent)
    return graph


def all_clustering_values(graph: nx.Graph) -> Iterable[float]:
    """Return clustering coefficients for all vertices in a graph."""
    return nx.clustering(graph).values()


@settings(deadline=None, max_examples=60)
@given(connected_weighted_graphs(min_nodes=2, max_nodes=8))
def test_mst_has_n_minus_1_edges(graph: nx.Graph) -> None:
    """
    Property: A minimum spanning tree of a connected graph with n vertices has exactly n-1 edges.

    Mathematical basis: Every tree on n vertices has n-1 edges, and every spanning tree is by
    definition both connected and acyclic while containing all original vertices. Therefore, if a
    minimum spanning tree algorithm is correct, its output must satisfy this structural identity no
    matter how the connected weighted input graph is generated.

    Test strategy: Hypothesis generates connected undirected graphs of varying sizes and densities.
    Connectivity is guaranteed by first building a random tree and then optionally adding extra
    weighted edges. The test runs NetworkX's minimum spanning tree algorithm and verifies the edge
    count against the theorem.

    Assumptions: Edge weights are positive integers, but the exact weight values do not matter for
    the n-1 result. The only essential precondition is that the input graph is connected.

    Why this matters: If the spanning tree has too few edges, some vertices were not connected. If
    it has too many edges, then a cycle remains and the result is not a tree. Either outcome would
    indicate a correctness bug in the implementation or an invalid interpretation of the API.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    assert tree.number_of_edges() == graph.number_of_nodes() - 1


@settings(deadline=None, max_examples=60)
@given(connected_weighted_graphs(min_nodes=2, max_nodes=8))
def test_mst_spans_all_original_vertices(graph: nx.Graph) -> None:
    """
    Property: A minimum spanning tree must include every vertex from the original connected graph
    and itself be a tree.

    Mathematical basis: A spanning tree is not merely any low-weight subgraph. It must span the
    entire vertex set and contain no cycles. These are the defining postconditions of the problem,
    independent of algorithm design details such as Kruskal or Prim style edge selection.

    Test strategy: Hypothesis again constructs connected weighted graphs with random topology. The
    test checks two postconditions on the NetworkX result: the node set is unchanged, and the output
    satisfies the tree predicate.

    Assumptions: Inputs are simple undirected connected graphs. No assumptions are made about edge
    uniqueness or graph density.

    Why this matters: An implementation could accidentally optimize weight while dropping a vertex or
    leaving a cycle behind. This test catches both classes of defect and complements the simpler
    edge-count property with a direct structural validation.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    assert set(tree.nodes()) == set(graph.nodes())
    assert nx.is_tree(tree)


@settings(deadline=None, max_examples=60)
@given(
    connected_weighted_graphs(min_nodes=2, max_nodes=8, distinct_weights=True),
    st.integers(min_value=1, max_value=10),
)
def test_positive_weight_scaling_preserves_mst_edges(graph: nx.Graph, factor: int) -> None:
    """
    Property: Multiplying every edge weight by the same positive constant preserves the minimum
    spanning tree whenever all original edge weights are distinct.

    Mathematical basis: Distinct edge weights imply that the minimum spanning tree is unique. A
    positive scaling factor preserves the total ordering of all edge weights, so every comparison the
    algorithm relies on is unchanged. The unique optimal tree must therefore remain the same.

    Test strategy: The generator creates connected graphs whose edge weights are guaranteed to be
    distinct. The test computes the MST once, rescales every edge weight by a positive integer, and
    computes the MST again. It then compares the resulting edge sets.

    Assumptions: Distinct weights are required because ties can legitimately lead to multiple valid
    minimum spanning trees with different edge sets. The scaling factor must be positive so weight
    order is preserved.

    Why this matters: This is a metamorphic property. If it fails, the algorithm may be relying on
    absolute weight magnitudes rather than order, or it may mishandle weighted comparisons during
    optimization.
    """
    original_tree = nx.minimum_spanning_tree(graph, weight="weight")

    scaled_graph = graph.copy()
    for u, v, data in scaled_graph.edges(data=True):
        data["weight"] *= factor

    scaled_tree = nx.minimum_spanning_tree(scaled_graph, weight="weight")
    assert edge_set(original_tree) == edge_set(scaled_tree)


@settings(deadline=None, max_examples=50)
@given(connected_weighted_graph_cases())
def test_dijkstra_matches_minimum_simple_path_weight(case: tuple[nx.Graph, int, int]) -> None:
    """
    Property: The Dijkstra shortest-path distance between two vertices equals the minimum total
    weight among all simple paths between those vertices when all edge weights are positive.

    Mathematical basis: In graphs with non-negative weights, an optimal walk can always be reduced
    to a simple path by removing cycles that only add non-negative cost. Therefore the shortest-path
    distance must equal the minimum weight over the finite set of simple source-to-target paths.

    Test strategy: Hypothesis generates small connected weighted graphs and a pair of distinct
    terminals. The graph size is intentionally capped so all simple paths can be enumerated without
    exploding combinatorially. The test compares NetworkX's Dijkstra result against the explicit
    minimum path weight computed from the full simple-path set.

    Assumptions: The graph is connected and all weights are strictly positive integers, satisfying
    Dijkstra's preconditions.

    Why this matters: This property checks the semantic definition of shortest path directly. A
    failure would indicate either an incorrect distance calculation or a mismatch between the API and
    the mathematical shortest-path problem.
    """
    graph, source, target = case
    shortest_distance = nx.dijkstra_path_length(graph, source, target, weight="weight")
    simple_path_weights = [
        path_weight(graph, path) for path in nx.all_simple_paths(graph, source=source, target=target)
    ]
    assert shortest_distance == min(simple_path_weights)


@settings(deadline=None, max_examples=50)
@given(connected_weighted_graph_cases())
def test_adding_isolated_node_preserves_existing_shortest_paths(
    case: tuple[nx.Graph, int, int]
) -> None:
    """
    Property: Adding an isolated vertex to a graph does not change shortest-path distances between
    vertices that were already present.

    Mathematical basis: An isolated vertex participates in no edges, so it cannot create a new path
    or alter the weight of any existing path between old vertices. The metric on the original vertex
    set should therefore remain unchanged.

    Test strategy: Hypothesis builds a connected weighted graph and chooses two existing terminals.
    The test measures their Dijkstra distance, adds one brand-new isolated node, and measures the
    same distance again.

    Assumptions: The chosen terminals lie in the original connected graph. The added vertex is truly
    isolated and has no incident edges.

    Why this matters: This is a clean metamorphic check for unwanted global side effects. If adding a
    disconnected node changes a pre-existing distance, the shortest-path implementation is depending
    on unrelated graph state and is almost certainly wrong.
    """
    graph, source, target = case
    baseline_distance = nx.dijkstra_path_length(graph, source, target, weight="weight")

    extended_graph = graph.copy()
    extended_graph.add_node(graph.number_of_nodes())

    assert (
        nx.dijkstra_path_length(extended_graph, source, target, weight="weight")
        == baseline_distance
    )


@settings(deadline=None, max_examples=50)
@given(reachable_weighted_digraph_cases())
def test_reversing_edges_reverses_directed_shortest_path_distances(
    case: tuple[nx.DiGraph, int, int]
) -> None:
    """
    Property: In a directed weighted graph, reversing every edge preserves shortest-path distance if
    the source and target are swapped.

    Mathematical basis: Every directed path from s to t in the original graph corresponds to a path
    from t to s in the reversed graph with exactly the same sequence of edge weights in reverse order.
    This defines a weight-preserving bijection between candidate paths, so the optimal distance must
    also be preserved.

    Test strategy: The generator creates a directed graph that is guaranteed to contain at least one
    source-to-target path by embedding a forward chain and then adding random extra weighted arcs.
    The test compares the Dijkstra distance in the original graph with the swapped-terminal distance
    in the reversed graph.

    Assumptions: Edge weights are positive integers and the chosen source can reach the chosen target
    in the original graph.

    Why this matters: This metamorphic property validates directional semantics. If it fails, the
    implementation may mishandle edge orientation or compute incorrect costs after graph reversal.
    """
    graph, source, target = case
    original_distance = nx.dijkstra_path_length(graph, source, target, weight="weight")
    reversed_distance = nx.dijkstra_path_length(
        graph.reverse(copy=True), target, source, weight="weight"
    )
    assert original_distance == reversed_distance


@settings(deadline=None, max_examples=50)
@given(st.integers(min_value=3, max_value=12))
def test_complete_graph_has_clustering_one(node_count: int) -> None:
    """
    Property: Every vertex in a complete graph with at least three vertices has clustering
    coefficient 1.

    Mathematical basis: In a complete graph, every pair of neighbors of a vertex is also adjacent.
    The local clustering coefficient is therefore the ratio of existing neighbor-neighbor edges to all
    possible neighbor-neighbor edges, which is exactly 1 for every vertex.

    Test strategy: Hypothesis varies the size of the complete graph and checks the clustering value
    for each vertex returned by NetworkX.

    Assumptions: The graph has at least three vertices so the coefficient is defined in the usual
    non-degenerate sense.

    Why this matters: This is a strong boundary-case oracle for clustering. Any value below 1 would
    show that the algorithm failed to recognize the densest possible local neighborhood structure.
    """
    graph = nx.complete_graph(node_count)
    assert all(value == 1.0 for value in all_clustering_values(graph))


@settings(deadline=None, max_examples=50)
@given(random_trees())
def test_tree_has_zero_clustering_everywhere(graph: nx.Graph) -> None:
    """
    Property: Every vertex in a tree has clustering coefficient 0.

    Mathematical basis: Trees contain no cycles, and in particular they contain no triangles. The
    clustering coefficient at a vertex measures how many pairs of its neighbors are connected to each
    other. In a tree, connecting two neighbors of the same vertex would create a cycle, so that count
    must always be zero.

    Test strategy: Hypothesis generates random labeled trees by attaching each new vertex to one
    earlier vertex, which guarantees acyclicity and connectivity. The test then checks the clustering
    coefficient reported for every vertex.

    Assumptions: The input is a simple undirected tree with at least three vertices.

    Why this matters: This property validates that the clustering implementation correctly detects the
    absence of triangles across a broad family of sparse graphs, not just a single hard-coded tree.
    """
    assert all(value == 0.0 for value in all_clustering_values(graph))