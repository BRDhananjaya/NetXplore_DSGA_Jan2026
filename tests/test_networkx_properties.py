"""
E0 251o Data Structures and Graph Analytics Project: Property-Based Testing for NetworkX

Team Members:
- Dhananjaya B R - SR-No: 13-19-02-19-52-25-1-26121
- Balla Malleswara Rao = SR-No: 13-19-02-19-52-25-1-26254
- Bharath Kannan M - SR-No: 13-19-02-19-52-25-1-26151

Algorithms Tested:
- Shortest Path (Dijkstra)
- Minimum Spanning Tree (MST)
- Connected Components
- Max Flow / Min Cut

Description:
This file contains property-based tests written with Hypothesis for a small
set of core NetworkX graph algorithms. The goal is to check mathematical
properties that should hold across many automatically generated graphs rather
than only a few fixed examples.

This work is submitted as part of E0 251o Data Structures and Graph Analytics course Project 

"""

from itertools import combinations

import networkx as nx
from hypothesis import HealthCheck, given, settings, strategies as st


# -----------------------------
# Graph Generators
# -----------------------------

@st.composite
def connected_graph(draw):
    n = draw(st.integers(min_value=3, max_value=12))
    p = draw(st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False))
    seed = draw(st.integers(min_value=0, max_value=10**6))

    graph = nx.erdos_renyi_graph(n, p, seed=seed)

    if not nx.is_connected(graph):
        components = [list(component) for component in nx.connected_components(graph)]
        for left, right in zip(components, components[1:]):
            graph.add_edge(left[0], right[0])

    return graph


@st.composite
def weighted_graph(draw):
    graph = draw(connected_graph())

    for u, v in graph.edges():
        weight = draw(st.integers(min_value=1, max_value=20))
        graph[u][v]["weight"] = weight
        graph[u][v]["capacity"] = weight

    return graph


@st.composite
def small_weighted_graph(draw):
    n = draw(st.integers(min_value=3, max_value=5))
    graph = nx.complete_graph(n)

    for u, v in graph.edges():
        graph[u][v]["weight"] = draw(st.integers(min_value=1, max_value=10))

    return graph


@st.composite
def directed_flow_graph(draw):
    n = draw(st.integers(min_value=3, max_value=8))
    p = draw(st.floats(min_value=0.2, max_value=0.7, allow_nan=False, allow_infinity=False))
    seed = draw(st.integers(min_value=0, max_value=10**6))

    graph = nx.gnp_random_graph(n, p, seed=seed, directed=True)

    for node in range(n - 1):
        graph.add_edge(node, node + 1)

    for u, v in graph.edges():
        graph[u][v]["capacity"] = draw(st.integers(min_value=1, max_value=20))

    return graph


common_settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)


# -----------------------------
# Helper Functions
# -----------------------------

def path_weight(graph, path):
    return sum(graph[u][v]["weight"] for u, v in zip(path, path[1:]))


def brute_force_mst_weight(graph):
    nodes = list(graph.nodes())
    edge_list = list(graph.edges())
    best_weight = None

    for chosen_edges in combinations(edge_list, len(nodes) - 1):
        candidate = nx.Graph()
        candidate.add_nodes_from(nodes)
        candidate.add_edges_from(chosen_edges)

        if nx.is_tree(candidate):
            total = sum(graph[u][v]["weight"] for u, v in chosen_edges)
            if best_weight is None or total < best_weight:
                best_weight = total

    return best_weight


# -----------------------------
# Shortest Path Tests
# -----------------------------

@given(connected_graph())
@common_settings
def test_triangle_inequality(graph):
    """
    Property: Unweighted shortest-path distance satisfies the triangle inequality.

    Mathematical reasoning:
    In any graph metric, the direct distance from u to v cannot be larger than
    the distance from u to w plus the distance from w to v. If a shorter route
    through w existed, then the supposed shortest path from u to v would not
    really be shortest.

    Graph generation:
    The test draws connected undirected Erdos-Renyi graphs with 3 to 12 nodes.
    If the random graph comes out disconnected, a few extra edges are added to
    join the components so every distance is defined.

    Assumptions:
    The graph is connected and unweighted, so each edge contributes one step to
    the path length.

    Why failure matters:
    A failure would mean NetworkX is returning distances that do not behave like
    a valid graph metric, which would point to a serious bug in shortest-path
    computation.
    """
    nodes = list(graph.nodes())
    u, v, w = nodes[:3]

    assert nx.shortest_path_length(graph, u, v) <= (
        nx.shortest_path_length(graph, u, w) + nx.shortest_path_length(graph, w, v)
    )


@given(connected_graph())
@common_settings
def test_isolated_node_invariance(graph):
    """
    Property: Adding an isolated node does not change distances among old nodes.

    Mathematical reasoning:
    A node with no incident edges cannot appear on any path between two existing
    vertices, so all previously computed shortest-path distances should remain
    unchanged.

    Graph generation:
    The base input is a connected undirected graph. The test then adds one new
    node with no edges and compares a distance before and after the mutation.

    Assumptions:
    The two nodes whose distance is checked already belong to the original graph.

    Why failure matters:
    If the distance changes after adding an isolated node, the algorithm is using
    irrelevant structure when it computes paths.
    """
    nodes = list(graph.nodes())
    source, target = nodes[0], nodes[-1]

    before = nx.shortest_path_length(graph, source, target)
    graph.add_node("isolated")
    after = nx.shortest_path_length(graph, source, target)

    assert before == after


@given(weighted_graph())
@common_settings
def test_shortest_path_weight_scaling(graph):
    """
    Property: Multiplying every edge weight by the same positive constant keeps
    the chosen shortest path unchanged.

    Mathematical reasoning:
    If every path cost is multiplied by the same positive number, the ordering of
    all candidate path costs stays the same. A minimum before scaling should stay
    a minimum after scaling.

    Graph generation:
    The test uses connected weighted graphs with positive integer weights so that
    Dijkstra's assumptions are satisfied.

    Assumptions:
    All edge weights are strictly positive.

    Why failure matters:
    A failure would suggest that path selection depends on the magnitude of the
    weights rather than their relative ordering.
    """
    nodes = list(graph.nodes())
    source, target = nodes[0], nodes[-1]

    first_path = nx.shortest_path(graph, source, target, weight="weight")

    scaled_graph = graph.copy()
    for u, v in scaled_graph.edges():
        scaled_graph[u][v]["weight"] *= 3

    scaled_path = nx.shortest_path(scaled_graph, source, target, weight="weight")

    assert first_path == scaled_path


@given(weighted_graph())
@common_settings
def test_shortest_path_symmetry(graph):
    """
    Property: In an undirected weighted graph, the shortest-path distance from u
    to v equals the distance from v to u.

    Mathematical reasoning:
    Every undirected path can be traversed in reverse with exactly the same edge
    weights, so the optimal cost must be symmetric.

    Graph generation:
    The test draws connected undirected graphs with positive edge weights.

    Assumptions:
    The graph is undirected. This property would not generally hold in directed
    graphs.

    Why failure matters:
    A failure would indicate that the implementation is treating undirected edges
    inconsistently when computing weighted distances.
    """
    nodes = list(graph.nodes())
    source, target = nodes[0], nodes[-1]

    assert nx.shortest_path_length(graph, source, target, weight="weight") == nx.shortest_path_length(
        graph, target, source, weight="weight"
    )


@given(weighted_graph())
@common_settings
def test_shortest_path_subpath_optimality(graph):
    """
    Property: Every consecutive edge on a shortest path is itself a shortest path
    between its two endpoints.

    Mathematical reasoning:
    This is the optimal-substructure idea behind shortest paths. If one edge on
    the chosen route could be replaced by a strictly cheaper path between the same
    endpoints, then the overall route would also become cheaper.

    Graph generation:
    The test uses connected weighted graphs and inspects the path returned by
    NetworkX between two selected nodes.

    Assumptions:
    All weights are positive, and the graph is simple enough that a single edge is
    a valid path between consecutive vertices of the returned route.

    Why failure matters:
    A failure would show that the algorithm returned a route containing a locally
    improvable segment, which contradicts shortest-path optimality.
    """
    nodes = list(graph.nodes())
    source, target = nodes[0], nodes[-1]
    route = nx.shortest_path(graph, source, target, weight="weight")

    for u, v in zip(route, route[1:]):
        assert nx.shortest_path_length(graph, u, v, weight="weight") == graph[u][v]["weight"]


@given(small_weighted_graph())
@settings(max_examples=25, deadline=None)
def test_dijkstra_matches_bruteforce(graph):
    """
    Property: Dijkstra's path length matches the true minimum obtained by checking
    every simple path on small graphs.

    Mathematical reasoning:
    On a finite graph, the shortest path is the minimum total weight over all
    simple source-to-target paths. For very small graphs we can compute that
    minimum directly and compare it to Dijkstra's result.

    Graph generation:
    The test uses complete weighted graphs on 3 to 5 nodes. These are small enough
    for brute-force enumeration but still rich enough to create many competing
    paths.

    Assumptions:
    Edge weights are positive, so Dijkstra is applicable.

    Why failure matters:
    If Dijkstra disagrees with the brute-force answer, then either the library or
    our assumptions about the weight handling are wrong.
    """
    nodes = list(graph.nodes())
    source, target = nodes[0], nodes[-1]

    dijkstra_distance = nx.dijkstra_path_length(graph, source, target, weight="weight")
    brute_force_distance = min(
        path_weight(graph, path) for path in nx.all_simple_paths(graph, source, target)
    )

    assert dijkstra_distance == brute_force_distance


@given(weighted_graph())
@common_settings
def test_shortest_path_consistent_length(graph):
    """
    Property: The reported shortest-path length equals the sum of edge weights on
    the reported shortest path.

    Mathematical reasoning:
    The path object and the distance object are two views of the same result. If
    they disagree, then at least one of them is being computed incorrectly.

    Graph generation:
    Connected weighted graphs are generated, then both the path and the numeric
    distance are requested for the same node pair.

    Assumptions:
    The graph remains unchanged between the two NetworkX calls.

    Why failure matters:
    A mismatch would indicate an inconsistency inside the shortest-path API.
    """
    nodes = list(graph.nodes())
    source, target = nodes[0], nodes[-1]

    route = nx.shortest_path(graph, source, target, weight="weight")
    distance = nx.shortest_path_length(graph, source, target, weight="weight")

    assert distance == path_weight(graph, route)


@given(connected_graph())
@common_settings
def test_adding_edge_does_not_increase_unweighted_distance(graph):
    """
    Property: Adding an undirected edge cannot increase shortest-path distance.

    Mathematical reasoning:
    Adding an edge only adds new candidate routes and never removes existing ones.
    Therefore, the optimal path length between any two fixed nodes can only stay
    the same or decrease.

    Graph generation:
    A connected graph is generated, two existing nodes are selected, and a direct
    edge is added between them before recomputing the distance.

    Assumptions:
    Distances are measured on an unweighted undirected graph where each edge has
    unit cost.

    Why failure matters:
    A failure would indicate that shortest-path computation is not monotonic with
    respect to edge additions, which contradicts a core graph property.
    """
    nodes = list(graph.nodes())
    source, target = nodes[0], nodes[-1]

    before = nx.shortest_path_length(graph, source, target)

    updated_graph = graph.copy()
    updated_graph.add_edge(source, target)
    after = nx.shortest_path_length(updated_graph, source, target)

    assert after <= before


# -----------------------------
# MST Tests
# -----------------------------

@given(weighted_graph())
@common_settings
def test_mst_edge_count(graph):
    """
    Property: A spanning tree on n vertices has exactly n - 1 edges.

    Mathematical reasoning:
    This is one of the basic characterizations of a tree. Fewer edges would leave
    the graph disconnected, while more edges would create a cycle.

    Graph generation:
    The input is a connected weighted graph, and NetworkX is asked to produce its
    minimum spanning tree.

    Assumptions:
    The original graph is connected, so a spanning tree exists.

    Why failure matters:
    A failure would mean the result is not actually a spanning tree.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")

    assert len(tree.edges()) == len(tree.nodes()) - 1


@given(weighted_graph())
@common_settings
def test_mst_acyclic(graph):
    """
    Property: The minimum spanning structure returned by NetworkX is acyclic.

    Mathematical reasoning:
    By definition, a tree contains no cycle. If a cycle were present, at least one
    edge on that cycle could be removed while keeping the graph connected.

    Graph generation:
    The test uses connected weighted graphs and checks the result with the standard
    NetworkX tree predicate.

    Assumptions:
    The spanning structure is computed on an undirected connected graph.

    Why failure matters:
    A cyclic result would contradict the definition of an MST.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")

    assert nx.is_tree(tree)


@given(weighted_graph())
@common_settings
def test_mst_idempotence(graph):
    """
    Property: Taking the minimum spanning tree of an MST should give back the same
    edge set.

    Mathematical reasoning:
    Once the graph is already a tree, there is only one spanning tree available:
    the tree itself. Running the MST algorithm again should therefore not change it.

    Graph generation:
    The test first computes an MST from a connected weighted graph and then applies
    the same operation to the resulting tree.

    Assumptions:
    Edge weights are preserved when the first MST is copied into the second call.

    Why failure matters:
    A failure would suggest that the algorithm does not behave consistently even on
    inputs that are already trees.
    """
    first_tree = nx.minimum_spanning_tree(graph, weight="weight")
    second_tree = nx.minimum_spanning_tree(first_tree, weight="weight")

    assert set(first_tree.edges()) == set(second_tree.edges())


@given(weighted_graph())
@common_settings
def test_mst_edge_removal_disconnects(graph):
    """
    Property: Removing any edge from a spanning tree disconnects it.

    Mathematical reasoning:
    Trees are minimally connected. Every edge is a bridge, so deleting one edge
    must split the tree into exactly two components.

    Graph generation:
    The test computes an MST and then removes one of its edges.

    Assumptions:
    The MST has at least one edge, which is guaranteed because the generated graph
    has at least three nodes.

    Why failure matters:
    If the graph stays connected after an edge removal, then the supposed MST was
    not actually a tree.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    u, v = next(iter(tree.edges()))
    tree.remove_edge(u, v)

    assert not nx.is_connected(tree)


@given(small_weighted_graph())
@settings(max_examples=20, deadline=None)
def test_mst_matches_bruteforce_weight(graph):
    """
    Property: On a small graph, the total weight of the MST matches the true minimum
    found by enumerating all spanning trees.

    Mathematical reasoning:
    A minimum spanning tree is defined as the spanning tree with the smallest total
    edge weight. Since the graphs here are tiny, we can check that definition
    directly instead of trusting the algorithm.

    Graph generation:
    The test uses complete weighted graphs on 3 to 5 nodes so every spanning tree
    candidate exists and can be compared.

    Assumptions:
    The graph is connected and small enough for exhaustive search.

    Why failure matters:
    A failure would be strong evidence that the MST routine is not actually finding
    a minimum-weight tree.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    mst_weight = sum(graph[u][v]["weight"] for u, v in tree.edges())

    assert mst_weight == brute_force_mst_weight(graph)


@given(weighted_graph())
@common_settings
def test_mst_cut_property(graph):
    """
    Property: If a cut has a unique lightest crossing edge, that edge must appear
    in the MST.

    Mathematical reasoning:
    This is the classic cut property for minimum spanning trees. The uniqueness
    condition matters because ties only guarantee that some minimum edge may appear,
    not necessarily a specific one.

    Graph generation:
    The graph is split into two non-empty vertex sets using the node order, and all
    crossing edges are examined.

    Assumptions:
    The chosen cut must have at least one crossing edge and a unique minimum weight.
    If those conditions are not met, the test skips the assertion for that example.

    Why failure matters:
    A failure would violate one of the central theorems used to justify MST
    algorithms such as Kruskal's and Prim's.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")

    nodes = list(graph.nodes())
    left = set(nodes[: len(nodes) // 2])
    right = set(nodes) - left
    crossing_edges = [
        (u, v)
        for u, v in graph.edges()
        if (u in left and v in right) or (u in right and v in left)
    ]

    if not crossing_edges:
        return

    min_weight = min(graph[u][v]["weight"] for u, v in crossing_edges)
    lightest_edges = [(u, v) for u, v in crossing_edges if graph[u][v]["weight"] == min_weight]

    if len(lightest_edges) != 1:
        return

    edge = lightest_edges[0]
    assert edge in tree.edges() or (edge[1], edge[0]) in tree.edges()


@given(weighted_graph())
@common_settings
def test_mst_spans_all_nodes(graph):
    """
    Property: The minimum spanning tree contains every vertex of the original graph.

    Mathematical reasoning:
    A spanning tree is not allowed to drop vertices. It is a tree over the full
    vertex set of the connected graph.

    Graph generation:
    The test builds an MST from a connected weighted graph and compares node sets.

    Assumptions:
    The input graph is connected.

    Why failure matters:
    A failure would mean the algorithm returned a tree, but not a spanning one.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")

    assert set(tree.nodes()) == set(graph.nodes())


@given(weighted_graph())
@common_settings
def test_mst_total_weight_not_more_than_original(graph):
    """
    Property: The MST total weight is no greater than the original graph weight.

    Mathematical reasoning:
    The original connected graph itself is a superset of many spanning structures.
    Since the MST minimizes total spanning-tree cost, its weight cannot exceed the
    sum of all edges in the original graph.

    Graph generation:
    A connected weighted graph is generated, then both the full edge-weight sum
    and the MST edge-weight sum are computed.

    Assumptions:
    All edge weights are positive integers and the graph is connected.

    Why failure matters:
    A heavier MST than the full graph would be mathematically inconsistent and
    would signal a severe problem in MST computation or weight handling.
    """
    tree = nx.minimum_spanning_tree(graph, weight="weight")
    mst_weight = sum(graph[u][v]["weight"] for u, v in tree.edges())
    original_weight = sum(graph[u][v]["weight"] for u, v in graph.edges())

    assert mst_weight <= original_weight


# -----------------------------
# Connected Components
# -----------------------------

@given(connected_graph())
@common_settings
def test_components_single_component(graph):
    """
    Property: A connected graph has exactly one connected component.

    Mathematical reasoning:
    This is the definition of connectivity. If every vertex can reach every other
    vertex, then all vertices must belong to the same component.

    Graph generation:
    The test uses connected undirected graphs generated by the helper strategy.

    Assumptions:
    Connectivity is enforced before the test runs.

    Why failure matters:
    A failure would mean NetworkX is misidentifying the basic connected structure of
    an undirected graph.
    """
    assert len(list(nx.connected_components(graph))) == 1


@given(connected_graph())
@common_settings
def test_all_nodes_reachable(graph):
    """
    Property: The node-connected component of any vertex in a connected graph is
    the entire vertex set.

    Mathematical reasoning:
    In a connected graph, starting from any node should reach every other node via
    some path, so the component containing that node must be the whole graph.

    Graph generation:
    The input is a connected undirected graph of random size and density.

    Assumptions:
    The chosen start node belongs to the graph, and the graph is connected.

    Why failure matters:
    A failure would suggest that the component routine is missing reachable nodes.
    """
    start = next(iter(graph.nodes()))

    assert set(nx.node_connected_component(graph, start)) == set(graph.nodes())


@given(st.integers(min_value=1, max_value=8))
def test_isolated_nodes_form_separate_components(count):
    """
    Property: A graph made of only isolated vertices has one component per vertex.

    Mathematical reasoning:
    With no edges, no vertex can reach any other vertex, so each node must form its
    own singleton component.

    Graph generation:
    The test generates the number of isolated vertices and constructs the graph
    explicitly.

    Assumptions:
    The graph contains nodes but no edges.

    Why failure matters:
    A failure would mean even the boundary case for connected components is handled
    incorrectly.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(count))

    assert len(list(nx.connected_components(graph))) == count


@given(st.integers(min_value=2, max_value=8))
def test_adding_bridge_reduces_component_count_by_one(count):
    """
    Property: Connecting two isolated components reduces component count by one.

    Mathematical reasoning:
    In an undirected graph, adding one edge between two distinct components merges
    exactly those components and leaves all others unchanged.

    Graph generation:
    The test starts from a graph of isolated nodes (one component per node), then
    adds a single edge between nodes 0 and 1.

    Assumptions:
    The graph has at least two nodes so the added edge joins distinct components.

    Why failure matters:
    A failure would mean connected-component updates under edge insertion are not
    behaving according to the graph connectivity definition.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    before = len(list(nx.connected_components(graph)))

    graph.add_edge(0, 1)
    after = len(list(nx.connected_components(graph)))

    assert after == before - 1


# -----------------------------
# Max Flow / Min Cut
# -----------------------------

@given(directed_flow_graph())
@common_settings
def test_maxflow_equals_mincut(graph):
    """
    Property: The maximum s-t flow value equals the minimum s-t cut capacity.

    Mathematical reasoning:
    This is exactly the max-flow min-cut theorem. It is one of the central results
    in network flow theory, so it is a natural property to test.

    Graph generation:
    The test uses directed graphs with positive capacities and an enforced path from
    source 0 to sink n - 1 so the flow problem is always meaningful.

    Assumptions:
    Capacities are non-negative integers and source and sink are distinct.

    Why failure matters:
    A failure would indicate a serious inconsistency between two theoretically
    equivalent NetworkX routines.
    """
    source, sink = 0, max(graph.nodes())
    max_flow_value, _ = nx.maximum_flow(graph, source, sink, capacity="capacity")
    min_cut_value, _ = nx.minimum_cut(graph, source, sink, capacity="capacity")

    assert max_flow_value == min_cut_value


@given(directed_flow_graph())
@common_settings
def test_flow_respects_capacity(graph):
    """
    Property: The flow sent on any directed edge never exceeds that edge's capacity.

    Mathematical reasoning:
    Capacity constraints are part of the definition of a feasible flow. Any edge
    carrying more than its capacity would make the solution invalid.

    Graph generation:
    Random directed graphs with positive integer capacities are generated.

    Assumptions:
    NetworkX returns the flow as a nested dictionary keyed by existing edges.

    Why failure matters:
    A failure would mean the library is producing an infeasible flow.
    """
    source, sink = 0, max(graph.nodes())
    _, flow_dict = nx.maximum_flow(graph, source, sink, capacity="capacity")

    for u, neighbors in flow_dict.items():
        for v, flow_value in neighbors.items():
            if graph.has_edge(u, v):
                assert flow_value <= graph[u][v]["capacity"]


@given(directed_flow_graph())
@common_settings
def test_flow_conservation(graph):
    """
    Property: At every internal vertex, incoming flow equals outgoing flow.

    Mathematical reasoning:
    Flow conservation says that intermediate vertices cannot create or destroy flow.
    Only the source can have net outflow and only the sink can have net inflow.

    Graph generation:
    The test uses directed capacity graphs with a guaranteed source-to-sink path.

    Assumptions:
    The checked vertices are neither the source nor the sink.

    Why failure matters:
    A failure would show that the returned flow violates the definition of a valid
    network flow.
    """
    source, sink = 0, max(graph.nodes())
    _, flow_dict = nx.maximum_flow(graph, source, sink, capacity="capacity")

    for node in graph.nodes():
        if node in (source, sink):
            continue

        inflow = sum(flow_dict[u].get(node, 0) for u in flow_dict)
        outflow = sum(flow_dict[node].values())
        assert inflow == outflow


@given(directed_flow_graph())
@common_settings
def test_increasing_capacity_does_not_reduce_max_flow(graph):
    """
    Property: Increasing capacities cannot reduce the maximum flow value.

    Mathematical reasoning:
    When capacities are increased, the original feasible flow is still feasible, so
    the optimal value after the change must be at least as large as before.

    Graph generation:
    The test starts with a random directed network and then increases the capacity
    of one existing edge.

    Assumptions:
    The graph has at least one edge, which is guaranteed by the generator because a
    source-to-sink chain is always inserted.

    Why failure matters:
    A failure would indicate that the max-flow routine violates a basic monotonicity
    property.
    """
    source, sink = 0, max(graph.nodes())
    before, _ = nx.maximum_flow(graph, source, sink, capacity="capacity")

    updated_graph = graph.copy()
    u, v = next(iter(updated_graph.edges()))
    updated_graph[u][v]["capacity"] += 5

    after, _ = nx.maximum_flow(updated_graph, source, sink, capacity="capacity")

    assert after >= before


@given(directed_flow_graph())
@common_settings
def test_zero_capacities_give_zero_maxflow(graph):
    """
    Property: If all capacities are zero, the maximum flow value is zero.

    Mathematical reasoning:
    With zero capacity on every edge, no positive flow can traverse any arc. The
    only feasible flow is the all-zero flow, so the optimum value must be zero.

    Graph generation:
    A directed flow graph is generated and then copied with every edge capacity
    overwritten to zero before running the max-flow algorithm.

    Assumptions:
    Source and sink are distinct valid nodes in the graph.

    Why failure matters:
    A non-zero result would violate feasibility constraints and indicate a serious
    issue in capacity handling.
    """
    source, sink = 0, max(graph.nodes())
    zero_graph = graph.copy()

    for u, v in zero_graph.edges():
        zero_graph[u][v]["capacity"] = 0

    max_flow_value, _ = nx.maximum_flow(zero_graph, source, sink, capacity="capacity")

    assert max_flow_value == 0


@given(directed_flow_graph())
@common_settings
def test_flow_value_equals_source_outflow(graph):
    """
    Property: Reported max-flow value equals net outflow from the source.

    Mathematical reasoning:
    By definition of flow value, |f| is the total outgoing flow from the source
    minus any incoming flow to the source.

    Graph generation:
    Directed capacity graphs are generated with a guaranteed path from source to
    sink, then NetworkX returns both the scalar value and the flow dictionary.

    Assumptions:
    The flow dictionary includes source adjacency entries for relevant edges.

    Why failure matters:
    If the scalar value disagrees with the represented source flow, the API outputs
    are internally inconsistent.
    """
    source, sink = 0, max(graph.nodes())
    max_flow_value, flow_dict = nx.maximum_flow(graph, source, sink, capacity="capacity")

    source_outflow = sum(flow_dict[source].values())
    source_inflow = sum(flow_dict[u].get(source, 0) for u in flow_dict)

    assert max_flow_value == source_outflow - source_inflow
