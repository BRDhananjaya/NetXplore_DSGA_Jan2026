"""
E0 2510 Project: Property-Based Testing for NetworkX

Team Members:
- Balla Malleswara Rao
- Dhananjaya B R
- Bharath Kannan M

Algorithms Tested:
- Shortest Path (Dijkstra)
- Minimum Spanning Tree (MST)
- Connected Components
- Max Flow / Min Cut

Description:
This project uses property-based testing via Hypothesis to verify
mathematical invariants and correctness properties of graph algorithms
implemented in NetworkX.
"""

import networkx as nx
from hypothesis import given, strategies as st, settings, HealthCheck


# -----------------------------
# Graph Generators
# -----------------------------

@st.composite
def connected_graph(draw):
    n = draw(st.integers(3, 15))
    p = draw(st.floats(0.3, 0.9))

    G = nx.erdos_renyi_graph(n, p)

    if not nx.is_connected(G):
        G = nx.complete_graph(n)

    return G


@st.composite
def weighted_graph(draw):
    G = draw(connected_graph())

    for u, v in G.edges():
        w = draw(st.integers(1, 20))
        G[u][v]["weight"] = w
        G[u][v]["capacity"] = w

    return G


@st.composite
def small_weighted_graph(draw):
    n = draw(st.integers(3, 5))
    G = nx.complete_graph(n)

    for u, v in G.edges():
        G[u][v]["weight"] = draw(st.integers(1, 10))

    return G


common_settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

# -----------------------------
# Shortest Path Tests
# -----------------------------

@given(connected_graph())
@common_settings
def test_triangle_inequality(G):
    nodes = list(G.nodes())
    u, v, w = nodes[:3]

    assert nx.shortest_path_length(G, u, v) <= (
        nx.shortest_path_length(G, u, w) +
        nx.shortest_path_length(G, w, v)
    )


@given(connected_graph())
@common_settings
def test_isolated_node_invariance(G):
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]

    before = nx.shortest_path_length(G, u, v)
    G.add_node("isolated")
    after = nx.shortest_path_length(G, u, v)

    assert before == after


@given(weighted_graph())
@common_settings
def test_shortest_path_weight_scaling(G):
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]

    path1 = nx.shortest_path(G, u, v, weight="weight")

    G2 = G.copy()
    for a, b in G2.edges():
        G2[a][b]["weight"] *= 2

    path2 = nx.shortest_path(G2, u, v, weight="weight")

    assert path1 == path2


@given(weighted_graph())
@common_settings
def test_shortest_path_scaling_only(G):
    """
    Property: Scaling all weights by a positive constant preserves shortest path.

    Mathematical reasoning:
    Multiplication preserves ordering of path costs.
    """
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]

    path1 = nx.shortest_path(G, u, v, weight="weight")

    G2 = G.copy()
    for a, b in G2.edges():
        G2[a][b]["weight"] *= 2

    path2 = nx.shortest_path(G2, u, v, weight="weight")

    assert path1 == path2


@given(weighted_graph())
@common_settings
def test_shortest_path_symmetry(G):
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]

    assert nx.shortest_path_length(G, u, v, weight="weight") == \
           nx.shortest_path_length(G, v, u, weight="weight")


@given(weighted_graph())
@common_settings
def test_shortest_path_subpath_optimality(G):
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]

    path = nx.shortest_path(G, u, v, weight="weight")

    if len(path) >= 2:
        a, b = path[0], path[1]
        assert nx.shortest_path_length(G, a, b, weight="weight") == G[a][b]["weight"]


# -----------------------------
# Dijkstra Validation
# -----------------------------

@given(small_weighted_graph())
@settings(max_examples=20, deadline=None)
def test_dijkstra_matches_bruteforce(G):
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]

    d1 = nx.dijkstra_path_length(G, u, v, weight="weight")

    paths = list(nx.all_simple_paths(G, u, v))
    best = min(
        sum(G[a][b]["weight"] for a, b in zip(p, p[1:]))
        for p in paths
    )

    assert d1 == best


# -----------------------------
# MST Tests
# -----------------------------

@given(weighted_graph())
@common_settings
def test_mst_edge_count(G):
    T = nx.minimum_spanning_tree(G, weight="weight")
    assert len(T.edges()) == len(T.nodes()) - 1


@given(weighted_graph())
@common_settings
def test_mst_acyclic(G):
    T = nx.minimum_spanning_tree(G, weight="weight")
    assert nx.is_tree(T)


@given(weighted_graph())
@common_settings
def test_mst_idempotence(G):
    T1 = nx.minimum_spanning_tree(G, weight="weight")
    T2 = nx.minimum_spanning_tree(T1, weight="weight")
    assert set(T1.edges()) == set(T2.edges())


@given(weighted_graph())
@common_settings
def test_mst_edge_removal_disconnects(G):
    T = nx.minimum_spanning_tree(G, weight="weight")

    if len(T.edges()) > 0:
        u, v = list(T.edges())[0]
        T.remove_edge(u, v)
        assert not nx.is_connected(T)


@given(weighted_graph())
@common_settings
def test_mst_weight_minimality(G):
    T = nx.minimum_spanning_tree(G, weight="weight")
    mst_weight = sum(G[u][v]["weight"] for u, v in T.edges())

    random_tree = nx.minimum_spanning_tree(G)
    random_weight = sum(G[u][v]["weight"] for u, v in random_tree.edges())

    assert mst_weight <= random_weight


@given(weighted_graph())
@common_settings
def test_mst_cut_property(G):
    T = nx.minimum_spanning_tree(G, weight="weight")

    nodes = list(G.nodes())
    A = set(nodes[:len(nodes)//2])
    B = set(nodes) - A

    edges = [(u, v) for u, v in G.edges()
             if (u in A and v in B) or (u in B and v in A)]

    if edges:
        min_edge = min(edges, key=lambda e: G[e[0]][e[1]]["weight"])
        assert min_edge in T.edges() or (min_edge[1], min_edge[0]) in T.edges()


# -----------------------------
# Connected Components
# -----------------------------

@given(connected_graph())
@common_settings
def test_components_single_component(G):
    assert len(list(nx.connected_components(G))) == 1


@given(connected_graph())
@common_settings
def test_all_nodes_reachable(G):
    nodes = list(G.nodes())
    reachable = nx.node_connected_component(G, nodes[0])
    assert set(reachable) == set(G.nodes())


@given(st.integers(1, 5))
def test_single_node_component(_):
    G = nx.Graph()
    G.add_node(0)
    assert len(list(nx.connected_components(G))) == 1


# -----------------------------
# Max Flow / Min Cut
# -----------------------------

@given(weighted_graph())
@common_settings
def test_maxflow_equals_mincut(G):
    nodes = list(G.nodes())
    s, t = nodes[0], nodes[-1]

    flow, _ = nx.maximum_flow(G, s, t, capacity="capacity")
    cut, _ = nx.minimum_cut(G, s, t, capacity="capacity")

    assert flow == cut


@given(weighted_graph())
@common_settings
def test_flow_respects_capacity(G):
    nodes = list(G.nodes())
    s, t = nodes[0], nodes[-1]

    _, flow_dict = nx.maximum_flow(G, s, t, capacity="capacity")

    for u in flow_dict:
        for v in flow_dict[u]:
            assert flow_dict[u][v] <= G[u][v]["capacity"]


@given(weighted_graph())
@common_settings
def test_flow_conservation(G):
    nodes = list(G.nodes())
    s, t = nodes[0], nodes[-1]

    _, flow_dict = nx.maximum_flow(G, s, t, capacity="capacity")

    for node in G.nodes():
        if node not in (s, t):
            inflow = sum(flow_dict[u][node] for u in flow_dict if node in flow_dict[u])
            outflow = sum(flow_dict[node][v] for v in flow_dict[node])
            assert inflow == outflow
@given(weighted_graph())
@common_settings
def test_shortest_path_consistent_length(G):
    """
    Property: Length of shortest path equals sum of edge weights along the path.

    Why this matters:
    Ensures path structure matches computed distance.
    """
    nodes = list(G.nodes())
    u, v = nodes[0], nodes[-1]

    path = nx.shortest_path(G, u, v, weight="weight")

    length = nx.shortest_path_length(G, u, v, weight="weight")

    computed = sum(G[a][b]["weight"] for a, b in zip(path, path[1:]))

    assert length == computed

@given(weighted_graph())
@common_settings
def test_mst_spans_all_nodes(G):
    """
    Property: MST includes all nodes of the graph.

    Why this matters:
    Ensures MST is truly spanning.
    """
    T = nx.minimum_spanning_tree(G, weight="weight")

    assert set(T.nodes()) == set(G.nodes())