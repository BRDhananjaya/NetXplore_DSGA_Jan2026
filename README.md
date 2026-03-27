# Property-Based Testing for NetworkX Graph Algorithms

## Team Members

- Dhananjaya B R
- Balla Malleswara Rao
- Bharath Kannan M

## Overview

This repository contains a single property-based testing file for selected NetworkX graph algorithms. The tests use Hypothesis to generate many graph instances automatically and then check properties that should always be true from graph theory.

The project focuses on writing tests that read like mathematical claims about the algorithms instead of fixed example-based checks.

## Algorithms Covered

- Shortest path and Dijkstra's algorithm
- Minimum spanning tree
- Connected components
- Maximum flow / minimum cut

## Property Types Used

- Invariants such as triangle inequality, symmetry, and flow conservation
- Postconditions such as MST edge count and spanning behavior
- Metamorphic properties such as weight scaling and adding isolated vertices
- Idempotence for repeated MST computation
- Boundary cases such as graphs with only isolated vertices
- Exhaustive cross-checks on small graphs using brute-force search

## Graph Generation Strategy

The tests generate several kinds of graphs depending on the algorithm:

- Connected undirected random graphs
- Connected weighted graphs with positive integer weights
- Very small weighted complete graphs for brute-force comparison
- Directed capacity graphs for flow and cut properties

The random generators use explicit seeds so they work cleanly with Hypothesis.

## How To Run

```powershell
.venv\Scripts\python.exe -m pytest
```

## Deliverable File

The main submission file is in [tests/test_networkx_properties.py](tests/test_networkx_properties.py).