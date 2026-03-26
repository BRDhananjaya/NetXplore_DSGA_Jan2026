# Property-Based Testing for NetworkX Graph Algorithms

This repository contains our E0 2510 project.
We used Hypothesis with NetworkX to test graph algorithms by checking properties on many generated graphs instead of testing only a few fixed cases.

## Team Members

- Dhananjaya B R
- Balla Malleswara Rao
- Bharath Kannan M

## What We Tested

- shortest path
- Dijkstra shortest path
- minimum spanning tree
- connected components
- max flow / min cut

## Properties We Checked

- triangle inequality for shortest paths
- shortest path symmetry in undirected graphs
- shortest path remains the same when all weights are scaled equally
- subpaths of a shortest path should also be shortest
- Dijkstra matches brute-force results on small graphs
- MST should be a tree, span all nodes, and satisfy cut/cycle properties
- connected graphs should have one component
- max flow should match min cut and obey capacity limits

## Graph Generation

For most of the tests, we generate connected graphs first and then add extra edges.
For weighted graphs, we assign positive integer weights to every edge.
Those same values are also used as capacities in the flow-related tests.

## Project Files

- [tests/test_networkx_properties.py](tests/test_networkx_properties.py) has the main property-based tests
- [src/netxplore_dsga_jan2026/__init__.py](src/netxplore_dsga_jan2026/__init__.py) is a minimal package file
- [pyproject.toml](pyproject.toml) has the project configuration

## Running the Project

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m pytest
```

If needed, you can also run:

```powershell
python -m ruff check .
```

## Libraries Used

- NetworkX
- Hypothesis
- pytest
- Ruff
