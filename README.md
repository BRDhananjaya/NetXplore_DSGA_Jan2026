# Property-Based Testing for NetworkX Graph Algorithms

## Overview

This repository uses Hypothesis and NetworkX to validate graph-algorithm behavior through mathematical properties rather than fixed example inputs. The test suite focuses on structural invariants, metamorphic properties, and theorem-level guarantees across randomly generated connected graphs.

## Team Members

- Dhananjaya B R
- Balla Malleswara Rao
- Bharath Kannan M

## Algorithms Covered

- Shortest path properties for unweighted and weighted graphs
- Dijkstra correctness against brute-force path enumeration
- Minimum spanning tree structure and optimality properties
- Connected-component behavior on connected and trivial graphs
- Max-flow and min-cut consistency with capacity constraints

## Properties Verified

- Triangle inequality for unweighted shortest paths
- Shortest-path symmetry in undirected weighted graphs
- Shortest-path stability under positive weight scaling
- Optimality of shortest-path subpaths
- Agreement between Dijkstra and brute-force search on small graphs
- MST edge-count, spanning, idempotence, cut, and cycle properties
- Single-component reachability for connected graphs
- Max-flow min-cut equality, flow conservation, and capacity bounds

## Graph Generation Strategy

The tests use Hypothesis-native graph builders instead of NetworkX random graph helpers. Each connected graph is constructed from a guaranteed spanning path plus a random set of extra edges, which keeps the strategies deterministic and avoids Hypothesis warnings about hidden randomness.

Weighted graphs assign positive integer edge weights and reuse those values as capacities for the flow tests.

## Setup

Create and activate a virtual environment, then install the project in editable mode with development dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

If you prefer a minimal install, [requirements.txt](requirements.txt) includes the core packages needed to run the tests.

## Running the Suite

```powershell
python -m pytest
```

Optional linting:

```powershell
ruff check .
```

## Project Structure

```text
src/
	netxplore_dsga_jan2026/
		__init__.py
tests/
	test_networkx_properties.py
pyproject.toml
README.md
requirements.txt
```

## Notes

The package under [src/netxplore_dsga_jan2026/__init__.py](src/netxplore_dsga_jan2026/__init__.py) is intentionally minimal. The main deliverable is the property-based test suite in [tests/test_networkx_properties.py](tests/test_networkx_properties.py).
