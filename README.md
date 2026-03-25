# NetXplore_DSGA_Jan2026

Property-based testing project for NetworkX graph algorithms using Hypothesis.

## Focus

This repository is set up for the course project on property-based testing.
The main deliverable is a single Python test file:

- `tests/test_networkx_properties.py`

It contains:

- all imports needed to run the tests
- helper functions and custom Hypothesis strategies for graph generation
- detailed property docstrings for each test
- property-based tests for multiple NetworkX algorithms

Algorithms currently covered:

- Dijkstra shortest path length
- minimum spanning tree
- clustering coefficient

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Test

```powershell
pytest
```

## Suggested next steps

1. Add more properties for connectivity, flow, or centrality algorithms.
2. Increase graph diversity with additional directed and weighted strategies.
3. Record any failing Hypothesis examples as evidence if a genuine NetworkX bug is found.
