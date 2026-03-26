# Property-Based Testing for NetworkX Graph Algorithms

## Overview

This project implements property-based testing for graph algorithms using the Hypothesis library and NetworkX.

Unlike traditional unit testing (which uses fixed inputs), property-based testing generates diverse random graphs and verifies mathematical properties that must always hold. This enables systematic testing across a wide range of graph structures, including edge cases that are difficult to anticipate manually.

---

## Algorithms Covered

We implemented property-based tests for the following algorithms:

- Shortest Path (unweighted and Dijkstra)
- Minimum Spanning Tree (MST)
- Connected Components
- Max Flow / Min Cut

---

## Property-Based Testing Approach

Instead of verifying outputs for specific inputs, we validate **fundamental algorithmic properties** derived from graph theory.

### Types of Properties Tested

#### 1. Invariants
Properties that always hold for valid inputs.

- Triangle inequality for shortest paths  
- Flow conservation in networks  

---

#### 2. Postconditions
Properties that must hold for outputs.

- MST has exactly (n − 1) edges  
- MST is acyclic (tree structure)  

---

#### 3. Metamorphic Properties
Relationships between outputs under transformations.

- Scaling edge weights does not change shortest path structure  
- Adding constant weight preserves shortest path  
- Adding isolated nodes does not affect distances  

---

#### 4. Idempotence
Repeated application yields same result.

- MST(MST(G)) = MST(G)

---

#### 5. Boundary Conditions
Edge cases that must be handled correctly.

- Single node graph  
- Small graphs for brute-force validation  

---

## Advanced Properties Implemented

To strengthen correctness guarantees, we included advanced theoretical properties:

- **Optimal substructure** of shortest paths  
- **Symmetry of shortest paths** in undirected graphs  
- **MST cut property**  
- **MST minimality (weight optimality)**  
- **Max-flow min-cut theorem**  
- **Flow capacity constraints**  
- **Flow conservation law**  

These properties ensure deeper validation beyond basic correctness.

---

## Graph Generation Strategy

We used Hypothesis to generate diverse graph inputs:

- Random graphs using Erdős–Rényi model  
- Connected graphs (forced connectivity)  
- Weighted graphs with positive weights  
- Small graphs for brute-force verification  

This ensures coverage across:

- Different sizes  
- Different densities  
- Various topologies  

---

## Key Insight

Property-based testing shifts focus from:

"Does this output match expected value?"  

to  

"What must always be true for any valid input?"

This approach provides stronger guarantees of correctness and can reveal subtle bugs.

---

## Tools Used

- Hypothesis (property-based testing)
- NetworkX (graph algorithms)
- pytest (test execution)

---

## Project Structure
tests/
└── test_networkx_properties.py