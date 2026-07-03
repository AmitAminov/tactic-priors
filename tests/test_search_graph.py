"""Graph utilities: best-first minimum-cost path on toy graphs with known answers."""

from __future__ import annotations

import networkx as nx
import numpy as np

from tactic_priors.search import (
    LEAN_DOJO_AVAILABLE,
    best_first_min_cost_path_to_leaf,
    dfs_find_path,
)


def weighted_digraph(edges):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(edges)
    return graph


def test_min_cost_path_prefers_cheaper_leaf():
    # 0 -> 1 -> 3 costs 1 + 1 = 2; 0 -> 2 costs 5. Leaves are 3 and 2.
    graph = weighted_digraph([(0, 1, 1.0), (1, 3, 1.0), (0, 2, 5.0)])
    assert best_first_min_cost_path_to_leaf(graph) == [0, 1, 3]


def test_min_cost_path_accumulates_costs():
    # Direct edge to a leaf (cost 4) beats the two-step path (cost 2 + 3 = 5).
    graph = weighted_digraph([(0, 1, 2.0), (1, 2, 3.0), (0, 3, 4.0)])
    assert best_first_min_cost_path_to_leaf(graph) == [0, 3]


def test_min_cost_path_diamond():
    # Two routes to the same leaf; the cheaper one must be returned.
    graph = weighted_digraph([(0, 1, 1.0), (0, 2, 2.0), (1, 3, 5.0), (2, 3, 1.0)])
    assert best_first_min_cost_path_to_leaf(graph) == [0, 2, 3]


def test_single_node_graph_returns_root():
    graph = nx.DiGraph()
    graph.add_node(0)
    assert best_first_min_cost_path_to_leaf(graph) == [0]


def test_dfs_find_path_reconstructs_proof_path():
    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    path = dfs_find_path(adjacency, 0, 3)
    assert path == [0, 1, 3]


def test_dfs_find_path_returns_none_when_unreachable():
    adjacency = np.array([[0, 1], [0, 0]])
    assert dfs_find_path(adjacency, 1, 0) is None


def test_search_module_imports_without_lean_dojo():
    """The import guard keeps graph utilities usable without lean_dojo."""
    assert isinstance(LEAN_DOJO_AVAILABLE, bool)
