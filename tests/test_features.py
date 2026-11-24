"""Unit tests for the 14 directed-graph features and Gini inequality."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aml_graph.features import FEATURES, gini, graph_features


def test_gini_edge_cases() -> None:
    assert gini([]) == 0.0
    assert gini([0.0, 0.0, 0.0, 0.0]) == 0.0
    assert gini([4.0]) == 0.0
    assert math.isclose(gini([5.0, 5.0, 5.0, 5.0]), 0.0, abs_tol=1e-12)
    # Single non-zero among n=4 has Gini = (n - 1) / n = 0.75
    assert math.isclose(gini([0.0, 0.0, 0.0, 10.0]), 0.75, rel_tol=1e-12)
    # Permutation and positive scale invariance
    base = gini([10.0, 0.0, 2.0, 4.0])
    assert math.isclose(base, gini([0.0, 2.0, 4.0, 10.0]), rel_tol=1e-12)
    assert math.isclose(base, gini(np.array([70.0, 0.0, 14.0, 28.0])), rel_tol=1e-12)


def test_graph_features_directed_path() -> None:
    # 1 -> 2 -> 3 -> 4
    feats = graph_features([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])
    assert list(feats.keys()) == FEATURES
    assert feats["n_nodes"] == 4
    assert feats["n_edges"] == 3
    assert math.isclose(feats["density"], 3 / 12)
    assert math.isclose(feats["edges_per_node"], 3 / 4)
    assert math.isclose(feats["max_in_deg_norm"], 1 / 3)
    assert math.isclose(feats["max_out_deg_norm"], 1 / 3)
    assert math.isclose(feats["hub_share"], 2 / 6)
    assert math.isclose(feats["frac_sources"], 0.25)
    assert math.isclose(feats["frac_sinks"], 0.25)
    assert math.isclose(feats["frac_passthrough"], 0.50)
    assert math.isclose(feats["longest_chain_norm"], 1.0)
    assert feats["has_cycle"] == 0.0
    assert feats["reciprocity"] == 0.0


def test_graph_features_directed_cycle() -> None:
    # 1 -> 2 -> 3 -> 1
    feats = graph_features([1, 2, 3], [(1, 2), (2, 3), (3, 1)])
    assert feats["n_nodes"] == 3
    assert feats["n_edges"] == 3
    assert math.isclose(feats["density"], 0.5)
    assert math.isclose(feats["edges_per_node"], 1.0)
    assert math.isclose(feats["max_in_deg_norm"], 0.5)
    assert math.isclose(feats["max_out_deg_norm"], 0.5)
    assert math.isclose(feats["hub_share"], 1 / 3)
    assert math.isclose(feats["deg_gini"], 0.0, abs_tol=1e-12)
    assert feats["frac_sources"] == 0.0
    assert feats["frac_sinks"] == 0.0
    assert feats["frac_passthrough"] == 1.0
    assert feats["longest_chain_norm"] == 0.0
    assert feats["has_cycle"] == 1.0
    assert feats["reciprocity"] == 0.0


def test_graph_features_star() -> None:
    # Outward star: 0 -> 1, 0 -> 2, 0 -> 3
    feats = graph_features([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)])
    assert feats["n_nodes"] == 4
    assert feats["n_edges"] == 3
    assert math.isclose(feats["max_out_deg_norm"], 1.0)
    assert math.isclose(feats["max_in_deg_norm"], 1 / 3)
    assert math.isclose(feats["hub_share"], 3 / 6)
    assert math.isclose(feats["frac_sources"], 0.25)
    assert math.isclose(feats["frac_sinks"], 0.75)
    assert feats["frac_passthrough"] == 0.0
    assert math.isclose(feats["longest_chain_norm"], 1 / 3)
    assert feats["has_cycle"] == 0.0
    assert feats["reciprocity"] == 0.0


def test_graph_features_bidirectional_pair_and_multiedge_selfloop_collapse() -> None:
    # Parallel edges and self-loops should be collapsed/dropped cleanly
    feats = graph_features([10, 20], [(10, 20), (10, 20), (20, 10), (10, 10)])
    assert feats["n_nodes"] == 2
    assert feats["n_edges"] == 2
    assert math.isclose(feats["density"], 1.0)
    assert math.isclose(feats["edges_per_node"], 1.0)
    assert math.isclose(feats["max_in_deg_norm"], 1.0)
    assert math.isclose(feats["max_out_deg_norm"], 1.0)
    assert math.isclose(feats["hub_share"], 0.5)
    assert feats["frac_passthrough"] == 1.0
    assert feats["longest_chain_norm"] == 0.0
    assert feats["has_cycle"] == 1.0
    assert math.isclose(feats["reciprocity"], 1.0)


@pytest.mark.parametrize("nodes", [[1], [1, 2, 3]])
def test_graph_features_isolated_nodes(nodes: list[int]) -> None:
    feats = graph_features(nodes, [])
    assert feats["n_nodes"] == len(nodes)
    assert feats["n_edges"] == 0
    assert feats["density"] == 0.0
    assert feats["edges_per_node"] == 0.0
    assert feats["hub_share"] == 0.0
    assert feats["deg_gini"] == 0.0
    assert feats["frac_sources"] == 0.0
    assert feats["frac_sinks"] == 0.0
    assert feats["frac_passthrough"] == 0.0
    assert feats["has_cycle"] == 0.0
    assert feats["reciprocity"] == 0.0
