"""Tests for transaction-disjoint, embargo, and account-disjoint split guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aml_graph.splits import (
    account_disjoint_partition,
    chronological_embargo_split,
    purged_split,
    validation_indices,
)


def test_purged_and_validation_splits_have_zero_shared_transactions() -> None:
    rows = []
    edge_sets: dict[str, np.ndarray] = {}
    node_sets: dict[str, np.ndarray] = {}
    for i in range(40):
        sg_id = f"sg_{i}"
        rows.append({"sg_id": sg_id, "label": int(i % 5 == 0), "t_start": i * 10})
        # Deliberately create overlapping edges between adjacent subgraphs
        edge_sets[sg_id] = np.array([i, i + 1], dtype=np.int64)
        node_sets[sg_id] = np.array([i % 7, (i + 1) % 7], dtype=np.int64)

    df = pd.DataFrame(rows)
    tr, te, info = purged_split(df, edge_sets, node_sets, temporal=True, seed=0, test_size=0.3)
    assert info["purged"] >= 1

    tr_edges = set().union(*(edge_sets[df["sg_id"].iloc[idx]].tolist() for idx in tr))
    te_edges = set().union(*(edge_sets[df["sg_id"].iloc[idx]].tolist() for idx in te))
    assert tr_edges.isdisjoint(te_edges)

    sets = {"ibm_edges": edge_sets, "ibm_nodes": node_sets, "ell_nodes": {}}
    fit_idx, val_idx, _ = validation_indices(
        "ibm", {"ibm": df}, sets, tr, seed=0, validation_fraction=0.25
    )
    fit_edges = set().union(*(edge_sets[df["sg_id"].iloc[idx]].tolist() for idx in fit_idx))
    val_edges = set().union(*(edge_sets[df["sg_id"].iloc[idx]].tolist() for idx in val_idx))
    assert fit_edges.isdisjoint(val_edges)
    assert fit_edges.isdisjoint(te_edges)
    assert val_edges.isdisjoint(te_edges)


def test_chronological_embargo_split_enforces_time_gap_and_edge_disjointness() -> None:
    rows = []
    edge_map: dict[str, np.ndarray] = {}
    for i in range(30):
        sg_id = f"c_{i:02d}"
        t_start = i * 100
        t_end = t_start + 30
        rows.append({"sg_id": sg_id, "label": int(i % 3 == 0), "t_start": t_start, "t_end": t_end})
        # Inject one cross-partition shared edge to verify purging
        shared = 9999 if i in (5, 20, 26) else i
        edge_map[sg_id] = np.array([i, shared], dtype=np.int64)

    df = pd.DataFrame(rows)
    embargo = 120
    groups, audit = chronological_embargo_split(df, edge_map, embargo_minutes=embargo)
    assert set(audit["partition"]) == {"train", "validation", "test"}

    lookup = df.set_index("sg_id")
    train_max_end = int(lookup.loc[groups["train"], "t_end"].max())
    val_min_start = int(lookup.loc[groups["validation"], "t_start"].min())
    val_max_end = int(lookup.loc[groups["validation"], "t_end"].max())
    test_min_start = int(lookup.loc[groups["test"], "t_start"].min())
    assert val_min_start > train_max_end + embargo
    assert test_min_start > val_max_end + embargo

    for p1, p2 in (("train", "validation"), ("train", "test"), ("validation", "test")):
        e1 = set().union(*(edge_map[k].tolist() for k in groups[p1]))
        e2 = set().union(*(edge_map[k].tolist() for k in groups[p2]))
        assert e1.isdisjoint(e2)
        assert "c_20" not in groups["validation"]
        assert "c_26" not in groups["test"]


def test_account_disjoint_partition_has_zero_shared_accounts() -> None:
    rows = []
    node_map: dict[str, np.ndarray] = {}
    # 15 disjoint account clusters + bridge graphs that couple clusters
    for i in range(30):
        sg_id = f"ad_{i:02d}"
        rows.append({"sg_id": sg_id, "label": int(i % 4 == 0), "t_start": i * 10})
        cluster_base = (i // 2) * 10
        node_map[sg_id] = np.array([cluster_base, cluster_base + 1], dtype=np.int64)

    # Add a bridge graph linking early cluster 0 and late cluster 140
    rows.append({"sg_id": "bridge", "label": 1, "t_start": 500})
    node_map["bridge"] = np.array([0, 140], dtype=np.int64)

    df = pd.DataFrame(rows)
    fit_i, val_i, test_i = account_disjoint_partition(df, node_map, train_frac=0.5, val_frac=0.25)
    assert len(fit_i) > 0 and len(val_i) > 0 and len(test_i) > 0

    fit_nodes = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in fit_i))
    val_nodes = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in val_i))
    test_nodes = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in test_i))

    assert fit_nodes.isdisjoint(val_nodes)
    assert fit_nodes.isdisjoint(test_nodes)
    assert val_nodes.isdisjoint(test_nodes)
