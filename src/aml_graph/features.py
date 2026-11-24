"""Core 14 structural directed-graph features and IBM transaction activity features."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd

from aml_graph.data import CC_ID, E_DST, E_SRC, N_CC, N_ID

if TYPE_CHECKING:
    from aml_graph.sampling import EdgeIndex

SIZE_COLS: list[str] = ["n_nodes", "n_edges"]


def gini(x: Iterable[float] | np.ndarray) -> float:
    """Compute Gini inequality coefficient of non-negative degree array x."""
    arr = np.sort(np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float))
    if arr.size == 0:
        return 0.0
    total = float(arr.sum())
    if total == 0.0:
        return 0.0
    n = len(arr)
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float((weights @ arr) / (n * total))


def graph_features(
    node_list: Iterable[int],
    edge_pairs: Iterable[tuple[int, int]],
) -> dict[str, float | int]:
    """Compute the 14 canonical topological features of a directed simple subgraph."""
    g = nx.DiGraph()
    g.add_nodes_from(node_list)
    g.add_edges_from((int(u), int(v)) for u, v in edge_pairs if u != v)
    n, m = g.number_of_nodes(), g.number_of_edges()
    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
            "edges_per_node": 0.0,
            "max_in_deg_norm": 0.0,
            "max_out_deg_norm": 0.0,
            "hub_share": 0.0,
            "deg_gini": 0.0,
            "frac_sources": 0.0,
            "frac_sinks": 0.0,
            "frac_passthrough": 0.0,
            "longest_chain_norm": 0.0,
            "has_cycle": 0.0,
            "reciprocity": 0.0,
        }
    in_deg = np.array([d for _, d in g.in_degree()], dtype=float)
    out_deg = np.array([d for _, d in g.out_degree()], dtype=float)
    sccs = list(nx.strongly_connected_components(g))
    chain = nx.dag_longest_path_length(nx.condensation(g, sccs))
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": m / (n * (n - 1)) if n > 1 else 0.0,
        "edges_per_node": m / n,
        "max_in_deg_norm": float(in_deg.max() / (n - 1)) if n > 1 else 0.0,
        "max_out_deg_norm": float(out_deg.max() / (n - 1)) if n > 1 else 0.0,
        "hub_share": float((in_deg + out_deg).max() / (2 * m)) if m else 0.0,
        "deg_gini": gini(in_deg + out_deg),
        "frac_sources": float(((in_deg == 0) & (out_deg > 0)).mean()),
        "frac_sinks": float(((out_deg == 0) & (in_deg > 0)).mean()),
        "frac_passthrough": float(((in_deg > 0) & (out_deg > 0)).mean()),
        "longest_chain_norm": chain / (n - 1) if n > 1 else 0.0,
        "has_cycle": float(any(len(c) > 1 for c in sccs)),
        "reciprocity": float(nx.reciprocity(g)) if m else 0.0,
    }


FEATURES: list[str] = list(graph_features([1, 2], [(1, 2)]).keys())


def build_feature_tables(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    idx: EdgeIndex,
    cc: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Build IBM and Elliptic2 subgraph feature DataFrames and membership dictionaries."""
    ibm_rows: list[dict[str, Any]] = []
    ibm_nodesets: dict[str, np.ndarray] = {}
    ibm_edgesets: dict[str, np.ndarray] = {}

    for g in positives + negatives:
        e = g["edges"]
        ibm_rows.append(
            {
                "sg_id": g["sg_id"],
                "dataset": "ibm",
                "label": int(g["label"]),
                "pattern": g["pattern"],
                "t_start": int(g["t0"]),
                **graph_features(g["nodes"], zip(idx.s[e], idx.d[e], strict=False)),
            }
        )
        ibm_nodesets[g["sg_id"]] = np.fromiter(g["nodes"], np.int64)
        ibm_edgesets[g["sg_id"]] = np.asarray(e, np.int64)
    ibm_feat = pd.DataFrame(ibm_rows)

    node_groups = nodes.groupby(N_CC)[N_ID].apply(np.asarray).to_dict()
    edge_groups = {k: grp[[E_SRC, E_DST]].to_numpy() for k, grp in edges.groupby("cc")}
    ell_rows: list[dict[str, Any]] = []
    ell_nodesets: dict[str, np.ndarray] = {}
    for cid, y in zip(cc[CC_ID], cc["label"], strict=False):
        nd = node_groups.get(cid)
        if nd is None:
            continue
        raw_edges = edge_groups.get(cid)
        edge_iter = map(tuple, raw_edges) if raw_edges is not None else []
        sg_id = f"ell_{cid}"
        ell_rows.append(
            {
                "sg_id": sg_id,
                "dataset": "elliptic2",
                "label": int(y),
                "pattern": "NA",
                "t_start": np.nan,
                **graph_features(nd, edge_iter),
            }
        )
        ell_nodesets[sg_id] = nd
    ell_feat = pd.DataFrame(ell_rows)

    const = (
        [f for f in FEATURES if ell_feat[f].value_counts(normalize=True).iloc[0] > 0.99]
        if len(ell_feat)
        else []
    )
    stats: dict[str, Any] = {
        "ibm_final_subgraphs": len(ibm_feat),
        "ibm_prevalence": round(float(ibm_feat["label"].mean()), 4) if len(ibm_feat) else 0.0,
        "elliptic2_final_subgraphs": len(ell_feat),
        "elliptic2_prevalence": round(float(ell_feat["label"].mean()), 4) if len(ell_feat) else 0.0,
        "elliptic2_subgraphs_without_edges": int((ell_feat["n_edges"] == 0).sum())
        if len(ell_feat)
        else 0,
        "features_nearly_constant_in_elliptic2": const,
    }
    sets = {"ibm_nodes": ibm_nodesets, "ibm_edges": ibm_edgesets, "ell_nodes": ell_nodesets}
    return ibm_feat, ell_feat, sets, stats


def transaction_activity_features(
    sg_id: str,
    edges_map: dict[str, np.ndarray],
    idx: EdgeIndex,
    start_by_id: dict[str, int],
    end_by_id: dict[str, int],
) -> dict[str, float | int | str]:
    """Compute the 6 IBM-only behavioral/temporal features without reading labels."""
    blind_idx = idx.label_blind_view()
    e = np.asarray(edges_map[sg_id], dtype=np.int64)
    n_transactions = int(e.size)
    if not n_transactions:
        return {
            "sg_id": sg_id,
            "n_transactions": 0,
            "transactions_per_hour": 0.0,
            "repeat_pair_fraction": 0.0,
            "dominant_pair_fraction": 0.0,
            "active_span_minutes": 0.0,
            "median_interarrival_minutes": 0.0,
        }
    pairs = np.column_stack((blind_idx.s[e], blind_idx.d[e]))
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    times = np.sort(blind_idx.t[e])
    duration = max(1, int(end_by_id[sg_id] - start_by_id[sg_id]))
    return {
        "sg_id": sg_id,
        "n_transactions": n_transactions,
        "transactions_per_hour": float(60 * n_transactions / duration),
        "repeat_pair_fraction": float(1 - len(counts) / n_transactions),
        "dominant_pair_fraction": float(counts.max() / n_transactions),
        "active_span_minutes": float(times[-1] - times[0]),
        "median_interarrival_minutes": float(np.median(np.diff(times)))
        if n_transactions > 1
        else 0.0,
    }
