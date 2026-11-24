"""Temporal, stratified, overlap-purged, rolling-origin, and account-component splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def purged_split(
    df: pd.DataFrame,
    purge_sets: dict[str, np.ndarray],
    node_sets: dict[str, np.ndarray],
    temporal: bool,
    seed: int,
    test_size: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Hold out a fraction of subgraphs and purge holdout members overlapping train."""
    if temporal:
        r = df["t_start"].rank(method="first").to_numpy()
        cut = (1 - test_size) * len(df)
        tr = np.where(r <= cut)[0]
        te = np.where(r > cut)[0]
    else:
        tr, te = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            stratify=df["label"],
            random_state=seed,
        )

    seen_items: set[int] = set()
    seen_nodes: set[int] = set()
    for sg in df["sg_id"].iloc[tr]:
        seen_items.update(purge_sets[sg].tolist())
        seen_nodes.update(node_sets[sg].tolist())

    keep = np.asarray(
        [seen_items.isdisjoint(purge_sets[sg].tolist()) for sg in df["sg_id"].iloc[te]],
        dtype=bool,
    )
    retained = te[keep]
    recurring = [
        not seen_nodes.isdisjoint(node_sets[sg].tolist()) for sg in df["sg_id"].iloc[retained]
    ]
    info: dict[str, Any] = {
        "train": len(tr),
        "test_before": len(te),
        "purged": int((~keep).sum()),
        "test_final": len(retained),
        "test_positives": int(df["label"].iloc[retained].sum()),
        "test_with_recurring_node": round(float(np.mean(recurring)), 3) if recurring else None,
    }
    return tr, retained, info


def make_splits(
    ibm_feat: pd.DataFrame,
    ell_feat: pd.DataFrame,
    sets: dict[str, dict[str, np.ndarray]],
    seed: int,
    test_size: float = 0.3,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Construct outer splits for both IBM (temporal) and Elliptic2 (stratified)."""
    return {
        "ibm": purged_split(
            ibm_feat, sets["ibm_edges"], sets["ibm_nodes"], True, seed, test_size=test_size
        ),
        "elliptic2": purged_split(
            ell_feat, sets["ell_nodes"], sets["ell_nodes"], False, seed, test_size=test_size
        ),
    }


def validation_indices(
    ds: str,
    data_map: dict[str, pd.DataFrame],
    sets: dict[str, dict[str, np.ndarray]],
    train_indices: np.ndarray,
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct a held-out inner validation split and purge overlaps with fitting rows."""
    train_arr = np.asarray(train_indices)
    sub_df = data_map[ds].iloc[train_arr].reset_index(drop=True)
    overlap = sets["ibm_edges"] if ds == "ibm" else sets["ell_nodes"]
    nodes_for_overlap = sets["ibm_nodes"] if ds == "ibm" else sets["ell_nodes"]
    fit_local, val_local, info = purged_split(
        sub_df,
        overlap,
        nodes_for_overlap,
        temporal=(ds == "ibm"),
        seed=seed,
        test_size=validation_fraction,
    )
    return train_arr[fit_local], train_arr[val_local], info


def chronological_embargo_split(
    df: pd.DataFrame,
    edge_map: dict[str, np.ndarray],
    embargo_minutes: int = 120,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Sort candidates chronologically, apply time embargo, and purge shared edges."""
    ordered = df.sort_values(["t_start", "sg_id"]).reset_index(drop=True)
    n = len(ordered)
    first = ordered.iloc[: int(0.60 * n)]
    second = ordered.iloc[int(0.60 * n) : int(0.80 * n)]
    third = ordered.iloc[int(0.80 * n) :]

    train_ids = first["sg_id"].tolist()
    train_latest_end = int(first["t_end"].max()) if len(first) else 0
    val_after_time = second[second["t_start"] > train_latest_end + embargo_minutes]

    seen_train_edges: set[int] = set()
    for ident in train_ids:
        seen_train_edges.update(edge_map[ident].tolist())

    val_ids = [
        ident
        for ident in val_after_time["sg_id"]
        if seen_train_edges.isdisjoint(edge_map[ident].tolist())
    ]
    if not val_ids:
        raise RuntimeError("Embargo removed all validation graphs.")

    val_max_end = int(ordered.set_index("sg_id").loc[val_ids, "t_end"].max())
    test_after_time = third[third["t_start"] > val_max_end + embargo_minutes]
    seen_previous_edges = set(seen_train_edges)
    for ident in val_ids:
        seen_previous_edges.update(edge_map[ident].tolist())

    test_ids = [
        ident
        for ident in test_after_time["sg_id"]
        if seen_previous_edges.isdisjoint(edge_map[ident].tolist())
    ]
    groups = {"train": train_ids, "validation": val_ids, "test": test_ids}
    audit: list[dict[str, Any]] = []
    lookup = df.set_index("sg_id")
    for name, ids in groups.items():
        labels = lookup.loc[ids, "label"].to_numpy(dtype=int) if ids else np.array([], dtype=int)
        audit.append(
            {
                "partition": name,
                "n": len(ids),
                "positive": int(labels.sum()),
                "negative": int((labels == 0).sum()),
                "positive_rate": float(labels.mean()) if labels.size else np.nan,
            }
        )
    return groups, pd.DataFrame(audit)


def account_disjoint_partition(
    df: pd.DataFrame,
    node_map: dict[str, np.ndarray],
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily assign subgraphs so zero accounts overlap between fit, val, and test."""
    ordered = df.sort_values(["t_start", "sg_id"]).reset_index()
    fit_i: list[int] = []
    val_i: list[int] = []
    test_i: list[int] = []
    seen_fit: set[int] = set()
    seen_val: set[int] = set()
    seen_test: set[int] = set()
    n = len(ordered)
    n_fit_target = int(train_frac * n)
    n_val_target = int((train_frac + val_frac) * n)

    for _, row in ordered.iterrows():
        orig_idx = int(row["index"])
        nodes_k = set(node_map[str(row["sg_id"])].tolist())
        in_fit = not seen_fit.isdisjoint(nodes_k)
        in_val = not seen_val.isdisjoint(nodes_k)
        in_test = not seen_test.isdisjoint(nodes_k)
        if (in_fit and in_val) or (in_fit and in_test) or (in_val and in_test):
            continue
        if in_fit or (not in_val and not in_test and len(fit_i) < n_fit_target):
            fit_i.append(orig_idx)
            seen_fit.update(nodes_k)
        elif in_val or (not in_test and len(fit_i) + len(val_i) < n_val_target):
            val_i.append(orig_idx)
            seen_val.update(nodes_k)
        else:
            test_i.append(orig_idx)
            seen_test.update(nodes_k)

    return np.asarray(fit_i, dtype=int), np.asarray(val_i, dtype=int), np.asarray(test_i, dtype=int)


# ---------------------------------------------------------------------------
# Phase 2 Splits: Rolling-Origin Temporal Folds & Account-Component Group Folds
# ---------------------------------------------------------------------------


def rolling_origin_splits(
    df: pd.DataFrame,
    edge_map: dict[str, np.ndarray],
    n_folds: int = 4,
    embargo_minutes: int = 480,
    first_train_quantile: float = 0.30,
    val_quantile_width: float = 0.12,
) -> list[dict[str, Any]]:
    """Construct >= 3 expanding-window temporal folds with embargo gaps and non-overlapping tests.

    Guarantees:
      1. `n_folds >= 3` strictly non-overlapping chronological test intervals.
      2. For every fold: `min(val.t_start) > max(train.t_end) + embargo_minutes` and
         `min(test.t_start) > max(val.t_end) + embargo_minutes`.
      3. Pairwise zero shared transaction IDs across `(train, val, test)` in every fold,
         and zero shared subgraph indices across `{test_0, ..., test_{K-1}}`.
    """
    if n_folds < 3:
        raise ValueError("rolling_origin_splits requires n_folds >= 3")

    t_start = df["t_start"].to_numpy(dtype=np.int64)
    t_end = df["t_end"].to_numpy(dtype=np.int64)
    sg_ids = df["sg_id"].astype(str).to_numpy()

    q_test_start = float(first_train_quantile + val_quantile_width)
    if q_test_start >= 0.85:
        raise ValueError("first_train_quantile + val_quantile_width leaves insufficient test range")

    test_quantiles = np.linspace(q_test_start, 1.0, n_folds + 1)
    used_test_indices: set[int] = set()
    folds: list[dict[str, Any]] = []

    for k in range(n_folds):
        q_val_hi = float(test_quantiles[k])
        q_train_hi = max(0.05, q_val_hi - float(val_quantile_width))
        q_test_hi = float(test_quantiles[k + 1])

        train_cut_t = int(np.quantile(t_start, q_train_hi))
        val_cut_t = int(np.quantile(t_start, q_val_hi))
        test_cut_t = (
            int(t_start.max()) if k == n_folds - 1 else int(np.quantile(t_start, q_test_hi))
        )

        train_idx = np.flatnonzero(t_end <= train_cut_t)
        if train_idx.size == 0:
            train_idx = np.flatnonzero(t_start <= train_cut_t)
        train_max_end = int(t_end[train_idx].max()) if train_idx.size else train_cut_t

        seen_train_edges: set[int] = set()
        for pos in train_idx:
            seen_train_edges.update(edge_map[sg_ids[pos]].tolist())

        val_candidates = np.flatnonzero(
            (t_start > train_max_end + int(embargo_minutes)) & (t_end <= val_cut_t)
        )
        if val_candidates.size == 0:
            # Fallback if val_cut_t was tighter than embargo: advance val window past embargo
            val_candidates = np.flatnonzero(
                (t_start > train_max_end + int(embargo_minutes))
                & (t_start <= val_cut_t + int(embargo_minutes))
            )
        val_idx = np.asarray(
            [
                p
                for p in val_candidates
                if seen_train_edges.isdisjoint(edge_map[sg_ids[p]].tolist())
            ],
            dtype=int,
        )
        val_max_end = (
            int(t_end[val_idx].max()) if val_idx.size else (train_max_end + int(embargo_minutes))
        )

        seen_train_val_edges = set(seen_train_edges)
        for pos in val_idx:
            seen_train_val_edges.update(edge_map[sg_ids[pos]].tolist())

        test_lo_bound = max(val_cut_t, val_max_end + int(embargo_minutes))
        test_candidates = np.flatnonzero((t_start > test_lo_bound) & (t_start <= test_cut_t))
        test_idx = np.asarray(
            [
                p
                for p in test_candidates
                if p not in used_test_indices
                and seen_train_val_edges.isdisjoint(edge_map[sg_ids[p]].tolist())
            ],
            dtype=int,
        )
        used_test_indices.update(test_idx.tolist())

        folds.append(
            {
                "fold": k,
                "regime": "rolling_origin",
                "fit_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "embargo_minutes": int(embargo_minutes),
                "train_max_end": train_max_end,
                "val_min_start": int(t_start[val_idx].min()) if val_idx.size else None,
                "val_max_end": val_max_end,
                "test_min_start": int(t_start[test_idx].min()) if test_idx.size else None,
                "test_max_end": int(t_end[test_idx].max()) if test_idx.size else None,
            }
        )
    return folds


def account_component_group_splits(
    df: pd.DataFrame,
    node_map: dict[str, np.ndarray],
    edge_map: dict[str, np.ndarray] | None = None,
    n_folds: int = 4,
) -> list[dict[str, Any]]:
    """Group subgraphs by connected components of shared accounts and partition into K folds.

    Every connected component of the 'subgraphs share an account (or transaction)' graph
    is placed 100% inside a single fold. Across fit, validation, and test in every fold:
      - Zero accounts are shared.
      - Zero transactions are shared.
      - 100% of subgraphs are retained (zero bridge-dropping bias).
    """
    if n_folds < 3:
        raise ValueError("account_component_group_splits requires n_folds >= 3 (fit, val, test)")

    n_sub = len(df)
    parent = np.arange(n_sub, dtype=np.int64)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first_sub_for_node: dict[int, int] = {}
    first_sub_for_edge: dict[int, int] = {}
    sg_ids = df["sg_id"].astype(str).tolist()
    labels = df["label"].to_numpy(dtype=int)

    for idx_i, sg in enumerate(sg_ids):
        for u in node_map[sg].tolist():
            iu = int(u)
            if iu in first_sub_for_node:
                union(idx_i, first_sub_for_node[iu])
            else:
                first_sub_for_node[iu] = idx_i
        if edge_map is not None and sg in edge_map:
            for e in edge_map[sg].tolist():
                ie = int(e)
                if ie in first_sub_for_edge:
                    union(idx_i, first_sub_for_edge[ie])
                else:
                    first_sub_for_edge[ie] = idx_i

    comp_members: dict[int, list[int]] = defaultdict(list)
    for idx_i in range(n_sub):
        comp_members[find(idx_i)].append(idx_i)

    # Sort components largest-first for deterministic balanced bin packing across K folds
    sorted_comps = sorted(
        comp_members.values(),
        key=lambda members: (-len(members), min(members)),
    )
    fold_bins: list[list[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    fold_pos = [0] * n_folds

    for members in sorted_comps:
        pos_in_comp = int(labels[members].sum())
        # Assign entire component to the lightest fold (tie-break by fewest positives)
        target_f = min(range(n_folds), key=lambda f: (fold_sizes[f], fold_pos[f], f))
        fold_bins[target_f].extend(members)
        fold_sizes[target_f] += len(members)
        fold_pos[target_f] += pos_in_comp

    fold_arrays = [np.asarray(sorted(b), dtype=int) for b in fold_bins]
    folds: list[dict[str, Any]] = []
    for k in range(n_folds):
        test_idx = fold_arrays[k]
        val_idx = fold_arrays[(k - 1) % n_folds]
        fit_parts = [fold_arrays[j] for j in range(n_folds) if j not in (k, (k - 1) % n_folds)]
        fit_idx = np.sort(np.concatenate(fit_parts)) if fit_parts else np.empty(0, dtype=int)
        folds.append(
            {
                "fold": k,
                "regime": "account_group",
                "fit_idx": fit_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "n_components_total": len(sorted_comps),
            }
        )
    return folds
