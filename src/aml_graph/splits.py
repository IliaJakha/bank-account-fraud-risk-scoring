"""Temporal, stratified, overlap-purged, embargo, and account-disjoint splits."""

from __future__ import annotations

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
