"""Indexed edge lookup, legacy positive/negative construction, and common-BFS sampler."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from aml_graph.data import IBM_COLS, account_key
from aml_graph.features import graph_features


class LabelAccessForbiddenError(RuntimeError):
    """Raised if a label-blind sampler or feature function attempts to access transaction labels."""


class EdgeIndex:
    """Fast CSR-like index over directed timestamped transactions."""

    def __init__(self, df: pd.DataFrame, n_accounts: int, *, _forbid_labels: bool = False) -> None:
        self.s: np.ndarray = df["s"].to_numpy(dtype=np.int64, copy=False)
        self.d: np.ndarray = df["d"].to_numpy(dtype=np.int64, copy=False)
        self._y: np.ndarray = df["is_laundering"].to_numpy(dtype=np.int64, copy=False)
        self._forbid_labels = _forbid_labels
        self.t: np.ndarray = df["timestamp"].to_numpy().astype("datetime64[m]").astype(np.int64)
        self.out_o: np.ndarray = np.argsort(self.s, kind="stable")
        self.out_p: np.ndarray = np.searchsorted(self.s[self.out_o], np.arange(n_accounts + 1))
        self.in_o: np.ndarray = np.argsort(self.d, kind="stable")
        self.in_p: np.ndarray = np.searchsorted(self.d[self.in_o], np.arange(n_accounts + 1))
        self.t_o: np.ndarray = np.argsort(self.t, kind="stable")
        self.t_sorted: np.ndarray = self.t[self.t_o]

    @property
    def y(self) -> np.ndarray:
        if self._forbid_labels:
            raise LabelAccessForbiddenError(
                "Transaction labels cannot be accessed in label-blind mode."
            )
        return self._y

    def label_blind_view(self) -> EdgeIndex:
        """Return a shallow view that raises LabelAccessForbiddenError if .y is accessed."""
        view = object.__new__(EdgeIndex)
        view.s = self.s
        view.d = self.d
        view._y = self._y
        view._forbid_labels = True
        view.t = self.t
        view.out_o = self.out_o
        view.out_p = self.out_p
        view.in_o = self.in_o
        view.in_p = self.in_p
        view.t_o = self.t_o
        view.t_sorted = self.t_sorted
        return view

    def out_e(self, u: int) -> np.ndarray:
        return self.out_o[self.out_p[u] : self.out_p[u + 1]]

    def in_e(self, u: int) -> np.ndarray:
        return self.in_o[self.in_p[u] : self.in_p[u + 1]]

    def induced(self, node_set: Iterable[int], t0: int, t1: int) -> np.ndarray:
        """Return transaction row indices among node_set occurring within [t0, t1]."""
        arr = np.fromiter(node_set, np.int64)
        if arr.size == 0:
            return np.empty(0, dtype=np.int64)
        e = np.concatenate([self.out_e(int(u)) for u in arr])
        return e[np.isin(self.d[e], arr) & (self.t[e] >= t0) & (self.t[e] <= t1)]


def build_ibm_positives(
    attempts: list[dict[str, Any]],
    acc_id: pd.Series,
    idx: EdgeIndex,
    min_window: int = 60,
) -> list[dict[str, Any]]:
    """Construct positive IBM subgraphs from documented laundering attempts (legacy Cell 11)."""
    positives: list[dict[str, Any]] = []
    for k, a in enumerate(attempts):
        rows = pd.DataFrame(a["rows"], columns=IBM_COLS)
        accts = pd.concat(
            [
                account_key(rows["from_bank"], rows["from_acct"]),
                account_key(rows["to_bank"], rows["to_acct"]),
            ]
        )
        node_set = set(acc_id.reindex(accts.unique()).dropna().astype(int))
        ts = pd.to_datetime(rows["timestamp"], format="%Y/%m/%d %H:%M")
        t0 = int(np.datetime64(ts.min(), "m").astype(np.int64))
        t1 = max(int(np.datetime64(ts.max(), "m").astype(np.int64)), t0 + int(min_window))
        if len(node_set) >= 2:
            positives.append(
                {
                    "sg_id": f"ibm_pos_{k}",
                    "label": 1,
                    "pattern": a["pattern"],
                    "t0": t0,
                    "t1": t1,
                    "nodes": node_set,
                    "edges": idx.induced(node_set, t0, t1),
                }
            )
    return positives


def grow_licit_bfs(
    idx: EdgeIndex,
    n_target: int,
    t0: int,
    t1: int,
    rng: np.random.Generator,
    tries: int = 10,
) -> set[int] | None:
    """Legacy licit-only BFS growth matching exact RNG call sequence from Cell 12."""
    lo = int(np.searchsorted(idx.t_sorted, t0))
    hi = int(np.searchsorted(idx.t_sorted, t1, "right"))
    best: set[int] | None = None
    for _ in range(tries):
        if hi <= lo:
            return None
        e0 = int(idx.t_o[rng.integers(lo, hi)])
        if idx.y[e0]:
            continue
        seen: set[int] = {int(idx.s[e0])}
        queue: list[int] = [int(idx.s[e0])]
        while queue and len(seen) < n_target:
            u = queue.pop(0)
            e = np.concatenate([idx.out_e(u), idx.in_e(u)])
            e = e[(idx.t[e] >= t0) & (idx.t[e] <= t1) & (idx.y[e] == 0)]
            nb = np.unique(np.concatenate([idx.s[e], idx.d[e]]))
            rng.shuffle(nb)
            for v in nb:
                if len(seen) >= n_target:
                    break
                iv = int(v)
                if iv not in seen:
                    seen.add(iv)
                    queue.append(iv)
        if best is None or len(seen) > len(best):
            best = seen
        if len(seen) >= n_target:
            break
    return best


def sample_ibm_negatives(
    positives: list[dict[str, Any]],
    idx: EdgeIndex,
    rng: np.random.Generator,
    neg_per_pos: int = 20,
    tries: int = 10,
    max_attempt_multiplier: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample size- and window-matched licit BFS negative subgraphs (legacy Cell 12)."""
    if not positives or idx.t.size == 0:
        return [], {
            "ibm_negative_subgraphs": 0,
            "ibm_negative_sampling_attempts": 0,
            "ibm_negatives_rejected_contained_laundering": 0,
            "ibm_negatives_rejected_too_small": 0,
        }
    t_min, t_max = int(idx.t.min()), int(idx.t.max())
    negatives: list[dict[str, Any]] = []
    rejected = 0
    too_small = 0
    target = int(neg_per_pos) * len(positives)
    max_attempts = int(max_attempt_multiplier) * target
    attempts_made = 0

    while len(negatives) < target and attempts_made < max_attempts:
        attempts_made += 1
        tmpl = positives[int(rng.integers(len(positives)))]
        dur = int(tmpl["t1"] - tmpl["t0"])
        t0 = int(rng.integers(t_min, max(t_max - dur, t_min + 1)))
        t1 = t0 + dur
        node_set = grow_licit_bfs(idx, len(tmpl["nodes"]), t0, t1, rng, tries=tries)
        if not node_set or len(node_set) != len(tmpl["nodes"]):
            too_small += 1
            continue
        e = idx.induced(node_set, t0, t1)
        if idx.y[e].any():
            rejected += 1
            continue
        negatives.append(
            {
                "sg_id": f"ibm_neg_{len(negatives)}",
                "label": 0,
                "pattern": "NONE",
                "t0": t0,
                "t1": t1,
                "nodes": node_set,
                "edges": e,
            }
        )

    stats = {
        "ibm_positive_subgraphs": len(positives),
        "ibm_negative_subgraphs": len(negatives),
        "ibm_negative_sampling_attempts": attempts_made,
        "ibm_negatives_rejected_contained_laundering": rejected,
        "ibm_negatives_rejected_too_small": too_small,
    }
    return negatives, stats


def common_bfs_candidate(
    idx: EdgeIndex,
    seed_edge: int,
    n_target: int,
    duration_min: int,
    generator: np.random.Generator,
) -> dict[str, Any] | None:
    """Construct a candidate subgraph using strictly label-blind time-restricted BFS."""
    blind = idx.label_blind_view()
    seed_edge = int(seed_edge)
    t_seed = int(blind.t[seed_edge])
    before = int(generator.integers(0, int(duration_min) + 1))
    t0 = t_seed - before
    t1 = t0 + int(duration_min)
    s0, d0 = int(blind.s[seed_edge]), int(blind.d[seed_edge])
    seen: set[int] = {s0, d0}
    queue: deque[int] = deque([s0, d0])

    while queue and len(seen) < n_target:
        u = queue.popleft()
        adjacent = np.concatenate([blind.out_e(u), blind.in_e(u)])
        if adjacent.size == 0:
            continue
        valid = adjacent[(blind.t[adjacent] >= t0) & (blind.t[adjacent] <= t1)]
        if valid.size == 0:
            continue
        neighbors = np.unique(np.concatenate([blind.s[valid], blind.d[valid]]))
        generator.shuffle(neighbors)
        for v in neighbors:
            vv = int(v)
            if vv not in seen:
                seen.add(vv)
                queue.append(vv)
                if len(seen) >= n_target:
                    break

    if len(seen) < 2:
        return None
    edges_found = blind.induced(seen, t0, t1)
    if edges_found.size == 0:
        return None
    return {
        "nodes": seen,
        "edges": np.asarray(edges_found, dtype=np.int64),
        "t0": t0,
        "t1": t1,
    }


def generate_common_candidates(
    idx: EdgeIndex,
    n_candidates: int = 2500,
    positive_seed_probability: float = 0.40,
    node_sizes: tuple[int, ...] = (3, 4, 5, 6, 8, 10),
    windows_min: tuple[int, ...] = (60, 180, 360),
    seed: int = 2026,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, tuple[int, int]],
    dict[str, Any],
]:
    """Generate common-construction BFS candidates and assign labels post-hoc (Extension 2)."""
    generator = np.random.default_rng(seed)
    positive_seed_indices = np.flatnonzero(np.asarray(idx.y) == 1)
    if positive_seed_indices.size == 0:
        raise RuntimeError("No annotated laundering transactions found in IBM data")

    rows: list[dict[str, Any]] = []
    nodes_by_id: dict[str, np.ndarray] = {}
    edges_by_id: dict[str, np.ndarray] = {}
    windows_by_id: dict[str, tuple[int, int]] = {}
    signatures: set[tuple[int, int, tuple[int, ...]]] = set()
    attempts = 0
    source_counts = {"uniform": 0, "positive_enriched": 0}
    max_tries = max(20 * int(n_candidates), 1000)

    while len(rows) < n_candidates and attempts < max_tries:
        attempts += 1
        mode = "positive_enriched" if generator.random() < positive_seed_probability else "uniform"
        if mode == "positive_enriched":
            seed_edge = int(generator.choice(positive_seed_indices))
        else:
            seed_edge = int(generator.integers(0, len(idx.y)))
        n_target = int(generator.choice(node_sizes))
        duration = int(generator.choice(windows_min))
        graph = common_bfs_candidate(idx, seed_edge, n_target, duration, generator)
        if graph is None:
            continue
        signature = (graph["t0"], graph["t1"], tuple(sorted(graph["nodes"])))
        if signature in signatures:
            continue
        signatures.add(signature)
        edges_idx = graph["edges"]
        label = int(np.any(np.asarray(idx.y)[edges_idx] == 1))
        sg_id = f"common_{len(rows):05d}"
        feats = graph_features(
            graph["nodes"], zip(idx.s[edges_idx], idx.d[edges_idx], strict=False)
        )
        rows.append(
            {
                "sg_id": sg_id,
                "label": label,
                "t_start": graph["t0"],
                "t_end": graph["t1"],
                "seed_source": mode,
                "target_nodes": n_target,
                "window_minutes": duration,
                **feats,
            }
        )
        nodes_by_id[sg_id] = np.asarray(sorted(graph["nodes"]), dtype=np.int64)
        edges_by_id[sg_id] = np.unique(edges_idx)
        windows_by_id[sg_id] = (graph["t0"], graph["t1"])
        source_counts[mode] += 1

    df = pd.DataFrame(rows)
    if df.empty or df["label"].nunique() < 2:
        raise RuntimeError("Common sampler lacks both labels.")
    return (
        df,
        nodes_by_id,
        edges_by_id,
        windows_by_id,
        {"attempts": attempts, "seed_counts": source_counts},
    )
