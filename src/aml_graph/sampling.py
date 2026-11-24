"""Indexed edge lookup, legacy samplers, and Phase 2 unified label-blind IPW sampler."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from aml_graph.data import IBM_COLS, account_key
from aml_graph.features import graph_features, transaction_activity_features


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


# ---------------------------------------------------------------------------
# Legacy Benchmark Samplers (preserved strictly for before/after comparison)
# ---------------------------------------------------------------------------


def build_legacy_ibm_positives(
    attempts: list[dict[str, Any]],
    acc_id: pd.Series,
    idx: EdgeIndex,
    min_window: int = 60,
) -> list[dict[str, Any]]:
    """Construct positive IBM subgraphs by clamping [t0, t1] to documented attempts (legacy)."""
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


def sample_legacy_ibm_negatives(
    positives: list[dict[str, Any]],
    idx: EdgeIndex,
    rng: np.random.Generator,
    neg_per_pos: int = 20,
    tries: int = 10,
    max_attempt_multiplier: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample size- and window-matched licit BFS negatives (legacy Cell 12)."""
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


# ---------------------------------------------------------------------------
# Phase 2 Unified Symmetric-Window & Label-Blind Samplers
# ---------------------------------------------------------------------------


def symmetric_seed_window(
    t_seed: int,
    duration_min: int,
    generator: np.random.Generator,
) -> tuple[int, int]:
    """Place a fixed-duration observation window around t_seed with uniform random offset."""
    dur = int(duration_min)
    offset = int(generator.integers(0, dur + 1))
    t0 = int(t_seed) - offset
    return t0, t0 + dur


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
    t0, t1 = symmetric_seed_window(int(blind.t[seed_edge]), int(duration_min), generator)
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


def build_ibm_positives(
    attempts: list[dict[str, Any]],
    acc_id: pd.Series,
    idx: EdgeIndex,
    min_window: int = 60,
    window_durations: tuple[int, ...] = (360, 720),
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Construct positive IBM subgraphs using the symmetric seed-offset window rule (Phase 2).

    Unlike `build_legacy_ibm_positives` (which clamped `[t0, t1]` to the first and last
    attempt transactions), this function picks one seed transaction from the attempt, draws
    a window duration from the fixed label-independent `window_durations` distribution
    (respecting `min_window`), and places `[t0, t1]` via `symmetric_seed_window`, growing the
    subgraph with `common_bfs_candidate` over `idx.label_blind_view()`.
    """
    rng = np.random.default_rng(seed)
    valid_durs = tuple(max(int(d), int(min_window)) for d in window_durations)
    positives: list[dict[str, Any]] = []

    for k, a in enumerate(attempts):
        rows = pd.DataFrame(a["rows"], columns=IBM_COLS)
        accts = pd.concat(
            [
                account_key(rows["from_bank"], rows["from_acct"]),
                account_key(rows["to_bank"], rows["to_acct"]),
            ]
        )
        node_set_ref = set(acc_id.reindex(accts.unique()).dropna().astype(int))
        if len(node_set_ref) < 2:
            continue
        ts = pd.to_datetime(rows["timestamp"], format="%Y/%m/%d %H:%M")
        t_seed_min = int(np.datetime64(ts.iloc[int(rng.integers(len(ts)))], "m").astype(np.int64))
        # Locate a matching seed edge in idx without consulting idx.y
        blind = idx.label_blind_view()
        lo = int(np.searchsorted(blind.t_sorted, t_seed_min))
        hi = int(np.searchsorted(blind.t_sorted, t_seed_min, "right"))
        cand_edges = blind.t_o[lo:hi]
        in_ref = cand_edges[
            np.isin(blind.s[cand_edges], list(node_set_ref))
            & np.isin(blind.d[cand_edges], list(node_set_ref))
        ]
        if in_ref.size == 0:
            continue
        seed_edge = int(in_ref[int(rng.integers(len(in_ref)))])
        dur = int(rng.choice(valid_durs))
        graph = common_bfs_candidate(blind, seed_edge, len(node_set_ref), dur, rng)
        if graph is None:
            continue
        positives.append(
            {
                "sg_id": f"ibm_pos_{k}",
                "label": 1,
                "pattern": a["pattern"],
                "t0": graph["t0"],
                "t1": graph["t1"],
                "nodes": graph["nodes"],
                "edges": graph["edges"],
            }
        )
    return positives


def sample_ibm_negatives(
    positives: list[dict[str, Any]],
    idx: EdgeIndex,
    rng: np.random.Generator,
    neg_per_pos: int = 20,
    window_durations: tuple[int, ...] = (360, 720),
    max_attempt_multiplier: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample negative IBM subgraphs using the exact same symmetric seed-window BFS rule."""
    if not positives or idx.t.size == 0:
        return [], {
            "ibm_negative_subgraphs": 0,
            "ibm_negative_sampling_attempts": 0,
            "ibm_negatives_rejected_contained_laundering": 0,
            "ibm_negatives_rejected_too_small": 0,
        }
    blind = idx.label_blind_view()
    n_total = len(blind.t)
    target = int(neg_per_pos) * len(positives)
    max_attempts = int(max_attempt_multiplier) * target
    attempts_made = 0
    rejected = 0
    too_small = 0
    negatives: list[dict[str, Any]] = []
    size_pool = [len(p["nodes"]) for p in positives]

    while len(negatives) < target and attempts_made < max_attempts:
        attempts_made += 1
        seed_edge = int(rng.integers(0, n_total))
        n_target = int(size_pool[int(rng.integers(len(size_pool)))])
        dur = int(rng.choice(window_durations))
        graph = common_bfs_candidate(blind, seed_edge, n_target, dur, rng)
        if graph is None or len(graph["nodes"]) < 2:
            too_small += 1
            continue
        # Label assigned strictly AFTER label-blind graph construction
        if idx.y[graph["edges"]].any():
            rejected += 1
            continue
        negatives.append(
            {
                "sg_id": f"ibm_neg_{len(negatives)}",
                "label": 0,
                "pattern": "NONE",
                "t0": graph["t0"],
                "t1": graph["t1"],
                "nodes": graph["nodes"],
                "edges": graph["edges"],
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


def index_documented_attempt_edges(
    attempts: list[dict[str, Any]],
    ibm: pd.DataFrame,
) -> dict[int, str]:
    """Map IBM transaction row index -> documented pattern name for post-hoc typology tagging."""
    if not attempts:
        return {}
    rec_rows: list[dict[str, Any]] = []
    for a in attempts:
        pat_name = str(a["pattern"])
        for r in a["rows"]:
            rec_rows.append(
                {
                    "timestamp": r[0],
                    "from_bank": r[1],
                    "from_acct": r[2],
                    "to_bank": r[3],
                    "to_acct": r[4],
                    "pattern": pat_name,
                }
            )
    pat_df = pd.DataFrame(rec_rows)
    pat_df["timestamp"] = pd.to_datetime(
        pat_df["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce"
    )
    pat_df["src"] = account_key(pat_df["from_bank"], pat_df["from_acct"])
    pat_df["dst"] = account_key(pat_df["to_bank"], pat_df["to_acct"])
    pat_lookup = pat_df.dropna(subset=["timestamp"]).drop_duplicates(["timestamp", "src", "dst"])

    laund_ibm = ibm.loc[ibm["is_laundering"] == 1, ["timestamp", "src", "dst"]].reset_index()
    merged = laund_ibm.merge(
        pat_lookup[["timestamp", "src", "dst", "pattern"]],
        on=["timestamp", "src", "dst"],
        how="inner",
    )
    return {int(row["index"]): str(row["pattern"]) for _, row in merged.iterrows()}


def sample_label_blind_ibm(
    idx: EdgeIndex,
    attempt_edge_patterns: dict[int, str] | None = None,
    n_candidates: int = 10000,
    positive_seed_probability: float = 0.25,
    node_sizes: tuple[int, ...] = (4, 6, 8, 10, 12),
    windows_min: tuple[int, ...] = (120, 240, 480),
    max_tries_multiplier: int = 15,
    seed: int = 2026,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """Phase 2 primary IBM sampler: one label-blind procedure + exact IPW weights + 20 features.

    Every candidate subgraph is grown via `common_bfs_candidate` on `idx.label_blind_view()`
    with target node count and window duration drawn from fixed distributions independent of
    transaction labels. Labels (`contains >= 1 laundering transaction`), documented attempt
    overlap, typology, 14 structural features, and 6 activity features are computed afterwards.

    Sampling probabilities & inverse-probability weights (IPW):
      - Uniform population seed distribution over all N transactions: q(e) = 1 / N.
      - Mixture proposal distribution with positive-enrichment alpha in [0, 1):
          p(e) = (1 - alpha)/N + alpha/M   if y_e == 1
          p(e) = (1 - alpha)/N             if y_e == 0
        where M = sum(y == 1).
      - Exact Horvitz-Thompson / importance weight: w(e) = q(e) / p(e) = (1 / N) / p(e).
    """
    if not (0.0 <= positive_seed_probability < 1.0):
        raise ValueError("positive_seed_probability must lie in [0.0, 1.0)")

    generator = np.random.default_rng(seed)
    y_all = np.asarray(idx.y, dtype=np.int64)
    n_total = int(y_all.size)
    pos_indices = np.flatnonzero(y_all == 1)
    m_pos = int(pos_indices.size)
    if m_pos == 0:
        raise RuntimeError("No annotated laundering transactions found in IBM data")

    alpha = float(positive_seed_probability)
    p_pos_seed = (1.0 - alpha) / n_total + (alpha / m_pos if alpha > 0.0 else 0.0)
    p_neg_seed = (1.0 - alpha) / n_total
    q_uniform = 1.0 / n_total

    blind = idx.label_blind_view()
    edge_pat_map = attempt_edge_patterns or {}

    rows: list[dict[str, Any]] = []
    nodes_by_id: dict[str, np.ndarray] = {}
    edges_by_id: dict[str, np.ndarray] = {}
    start_by_id: dict[str, int] = {}
    end_by_id: dict[str, int] = {}
    signatures: set[tuple[int, int, tuple[int, ...]]] = set()

    attempts_made = 0
    max_tries = max(int(max_tries_multiplier) * int(n_candidates), 1000)
    seed_mode_counts = {"uniform": 0, "positive_enriched": 0}

    while len(rows) < n_candidates and attempts_made < max_tries:
        attempts_made += 1
        if alpha > 0.0 and generator.random() < alpha:
            mode = "positive_enriched"
            seed_edge = int(generator.choice(pos_indices))
        else:
            mode = "uniform"
            seed_edge = int(generator.integers(0, n_total))

        n_target = int(generator.choice(node_sizes))
        duration = int(generator.choice(windows_min))
        graph = common_bfs_candidate(blind, seed_edge, n_target, duration, generator)
        if graph is None:
            continue

        signature = (graph["t0"], graph["t1"], tuple(sorted(graph["nodes"])))
        if signature in signatures:
            continue
        signatures.add(signature)

        edges_idx = np.unique(graph["edges"])
        sg_id = f"lb_{len(rows):05d}"

        # Exact seed inclusion probability and importance weight relative to uniform seeding
        seed_is_pos = int(y_all[seed_edge] == 1)
        samp_prob = p_pos_seed if seed_is_pos else p_neg_seed
        ipw = q_uniform / samp_prob

        # Post-hoc label and documented attempt / typology lookup
        laund_mask = y_all[edges_idx] == 1
        n_laund = int(laund_mask.sum())
        label = int(n_laund > 0)
        matched_patterns = [edge_pat_map[int(e)] for e in edges_idx if int(e) in edge_pat_map]
        attempt_overlap = int(len(matched_patterns) > 0)
        if matched_patterns:
            typology = Counter(matched_patterns).most_common(1)[0][0]
        elif label == 1:
            typology = "UNLISTED_LAUNDERING"
        else:
            typology = "NONE"

        nodes_by_id[sg_id] = np.asarray(sorted(graph["nodes"]), dtype=np.int64)
        edges_by_id[sg_id] = edges_idx
        start_by_id[sg_id] = int(graph["t0"])
        end_by_id[sg_id] = int(graph["t1"])

        s_feats = graph_features(
            graph["nodes"], zip(blind.s[edges_idx], blind.d[edges_idx], strict=False)
        )
        a_feats = transaction_activity_features(sg_id, edges_by_id, blind, start_by_id, end_by_id)
        a_feats_clean = {k: v for k, v in a_feats.items() if k != "sg_id"}

        rows.append(
            {
                "sg_id": sg_id,
                "dataset": "ibm_label_blind",
                "label": label,
                "n_laundering_tx": n_laund,
                "documented_attempt_overlap": attempt_overlap,
                "pattern": typology,
                "seed_edge": seed_edge,
                "seed_is_laundering": seed_is_pos,
                "seed_source": mode,
                "sampling_prob": samp_prob,
                "ipw_weight": ipw,
                "t_start": int(graph["t0"]),
                "t_end": int(graph["t1"]),
                "target_nodes": n_target,
                "window_minutes": duration,
                **s_feats,
                **a_feats_clean,
            }
        )
        seed_mode_counts[mode] += 1

    df = pd.DataFrame(rows)
    if df.empty or df["label"].nunique() < 2:
        raise RuntimeError("Label-blind sampler failed to collect both classes.")

    unweighted_prev = float(df["label"].mean())
    weighted_prev = float(np.average(df["label"].to_numpy(), weights=df["ipw_weight"].to_numpy()))
    info = {
        "n_candidates": len(df),
        "attempts_made": attempts_made,
        "positive_seed_probability": alpha,
        "seed_counts": seed_mode_counts,
        "unweighted_prevalence": round(unweighted_prev, 6),
        "ipw_estimated_population_prevalence": round(weighted_prev, 6),
        "positives": int(df["label"].sum()),
        "negatives": int((df["label"] == 0).sum()),
        "positives_overlapping_documented_attempts": int(
            df.loc[df["label"] == 1, "documented_attempt_overlap"].sum()
        ),
    }
    return df, nodes_by_id, edges_by_id, info


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
    """Compatibility wrapper for Extension 2 common-BFS sensitivity analysis."""
    df, nodes_by_id, edges_by_id, info = sample_label_blind_ibm(
        idx=idx,
        attempt_edge_patterns=None,
        n_candidates=n_candidates,
        positive_seed_probability=positive_seed_probability,
        node_sizes=node_sizes,
        windows_min=windows_min,
        max_tries_multiplier=20,
        seed=seed,
    )
    windows_by_id = {
        str(r["sg_id"]): (int(r["t_start"]), int(r["t_end"])) for _, r in df.iterrows()
    }
    return df, nodes_by_id, edges_by_id, windows_by_id, info
