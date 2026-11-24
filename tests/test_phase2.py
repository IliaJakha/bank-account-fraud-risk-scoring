"""Unit tests for Phase 2 label-blind IPW sampling, multi-fold splits, and statistical tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from aml_graph.data import IBM_COLS
from aml_graph.phase2 import (
    delong_roc_test,
    paired_bootstrap_comparison,
    score_unweighted_and_ipw,
)
from aml_graph.sampling import EdgeIndex, sample_label_blind_ibm
from aml_graph.splits import account_component_group_splits, rolling_origin_splits


def test_label_blind_ipw_weights_and_posthoc_typology() -> None:
    base = pd.Timestamp("2022-09-01 00:00:00")
    rows = []
    # 40 transactions across 10 disjoint triads; 4 positive triads, 6 licit triads
    for k in range(10):
        is_pos = int(k < 4)
        for step, (a, b) in enumerate(((0, 1), (1, 2), (2, 0), (0, 2))):
            ts = (base + pd.Timedelta(minutes=k * 120 + step * 10)).strftime("%Y/%m/%d %H:%M")
            rows.append(
                [
                    ts,
                    "1",
                    f"N{k}_{a}",
                    "1",
                    f"N{k}_{b}",
                    100.0,
                    "US Dollar",
                    100.0,
                    "US Dollar",
                    "ACH",
                    is_pos,
                ]
            )
    ibm = pd.DataFrame(rows, columns=IBM_COLS)
    ibm["timestamp"] = pd.to_datetime(ibm["timestamp"], format="%Y/%m/%d %H:%M")
    ibm["is_laundering"] = ibm["is_laundering"].astype(int)
    ibm["src"] = ibm["from_bank"] + "_" + ibm["from_acct"]
    ibm["dst"] = ibm["to_bank"] + "_" + ibm["to_acct"]
    codes, accounts = pd.factorize(pd.concat([ibm["src"], ibm["dst"]]))
    ibm["s"] = codes[: len(ibm)]
    ibm["d"] = codes[len(ibm) :]

    idx = EdgeIndex(ibm, len(accounts))
    # Verify exact Horvitz-Thompson proposal normalization: sum_e p(e) == 1 and p(e)*w(e) == 1/N
    n_total = len(ibm)
    m_pos = int(ibm["is_laundering"].sum())
    alpha = 0.40
    p_pos = (1.0 - alpha) / n_total + alpha / m_pos
    p_neg = (1.0 - alpha) / n_total
    assert math.isclose(m_pos * p_pos + (n_total - m_pos) * p_neg, 1.0, rel_tol=1e-12)

    df, _, _, info = sample_label_blind_ibm(
        idx=idx,
        attempt_edge_patterns={0: "CYCLE", 1: "CYCLE"},
        n_candidates=25,
        positive_seed_probability=alpha,
        node_sizes=(3,),
        windows_min=(60,),
        seed=123,
    )
    assert len(df) > 0
    assert "ipw_weight" in df.columns and "sampling_prob" in df.columns
    for _, r in df.iterrows():
        expected_p = p_pos if int(r["seed_is_laundering"]) == 1 else p_neg
        assert math.isclose(float(r["sampling_prob"]), expected_p, rel_tol=1e-12)
        assert math.isclose(float(r["ipw_weight"]), (1.0 / n_total) / expected_p, rel_tol=1e-12)
    assert info["n_candidates"] == len(df)


def test_rolling_origin_splits_three_non_overlapping_embargoed_folds() -> None:
    rows = []
    edge_map: dict[str, np.ndarray] = {}
    for i in range(120):
        sg = f"ro_{i:03d}"
        t0 = i * 60
        t1 = t0 + 30
        rows.append({"sg_id": sg, "label": int(i % 4 == 0), "t_start": t0, "t_end": t1})
        edge_map[sg] = np.array([i * 2, i * 2 + 1], dtype=np.int64)
    df = pd.DataFrame(rows)

    folds = rolling_origin_splits(
        df,
        edge_map,
        n_folds=3,
        embargo_minutes=90,
        first_train_quantile=0.25,
        val_quantile_width=0.15,
    )
    assert len(folds) == 3

    seen_tests: set[int] = set()
    for f in folds:
        fit_i, val_i, test_i = f["fit_idx"], f["val_idx"], f["test_idx"]
        assert len(fit_i) > 0 and len(val_i) > 0 and len(test_i) > 0
        # Embargo invariants
        assert int(df["t_start"].iloc[val_i].min()) > int(df["t_end"].iloc[fit_i].max()) + 90
        assert int(df["t_start"].iloc[test_i].min()) > int(df["t_end"].iloc[val_i].max()) + 90
        # Non-overlapping test periods across folds
        assert seen_tests.isdisjoint(test_i.tolist())
        seen_tests.update(test_i.tolist())
        # Transaction-edge disjointness
        ef = set().union(*(edge_map[df["sg_id"].iloc[k]].tolist() for k in fit_i))
        ev = set().union(*(edge_map[df["sg_id"].iloc[k]].tolist() for k in val_i))
        et = set().union(*(edge_map[df["sg_id"].iloc[k]].tolist() for k in test_i))
        assert ef.isdisjoint(ev) and ef.isdisjoint(et) and ev.isdisjoint(et)


def test_account_component_group_splits_keeps_connected_components_intact() -> None:
    rows = []
    node_map: dict[str, np.ndarray] = {}
    edge_map: dict[str, np.ndarray] = {}
    # 12 components of 3 subgraphs each; subgraphs inside a component chain accounts
    for comp in range(12):
        base_acc = comp * 100
        for step in range(3):
            idx_row = len(rows)
            sg = f"ag_{idx_row:03d}"
            rows.append(
                {"sg_id": sg, "label": int((comp + step) % 3 == 0), "t_start": idx_row * 10}
            )
            node_map[sg] = np.array([base_acc + step, base_acc + step + 1], dtype=np.int64)
            edge_map[sg] = np.array([idx_row], dtype=np.int64)
    df = pd.DataFrame(rows)

    folds = account_component_group_splits(df, node_map, edge_map, n_folds=4)
    assert len(folds) == 4
    all_test = np.concatenate([f["test_idx"] for f in folds])
    # 100% retention and exact partition of indices across the 4 test folds
    assert sorted(all_test.tolist()) == list(range(len(df)))

    for f in folds:
        nf = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in f["fit_idx"]))
        nv = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in f["val_idx"]))
        nt = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in f["test_idx"]))
        assert nf.isdisjoint(nv) and nf.isdisjoint(nt) and nv.isdisjoint(nt)


def test_min_30_positives_guard_and_delong_paired_tests() -> None:
    # 29 positives must be marked not_estimable_lt_30_positives under default min_test_positives=30
    y_small = np.array([1] * 29 + [0] * 100)
    p_small = np.linspace(0.1, 0.9, len(y_small))
    w_small = np.ones_like(p_small)
    res_small = score_unweighted_and_ipw(y_small, p_small, 0.5, w_small, min_test_positives=30)
    assert res_small["status"] == "not_estimable_lt_30_positives"
    assert math.isnan(res_small["pr_auc"])

    # 30 positives is estimable (use noisy overlapping scores so DeLong covariance > 0)
    rng = np.random.default_rng(42)
    y_ok = np.array([1] * 30 + [0] * 100)
    p_a = np.concatenate([rng.uniform(0.45, 0.95, 30), rng.uniform(0.05, 0.65, 100)])
    p_b = rng.uniform(0.2, 0.8, 130)
    res_ok = score_unweighted_and_ipw(y_ok, p_a, 0.5, np.ones_like(p_a), min_test_positives=30)
    assert res_ok["status"] == "ok" and res_ok["pr_auc"] > 0.6

    dl_same = delong_roc_test(y_ok, p_a, p_a)
    assert math.isclose(dl_same["delta_auc"], 0.0, abs_tol=1e-12)
    assert math.isclose(dl_same["delong_p"], 1.0, abs_tol=1e-12)

    paired = paired_bootstrap_comparison(
        y_ok, p_a, p_b, weights=np.ones_like(p_a), n_boot=200, seed=0
    )
    assert paired["delta_ap"] > 0.2
    assert paired["delong_p"] < 0.001
