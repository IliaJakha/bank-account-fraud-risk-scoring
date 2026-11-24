"""Unit tests for Phase 2 IPW/Kish n_eff, multi-fold splits, and fold-stratified tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aml_graph.data import IBM_COLS
from aml_graph.phase2 import (
    delong_roc_test,
    fold_stratified_bootstrap_summary,
    fold_stratified_paired_comparison,
    kish_effective_sample_size,
    paired_bootstrap_comparison,
    score_unweighted_and_ipw,
)
from aml_graph.sampling import EdgeIndex, sample_label_blind_ibm
from aml_graph.splits import account_component_group_splits, rolling_origin_splits


def test_label_blind_ipw_weights_and_posthoc_typology() -> None:
    base = pd.Timestamp("2022-09-01 00:00:00")
    rows = []
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
        assert int(df["t_start"].iloc[val_i].min()) > int(df["t_end"].iloc[fit_i].max()) + 90
        assert int(df["t_start"].iloc[test_i].min()) > int(df["t_end"].iloc[val_i].max()) + 90
        assert seen_tests.isdisjoint(test_i.tolist())
        seen_tests.update(test_i.tolist())
        ef = set().union(*(edge_map[df["sg_id"].iloc[k]].tolist() for k in fit_i))
        ev = set().union(*(edge_map[df["sg_id"].iloc[k]].tolist() for k in val_i))
        et = set().union(*(edge_map[df["sg_id"].iloc[k]].tolist() for k in test_i))
        assert ef.isdisjoint(ev) and ef.isdisjoint(et) and ev.isdisjoint(et)


def test_account_component_group_splits_keeps_connected_components_intact() -> None:
    rows = []
    node_map: dict[str, np.ndarray] = {}
    edge_map: dict[str, np.ndarray] = {}
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
    assert sorted(all_test.tolist()) == list(range(len(df)))

    for f in folds:
        nf = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in f["fit_idx"]))
        nv = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in f["val_idx"]))
        nt = set().union(*(node_map[df["sg_id"].iloc[k]].tolist() for k in f["test_idx"]))
        assert nf.isdisjoint(nv) and nf.isdisjoint(nt) and nv.isdisjoint(nt)


def test_kish_effective_sample_size_and_ipw_estimability_guard() -> None:
    assert kish_effective_sample_size(np.array([])) == 0.0
    assert math.isclose(kish_effective_sample_size(np.ones(50) * 3.5), 50.0, rel_tol=1e-12)
    # 5 heavy weights of 100 and 95 tiny weights of 0.01 -> n_eff ~ 5.0
    skewed_w = np.array([100.0] * 5 + [0.01] * 95)
    assert 5.0 <= kish_effective_sample_size(skewed_w) < 5.02

    # 40 positives (>=30 unweighted), but only 5 uniform positives -> kish_n_eff_pos < 30
    rng = np.random.default_rng(9)
    y = np.array([1] * 40 + [0] * 100)
    p = np.concatenate([rng.uniform(0.5, 0.9, 40), rng.uniform(0.1, 0.5, 100)])
    w = np.array([100.0] * 5 + [0.01] * 35 + [1.0] * 100)
    seeds = np.array(["uniform"] * 5 + ["positive_enriched"] * 35 + ["uniform"] * 100)

    res = score_unweighted_and_ipw(
        y, p, 0.5, w, seed_sources=seeds, min_test_positives=30, min_effective_positives=30.0
    )
    assert res["status"] == "ok"
    assert res["n_uniform_positives"] == 5
    assert res["kish_n_eff_positives"] < 30.0
    assert res["ipw_status"] == "not_estimable_lt_30_eff_positives"
    assert math.isnan(res["ipw_pr_auc"]) and math.isnan(res["ipw_roc_auc"])


def test_fold_stratified_paired_test_prevents_prevalence_shift_inversion() -> None:
    """Demonstrate why cross-fold raw probability pooling fails under prevalence shift."""
    # Fold 0: low prevalence (20%), high score offset [0.70..0.95] -> AUC(A) > AUC(B)
    # Fold 1: high prevalence (75%), lower offset [0.10..0.40] -> AUC(A) > AUC(B)
    rng = np.random.default_rng(2026)
    y0 = np.array([1] * 30 + [0] * 120)
    pa0 = np.concatenate([rng.uniform(0.82, 0.95, 30), rng.uniform(0.70, 0.86, 120)])
    pb0 = rng.uniform(0.70, 0.90, 150)

    y1 = np.array([1] * 120 + [0] * 40)
    pa1 = np.concatenate([rng.uniform(0.22, 0.40, 120), rng.uniform(0.10, 0.26, 40)])
    pb1 = rng.uniform(0.10, 0.35, 160)

    # Naive pooling inverts AUC(pa) below 0.5 because Fold 0 negatives > Fold 1 positives!
    pooled_auc = roc_auc_score(np.concatenate([y0, y1]), np.concatenate([pa0, pa1]))
    assert pooled_auc < 0.40

    # Fold-stratified paired test computes deltas within each fold and recovers true AUC > 0.85!
    comb, per_fold = fold_stratified_paired_comparison(
        [(y0, pa0, pb0), (y1, pa1, pb1)], n_boot=300, seed=0
    )
    assert len(per_fold) == 2
    assert comb["auc_a_mean"] > 0.85
    assert comb["delta_auc"] > 0.25
    assert comb["delong_p"] < 1e-6
    assert comb["delta_ap"] > 0.15

    # Also verify clipped bootstrap CIs lie strictly inside valid bounds
    boot = fold_stratified_bootstrap_summary([(y0, pa0, None), (y1, pa1, None)], n_boot=200, seed=0)
    assert 0.0 <= boot["ap_ci95_lo"] <= boot["ap_mean"] <= boot["ap_ci95_hi"] <= 1.0
    assert 0.0 <= boot["roc_auc_ci95_lo"] <= boot["roc_auc_mean"] <= boot["roc_auc_ci95_hi"] <= 1.0
    assert 0.0 <= boot["lift_ci95_lo"] <= boot["lift_mean"] <= boot["lift_ci95_hi"]

    dl_same = delong_roc_test(y0, pa0, pa0)
    assert math.isclose(dl_same["delta_auc"], 0.0, abs_tol=1e-12)
    assert math.isclose(dl_same["delong_p"], 1.0, abs_tol=1e-12)
    single = paired_bootstrap_comparison(y0, pa0, pb0, n_boot=100, seed=0)
    assert single["delta_auc"] > 0.20
